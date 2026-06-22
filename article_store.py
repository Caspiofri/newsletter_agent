import hashlib
from datetime import date, timedelta

from bs4 import BeautifulSoup
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from models import Article

_embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
_client = PersistentClient(path="./chroma_db")

SIMILARITY_THRESHOLD = 0.82  # cosine similarity; distance = 1 - similarity
DEDUPE_DAYS = 3
RETENTION_DAYS = 14


def _collection(profile: str):
    return _client.get_or_create_collection(
        name=profile,
        embedding_function=_embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def _article_text(article: Article) -> str:
    plain = BeautifulSoup(article.content, "html.parser").get_text()[:500]
    return f"{article.name}. {plain}"


def store_articles(articles: list[Article], profile: str) -> None:
    if not articles:
        return
    col = _collection(profile)
    today_int = int(date.today().strftime("%Y%m%d"))
    col.upsert(
        ids=[hashlib.md5(f"{today_int}_{a.name}".encode()).hexdigest() for a in articles],
        documents=[_article_text(a) for a in articles],
        metadatas=[{"date": today_int, "name": a.name} for a in articles],
    )


def purge_old(profile: str, days: int = RETENTION_DAYS) -> int:
    col = _collection(profile)
    cutoff = int((date.today() - timedelta(days=days)).strftime("%Y%m%d"))
    results = col.get(where={"date": {"$lt": cutoff}}, include=[])
    ids = results["ids"]
    if ids:
        col.delete(ids=ids)
    return len(ids)


def filter_seen(articles: list[Article], profile: str, days: int = DEDUPE_DAYS) -> list[Article]:
    if not articles:
        return articles

    col = _collection(profile)
    if col.count() == 0:
        return articles

    cutoff = int((date.today() - timedelta(days=days)).strftime("%Y%m%d"))
    distance_threshold = 1.0 - SIMILARITY_THRESHOLD

    try:
        results = col.query(
            query_texts=[_article_text(a) for a in articles],
            n_results=1,
            where={"date": {"$gte": cutoff}},
            include=["distances"],
        )
    except Exception:
        return articles

    unseen = []
    for article, distances in zip(articles, results["distances"]):
        min_dist = min(distances) if distances else 1.0
        if min_dist >= distance_threshold:
            unseen.append(article)

    return unseen
