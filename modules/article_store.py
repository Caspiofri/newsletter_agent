import hashlib
from datetime import date, timedelta

import numpy as np
from bs4 import BeautifulSoup
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config.settings import CHROMA_PATH
from modules.models import Article

_embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
_client = PersistentClient(path=str(CHROMA_PATH))

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


def upgrade_bad_links(articles: list[Article]) -> list[Article]:
    """Give a story with a bad link the URL of the same story from another newsletter, when that one is good.

    "Same story" = embedding similarity >= SIMILARITY_THRESHOLD, as in the dedupe step.
    """
    bad = [i for i, a in enumerate(articles) if a.link_status != "ok"]
    good = [i for i, a in enumerate(articles) if a.link_status == "ok"]
    if not bad or not good:
        return articles
    emb = np.array(_embed_fn([_article_text(a) for a in articles]), dtype=float)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    sims = emb[bad] @ emb[good].T
    out = list(articles)
    for row, i in enumerate(bad):
        k = int(np.argmax(sims[row]))
        if sims[row][k] >= SIMILARITY_THRESHOLD:
            out[i] = articles[i].model_copy(update={"url": articles[good[k]].url, "link_status": "ok"})
    return out


def _dedup_batch(articles: list[Article]) -> list[Article]:
    """Drop near-duplicate articles within the current batch (same story, different headlines)."""
    if len(articles) <= 1:
        return articles
    embeddings = _embed_fn([_article_text(a) for a in articles])
    keep = [True] * len(articles)
    for i in range(len(articles)):
        if not keep[i]:
            continue
        ei = embeddings[i]
        norm_i = sum(x * x for x in ei) ** 0.5
        for j in range(i + 1, len(articles)):
            if not keep[j]:
                continue
            ej = embeddings[j]
            norm_j = sum(x * x for x in ej) ** 0.5
            dot = sum(x * y for x, y in zip(ei, ej))
            sim = dot / (norm_i * norm_j) if norm_i and norm_j else 0.0
            if sim >= SIMILARITY_THRESHOLD:
                keep[j] = False
    return [a for a, k in zip(articles, keep) if k]


def filter_seen(articles: list[Article], profile: str, days: int = DEDUPE_DAYS) -> list[Article]:
    if not articles:
        return articles

    articles = _dedup_batch(articles)
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
