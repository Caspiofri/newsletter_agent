import hashlib
from datetime import date, timedelta

import numpy as np
from bs4 import BeautifulSoup
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langsmith import traceable

from config.settings import CHROMA_PATH, DEDUPE_NEAR_MISS_MIN
from modules.models import Article

_embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
_client = PersistentClient(path=str(CHROMA_PATH))

SIMILARITY_THRESHOLD = 0.82  # cosine similarity; distance = 1 - similarity
DEDUPE_DAYS = 7  # embedding window; an already-sent URL is dropped for all of RETENTION_DAYS
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
        metadatas=[{"date": today_int, "name": a.name, "url": a.url or ""} for a in articles],
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


def _sent_on(date_int) -> str:
    """Stored YYYYMMDD int -> "YYYY-MM-DD"; "" for records without a date."""
    text = str(date_int or "")
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}" if len(text) == 8 else ""


def _dedupe_trace_inputs(inputs: dict) -> dict:
    """Show titles and the window in the trace instead of Article objects."""
    return {"queries": [a.name for a in inputs.get("articles", [])],
            "window_days": inputs.get("days", DEDUPE_DAYS)}


def _dedupe_trace_outputs(result) -> dict:
    """Show the lookup as scored documents: the matched sent title with its similarity."""
    if not isinstance(result, tuple) or len(result) != 2:  # langsmith passes None when the run failed
        return {"documents": []}
    _, matches = result
    return {"documents": [{
        "page_content": m["matched_title"],
        "metadata": {"query_title": m["title"], "similarity": m["similarity"],
                     "matched_sent_on": m["matched_sent_on"], "dropped": m["dropped"]},
    } for m in matches]}


def _drop_sent_urls(col, articles: list[Article]) -> tuple[list[Article], list[dict]]:
    """Survivors plus one match per article whose exact URL was already sent, at any age still in the store.

    Records stored before the URL was kept have none, so they only match by embedding.
    """
    urls = sorted({a.url for a in articles if a.url})
    if not urls:
        return articles, []
    try:
        sent = col.get(where={"url": {"$in": urls}}, include=["metadatas"])["metadatas"]
    except Exception:
        return articles, []
    latest = {}
    for meta in sent:
        if meta and (meta.get("date") or 0) >= (latest.get(meta.get("url"), {}).get("date") or 0):
            latest[meta.get("url")] = meta
    unseen, matches = [], []
    for article in articles:
        meta = latest.get(article.url) if article.url else None
        if meta is None:
            unseen.append(article)
            continue
        matches.append({"title": article.name, "url": article.url, "similarity": 1.0,
                        "matched_title": meta.get("name", ""), "matched_sent_on": _sent_on(meta.get("date")),
                        "dropped": True, "matched_by": "url"})
    return unseen, matches


@traceable(run_type="retriever", name="dedupe_lookup",
           process_inputs=_dedupe_trace_inputs, process_outputs=_dedupe_trace_outputs)
def filter_seen_detailed(articles: list[Article], profile: str,
                         days: int = DEDUPE_DAYS) -> tuple[list[Article], list[dict]]:
    """Survivors plus one record per article already sent: by exact URL, or a near-miss or duplicate by embedding."""
    if not articles:
        return articles, []

    articles = _dedup_batch(articles)
    if not articles:
        return articles, []

    col = _collection(profile)
    if col.count() == 0:
        return articles, []

    articles, matches = _drop_sent_urls(col, articles)
    if not articles:
        return articles, matches

    cutoff = int((date.today() - timedelta(days=days)).strftime("%Y%m%d"))
    distance_threshold = 1.0 - SIMILARITY_THRESHOLD

    try:
        results = col.query(
            query_texts=[_article_text(a) for a in articles],
            n_results=1,
            where={"date": {"$gte": cutoff}},
            include=["distances", "metadatas"],
        )
    except Exception:
        return articles, matches

    unseen = []
    for article, distances, metadatas in zip(articles, results["distances"], results["metadatas"]):
        min_dist = min(distances) if distances else 1.0
        if min_dist >= distance_threshold:
            unseen.append(article)
        if distances:
            similarity = round(1.0 - min_dist, 3)
            if similarity >= DEDUPE_NEAR_MISS_MIN:
                meta = metadatas[distances.index(min_dist)] or {}
                matches.append({
                    "title": article.name,
                    "url": article.url,
                    "similarity": similarity,
                    "matched_title": meta.get("name", ""),
                    "matched_sent_on": _sent_on(meta.get("date")),
                    "dropped": min_dist < distance_threshold,
                    "matched_by": "embedding",
                })

    return unseen, matches


def filter_seen(articles: list[Article], profile: str, days: int = DEDUPE_DAYS) -> list[Article]:
    return filter_seen_detailed(articles, profile, days)[0]
