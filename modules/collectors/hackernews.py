"""Hacker News collector: recent, well-voted stories matching the configured queries (Algolia API)."""
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import requests
from bs4 import BeautifulSoup

from modules.models import Article

API_URL = "https://hn.algolia.com/api/v1/search_by_date"


def parse_hits(hits: list[dict], subject: str) -> list[Article]:
    articles = []
    for hit in hits:
        title = (hit.get("title") or "").strip()
        if not title:
            continue
        object_id = hit.get("objectID")
        url = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
        text = BeautifulSoup(hit.get("story_text") or "", "html.parser").get_text(" ", strip=True)
        stats = f"{hit.get('points', 0)} points, {hit.get('num_comments') or 0} comments"
        articles.append(Article(
            name=title,
            subject=subject,
            author=hit.get("author"),
            published_at=datetime.fromtimestamp(hit["created_at_i"]).date() if hit.get("created_at_i") else date.today(),
            content=text[:2000],
            signal=stats,
            url=url,
            origin="hackernews",
        ))
    return articles


def _search(query: str, max_age_days: int, min_points: int) -> list[dict]:
    since = int(time.time()) - max_age_days * 86400
    resp = requests.get(API_URL, params={
        "tags": "story",
        "query": query,
        "numericFilters": f"created_at_i>{since},points>{min_points}",
        "hitsPerPage": 20,
    }, timeout=15)
    resp.raise_for_status()
    return resp.json()["hits"]


def collect(queries: list[str], subject: str, max_age_days: int = 2, min_points: int = 20) -> list[Article]:
    """Articles for every query, de-duplicated by URL. A failing query is skipped, not fatal."""
    if not queries:
        return []

    def one(query: str) -> list[Article]:
        try:
            return parse_hits(_search(query, max_age_days, min_points), subject)
        except (requests.RequestException, KeyError, ValueError) as e:
            print(f"[hackernews] query '{query}' failed: {e}")
            return []

    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        results = list(pool.map(one, queries))
    by_url = {a.url: a for batch in results for a in batch}
    return list(by_url.values())
