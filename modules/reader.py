"""Reader: fetches the full text of shortlisted articles so ranking and summarizing see more than a teaser."""
from concurrent.futures import ThreadPoolExecutor

import requests
import trafilatura

from modules.models import Article

FULL_TEXT_MIN_CHARS = 1500
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
}


def fetch_text(url: str, timeout: int = 10) -> str | None:
    """Main article text of a page, or None when it can't be fetched or has no extractable text."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout, allow_redirects=True)
        if resp.status_code != 200 or "html" not in resp.headers.get("content-type", "html"):
            return None
        return trafilatura.extract(resp.text, include_comments=False, include_tables=False) or None
    except Exception:  # network errors and parser errors both mean "keep the teaser"
        return None


def read_article(article: Article, max_chars: int, fetch=fetch_text) -> Article:
    """read_status: "full" (>= FULL_TEXT_MIN_CHARS fetched), "partial" (less), "snippet" (fetch failed, teaser kept)."""
    text = fetch(article.url)
    if not text or len(text) <= len(article.content):
        return article.model_copy(update={"read_status": "snippet"})
    status = "full" if len(text) >= FULL_TEXT_MIN_CHARS else "partial"
    return article.model_copy(update={"content": text[:max_chars], "read_status": status})


def read_articles(articles: list[Article], max_chars: int = 6000, max_workers: int = 8) -> list[Article]:
    if not articles:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda a: read_article(a, max_chars), articles))
