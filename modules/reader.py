"""Reader: fetches the full text of shortlisted articles so ranking and summarizing see more than a teaser."""
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests
import trafilatura

from modules.models import Article

FULL_TEXT_MIN_CHARS = 1500
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
}
# where the abstract ends on an arxiv.org/abs page (the rest is site chrome)
_ARXIV_CUT_MARKERS = ("Comments:", "Subjects:", "Cite as:", "Submission history", "Bibliographic", "Bookmark",
                      "Additional Features", "arXivLabs")
_ARXIV_CUT_RE = re.compile(r"^[ \t]*(?:%s)" % "|".join(map(re.escape, _ARXIV_CUT_MARKERS)), re.M)


def clean_page_text(final_url: str, text: str) -> str:
    """For an arxiv.org/abs page keep the title, submission date and abstract; anything else comes back unchanged."""
    try:
        parsed = urlparse(final_url)
        if parsed.hostname not in ("arxiv.org", "www.arxiv.org") or not parsed.path.startswith("/abs/"):
            return text
        start = text.find("Abstract:")
        if start < 0:
            return text
        abstract = text[start:]
        # markers count only at a line start, after the abstract's own line (the words also occur in running text)
        first, _, rest = abstract.partition("\n")
        cut = _ARXIV_CUT_RE.search(rest)
        abstract = (first + "\n" + rest[:cut.start()] if cut else abstract).strip()
        head = text[:start]
        title = re.search(r"^\s*Title:.*$", head, re.M)
        submitted = re.search(r"^\s*\[Submitted on[^\n]*$", head, re.M)
        lines = [m.group(0).strip() for m in (title, submitted) if m] + [abstract]
        # without a title or submitted line only the abstract was found: not a page we understand, keep the raw text
        return "\n".join(lines) if len(lines) > 1 else text
    except Exception:  # cleaning is best effort; the raw text is always usable
        return text


def fetch_text(url: str, timeout: int = 10) -> str | None:
    """Main article text of a page, or None when it can't be fetched or has no extractable text."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout, allow_redirects=True)
        if resp.status_code != 200 or "html" not in resp.headers.get("content-type", "html"):
            return None
        text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        return clean_page_text(resp.url, text) if text else None
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
