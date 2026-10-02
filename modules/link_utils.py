import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup

MAX_CANDIDATES = 120
CONTEXT_CHARS = 300

_JUNK_TEXT = re.compile(
    r"unsubscribe|manage (your )?(preferences|subscription)|view (this )?(email )?in (your )?browser|"
    r"privacy policy|terms of (service|use)|update (your )?preferences|forward to a friend|"
    r"upgrade to paid|subscribe here|sign up here|^like$|^comment$|^restack$|^share$",
    re.IGNORECASE,
)
_JUNK_HREF = re.compile(
    r"unsubscribe|/subscribe|/signup|/sign-up|preferences|mailto:|tel:|"
    r"(twitter|x|facebook|linkedin|instagram|youtube)\.com/?$|"
    r"list-manage\.com|/action/disable",
    re.IGNORECASE,
)
_DEAD_STATUSES = {404, 410}


_CL0 = re.compile(r"^https?://[^/]*tldrnewsletter\.com/CL0/(https?[:%][^/]+)/", re.IGNORECASE)


def unwrap_tracking(url: str) -> str:
    """Unwrap TLDR-style click-tracking links (.../CL0/<percent-encoded target>/...) to the real target."""
    m = _CL0.match(url)
    return unquote(m.group(1)) if m else url


def extract_candidate_links(html: str) -> list[dict]:
    """Real <a href> links found in the email HTML, with anchor text and nearby context.

    The LLM picks from this list by id, so it can never return a URL that is not in the email.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    seen: set[str] = set()
    candidates: list[dict] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if urlparse(href).scheme not in ("http", "https") or href in seen:
            continue
        text = a.get_text(" ", strip=True)
        if _JUNK_HREF.search(href) or _JUNK_TEXT.search(text):
            continue
        parent = a.find_parent(["td", "p", "div", "li"]) or a
        context = parent.get_text(" ", strip=True)[:CONTEXT_CHARS]
        if not text and not context:
            continue
        seen.add(href)
        candidates.append({"id": len(candidates), "text": text[:150], "context": context, "url": unwrap_tracking(href)})
        if len(candidates) >= MAX_CANDIDATES:
            break
    return candidates


def email_plain_text(html: str, limit: int = 6000) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)[:limit]


def check_link(url: str, timeout: int = 8) -> str:
    """Classify a link: "dead" (404/410), "unverified" (Cloudflare bot challenge, so we cannot tell), else "ok".

    Other 403s and network errors count as "ok": many sites block scripts but work in a browser.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; newsletter-agent/1.0)"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
        resp.close()
    except requests.RequestException:
        return "ok"
    if resp.status_code in _DEAD_STATUSES:
        return "dead"
    if resp.headers.get("cf-mitigated", "").lower() == "challenge":
        return "unverified"
    return "ok"


def classify_links(urls: list[str], max_workers: int = 16) -> dict[str, str]:
    """Map each url to its check_link status."""
    unique = list(dict.fromkeys(urls))
    if not unique:
        return {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(zip(unique, pool.map(check_link, unique)))
