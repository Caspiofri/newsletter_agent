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


_TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid", "_hsenc", "_hsmi", "ref", "source"}
# Links that hide the real article address behind a per-email redirect id: (host, path prefix).
_REDIRECTORS = (("substack.com", "/redirect/"), ("www.substack.com", "/redirect/"))


def _is_redirector(url: str) -> bool:
    p = urlparse(url)
    return any(p.hostname == host and p.path.startswith(prefix) for host, prefix in _REDIRECTORS)


def _strip_tracking(url: str) -> str:
    """Drop the fragment and tracking query parameters (utm_*, fbclid, ...); other parameters keep their order."""
    p = urlparse(unwrap_tracking(url))
    kept = [kv for kv in p.query.split("&")
            if kv and not (kv.split("=", 1)[0].lower().startswith("utm_") or kv.split("=", 1)[0].lower() in _TRACKING_PARAMS)]
    return p._replace(query="&".join(kept), fragment="").geturl()


def canonical_url(original: str, final_url: str) -> str:
    """The one address an article should be known by.

    A known redirector (e.g. substack.com/redirect/<per-email id>) is replaced by the address it led to;
    everything else keeps its own URL. Tracking parameters and the fragment are stripped either way.
    Never raises: on any problem the original comes back.
    """
    try:
        if final_url and _is_redirector(original) and urlparse(final_url).hostname != urlparse(original).hostname:
            return _strip_tracking(final_url)
        return _strip_tracking(original)
    except Exception:
        return original


def resolve_link(url: str, timeout: int = 8) -> tuple[str, str]:
    """Follow redirects once and return (status, final_url).

    status: "dead" (404/410), "unverified" (Cloudflare bot challenge, so we cannot tell), else "ok".
    Other 403s and network errors count as "ok": many sites block scripts but work in a browser.
    final_url is where the redirects ended, or url itself when the request failed.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; newsletter-agent/1.0)"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
        resp.close()
    except requests.RequestException:
        return "ok", url
    final = getattr(resp, "url", None) or url
    if resp.status_code in _DEAD_STATUSES:
        return "dead", final
    if resp.headers.get("cf-mitigated", "").lower() == "challenge":
        return "unverified", final
    return "ok", final


def check_link(url: str, timeout: int = 8) -> str:
    """Classify a link: "dead", "unverified" or "ok" (see resolve_link)."""
    return resolve_link(url, timeout)[0]


def resolve_links(urls: list[str], max_workers: int = 16) -> dict[str, tuple[str, str]]:
    """Map each url to its resolve_link (status, final_url)."""
    unique = list(dict.fromkeys(urls))
    if not unique:
        return {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(zip(unique, pool.map(resolve_link, unique)))


def classify_links(urls: list[str], max_workers: int = 16) -> dict[str, str]:
    """Map each url to its check_link status."""
    unique = list(dict.fromkeys(urls))
    if not unique:
        return {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(zip(unique, pool.map(check_link, unique)))
