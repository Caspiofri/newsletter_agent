"""Medium collector: polls public tag RSS feeds (medium.com/feed/tag/<tag>). Medium has no API.

Tag feeds carry only a teaser, and Medium answers 403 to page fetches, but each author's and publication's own feed
carries the full text of its non-paywalled posts, so the teasers are upgraded from those feeds."""
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup

from modules.models import Article
from modules.reader import FULL_TEXT_MIN_CHARS

FEED_URL = "https://medium.com/feed/tag/{tag}"
FULL_TEXT_MAX_CHARS = 6000  # the cap the reader applies to fetched pages
_CONTENT = "{http://purl.org/rss/1.0/modules/content/}encoded"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; newsletter-agent/1.0)"}


def _clean_url(url: str) -> str:
    """Drop the ?source=rss tracking query."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


def parse_feed(xml_text: str, subject: str, max_age_days: int = 2, today: date | None = None) -> list[Article]:
    today = today or date.today()
    oldest = today - timedelta(days=max_age_days)
    dc = "{http://purl.org/dc/elements/1.1/}"
    articles = []
    for item in ET.fromstring(xml_text).findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = item.findtext("pubDate")
        if not title or not link or not pub:
            continue
        published_at = parsedate_to_datetime(pub).date()
        if published_at < oldest:
            continue
        teaser = BeautifulSoup(item.findtext("description") or "", "html.parser").get_text(" ", strip=True)
        articles.append(Article(
            name=title,
            subject=subject,
            author=item.findtext(f"{dc}creator"),
            published_at=published_at,
            content=teaser[:2000],
            url=_clean_url(link),
            origin="medium",
        ))
    return articles


def feed_url_for(article_url: str) -> str:
    """The feed of the author (medium.com/@user/...), publication (medium.com/<pub>/...) or custom-domain
    publication that published an article."""
    parts = urlsplit(article_url)
    if parts.netloc == "medium.com":
        return f"https://medium.com/feed/{parts.path.strip('/').split('/')[0]}"
    return f"https://{parts.netloc}/feed"


def parse_full_texts(xml_text: str) -> dict[str, str]:
    """Clean URL -> plain text of every item that carries its full body (paywalled posts carry none)."""
    texts = {}
    for item in ET.fromstring(xml_text).findall("./channel/item"):
        body = item.findtext(_CONTENT)
        link = (item.findtext("link") or "").strip()
        if body and link:
            texts[_clean_url(link)] = BeautifulSoup(body, "html.parser").get_text("\n", strip=True)
    return texts


def _fetch(url: str) -> str:
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def _fetch_tag(tag: str) -> str:
    return _fetch(FEED_URL.format(tag=tag))


def with_full_text(articles: list[Article], fetch=_fetch) -> list[Article]:
    """Replace teasers with the full text from each author's or publication's feed, one fetch per feed. Articles
    the feeds don't carry in full, or whose feed fails, keep their teaser."""
    feeds = sorted({feed_url_for(a.url) for a in articles})

    def one(feed: str) -> dict[str, str]:
        try:
            return parse_full_texts(fetch(feed))
        except (requests.RequestException, ET.ParseError) as e:
            print(f"[medium] feed {feed} failed: {e}")
            return {}

    texts = {}
    with ThreadPoolExecutor(max_workers=4) as pool:  # Medium answers 429 to bursts of feed requests
        for found in pool.map(one, feeds):
            texts.update(found)
    out = []
    for a in articles:
        text = texts.get(a.url, "")
        if len(text) > len(a.content):
            status = "full" if len(text) >= FULL_TEXT_MIN_CHARS else "partial"
            a = a.model_copy(update={"content": text[:FULL_TEXT_MAX_CHARS], "read_status": status})
        out.append(a)
    return out


def collect(tags: list[str], subject: str, max_age_days: int = 2) -> list[Article]:
    """Articles from every tag feed, de-duplicated by URL. A failing feed is skipped, not fatal."""
    if not tags:
        return []

    def one(tag: str) -> list[Article]:
        try:
            return parse_feed(_fetch_tag(tag), subject, max_age_days)
        except (requests.RequestException, ET.ParseError) as e:
            print(f"[medium] tag '{tag}' failed: {e}")
            return []

    with ThreadPoolExecutor(max_workers=len(tags)) as pool:
        results = list(pool.map(one, tags))
    by_url = {a.url: a for batch in results for a in batch}
    articles = with_full_text(list(by_url.values()))
    print(f"[medium] full text for {sum(a.read_status != 'snippet' for a in articles)} of {len(articles)} articles")
    return articles
