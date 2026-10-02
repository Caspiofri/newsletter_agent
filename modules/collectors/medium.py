"""Medium collector: polls public tag RSS feeds (medium.com/feed/tag/<tag>). Medium has no API."""
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup

from modules.models import Article

FEED_URL = "https://medium.com/feed/tag/{tag}"
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


def _fetch_tag(tag: str) -> str:
    resp = requests.get(FEED_URL.format(tag=tag), headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


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
    return list(by_url.values())
