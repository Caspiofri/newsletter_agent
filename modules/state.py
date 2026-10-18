import operator
from typing import Annotated, TypedDict, Optional
from modules.models import Article, NewsletterData


class DigestState(TypedDict):
    gmail_label: str
    subject: list[str]
    target_audience: str
    experience_level: str
    digest_name: str
    recipient: str
    sender: str
    top_k: int
    medium_tags: list[str]
    hn_queries: list[str]
    articles: list[Article]
    candidates: list[Article]  # per-source shortlist, read and ranked into top_articles
    # written in parallel by the collector nodes; combined with `articles` in merge_sources
    extra_articles: Annotated[list[Article], operator.add]
    top_articles: list[Article]
    # evidence for the trace; each node after merge writes {**state.get("run_stats", {}), ...}
    run_stats: dict[str, int]
    dedupe_drops: list[dict]
    rank_scores: list[dict]
    newsletter_data: Optional[NewsletterData]
    summary: str
    email_status: str
    tries: int          # transient failure retries (API errors)
    max_tries: int      # hard cap on failure retries
    days_back: int      # current search window in days
    max_days_back: int  # max time window before giving up