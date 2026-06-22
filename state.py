from typing import TypedDict, Optional
from models import Article, NewsletterData


class DigestState(TypedDict):
    gmail_label: str
    subject: list[str]
    target_audience: str
    experience_level: str
    digest_name: str
    recipient: str
    sender: str
    top_k: int
    articles: list[Article]
    top_articles: list[Article]
    newsletter_data: Optional[NewsletterData]
    summary: str
    email_status: str
    tries: int          # transient failure retries (API errors)
    max_tries: int      # hard cap on failure retries
    days_back: int      # current search window in days
    max_days_back: int  # max time window before giving up