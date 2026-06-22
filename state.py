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
    tries: int
    max_tries: int
    article_count_last: int
    days_back: int