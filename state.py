from typing import TypedDict
from models import Article
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
    summary: str
    email_status: str
    tries: int       
    max_tries: int   