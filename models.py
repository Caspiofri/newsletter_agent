from pydantic import BaseModel
from typing import Optional
from datetime import date


class Article(BaseModel):
    subject: str
    name: str
    published_at: date
    content: str
    url: Optional[str] = None
    author: Optional[str] = None


class ArticleCard(BaseModel):
    title: str          # Hebrew title with emoji
    source: str
    brief: str          # "השורה התחתונה" — 1-2 sentence factual summary
    personalization: str  # "איך זה פוגש אותך" — audience-specific angle
    url: str


class BriefItem(BaseModel):
    summary: str        # 1-sentence Hebrew summary
    url: str


class NewsletterData(BaseModel):
    cards: list[ArticleCard]
    brief_items: list[BriefItem]
    actions: list[str]