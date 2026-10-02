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
    origin: str = "inbox"  # which collector found it: "inbox" | "medium" | "hackernews"
    read_status: str = "snippet"  # "full" | "partial" | "snippet" (see reader.read_article)
    signal: str = ""  # source-specific quality hint, e.g. "193 points, 40 comments" for Hacker News
    link_status: str = "ok"  # "ok" | "dead" (404/410) | "unverified" (bot challenge, can't be checked)


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