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