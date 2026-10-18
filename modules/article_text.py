from bs4 import BeautifulSoup

from config import settings
from modules.models import Article


def article_text(a: Article, max_chars: int = settings.READ_MAX_CHARS) -> str:
    """Plain text of an article's content: tags stripped, whitespace collapsed, cut to max_chars."""
    text = BeautifulSoup(a.content, "html.parser").get_text(" ")
    return " ".join(text.split())[:max_chars]
