"""Offline tests for the shared article text view. Run from the repo root: python -m tests.test_article_text"""
from datetime import date

from config import settings
from modules.article_text import article_text
from modules.models import Article


def _art(content):
    return Article(subject="s", name="n", published_at=date.today(), content=content)


def test_strips_html():
    assert article_text(_art("<p>Hi <b>there</b></p>")) == "Hi there"


def test_truncates_to_max_chars():
    assert len(article_text(_art("a" * 7000), max_chars=6000)) == 6000


def test_default_cap_is_read_max_chars():
    assert len(article_text(_art("a" * 7000))) == settings.READ_MAX_CHARS


def test_empty_content():
    assert article_text(_art("")) == ""


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
