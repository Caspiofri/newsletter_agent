"""Offline tests for the text-length settings of the shortlist and the ranker. Run: python -m tests.test_text_levers"""
import json
from datetime import date
from unittest.mock import patch

from config import settings
from modules import nodes, ranking_report
from modules.models import Article

TODAY = date(2026, 10, 18)


def _art(content, name="T"):
    return Article(subject="AI", name=name, published_at=TODAY, content=content)


def _rank_prompt(content):
    seen = []

    def fake(prompt):
        seen.append(prompt)
        return '[{"id": 0, "score": 7, "reason": "ok"}]'

    state = {"candidates": [_art(content)], "top_k": 1, "subject": "AI", "target_audience": "devs",
             "experience_level": "mid", "digest_name": "t", "articles": []}
    with patch.object(nodes, "_call_gemini", side_effect=fake), patch.object(ranking_report, "save", return_value="x"):
        nodes.rank_articles(state)
    return seen[0]


def test_rank_prompt_includes_text_beyond_2500_chars():
    content = "<p>" + "a " * 2450 + "MARKER4900</p>" + " b" * 50
    assert "MARKER4900" in _rank_prompt(content)
    assert "<p>" not in _rank_prompt(content)


def test_rank_text_follows_the_setting():
    content = "a " * 75 + "MARKER150 " + "b " * 100
    with patch.object(settings, "RANK_TEXT_CHARS", 100):
        assert "MARKER150" not in _rank_prompt(content)
    assert "MARKER150" in _rank_prompt(content)


def _shortlist_prompt(articles):
    seen = []
    with patch.object(nodes, "_call_gemini", side_effect=lambda p: seen.append(p) or "[]"):
        nodes.fillter_articles({"articles": articles, "top_k": 2, "subject": "AI"})
    return seen[0]


def _items(prompt):
    start = prompt.index("[{")
    return json.JSONDecoder().raw_decode(prompt[start:])[0]


def test_shortlist_prompt_includes_teaser():
    items = _items(_shortlist_prompt([_art("word " * 500)]))
    assert items[0]["teaser"]
    assert len(items[0]["teaser"]) <= settings.SHORTLIST_TEASER_CHARS
    with patch.object(settings, "SHORTLIST_TEASER_CHARS", 50):
        assert len(_items(_shortlist_prompt([_art("word " * 500)]))[0]["teaser"]) <= 50


def test_shortlist_teaser_may_be_empty():
    assert _items(_shortlist_prompt([_art("")]))[0]["teaser"] == ""


def test_labeling_page_offers_full_text_mattered_tag():
    assert "full text mattered" in ranking_report.render_html([], "p", "r")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
