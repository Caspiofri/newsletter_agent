"""Offline tests for summraize id resolution. Run from the repo root: python -m tests.test_summarize_ids"""
import json
from datetime import date
from unittest.mock import patch

from modules import nodes
from modules.models import Article


_seq = iter(range(10**6))


def _art(url="https://a.example/x", content="body", link_status="ok", origin="inbox", read_status="full",
         author=None, signal=""):
    return Article(subject="s", name=f"Title-{next(_seq)}", author=author, signal=signal, published_at=date.today(), content=content,
                   url=url, link_status=link_status, origin=origin, read_status=read_status)


def _state(articles):
    return {"subject": "AI", "target_audience": "dev", "top_k": len(articles), "top_articles": articles,
            "experience_level": "junior", "digest_name": "Digest"}


def _card(id, **kw):
    return {"id": id, "title": "t", "source": "s", "brief": "b", "personalization": "p", **kw}


def _run(articles, payload):
    """Run summraize with a fixed model reply; returns (result, captured prompt)."""
    seen = {}

    def fake(prompt, config=None):
        seen["prompt"] = prompt
        return json.dumps(payload)

    with patch.object(nodes, "_call_gemini", fake):
        return nodes.summraize(_state(articles)), seen["prompt"]


def test_card_url_comes_from_article_not_llm():
    arts = [_art("https://real.example/0")]
    out, _ = _run(arts, {"cards": [_card(0, url="https://invented.example/z")], "brief_items": [], "actions": ["a"]})
    assert out["newsletter_data"].cards[0].url == "https://real.example/0"
    assert "https://real.example/0" in out["summary"]
    assert "invented.example" not in out["summary"]


def test_prompt_contains_text_beyond_2000_chars():
    arts = [_art(content="a " * 2449 + "ZMARK " + "b " * 1000)]
    _, prompt = _run(arts, {"cards": [_card(0)], "brief_items": [], "actions": ["a"]})
    assert "ZMARK" in prompt
    assert '"id": 0' in prompt


def test_prompt_carries_article_title_author_signal_not_topic():
    arts = [_art("https://a.example/0", author="Jane Doe", signal="193 points, 40 comments")]
    _, prompt = _run(arts, {"cards": [_card(0)], "brief_items": [], "actions": ["a"]})
    assert f'"title": "{arts[0].name}"' in prompt
    assert '"title": "s"' not in prompt
    assert "Jane Doe" in prompt and "193 points, 40 comments" in prompt


def test_dead_link_article_renders_disabled():
    arts = [_art("https://dead.example/0", link_status="dead")]
    out, _ = _run(arts, {"cards": [_card(0)], "brief_items": [], "actions": ["a"]})
    assert "https://dead.example/0" not in out["summary"]


def test_none_url_article_does_not_crash():
    arts = [_art(url=None)]
    out, _ = _run(arts, {"cards": [_card(0)], "brief_items": [], "actions": ["a"]})
    assert out["newsletter_data"].cards[0].url == ""
    assert 'href=""' not in out["summary"]


def test_odd_ids_are_dropped_not_crash():
    arts = [_art("https://a.example/0"), _art("https://a.example/1")]
    payload = {"cards": [_card("x"), _card(True), _card(1.5), _card(None), _card(1)],
               "brief_items": [{"id": "0", "summary": "s"}, {"summary": "s"}, {"id": 0, "summary": "s"}],
               "actions": ["a"]}
    out, _ = _run(arts, payload)
    data = out["newsletter_data"]
    assert [c.id for c in data.cards] == [1]
    assert [b.id for b in data.brief_items] == [0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
