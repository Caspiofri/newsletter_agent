"""Offline tests for the faithfulness judge. Run from the repo root: python -m tests.test_trace_judge"""
from unittest.mock import patch

# trace_judge builds a Vertex client at import; patch it so tests need no credentials
with patch("google.auth.default", return_value=(object(), "p")), patch("google.genai.Client"):
    from modules import trace_judge

from modules.article_text import article_text
from modules.models import Article, ArticleCard, NewsletterData
from modules.trace_judge import SKIPPED
from datetime import date


def _article(name, content):
    return Article(subject=f"{name} title", name=name, url=f"https://{name}.example/a",
                    content=content, published_at=date(2026, 1, 1))


def _card(id, brief):
    return ArticleCard(id=id, title="t", source="s", brief=brief, personalization="p")


def _run(articles, cards):
    seen = {}

    def fake(criterion, rubric, content):
        seen["content"] = content
        return {"score": 4, "reasoning": "ok"}

    data = None if cards is None else NewsletterData(cards=cards, brief_items=[], actions=[])
    with patch.object(trace_judge, "_llm_judge", fake):
        score = trace_judge._judge_content_faithfulness(articles, data)
    return seen.get("content"), score


def test_card_paired_with_its_own_source_only():
    arts = [_article("alpha", "ALPHA-BODY"), _article("beta", "BETA-BODY")]
    content, _ = _run(arts, [_card(1, "brief about beta")])
    assert "BETA-BODY" in content and "brief about beta" in content
    assert "ALPHA-BODY" not in content, content


def test_judge_sees_text_beyond_600_chars():
    arts = [_article("alpha", "x" * 700 + "LATE-DETAIL")]
    content, _ = _run(arts, [_card(0, "b")])
    assert "LATE-DETAIL" in content


def test_card_without_id_gets_all_sources():
    arts = [_article("alpha", "ALPHA-BODY"), _article("beta", "BETA-BODY")]
    for bad in (None, 5, -1):
        content, _ = _run(arts, [_card(bad, "b")])
        assert "ALPHA-BODY" in content and "BETA-BODY" in content, bad


def test_no_cards_is_skipped():
    content, score = _run([_article("alpha", "A")], [])
    assert content is None and score.score == SKIPPED
    _, score = _run([_article("alpha", "A")], None)
    assert score.score == SKIPPED


def test_details_store_full_text():
    a = _article("alpha", "y" * 1500)
    _, score = _run([a], [_card(0, "b")])
    src = score.details["source_articles"][0]
    assert src["content"] == article_text(a) and len(src["content"]) > 600
    assert src["name"] == "alpha" and src["url"] == a.url


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
