"""Offline tests for id-based shortlisting. Run from the repo root: python -m tests.test_shortlist_ids"""
from datetime import date
from unittest.mock import patch

from modules import nodes
from modules.models import Article

TODAY = date(2026, 10, 18)


def _art(name):
    return Article(subject="AI", name=name, published_at=TODAY, content="c")


def test_pick_by_ids_filters_bad_and_duplicates():
    pool = [_art("a"), _art("b"), _art("c")]
    picked = nodes._pick_by_ids([2, 2, 9, "x", 0], pool, 5)
    assert [a.name for a in picked] == ["c", "a"], picked


def test_pick_ignores_bools_and_negatives():
    pool = [_art("a"), _art("b")]
    assert nodes._pick_by_ids([True, -1, 1], pool, 5) == [pool[1]]


def test_pick_respects_limit():
    pool = [_art("a"), _art("b"), _art("c")]
    picked = nodes._pick_by_ids([0, 1, 2], pool, 2)
    assert [a.name for a in picked] == ["a", "b"], picked


def test_fillter_articles_returns_selected_by_id():
    same = [_art("Same title"), _art("Same title")]
    state = {"articles": same, "top_k": 3, "subject": "AI"}
    with patch.object(nodes, "_call_gemini", return_value="[1]"):
        out = nodes.fillter_articles(state)
    assert len(out["top_articles"]) == 1
    assert out["top_articles"][0] is same[1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
