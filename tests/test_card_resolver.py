"""Offline tests for resolve_cards. Run from the repo root: python -m tests.test_card_resolver"""
from datetime import date

from modules.card_resolver import resolve_cards
from modules.models import Article, ArticleCard, BriefItem, NewsletterData


def _art(url="https://a.example/x", origin="inbox", read_status="full"):
    return Article(subject="s", name="n", published_at=date.today(), content="",
                   url=url, origin=origin, read_status=read_status)


def _card(id, brief="b"):
    return ArticleCard(id=id, title="t", source="s", brief=brief, personalization="p")


def _data(cards=(), briefs=()):
    return NewsletterData(cards=list(cards), brief_items=list(briefs), actions=["a"])


def test_fills_url_from_article_id():
    arts = [_art("https://a.example/0"), _art("https://a.example/1")]
    out = resolve_cards(_data([_card(1)], [BriefItem(id=0, summary="x")]), arts)
    assert out.cards[0].url == "https://a.example/1"
    assert out.brief_items[0].url == "https://a.example/0"
    assert out.actions == ["a"]


def test_unknown_and_missing_ids_dropped():
    out = resolve_cards(_data([_card(7), _card(None), _card(-1)],
                              [BriefItem(id=7, summary="x"), BriefItem(summary="y")]), [_art()])
    assert out.cards == [] and out.brief_items == []


def test_duplicate_id_keeps_first():
    out = resolve_cards(_data([_card(0, "first"), _card(0, "second")],
                              [BriefItem(id=0, summary="x")]), [_art()])
    assert [c.brief for c in out.cards] == ["first"]
    assert out.brief_items == []


def test_none_url_becomes_empty_string():
    out = resolve_cards(_data([_card(0)]), [_art(url=None)])
    assert out.cards[0].url == ""


def test_external_snippet_card_demoted_to_brief():
    arts = [_art(origin="medium", read_status="snippet")]
    out = resolve_cards(_data([_card(0, "the brief")]), arts)
    assert out.cards == []
    assert out.brief_items[0].summary == "the brief"
    assert out.brief_items[0].id == 0
    assert out.brief_items[0].url == "https://a.example/x"


def test_inbox_snippet_card_kept():
    out = resolve_cards(_data([_card(0)]), [_art(origin="inbox", read_status="snippet")])
    assert len(out.cards) == 1 and out.brief_items == []


def test_external_full_card_kept():
    out = resolve_cards(_data([_card(0)]), [_art(origin="medium", read_status="full")])
    assert len(out.cards) == 1 and out.brief_items == []


def test_inputs_not_mutated_and_demoted_follow_brief_items():
    arts = [_art(origin="medium", read_status="snippet"), _art("https://a.example/1")]
    data = _data([_card(0)], [BriefItem(id=1, summary="x")])
    out = resolve_cards(data, arts)
    assert [b.id for b in out.brief_items] == [1, 0]
    assert data.cards[0].url == "" and data.brief_items[0].url == "" and len(data.cards) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
