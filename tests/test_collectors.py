"""Offline tests for the Medium/HN collectors and the parallel fan-out wiring. Run: python -m tests.test_collectors"""
import importlib
from datetime import date
from unittest.mock import patch

from modules.collectors import hackernews, medium
from modules.models import Article

TODAY = date(2026, 10, 2)
FEED = """<?xml version="1.0"?><rss xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0"><channel>
<item><title>Fresh post</title><description><![CDATA[<p>Teaser <b>text</b></p>]]></description>
<link>https://a.medium.com/fresh-abc123?source=rss------llm-5</link><dc:creator>Ann</dc:creator>
<pubDate>Fri, 02 Oct 2026 09:12:01 GMT</pubDate></item>
<item><title>Old post</title><description>old</description><link>https://a.medium.com/old-def456?source=rss</link>
<pubDate>Mon, 01 Jun 2026 09:12:01 GMT</pubDate></item>
<item><title></title><link>https://a.medium.com/no-title</link><pubDate>Fri, 02 Oct 2026 09:12:01 GMT</pubDate></item>
</channel></rss>"""


def test_medium_parse_feed():
    arts = medium.parse_feed(FEED, "AI", max_age_days=2, today=TODAY)
    assert [a.name for a in arts] == ["Fresh post"]
    a = arts[0]
    assert a.url == "https://a.medium.com/fresh-abc123"  # tracking query stripped
    assert a.content == "Teaser text" and a.author == "Ann" and a.origin == "medium"


def test_hackernews_parse_hits():
    hits = [
        {"title": "Show HN: Thing", "url": "https://github.com/x/y", "points": 50, "num_comments": 7,
         "objectID": "1", "author": "bob", "created_at_i": 1790000000},
        {"title": "Ask HN: text post", "url": None, "points": 30, "num_comments": None,
         "objectID": "2", "story_text": "<p>hello</p>", "author": "eve", "created_at_i": 1790000000},
        {"title": "", "objectID": "3"},
    ]
    arts = hackernews.parse_hits(hits, "AI")
    assert [a.name for a in arts] == ["Show HN: Thing", "Ask HN: text post"]
    assert arts[0].url == "https://github.com/x/y" and arts[0].origin == "hackernews"
    assert arts[1].url == "https://news.ycombinator.com/item?id=2" and "hello" in arts[1].content


def test_failing_source_is_skipped_not_fatal():
    import requests
    with patch("modules.collectors.medium._fetch_tag", side_effect=requests.ConnectionError("boom")):
        assert medium.collect(["llm", "rag"], "AI") == []
    with patch("modules.collectors.hackernews._search", side_effect=requests.ConnectionError("boom")):
        assert hackernews.collect(["LLM"], "AI") == []


def _art(name, url, origin):
    return Article(subject="AI", name=name, published_at=TODAY, content="c", url=url, origin=origin)


def _run_graph(inbox: list[Article]):
    """Run the real graph with every node that touches Gmail/LLM/Chroma stubbed out."""
    from modules import nodes
    medium_art, hn_art = _art("M", "https://m.example/1", "medium"), _art("H", "https://h.example/1", "hackernews")
    stubs = dict(
        fetch_articles=lambda s: {"articles": inbox},
        extract_sub_articles=lambda s: {"articles": list(s["articles"])},
        collect_medium=lambda s: {"extra_articles": [medium_art]},
        collect_hackernews=lambda s: {"extra_articles": [hn_art]},
        dedupe_articles=lambda s: {},
        shortlist_articles=lambda s: {"candidates": s["articles"]},
        read_articles=lambda s: {},
        rank_articles=lambda s: {"top_articles": s["candidates"]},
        summraize=lambda s: {"summary": "html", "newsletter_data": None},
        send_email=lambda s: {"email_status": "success"},
    )
    with patch.multiple(nodes, **stubs), \
         patch("modules.link_utils.classify_links", side_effect=lambda urls: {u: "ok" for u in urls}):
        from modules import graph as graph_mod
        graph_mod = importlib.reload(graph_mod)
        state = {"subject": "AI", "tries": 0, "max_tries": 3, "days_back": 1, "max_days_back": 2,
                 "medium_tags": ["llm"], "hn_queries": ["LLM"], "articles": [], "extra_articles": [],
                 "top_articles": [], "candidates": []}
        import asyncio
        return asyncio.run(graph_mod.graph.ainvoke(state, config={"configurable": {"thread_id": "t"}}))


def test_graph_joins_all_collectors():
    out = _run_graph([_art("I", "https://i.example/1", "inbox")])
    assert out["email_status"] == "success"
    assert sorted(a.origin for a in out["articles"]) == ["hackernews", "inbox", "medium"]


def test_empty_inbox_does_not_stop_the_run():
    out = _run_graph([])  # inbox finds nothing, window exhausts, other collectors still deliver
    assert out["email_status"] == "success"
    assert sorted(a.origin for a in out["articles"]) == ["hackernews", "medium"]


def test_merge_dedupes_by_url():
    from modules import nodes
    pool = {"articles": [_art("A", "https://x.example/1", "inbox")],
            "extra_articles": [_art("A again", "https://x.example/1", "medium"), _art("B", "https://x.example/2", "medium")]}
    with patch("modules.link_utils.classify_links", side_effect=lambda urls: {u: "ok" for u in urls}):
        out = nodes.merge_sources(pool)["articles"]
    assert [a.url for a in out] == ["https://x.example/1", "https://x.example/2"]
    assert out[0].origin == "inbox"  # first source wins


def test_reader_statuses():
    from modules import reader
    base = _art("A", "https://a.example", "inbox")  # content "c"
    assert reader.read_article(base, 6000, fetch=lambda u: None).read_status == "snippet"
    assert reader.read_article(base, 6000, fetch=lambda u: "x" * 300).read_status == "partial"
    full = reader.read_article(base, 100, fetch=lambda u: "y" * 3000)
    assert full.read_status == "full" and len(full.content) == 100
    # a fetched text shorter than the teaser we already have is not an upgrade
    teaser = base.model_copy(update={"content": "t" * 500})
    r = reader.read_article(teaser, 6000, fetch=lambda u: "z" * 50)
    assert r.read_status == "snippet" and r.content == "t" * 500


def test_select_top_ranks_filters_and_ignores_garbage():
    from modules import nodes
    cands = [_art(n, f"https://x.example/{n}", "inbox") for n in "abcd"]
    scored = [{"id": 0, "score": 6}, {"id": 1, "score": 9}, {"id": 2, "score": 3}, {"id": 99, "score": 10},
              {"id": 3, "score": "high"}, "junk", {"score": 8}]
    out = nodes._select_top(scored, cands, top_k=5, min_score=5)
    assert [a.name for a in out] == ["b", "a"]  # best first; below-min, unknown id, bad rows dropped
    assert [a.name for a in nodes._select_top(scored, cands, top_k=1, min_score=5)] == ["b"]


def test_shortlist_screens_each_source_with_its_own_quota():
    from modules import nodes
    calls = {}

    def fake_filter(state):
        origin = state["articles"][0].origin
        calls[origin] = state["top_k"]
        return {"top_articles": state["articles"][: state["top_k"]]}

    arts = [_art(f"m{i}", f"https://m.example/{i}", "medium") for i in range(10)] + \
           [_art(f"h{i}", f"https://h.example/{i}", "hackernews") for i in range(10)] + \
           [_art("i0", "https://i.example/0", "inbox")]
    with patch.object(nodes, "fillter_articles", side_effect=fake_filter):
        out = nodes.shortlist_articles({"articles": arts})["candidates"]
    assert calls == {"medium": 6, "hackernews": 6, "inbox": 10}
    assert sorted({a.origin for a in out}) == ["hackernews", "inbox", "medium"] and len(out) == 13


def test_snippet_only_articles_are_capped():
    from modules import nodes
    cands = [_art("full", "https://x.example/a", "inbox").model_copy(update={"read_status": "full"}),
             _art("snip", "https://x.example/b", "hackernews")]  # read_status defaults to "snippet"
    scored = [{"id": 0, "score": 8}, {"id": 1, "score": 9}]
    assert nodes._final_scores(scored, cands, snippet_cap=6) == {0: 8, 1: 6}
    # the capped snippet article now ranks below the fully read one
    assert [a.name for a in nodes._select_top(scored, cands, 5, 5, snippet_cap=6)] == ["full", "snip"]
    assert [a.name for a in nodes._select_top(scored, cands, 5, 7, snippet_cap=6)] == ["full"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
