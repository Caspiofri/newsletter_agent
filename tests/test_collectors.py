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


AUTHOR_FEED = """<?xml version="1.0"?><rss xmlns:content="http://purl.org/rss/1.0/modules/content/" version="2.0"><channel>
<item><title>Fresh post</title><link>https://a.medium.com/fresh-abc123?source=rss-1</link>
<content:encoded><![CDATA[<h3>Intro</h3><p>""" + "word " * 400 + """</p>]]></content:encoded></item>
<item><title>Member only</title><link>https://a.medium.com/paid-zzz?source=rss-1</link></item>
</channel></rss>"""


def test_medium_feed_url_for_article():
    assert medium.feed_url_for("https://medium.com/@ann/fresh-abc123") == "https://medium.com/feed/@ann"
    assert medium.feed_url_for("https://medium.com/illumination/post-1") == "https://medium.com/feed/illumination"
    assert medium.feed_url_for("https://pub.towardsai.net/post-1") == "https://pub.towardsai.net/feed"


def test_medium_parse_full_texts():
    texts = medium.parse_full_texts(AUTHOR_FEED)
    assert set(texts) == {"https://a.medium.com/fresh-abc123"}  # an item without content:encoded is left out
    assert texts["https://a.medium.com/fresh-abc123"].startswith("Intro")


def test_medium_with_full_text_upgrades_teasers():
    arts = medium.parse_feed(FEED, "AI", max_age_days=2, today=TODAY)
    paid = arts[0].model_copy(update={"name": "Member only", "url": "https://a.medium.com/paid-zzz"})
    out = medium.with_full_text(arts + [paid], fetch=lambda url: AUTHOR_FEED)
    assert out[0].read_status == "full" and out[0].content.startswith("Intro")
    assert len(out[0].content) <= medium.FULL_TEXT_MAX_CHARS
    assert out[1].read_status == "snippet" and out[1].content == "Teaser text"  # paywalled: teaser and link kept


def test_medium_with_full_text_survives_failing_feed():
    import requests
    arts = medium.parse_feed(FEED, "AI", max_age_days=2, today=TODAY)

    def boom(url):
        raise requests.ConnectionError("boom")
    assert medium.with_full_text(arts, fetch=boom) == arts


def test_reader_keeps_full_text_from_feed():
    from modules import reader
    fed = _art("A", "https://a.medium.com/x", "medium").model_copy(
        update={"content": "f" * 3000, "read_status": "full"})
    called = []
    r = reader.read_article(fed, 6000, fetch=lambda u: called.append(u))
    assert r == fed and called == []  # no page fetch (Medium answers 403 anyway)
    part = fed.model_copy(update={"content": "p" * 800, "read_status": "partial"})
    assert reader.read_article(part, 6000, fetch=lambda u: None).read_status == "partial"


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
         patch("modules.link_utils.resolve_links", side_effect=lambda urls: {u: ("ok", u) for u in urls}):
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
    with patch("modules.link_utils.resolve_links", side_effect=lambda urls: {u: ("ok", u) for u in urls}):
        out = nodes.merge_sources(pool)["articles"]
    assert [a.url for a in out] == ["https://x.example/1", "https://x.example/2"]
    assert out[0].origin == "inbox"  # first source wins


def test_merge_collapses_two_redirects_to_one_article():
    from modules import nodes
    r1, r2 = "https://substack.com/redirect/1/a", "https://substack.com/redirect/2/b"
    real = "https://x.substack.com/p/post"
    pool = {"articles": [_art("A", r1, "inbox")],
            "extra_articles": [_art("A again", r2, "inbox"), _art("B", "https://x.example/2", "medium")]}
    resolved = {r1: ("ok", real + "?utm_source=a"), r2: ("ok", real + "?utm_source=b"), "https://x.example/2": ("ok", "https://x.example/2")}
    with patch("modules.link_utils.resolve_links", return_value=resolved), \
         patch("modules.article_store.upgrade_bad_links", side_effect=lambda a: a):
        out = nodes.merge_sources(pool)
    assert [a.url for a in out["articles"]] == [real, "https://x.example/2"]
    assert out["articles"][0].name == "A"  # first seen wins
    assert out["run_stats"] == {"merged": 2}


def test_merge_keeps_status_dead_with_canonical_url():
    from modules import nodes
    r1 = "https://substack.com/redirect/1/a"
    pool = {"articles": [_art("A", r1, "inbox")], "extra_articles": []}
    with patch("modules.link_utils.resolve_links", return_value={r1: ("dead", "https://x.example/gone?ref=1")}), \
         patch("modules.article_store.upgrade_bad_links", side_effect=lambda a: a):
        out = nodes.merge_sources(pool)["articles"]
    assert out[0].url == "https://x.example/gone" and out[0].link_status == "dead"


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


def test_ranking_report_rows_and_html():
    import json, re, tempfile
    from pathlib import Path
    from modules import ranking_report
    cands = [_art("top", "https://x.example/1", "inbox"), _art("mid", "https://x.example/2", "medium")]
    pool = cands + [_art(f"other{i}", f"https://o.example/{i}", "inbox") for i in range(30)]
    rows = ranking_report.build_rows(cands, {0: 9, 1: 5}, {0: "great"}, [cands[0]], pool, sample_size=10, seed="s")
    assert [r["group"] for r in rows] == ["shortlisted"] * 2 + ["sample"] * 10
    assert rows[0]["picked"] and rows[0]["score"] == 9 and rows[0]["reason"] == "great" and not rows[1]["picked"]
    assert not {r["url"] for r in rows[2:]} & {"https://x.example/1", "https://x.example/2"}  # sample never repeats shortlisted
    again = ranking_report.build_rows(cands, {}, {}, [], pool, sample_size=10, seed="s")
    assert [r["url"] for r in again[2:]] == [r["url"] for r in rows[2:]]  # seeded, so reproducible
    rows[0]["title"] = "evil </script><b>x"
    with tempfile.TemporaryDirectory() as d:
        path = ranking_report.save("ai", rows, Path(d))
        html = path.read_text(encoding="utf-8")
        assert path.with_suffix(".json").exists()
    assert html.count("</script>") == 2  # data block + code block; the injected one was escaped
    data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1).replace("<\\/", "</"))
    assert len(data["items"]) == 12 and data["items"][0]["title"].startswith("evil")


def _sample_pool():
    cands = [_art("top", "https://x.example/1", "inbox"), _art("mid", "https://x.example/2", "medium")]
    return cands, cands + [_art(f"other{i}", f"https://o.example/{i}", "inbox") for i in range(30)]


def _fake_read(arts):
    return [a.model_copy(update={"content": "full article text " * 50, "read_status": "full"}) for a in arts]


def test_sample_rows_get_read_text():
    from modules import ranking_report
    cands, pool = _sample_pool()
    rows = ranking_report.build_rows(cands, {0: 9, 1: 5}, {}, [cands[0]], pool, sample_size=10, seed="s", read_sample=_fake_read)
    sample = [r for r in rows if r["group"] == "sample"]
    assert len(sample) == 10 and all(r["read_status"] == "full" and r["content"].startswith("full article") for r in sample)
    short = [r for r in rows if r["group"] == "shortlisted"]
    assert all(r["read_status"] == "snippet" and r["content"] == "c" for r in short)


def test_sample_selection_is_unchanged_by_reading():
    from modules import ranking_report
    cands, pool = _sample_pool()
    plain = ranking_report.build_rows(cands, {}, {}, [], pool, sample_size=10, seed="s")
    read = ranking_report.build_rows(cands, {}, {}, [], pool, sample_size=10, seed="s", read_sample=_fake_read)
    assert [r["url"] for r in read] == [r["url"] for r in plain]


def test_read_sample_failure_keeps_teaser():
    from modules import ranking_report
    cands, pool = _sample_pool()

    def boom(arts):
        raise RuntimeError("network down")

    def short(arts):
        return arts[:1]
    for bad in (boom, short):
        rows = ranking_report.build_rows(cands, {}, {}, [], pool, sample_size=10, seed="s", read_sample=bad)
        sample = [r for r in rows if r["group"] == "sample"]
        assert len(sample) == 10 and all(r["content"] == "c" and r["read_status"] == "snippet" for r in sample)


def test_read_sample_only_called_with_sample_articles():
    from modules import ranking_report
    cands, pool = _sample_pool()
    seen = []

    def spy(arts):
        seen.append(list(arts))
        return arts
    rows = ranking_report.build_rows(cands, {}, {}, [], pool, sample_size=10, seed="s", read_sample=spy)
    assert len(seen) == 1 and [a.url for a in seen[0]] == [r["url"] for r in rows if r["group"] == "sample"]
    assert not {a.url for a in seen[0]} & {a.url for a in cands}


def test_import_labels_merges_by_url():
    from evals import import_labels
    golden = [{"url": "https://a", "label": "skip", "title": "A"}]
    exported = {"run_id": "r2", "labels": [
        {"url": "https://a", "label": "must", "title": "A"},       # relabel
        {"url": "https://b", "label": "worth", "title": "B"},      # new
        {"url": "https://c", "label": "bogus", "title": "C"},      # invalid label ignored
        {"url": "", "label": "must"}]}                              # no url ignored
    merged, added, updated = import_labels.merge(golden, exported, "2026-10-02")
    assert (added, updated) == (1, 1)
    assert {g["url"]: g["label"] for g in merged} == {"https://a": "must", "https://b": "worth"}


def test_import_labels_keeps_a_rows_own_source_run():
    from evals import import_labels
    exported = {"run_id": "hub_1", "labels": [{"url": "https://a", "label": "must", "source_run": "20261001_070000"},
                                              {"url": "https://b", "label": "skip"}]}
    merged, _, _ = import_labels.merge([], exported, "2026-10-18")
    assert {g["url"]: g["source_run"] for g in merged} == {"https://a": "20261001_070000", "https://b": "hub_1"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
