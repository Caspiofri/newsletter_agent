"""Offline tests for link extraction. Run from the repo root: python -m tests.test_link_utils"""
from datetime import date
from unittest.mock import patch

from modules import link_utils
from modules.models import Article

SEQUENCE_LIKE = """<html><head><style>""" + ".x{color:red}" * 3000 + """</style></head><body>
<a href="https://substack.com/redirect/2/abc">Subscribe here</a>
<table><tr><td><a href="https://open.substack.com/pub/thesequence/p/real-post-slug?utm_source=email">READ IN APP</a>
Opinion 943: When Compute Gets a Futures Market</td></tr></table>
<a href="https://substack.com/redirect/2/xyz">Unsubscribe</a>
<a href="mailto:a@b.com">mail</a>
<a href="https://open.substack.com/pub/thesequence/p/real-post-slug?utm_source=email">dup</a>
<a href="https://example.com/other-article">Other article</a>
</body></html>"""


def test_candidates_come_from_html_and_skip_junk():
    cands = link_utils.extract_candidate_links(SEQUENCE_LIKE)
    urls = [c["url"] for c in cands]
    assert urls == [
        "https://open.substack.com/pub/thesequence/p/real-post-slug?utm_source=email",
        "https://example.com/other-article",
    ], urls
    assert [c["id"] for c in cands] == [0, 1]


def test_links_found_even_when_html_head_is_huge():
    # regression: old code sent only the first 20k chars, which here is all CSS
    assert len(SEQUENCE_LIKE[:20000]) == 20000 and "<a " not in SEQUENCE_LIKE[:20000]
    assert link_utils.extract_candidate_links(SEQUENCE_LIKE)


def test_hallucinated_id_is_dropped():
    from modules import nodes
    email = Article(subject="s", name="n", published_at=date.today(), content="")
    cands = [{"id": 0, "text": "t", "context": "c", "url": "https://real.example/a"}]
    items = [
        {"id": 0, "title": "Real", "snippet": "x"},
        {"id": 7, "title": "Invented", "snippet": "y"},
        {"id": 0, "title": "Duplicate", "snippet": "z"},
        {"title": "No id", "snippet": "z"},
        "garbage",
    ]
    out = nodes._map_selection_to_articles(items, cands, email)
    assert [a.name for a in out] == ["Real"] and out[0].url == "https://real.example/a"


def _resp(code, headers=None):
    class R:
        status_code = code
        def __init__(self): self.headers = headers or {}
        def close(self): pass
    return R()


def test_check_link_classification():
    for code, expected in [(404, "dead"), (410, "dead"), (200, "ok"), (403, "ok"), (405, "ok"), (500, "ok")]:
        with patch("modules.link_utils.requests.get", return_value=_resp(code)):
            assert link_utils.check_link("https://x.example") == expected, code
    # Cloudflare bot challenge: we cannot verify the link
    with patch("modules.link_utils.requests.get", return_value=_resp(403, {"cf-mitigated": "challenge"})):
        assert link_utils.check_link("https://x.example") == "unverified"
    with patch("modules.link_utils.requests.get", side_effect=link_utils.requests.ConnectionError()):
        assert link_utils.check_link("https://x.example") == "ok"


def _art(name, url, status="ok", content=""):
    return Article(subject="s", name=name, published_at=date.today(), content=content, url=url, link_status=status)


def test_bad_link_upgraded_from_same_story_elsewhere():
    from modules import article_store
    arts = [
        _art("Fermion releases 164MB speech model that beats Whisper large", "https://tracker.example/c?x=1", "unverified"),
        _art("Fermion ships a 164MB speech model, beating Whisper large", "https://direct.example/fermion"),
        _art("Rust 2.0 released with new borrow checker", "https://direct.example/rust"),
    ]
    out = article_store.upgrade_bad_links(arts)
    assert out[0].url == "https://direct.example/fermion" and out[0].link_status == "ok"
    assert out[0].name == arts[0].name  # keeps its own title/content, only the link changes


def test_bad_link_kept_when_no_same_story():
    from modules import article_store
    arts = [
        _art("Fermion releases 164MB speech model", "https://tracker.example/c?x=1", "unverified"),
        _art("Rust 2.0 released with new borrow checker", "https://direct.example/rust"),
    ]
    out = article_store.upgrade_bad_links(arts)
    assert out[0].url == "https://tracker.example/c?x=1" and out[0].link_status == "unverified"


def test_render_grays_out_bad_links():
    from modules import nodes
    from modules.models import NewsletterData, ArticleCard, BriefItem
    data = NewsletterData(
        cards=[ArticleCard(title="t", source="s", brief="b", personalization="p", url="https://bad.example"),
               ArticleCard(title="t2", source="s", brief="b", personalization="p", url="https://good.example")],
        brief_items=[BriefItem(summary="x", url="https://bad.example")], actions=["a"])
    html = nodes._render_html(data, "T", "d", frozenset({"https://bad.example"}))
    assert 'href="https://bad.example"' not in html
    assert 'href="https://good.example"' in html
    assert html.count("הקישור אינו זמין") == 2


def test_unwrap_tldr_tracking():
    wrapped = "https://tracking.tldrnewsletter.com/CL0/https:%2F%2Fbuf.build%2Fblog%2Fgrpc-vs-rest%3Futm_source=tldrdev/1/0100abc/xyz=1"
    assert link_utils.unwrap_tracking(wrapped) == "https://buf.build/blog/grpc-vs-rest?utm_source=tldrdev"
    assert link_utils.unwrap_tracking("https://example.com/a") == "https://example.com/a"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
