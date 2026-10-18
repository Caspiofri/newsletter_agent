"""Offline tests for stripping arXiv page chrome from fetched text. Run from the repo root: python -m tests.test_reader_arxiv"""
from pathlib import Path
from unittest.mock import MagicMock, patch

from modules import reader

FIXTURE = (Path(__file__).parent / "fixtures" / "arxiv_abs_harness2.txt").read_text(encoding="utf-8")
ABS_URL = "https://arxiv.org/abs/2610.01234"


def test_clean_keeps_title_and_abstract():
    out = reader.clean_page_text(ABS_URL, FIXTURE)
    assert "Harness^2: Recursive Agent Harnessing for an Open World" in out, out
    assert "Submitted on 9 Oct 2026" in out, out
    assert "LLM agents' capabilities depend not only on their underlying models" in out, out
    assert "with further gains as the task-level recursion budget increases." in out, out


def test_clean_drops_site_chrome():
    out = reader.clean_page_text(ABS_URL, FIXTURE)
    for junk in ("arXivLabs", "View PDF", "HTML (experimental)", "Bibliographic", "Computer Science >"):
        assert junk not in out, junk


def test_clean_is_much_shorter():
    out = reader.clean_page_text(ABS_URL, FIXTURE)
    assert len(out) < 0.7 * len(FIXTURE), (len(out), len(FIXTURE))


def test_non_arxiv_page_unchanged():
    assert reader.clean_page_text("https://example.com/abs/1", FIXTURE) == FIXTURE
    assert reader.clean_page_text("https://arxiv.org/pdf/2610.01234", FIXTURE) == FIXTURE


def test_arxiv_without_abstract_marker_unchanged():
    text = "Some arXiv listing page\nView PDF\narXivLabs"
    assert reader.clean_page_text(ABS_URL, text) == text
    assert reader.clean_page_text(ABS_URL, "") == ""


def test_fetch_text_cleans_by_final_url():
    resp = MagicMock(status_code=200, url=ABS_URL, text="<html></html>")
    resp.headers = {"content-type": "text/html"}
    with patch("modules.reader.requests.get", return_value=resp) as get, \
            patch("modules.reader.trafilatura.extract", return_value=FIXTURE):
        out = reader.fetch_text("https://substack.com/redirect/abc")
    assert get.call_args[0][0] == "https://substack.com/redirect/abc"
    assert "arXivLabs" not in out and "Abstract:" in out, out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
