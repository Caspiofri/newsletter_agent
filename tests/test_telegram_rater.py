"""Offline tests for the Telegram rating bot. Run from the repo root: python -m tests.test_telegram_rater
No network and no real token: the Bot API is a callable passed in and faked."""
import random

import requests

from evals import telegram_api, telegram_select

TOKEN = "123456:SECRET-test-token"


class _Resp:
    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


def test_api_returns_result():
    calls = []

    def post(url, json=None, timeout=None):
        calls.append((url, json, timeout))
        return _Resp({"ok": True, "result": {"message_id": 7}})

    api = telegram_api.make_api(TOKEN, post=post, timeout=5)
    assert api("sendMessage", {"chat_id": "1", "text": "hi"}) == {"message_id": 7}
    assert calls == [(f"https://api.telegram.org/bot{TOKEN}/sendMessage", {"chat_id": "1", "text": "hi"}, 5)]


def test_api_raises_on_not_ok():
    api = telegram_api.make_api(TOKEN, post=lambda url, json=None, timeout=None:
                                _Resp({"ok": False, "description": "Bad Request: message is too long"}))
    try:
        api("sendMessage", {})
    except telegram_api.TelegramError as e:
        assert "message is too long" in str(e)
    else:
        raise AssertionError("no TelegramError")


def test_api_error_hides_token():
    def post(url, json=None, timeout=None):
        raise requests.ConnectionError(f"Max retries exceeded with url: {url}")

    api = telegram_api.make_api(TOKEN, post=post)
    try:
        api("getUpdates", {})
    except telegram_api.TelegramError as e:
        assert TOKEN not in str(e) and "SECRET" not in str(e)
        assert "getUpdates" in str(e)
    else:
        raise AssertionError("no TelegramError")


# ---- daily selection and message format

def _item(url, picked=False, score=None, group="shortlisted", title=None, content="Some article text.", **kw):
    return {"title": title or f"title {url}", "url": url, "origin": "medium", "author": "", "published_at": "2026-10-20",
            "read_status": "full", "link_status": "ok", "signal": "", "group": group, "score": score,
            "reason": "", "picked": picked, "content": content, **kw}


def _urls(rows):
    return [r["url"] for r in rows]


def test_select_picks_first_then_slots():
    items = [_item(f"p{i}", picked=True, score=8) for i in range(5)] + [
        _item("low2", score=2), _item("near4", score=4.0), _item("near45", score=4.5), _item("low1", score=1),
        _item("mid3", score=3), _item("s1", group="sample"), _item("s2", group="sample")]
    got = telegram_select.select(items, set(), min_score=5, extra=3)
    assert _urls(got) == ["p0", "p1", "p2", "p3", "p4", "near45", "low1", "s1"]


def test_select_fills_empty_slot_from_next():
    # no near-cutoff candidate: slot a takes the lowest-scored drop, slot b the next lowest, slot c the sample
    items = [_item("p0", picked=True, score=9), _item("low1", score=1), _item("low2", score=2),
             _item("s1", group="sample")]
    assert _urls(telegram_select.select(items, set(), min_score=5, extra=3)) == ["p0", "low1", "low2", "s1"]
    # no sample: slot c falls back to any remaining non-pick
    items = [_item("p0", picked=True, score=9), _item("near", score=4.2), _item("low1", score=1), _item("x", score=3)]
    assert _urls(telegram_select.select(items, set(), min_score=5, extra=3)) == ["p0", "near", "low1", "x"]


def test_select_skips_labeled_urls():
    items = [_item("p0", picked=True, score=9), _item("p1", picked=True, score=8), _item("near", score=4.5),
             _item("low", score=1), _item("s1", group="sample"), _item("s1", group="sample")]
    got = telegram_select.select(items, {"p1", "near"}, min_score=5, extra=3)
    assert _urls(got) == ["p0", "low", "s1"]


def test_select_short_report_no_picks():
    items = [_item("a", score=4.5), _item("b", group="sample")]
    assert _urls(telegram_select.select(items, set(), min_score=5, extra=3)) == ["a", "b"]
    assert telegram_select.select([], set(), min_score=5) == []


def test_recheck_due_and_not_due():
    golden = [{"url": "g1", "label": "must"}, {"url": "g2", "label": "skip"}]
    rng = random.Random(0)
    assert telegram_select.pick_recheck(golden, None, "2026-10-21", rng=rng) in golden
    assert telegram_select.pick_recheck(golden, "2026-10-14", "2026-10-21", every_days=7, rng=rng) in golden
    assert telegram_select.pick_recheck(golden, "2026-10-15", "2026-10-21", every_days=7, rng=rng) is None


def test_recheck_empty_golden_none():
    assert telegram_select.pick_recheck([], None, "2026-10-21") is None


def test_format_marks_non_digest_and_escapes_html():
    text, markup = telegram_select.format_message(_item("https://x/?a=1&b=2", title="a<b & c", content="w " * 400))
    assert "<b>a&lt;b &amp; c</b>" in text
    assert "🔍 not in today's digest" in text
    assert '<a href="https://x/?a=1&amp;b=2">Read article</a>' in text
    assert "w w w" in text and len(text) < 600
    picked_text, _ = telegram_select.format_message(_item("u", picked=True, score=8, content="<p>Hello <b>x</b></p>"))
    assert "not in today's digest" not in picked_text and "Hello x" in picked_text
    recheck_text, _ = telegram_select.format_message(_item("u", picked=True), recheck=True)
    assert "🔁 re-check" in recheck_text
    long_text, _ = telegram_select.format_message(_item("u", title="t" * 5000, content="c" * 9000))
    assert len(long_text) < 4096


def test_format_buttons_callback_data():
    _, markup = telegram_select.format_message(_item("u"))
    buttons = markup["inline_keyboard"][0]
    assert [b["callback_data"] for b in buttons] == ["rate:must", "rate:worth", "rate:skip"]
    assert [b["text"] for b in buttons] == ["⭐ Must-read", "👍 Worth it", "👎 Skip"]


def test_rated_text_replaces_previous_check():
    text, _ = telegram_select.format_message(_item("u"))
    once = telegram_select.rated_markup_text(text, "must")
    assert once.startswith(text) and once.endswith("✅ ⭐ Must-read")
    twice = telegram_select.rated_markup_text(once, "skip")
    assert twice.endswith("✅ 👎 Skip") and "Must-read" not in twice.split("Read article")[-1]
    assert twice.count("✅") == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
