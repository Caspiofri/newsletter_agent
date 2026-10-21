"""Offline tests for the Telegram rating bot. Run from the repo root: python -m tests.test_telegram_rater
No network and no real token: the Bot API is a callable passed in and faked."""
import json
import random
import tempfile
from pathlib import Path

import requests

from evals import telegram_api, telegram_rater, telegram_select

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


# ---- update handling

CHAT = "42"
TODAY = "2026-10-21"


class FakeApi:
    """Records every call; `fail` maps a method name to an exception to raise."""
    def __init__(self, results=None, fail=None):
        self.calls, self.results, self.fail = [], results or {}, fail or {}

    def __call__(self, method, params):
        self.calls.append((method, params))
        if method in self.fail:
            raise self.fail[method]
        r = self.results.get(method, True)
        return r(params) if callable(r) else r

    def methods(self):
        return [m for m, _ in self.calls]


def _paths(d):
    d = Path(d)
    return {"golden": d / "golden.json", "log": d / "labels.jsonl", "recheck": d / "recheck.jsonl"}


def _state_with(message_id=100, row=None, recheck=False, label=None, note=""):
    row = row or _item("https://a", score=4.5, group="shortlisted", title="Article A")
    text, _ = telegram_select.format_message(row, recheck=recheck)
    return {"offset": 0, "last_recheck": None,
            "sent": {str(message_id): {"row": row, "source_run": "20261021_070000", "recheck": recheck,
                                       "text": text, "label": label, "note": note}}}


def _tap(data="rate:must", message_id=100, chat=CHAT, update_id=1):
    return {"update_id": update_id, "callback_query": {"id": f"cb{update_id}", "data": data,
                                                       "message": {"message_id": message_id, "chat": {"id": int(chat)}}}}


def _msg(text, chat=CHAT, reply_to=None, update_id=1):
    m = {"message_id": 900 + update_id, "chat": {"id": int(chat)}, "text": text}
    if reply_to is not None:
        m["reply_to_message"] = {"message_id": reply_to}
    return {"update_id": update_id, "message": m}


def _lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _golden(paths):
    return json.loads(paths["golden"].read_text(encoding="utf-8")) if paths["golden"].exists() else []


def test_tap_labels_and_merges_golden():
    with tempfile.TemporaryDirectory() as d:
        paths, state, api = _paths(d), _state_with(), FakeApi()
        telegram_rater.handle_update(_tap("rate:must"), api, CHAT, state, paths, TODAY)
        assert api.methods() == ["answerCallbackQuery", "editMessageText"]
        assert api.calls[0][1]["text"] == "Saved: ⭐ Must-read"
        edit = api.calls[1][1]
        assert edit["message_id"] == 100 and edit["text"].endswith("✅ ⭐ Must-read") and edit["parse_mode"] == "HTML"
        assert [b["callback_data"] for b in edit["reply_markup"]["inline_keyboard"][0]] == \
            ["rate:must", "rate:worth", "rate:skip"]
        assert state["sent"]["100"]["label"] == "must"
        log = _lines(paths["log"])
        assert len(log) == 1 and {k: log[0][k] for k in ("message_id", "url", "label", "note", "source_run")} == \
            {"message_id": 100, "url": "https://a", "label": "must", "note": "", "source_run": "20261021_070000"}
        [g] = _golden(paths)
        assert g["url"] == "https://a" and g["label"] == "must" and g["labeled_with"] == "snapshot"
        assert g["ranker_picked"] is False and g["ranker_score"] == 4.5 and g["group"] == "shortlisted"
        assert g["source_run"] == "20261021_070000" and g["labeled_on"] == TODAY and g["tag"] == ""
        assert g["title"] == "Article A" and g["origin"] == "medium" and g["content"]


def test_second_tap_changes_label_one_row_two_log_lines():
    with tempfile.TemporaryDirectory() as d:
        paths, state, api = _paths(d), _state_with(), FakeApi()
        telegram_rater.handle_update(_tap("rate:must"), api, CHAT, state, paths, TODAY)
        telegram_rater.handle_update(_tap("rate:skip", update_id=2), api, CHAT, state, paths, TODAY)
        assert [g["label"] for g in _golden(paths)] == ["skip"]
        assert [x["label"] for x in _lines(paths["log"])] == ["must", "skip"]
        assert api.calls[-1][1]["text"].endswith("✅ 👎 Skip") and api.calls[-1][1]["text"].count("✅") == 1


def test_unknown_message_answered_nothing_written():
    with tempfile.TemporaryDirectory() as d:
        paths, state, api = _paths(d), _state_with(), FakeApi()
        telegram_rater.handle_update(_tap("rate:must", message_id=555), api, CHAT, state, paths, TODAY)
        assert api.methods() == ["answerCallbackQuery"] and api.calls[0][1]["text"] == "unknown article"
        assert not paths["log"].exists() and not paths["golden"].exists()


def test_foreign_chat_ignored():
    with tempfile.TemporaryDirectory() as d:
        paths, state, api = _paths(d), _state_with(), FakeApi()
        telegram_rater.handle_update(_tap("rate:must", chat="7"), api, CHAT, state, paths, TODAY)
        telegram_rater.handle_update(_msg("/progress", chat="7"), api, CHAT, state, paths, TODAY)
        telegram_rater.handle_update(_msg("/start", chat="7"), api, CHAT, state, paths, TODAY)
        assert api.calls == [] and not paths["golden"].exists() and state["sent"]["100"]["label"] is None


def test_note_before_tap_is_attached():
    with tempfile.TemporaryDirectory() as d:
        paths, state, api = _paths(d), _state_with(), FakeApi()
        telegram_rater.handle_update(_msg("great on evals", reply_to=100), api, CHAT, state, paths, TODAY)
        assert api.calls[-1] == ("sendMessage", {"chat_id": CHAT, "text": "Note saved", "reply_to_message_id": 901})
        assert not paths["golden"].exists()
        telegram_rater.handle_update(_tap("rate:worth", update_id=2), api, CHAT, state, paths, TODAY)
        [g] = _golden(paths)
        assert g["label"] == "worth" and g["note"] == "great on evals"
        assert _lines(paths["log"])[-1]["note"] == "great on evals"


def test_note_after_tap_remerges():
    with tempfile.TemporaryDirectory() as d:
        paths, state, api = _paths(d), _state_with(), FakeApi()
        telegram_rater.handle_update(_tap("rate:must"), api, CHAT, state, paths, TODAY)
        telegram_rater.handle_update(_msg("too basic", reply_to=100, update_id=2), api, CHAT, state, paths, TODAY)
        [g] = _golden(paths)
        assert g["label"] == "must" and g["note"] == "too basic"
        assert state["sent"]["100"]["note"] == "too basic"


def test_recheck_tap_goes_to_recheck_file_not_golden():
    with tempfile.TemporaryDirectory() as d:
        paths = _paths(d)
        golden = [{"url": "https://a", "title": "Article A", "label": "must", "source_run": "20261001_070000"}]
        paths["golden"].write_text(json.dumps(golden), encoding="utf-8")
        state, api = _state_with(recheck=True), FakeApi()
        telegram_rater.handle_update(_tap("rate:skip"), api, CHAT, state, paths, TODAY)
        assert _golden(paths) == golden and not paths["log"].exists()
        [r] = _lines(paths["recheck"])
        assert r["url"] == "https://a" and r["label"] == "skip"
        assert api.methods() == ["answerCallbackQuery", "editMessageText"]


def test_progress_counts():
    with tempfile.TemporaryDirectory() as d:
        paths, api = _paths(d), FakeApi()
        labels = ["must"] * 3 + ["worth"] * 2 + ["skip"]
        paths["golden"].write_text(json.dumps([{"url": f"u{i}", "label": l} for i, l in enumerate(labels)]),
                                   encoding="utf-8")
        telegram_rater.handle_update(_msg("/progress"), api, CHAT, _state_with(), paths, TODAY)
        assert api.calls[-1][1]["text"] == "6 / 80 · must 3 · worth 2 · skip 1"


def test_start_without_chat_id_reports_id():
    with tempfile.TemporaryDirectory() as d:
        paths, api = _paths(d), FakeApi()
        telegram_rater.handle_update(_msg("/start", chat="777"), api, None, _state_with(), paths, TODAY)
        assert api.calls == [("sendMessage", {"chat_id": "777", "text":
                                              "Your chat id is 777. Put TELEGRAM_CHAT_ID=777 in .env"})]
        # without a chat id nothing else is served
        telegram_rater.handle_update(_tap("rate:must", chat="777"), api, None, _state_with(), paths, TODAY)
        assert len(api.calls) == 1 and not paths["golden"].exists()
        # with a chat id, /start answers a help line
        telegram_rater.handle_update(_msg("/start"), api, CHAT, _state_with(), paths, TODAY)
        assert api.calls[-1][1]["chat_id"] == CHAT and "/progress" in api.calls[-1][1]["text"]


def test_state_roundtrip_and_missing_file():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sub" / "state.json"
        assert telegram_rater.load_state(path) == {"offset": 0, "last_recheck": None, "sent": {}}
        state = _state_with()
        state["offset"] = 12
        telegram_rater.save_state(path, state)
        assert telegram_rater.load_state(path) == state
        assert [p.name for p in path.parent.iterdir()] == ["state.json"]  # no temp file left behind


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
