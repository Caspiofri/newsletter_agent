"""Offline tests for the Telegram rating bot. Run from the repo root: python -m tests.test_telegram_rater
No network and no real token: the Bot API is a callable passed in and faked."""
import json
import os
import random
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

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
    return {"golden": d / "golden.json", "log": d / "labels.jsonl", "recheck": d / "recheck.jsonl", "out_dir": d}


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
        # a rated message collapses to one line: label and linked title, with a Change button
        assert edit["message_id"] == 100 and edit["text"].startswith("✅ ⭐ Must-read · ") and edit["parse_mode"] == "HTML"
        assert '<a href="https://a">Article A</a>' in edit["text"] and "\n" not in edit["text"]
        assert [b["callback_data"] for b in edit["reply_markup"]["inline_keyboard"][0]] == ["edit"]
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
        assert api.calls[-1][1]["text"].startswith("✅ 👎 Skip · ") and api.calls[-1][1]["text"].count("✅") == 1


def test_change_expands_the_rated_message():
    with tempfile.TemporaryDirectory() as d:
        paths, state, api = _paths(d), _state_with(), FakeApi()
        telegram_rater.handle_update(_tap("rate:worth"), api, CHAT, state, paths, TODAY)
        telegram_rater.handle_update(_tap("edit", update_id=2), api, CHAT, state, paths, TODAY)
        assert api.methods()[-2:] == ["answerCallbackQuery", "editMessageText"]
        edit = api.calls[-1][1]
        # the full card is back, showing the current label, with the three rating buttons
        assert edit["text"].startswith(state["sent"]["100"]["text"]) and edit["text"].endswith("✅ 👍 Worth it")
        assert [b["callback_data"] for b in edit["reply_markup"]["inline_keyboard"][0]] == \
            ["rate:must", "rate:worth", "rate:skip"]
        assert len(_lines(paths["log"])) == 1 and [g["label"] for g in _golden(paths)] == ["worth"]  # nothing written


def test_collapsed_title_is_escaped_and_shortened():
    row = _item("https://a?x=1&y=2", title="a<b & " + "c" * 200)
    text = telegram_select.collapsed_text(row, "skip")
    assert text.startswith("✅ 👎 Skip · ") and "a&lt;b &amp; " in text and 'href="https://a?x=1&amp;y=2"' in text
    assert len(text) < 200 and text.endswith("…</a>")


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


def _post_spy(calls, api=None, fail=None):
    # stands in for human_feedback.post_picks_precision; records (golden, run, Telegram calls made so far)
    def post(golden, source_run, client=None):
        calls.append((golden, source_run, len(api.calls) if api else None))
        if fail:
            raise fail
        return True
    return post


def test_pick_tap_posts_human_score_after_telegram():
    with tempfile.TemporaryDirectory() as d:
        paths, api, posted = _paths(d), FakeApi(), []
        state = _state_with(row=_item("https://a", picked=True, score=8.0, group="picked", title="Article A"))
        with patch.object(telegram_rater.human_feedback, "post_picks_precision", _post_spy(posted, api)):
            telegram_rater.handle_update(_tap("rate:worth"), api, CHAT, state, paths, TODAY)
        [(golden, run, telegram_calls)] = posted
        assert run == "20261021_070000" and telegram_calls == 2  # answer and edit went out first
        assert [(g["url"], g["label"], g["ranker_picked"]) for g in golden] == [("https://a", "worth", True)]
        assert golden == _golden(paths)


def test_non_pick_and_recheck_taps_do_not_post():
    with tempfile.TemporaryDirectory() as d:
        paths, posted = _paths(d), []
        with patch.object(telegram_rater.human_feedback, "post_picks_precision", _post_spy(posted)):
            telegram_rater.handle_update(_tap("rate:must"), FakeApi(), CHAT, _state_with(), paths, TODAY)
            recheck = _state_with(row=_item("https://a", picked=True, score=8.0), recheck=True)
            telegram_rater.handle_update(_tap("rate:must"), FakeApi(), CHAT, recheck, paths, TODAY)
            telegram_rater.handle_update(_tap("edit", update_id=2), FakeApi(), CHAT, recheck, paths, TODAY)
        assert posted == []


def test_pick_tap_survives_feedback_error():
    with tempfile.TemporaryDirectory() as d:
        paths, api, posted = _paths(d), FakeApi(), []
        state = _state_with(row=_item("https://a", picked=True, score=8.0))
        with patch.object(telegram_rater.human_feedback, "post_picks_precision",
                          _post_spy(posted, api, fail=RuntimeError("langsmith down"))):
            telegram_rater.handle_update(_tap("rate:must"), api, CHAT, state, paths, TODAY)
        assert len(posted) == 1 and api.methods() == ["answerCallbackQuery", "editMessageText"]
        assert _golden(paths)[0]["label"] == "must"


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


# ---- send, poll, CLI

def _report(out, profile, ts, items):
    path = out / f"ranking_{profile}_{ts}.json"
    path.write_text(json.dumps({"profile": profile, "run_id": ts, "items": items}), encoding="utf-8")
    return path


def _report_items():
    return [_item(f"https://p{i}", picked=True, score=8) for i in range(5)] + [
        _item("https://near", score=4.5), _item("https://low", score=1), _item("https://s", group="sample")]


def _sender(start=1):
    """A sendMessage result with increasing message ids."""
    ids = iter(range(start, start + 1000))
    return lambda params: {"message_id": next(ids)}


def test_todays_reports_latest_per_profile():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        _report(out, "ai_engineering_daily", "20261021_070001", [])
        latest = _report(out, "ai_engineering_daily", "20261021_093000", [])
        med = _report(out, "medical", "20261021_070500", [])
        _report(out, "ai_engineering_daily", "20261020_070000", [])
        (out / "ranking_labels_ai_20261021_080000.json").write_text("{}", encoding="utf-8")
        (out / "ranking_ai_engineering_daily_20261021_093000.html").write_text("", encoding="utf-8")
        assert sorted(telegram_rater.todays_reports(out, "2026-10-21")) == sorted([latest, med])
        assert telegram_rater.todays_reports(out, "2026-10-22") == []


def test_send_posts_selection_and_records_state():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        report = _report(d, "ai", "20261021_070000", _report_items())
        golden = d / "golden.json"
        golden.write_text(json.dumps([{"url": "https://p4", "label": "must", "source_run": "x"}]), encoding="utf-8")
        api = FakeApi({"sendMessage": _sender()})
        with patch.object(telegram_select, "pick_recheck", return_value=None):
            n = telegram_rater.send([report], api, CHAT, d / "state.json", golden, TODAY)
        assert n == 7 and api.methods() == ["sendMessage"] * 7
        first = api.calls[0][1]
        assert first["chat_id"] == CHAT and first["parse_mode"] == "HTML" and first["disable_web_page_preview"] is True
        assert first["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "rate:must"
        state = telegram_rater.load_state(d / "state.json")
        assert [e["row"]["url"] for e in state["sent"].values()] == \
            ["https://p0", "https://p1", "https://p2", "https://p3", "https://near", "https://low", "https://s"]
        e = state["sent"]["1"]
        assert e["source_run"] == "20261021_070000" and e["recheck"] is False and e["label"] is None
        assert e["note"] == "" and e["text"] == first["text"]


def test_send_skips_failed_article_and_continues():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        report = _report(d, "ai", "20261021_070000", _report_items())
        ids = _sender()

        def send_message(params):
            if "p1" in params["text"]:
                raise telegram_api.TelegramError("sendMessage: Bad Request: message is too long")
            return ids(params)

        api = FakeApi({"sendMessage": send_message})
        with patch.object(telegram_select, "pick_recheck", return_value=None):
            n = telegram_rater.send([report], api, CHAT, d / "state.json", d / "golden.json", TODAY)
        assert n == 7 and len(api.calls) == 8
        urls = [e["row"]["url"] for e in telegram_rater.load_state(d / "state.json")["sent"].values()]
        assert "https://p1" not in urls and len(urls) == 7


def test_send_does_not_resend_same_run():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        report = _report(d, "ai", "20261021_070000", _report_items())
        api = FakeApi({"sendMessage": _sender()})
        with patch.object(telegram_select, "pick_recheck", return_value=None):
            assert telegram_rater.send([report], api, CHAT, d / "state.json", d / "golden.json", TODAY) == 8
            assert telegram_rater.send([report], api, CHAT, d / "state.json", d / "golden.json", TODAY) == 0
        assert len(api.calls) == 8


def test_send_adds_one_recheck_when_due():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        reports = [_report(d, "ai", "20261021_070000", _report_items()),
                   _report(d, "medical", "20261021_070500", [_item("https://m", picked=True, score=9)])]
        golden = d / "golden.json"
        golden.write_text(json.dumps([{"url": "https://g", "title": "Old one", "label": "worth", "origin": "inbox",
                                       "group": "sample", "content": "old text", "source_run": "20261001_070000",
                                       "ranker_picked": False, "ranker_score": None}]), encoding="utf-8")
        api = FakeApi({"sendMessage": _sender()})
        assert telegram_rater.send(reports, api, CHAT, d / "state.json", golden, TODAY) == 10
        rechecks = [c for c in api.calls if "🔁 re-check" in c[1]["text"]]
        assert len(rechecks) == 1 and "Old one" in rechecks[0][1]["text"] and "worth" not in rechecks[0][1]["text"]
        state = telegram_rater.load_state(d / "state.json")
        [entry] = [e for e in state["sent"].values() if e["recheck"]]
        assert entry["row"]["url"] == "https://g" and entry["source_run"] == "20261001_070000"
        assert state["last_recheck"] == TODAY
        # not due again the same week
        telegram_rater.send([_report(d, "ai", "20261021_090000", [_item("https://n", picked=True, score=9)])],
                            api, CHAT, d / "state.json", golden, "2026-10-23")
        assert sum("🔁 re-check" in c[1]["text"] for c in api.calls) == 1


def test_poll_once_advances_offset():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        paths = _paths(d)
        telegram_rater.save_state(d / "state.json", _state_with())
        api = FakeApi({"getUpdates": [_tap("rate:worth", update_id=17), _msg("/progress", update_id=18)]})
        telegram_rater.poll(api, CHAT, d / "state.json", paths, once=True, sleep=lambda s: None)
        assert api.calls[0] == ("getUpdates", {"offset": 0, "timeout": 50,
                                               "allowed_updates": ["message", "callback_query"]})
        state = telegram_rater.load_state(d / "state.json")
        assert state["offset"] == 19 and state["sent"]["100"]["label"] == "worth"
        assert [g["label"] for g in _golden(paths)] == ["worth"]


def test_poll_survives_api_error():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        results = iter([telegram_api.TelegramError("getUpdates: ConnectionError"), [_msg("/progress", update_id=3)]])

        def get_updates(params):
            r = next(results)
            if isinstance(r, Exception):
                raise r
            return r

        slept = []
        api = FakeApi({"getUpdates": get_updates})
        telegram_rater.poll(api, CHAT, d / "state.json", _paths(d), once=True, sleep=slept.append)
        assert slept == [5] and api.methods() == ["getUpdates", "getUpdates", "sendMessage"]
        assert telegram_rater.load_state(d / "state.json")["offset"] == 4


def test_cli_send_without_token_exits_zero():
    env = {k: v for k, v in os.environ.items() if k not in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")}
    with patch.dict(os.environ, env, clear=True), \
            patch.object(telegram_rater, "make_api", side_effect=AssertionError("no API without a token")):
        assert telegram_rater.main(["send"]) == 0


# ---- /more

def _more_reports(d):
    """Two days of reports (two profiles on the older day); the newer report repeats one URL of the older one."""
    _report(d, "ai", "20261019_070000", [
        _item("https://old-p0", picked=True, score=8), _item("https://old-near", score=4.6),
        _item("https://old-low", score=0.5), _item("https://old-s", group="sample"),
        _item("https://dup", picked=True, score=9, title="old dup")])
    _report(d, "medical", "20261019_080000", [_item("https://med-p0", picked=True, score=9)])
    _report(d, "ai", "20261020_070000", [
        _item("https://new-p0", picked=True, score=8), _item("https://new-p1", picked=True, score=7),
        _item("https://new-near", score=4.2), _item("https://new-low", score=2),
        _item("https://new-s", group="sample"), _item("https://dup", score=4.1, title="new dup")])
    (d / "ranking_labels_ai_20261021_080000.json").write_text(json.dumps({"run_id": "x", "items": [
        _item("https://label-file", picked=True, score=9)]}), encoding="utf-8")


def _more(d, text="/more", api=None, state=None):
    paths = _paths(d)
    state = state if state is not None else {"offset": 0, "last_recheck": None, "sent": {}}
    api = api or FakeApi({"sendMessage": _sender(1000)})
    telegram_rater.handle_update(_msg(text), api, CHAT, state, paths, TODAY)
    return api, state


def _sent_urls(api):
    return [re.search(r'href="([^"]+)">Read article', p["text"]).group(1) for m, p in api.calls
            if m == "sendMessage" and "Read article" in p["text"]]


def test_more_default_five_rotates_slots_newest_first():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _more_reports(d)
        api, state = _more(d)
        assert api.calls[0] == ("sendMessage", {"chat_id": CHAT, "text":
                                "Sending 5 more (picks 2 · near cutoff 1 · low score 1 · never ranked 1)"})
        # round 1: one per slot, newest report first; round 2: the next pick
        assert _sent_urls(api) == ["https://new-p0", "https://new-near", "https://new-low", "https://new-s",
                                   "https://new-p1"]
        first = api.calls[1][1]
        assert first["parse_mode"] == "HTML" and first["reply_markup"] == telegram_select.rating_markup()


def test_more_n_and_clamp():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _more_reports(d)
        api, _ = _more(d, "/more 3")
        assert _sent_urls(api) == ["https://new-p0", "https://new-near", "https://new-low"]
        assert api.calls[0][1]["text"] == "Sending 3 more (picks 1 · near cutoff 1 · low score 1)"
        assert len(_sent_urls(_more(d, "/more 0")[0])) == 1
        assert len(_sent_urls(_more(d, "/more abc")[0])) == 5
        # 50 clamps to 20; fewer than 20 exist, so all of them are sent: 11 unique URLs, the label file ignored
        urls = _sent_urls(_more(d, "/more 50")[0])
        assert len(urls) == 11 and "https://label-file" not in urls


def test_more_full_rotation_order_across_reports():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _more_reports(d)
        api, _ = _more(d, "/more 20")
        # dup is in the newest report as a near-cutoff drop (4.1), so it is asked once, from there
        assert _sent_urls(api) == [
            "https://new-p0", "https://new-near", "https://new-low", "https://new-s",
            "https://new-p1", "https://dup", "https://old-low", "https://old-s",
            "https://med-p0", "https://old-near",
            "https://old-p0"]
        assert api.calls[0][1]["text"] == "Sending 11 more (picks 4 · near cutoff 3 · low score 2 · never ranked 2)"


def test_more_excludes_labeled_and_already_sent():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _more_reports(d)
        paths = _paths(d)
        paths["golden"].write_text(json.dumps([{"url": "https://new-p0", "label": "must"},
                                               {"url": "https://new-near", "label": "skip"}]), encoding="utf-8")
        state = _state_with(row=_item("https://new-low", score=2))
        state["sent"]["101"] = {**state["sent"]["100"], "row": _item("https://new-s", group="sample"),
                                "source_run": "20261020_070000"}
        api, _ = _more(d, "/more 4", state=state)
        assert _sent_urls(api) == ["https://new-p1", "https://dup", "https://old-low", "https://old-s"]


def test_more_dedupes_keeping_newest_row_and_source_run():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _more_reports(d)
        api, state = _more(d, "/more 20")
        entries = {e["row"]["url"]: e for e in state["sent"].values()}
        assert len(entries) == len(state["sent"]) == 11
        assert entries["https://dup"]["row"]["title"] == "new dup"
        assert entries["https://dup"]["source_run"] == "20261020_070000"
        assert entries["https://old-p0"]["source_run"] == "20261019_070000"
        assert entries["https://med-p0"]["source_run"] == "20261019_080000"
        e = entries["https://new-p0"]
        assert e["recheck"] is False and e["label"] is None and e["note"] == ""
        assert e["text"] == telegram_select.format_message(e["row"])[0]


def test_more_nothing_left():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _more_reports(d)
        _, state = _more(d, "/more 20")
        api, _ = _more(d, "/more", state=state)
        assert api.calls == [("sendMessage", {"chat_id": CHAT, "text":
                                              "Nothing left to rate — new articles arrive with tomorrow's run."})]
    with tempfile.TemporaryDirectory() as d:  # no reports at all
        api, _ = _more(Path(d))
        assert api.calls[0][1]["text"].startswith("Nothing left to rate")


def test_more_failed_send_continues():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _more_reports(d)
        ids = _sender(1000)

        def send_message(params):
            if "new-near" in params["text"]:
                raise telegram_api.TelegramError("sendMessage: Bad Request")
            return ids(params)

        api, state = _more(d, "/more 3", api=FakeApi({"sendMessage": send_message}))
        assert len(api.calls) == 4
        assert [e["row"]["url"] for e in state["sent"].values()] == ["https://new-p0", "https://new-low"]


def test_more_from_poll_persists_state():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _more_reports(d)
        api = FakeApi({"getUpdates": [_msg("/more 2", update_id=5)], "sendMessage": _sender(1000)})
        telegram_rater.poll(api, CHAT, d / "state.json", _paths(d), once=True, sleep=lambda s: None)
        state = telegram_rater.load_state(d / "state.json")
        assert [e["row"]["url"] for e in state["sent"].values()] == ["https://new-p0", "https://new-near"]
        assert state["offset"] == 6


def test_more_low_sends_lowest_scored_drops_across_reports():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _more_reports(d)
        api, _ = _more(d, "/more low")
        # all drops from every report, lowest score first; picks and never-ranked samples are left out
        assert _sent_urls(api) == ["https://old-low", "https://new-low", "https://dup", "https://new-near",
                                   "https://old-near"]
        assert api.calls[0][1]["text"] == "Sending 5 more (low score 5)"


def test_more_low_n_either_order_and_excludes_labeled():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _more_reports(d)
        assert _sent_urls(_more(d, "/more low 2")[0]) == ["https://old-low", "https://new-low"]
        assert _sent_urls(_more(d, "/more 2 low")[0]) == ["https://old-low", "https://new-low"]
        assert _sent_urls(_more(d, "/more LOW 1")[0]) == ["https://old-low"]
        _paths(d)["golden"].write_text(json.dumps([{"url": "https://old-low", "label": "skip"}]), encoding="utf-8")
        _, state = _more(d, "/more low 1")
        assert [e["row"]["url"] for e in state["sent"].values()] == ["https://new-low"]
        api, state = _more(d, "/more low", api=FakeApi({"sendMessage": _sender(2000)}), state=state)
        assert _sent_urls(api) == ["https://dup", "https://new-near", "https://old-near"]
        api, _ = _more(d, "/more low", api=FakeApi({"sendMessage": _sender(3000)}), state=state)
        assert api.calls == [("sendMessage", {"chat_id": CHAT, "text": telegram_rater.NOTHING_LEFT})]


def test_help_mentions_more():
    with tempfile.TemporaryDirectory() as d:
        api = FakeApi()
        telegram_rater.handle_update(_msg("hello"), api, CHAT, _state_with(), _paths(d), TODAY)
        assert "/more" in api.calls[-1][1]["text"] and "/more low" in api.calls[-1][1]["text"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
