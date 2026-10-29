"""Telegram rating bot: one tap on an article message labels it, and the label lands in evals/ranking_golden.json.

Usage (from the repo root; TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env):
  python -m evals.telegram_rater setup   # once: press Start in the bot chat, the bot replies with your chat id
  python -m evals.telegram_rater send    # post today's articles (main.py does this after every run)
  python -m evals.telegram_rater poll    # long-running: reads the taps (deploy/telegram-rater.service)

State (evals/telegram_state.json) maps each sent message_id to the full report row, so a tap carries the same
fields the labeling hub exports. Every tap is appended to evals/telegram_labels.jsonl (the raw log, never rewritten)
and merged into the golden set through evals.import_labels.merge. Re-check taps go to evals/telegram_recheck.jsonl
and never touch the golden set. A tap on a digest pick also posts the run's picks precision to LangSmith
(evals.human_feedback).
"""
import argparse
import fcntl
import json
import os
import re
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

from config import settings  # also loads .env
from config.settings import EVALS_DIR, OUTPUT_DIR
from evals import human_feedback, telegram_select
from evals.import_labels import GOLDEN_RANKING_PATH, VALID_LABELS, merge
from evals.label_hub import DEFAULT_TARGET
from evals.telegram_api import TelegramError, make_api
from evals.telegram_select import (BUTTONS, change_markup, collapsed_text, format_message, rated_markup_text,
                                   rating_markup)

STATE_PATH = EVALS_DIR / "telegram_state.json"
LOG_PATH = EVALS_DIR / "telegram_labels.jsonl"
RECHECK_PATH = EVALS_DIR / "telegram_recheck.jsonl"
PATHS = {"golden": GOLDEN_RANKING_PATH, "log": LOG_PATH, "recheck": RECHECK_PATH, "out_dir": OUTPUT_DIR}
_REPORT_RE = re.compile(r"^ranking_(.+)_(\d{8})_(\d{6})\.json$")
POLL_TIMEOUT = 50  # seconds getUpdates waits for an update; the client timeout (60) must stay above it
HELP = ("Tap a button under an article to label it; reply to an article to add a note; /progress for the counts; "
        "/more (or /more N) for more articles to rate; /more low (or /more low N) for only the lowest-scored ones.")
MORE_DEFAULT, MORE_MAX = 5, 20
NOTHING_LEFT = "Nothing left to rate — new articles arrive with tomorrow's run."


def _write_atomic(path: Path, data) -> None:
    # write-to-temp then rename, so a crash never leaves half a file
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@contextmanager
def _locked(state_path: Path):
    # send (cron) and poll (service) both rewrite the state file; a lock file keeps one from undoing the other
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path.with_name(state_path.name + ".lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"offset": 0, "last_recheck": None, "sent": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    _write_atomic(path, state)


def _load_golden(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def golden_row(entry: dict, label: str) -> dict:
    """A sent entry as a golden-set row, with the fields the labeling hub exports."""
    row = entry["row"]
    return {"url": row["url"], "title": row.get("title", ""), "label": label, "tag": "", "note": entry.get("note", ""),
            "labeled_with": "snapshot", "source_run": entry["source_run"], "origin": row.get("origin", ""),
            "group": row.get("group", ""), "read_status": row.get("read_status", ""),
            "content": row.get("content", ""), "ranker_picked": row.get("picked"), "ranker_score": row.get("score")}


def _record(message_id: int, entry: dict, paths: dict, today: str) -> list[dict] | None:
    """Log the entry's current label (and note) and, unless it is a re-check, merge it into the golden set.
    Returns the merged golden set (None for a re-check)."""
    line = {"at": datetime.now().isoformat(timespec="seconds"), "message_id": message_id, "url": entry["row"]["url"],
            "label": entry["label"], "note": entry.get("note", ""), "source_run": entry["source_run"]}
    if entry.get("recheck"):
        _append(paths["recheck"], line)
        return None
    _append(paths["log"], line)
    golden = _load_golden(paths["golden"])
    merged, _, _ = merge(golden, {"run_id": entry["source_run"], "labels": [golden_row(entry, entry["label"])]}, today)
    _write_atomic(paths["golden"], merged)
    return merged


def _call(api, method: str, params: dict) -> None:
    # a failed answer or edit (query too old, "message is not modified") must not lose the label already saved
    try:
        api(method, params)
    except TelegramError as e:
        print(f"[telegram] {e}")


def _progress(golden_path: Path) -> str:
    golden = [g for g in _load_golden(golden_path) if g.get("label") in VALID_LABELS]
    counts = " · ".join(f"{k} {sum(g['label'] == k for g in golden)}" for k in BUTTONS)
    return f"{len(golden)} / {DEFAULT_TARGET} · {counts}"


def _on_callback(q: dict, api, state: dict, paths: dict, today: str) -> None:
    message = q.get("message") or {}
    message_id = message.get("message_id")
    entry = state["sent"].get(str(message_id))
    data = q.get("data") or ""
    label = data.removeprefix("rate:")
    edit = lambda text, markup: _call(api, "editMessageText", {
        "chat_id": message["chat"]["id"], "message_id": message_id, "text": text, "parse_mode": "HTML",
        "disable_web_page_preview": True, "reply_markup": markup})
    if entry is not None and data == "edit":
        # expand a collapsed message back to the full card; nothing is written until the next rating tap
        _call(api, "answerCallbackQuery", {"callback_query_id": q["id"], "text": "Pick a new rating"})
        text = rated_markup_text(entry["text"], entry["label"]) if entry.get("label") else entry["text"]
        edit(text, rating_markup())
        return
    if entry is None or label not in VALID_LABELS:
        _call(api, "answerCallbackQuery", {"callback_query_id": q["id"], "text": "unknown article"})
        return
    entry["label"] = label
    golden = _record(message_id, entry, paths, today)
    _call(api, "answerCallbackQuery", {"callback_query_id": q["id"], "text": f"Saved: {BUTTONS[label]}"})
    edit(collapsed_text(entry["row"], label), change_markup())
    if golden is not None and entry["row"].get("picked"):
        # after Telegram has its answer: LangSmith being slow or down must never cost the tap
        try:
            human_feedback.post_picks_precision(golden, entry["source_run"])
        except Exception as e:
            print(f"[warn] human picks score: {e}")


def _more_args(text: str) -> tuple[int, bool]:
    # "/more", "/more 3", "/more@bot 3", "/more low 3", "/more 3 low": a missing or non-number N is the
    # default, any number is clamped; "low" anywhere asks for low-scored drops only
    args = text.split()[1:]
    low = any(a.lower() == "low" for a in args)
    nums = [a for a in args if a.lower() != "low"]
    try:
        n = int(nums[0]) if nums else MORE_DEFAULT
    except ValueError:
        n = MORE_DEFAULT
    return max(1, min(MORE_MAX, n)), low


def _send_more(text: str, api, chat: str, state: dict, paths: dict) -> None:
    """/more: N extra articles from all reports, newest first (/more low: lowest-scored drops first),
    skipping labeled and already-sent URLs."""
    skip = {g["url"] for g in _load_golden(paths["golden"]) if g.get("url")}
    skip |= {e["row"]["url"] for e in state["sent"].values()}
    reports = [_read_report(p) for p in all_reports(paths["out_dir"])]
    n, low = _more_args(text)
    if low:
        chosen = telegram_select.select_low(reports, skip, n)
    else:
        chosen = telegram_select.select_more(reports, skip, settings.RANK_MIN_SCORE, n)
    if not chosen:
        _call(api, "sendMessage", {"chat_id": chat, "text": NOTHING_LEFT})
        return
    counts = " · ".join(f"{s} {n}" for s in telegram_select.MORE_SLOTS
                        if (n := sum(name == s for name, _, _ in chosen)))
    _call(api, "sendMessage", {"chat_id": chat, "text": f"Sending {len(chosen)} more ({counts})"})
    for _, row, run in chosen:
        _send_one(api, chat, state, row, run, recheck=False)  # a failed article is printed and skipped


def _on_message(m: dict, api, chat: str, state: dict, paths: dict, today: str) -> None:
    text = (m.get("text") or "").strip()
    reply = lambda t, **kw: _call(api, "sendMessage", {"chat_id": chat, "text": t, **kw})
    replied = (m.get("reply_to_message") or {}).get("message_id")
    if replied is not None and not text.startswith("/"):
        entry = state["sent"].get(str(replied))
        if entry is None:
            reply("unknown article", reply_to_message_id=m["message_id"])
            return
        entry["note"] = text
        if entry.get("label"):
            _record(replied, entry, paths, today)
        reply("Note saved", reply_to_message_id=m["message_id"])
    elif text.startswith("/progress"):
        reply(_progress(paths["golden"]))
    elif text.startswith("/more"):
        _send_more(text, api, chat, state, paths)
    else:
        reply(HELP)


def handle_update(update: dict, api, chat_id: str | None, state: dict, paths: dict, today: str) -> None:
    """Act on one getUpdates item. Only chat_id is served; without one, /start replies with the chat id for setup."""
    q = update.get("callback_query")
    m = update.get("message")
    chat = str(((q or {}).get("message") or m or {}).get("chat", {}).get("id", ""))
    if chat_id is None:
        if m and (m.get("text") or "").startswith("/start"):
            api("sendMessage", {"chat_id": chat, "text": f"Your chat id is {chat}. Put TELEGRAM_CHAT_ID={chat} in .env"})
        return
    if chat != str(chat_id):
        return
    if q:
        _on_callback(q, api, state, paths, today)
    elif m:
        _on_message(m, api, chat, state, paths, today)


# ---- send, poll, CLI

def todays_reports(out_dir: Path, today: str) -> list[Path]:
    """The latest ranking report of each profile written today (exported label files are not reports)."""
    day = today.replace("-", "")
    latest: dict[str, tuple[str, Path]] = {}
    for p in out_dir.glob(f"ranking_*_{day}_*.json"):
        m = _REPORT_RE.match(p.name)
        if not m or m.group(2) != day or m.group(1).startswith("labels_"):
            continue
        if m.group(1) not in latest or m.group(3) > latest[m.group(1)][0]:
            latest[m.group(1)] = (m.group(3), p)
    return [p for _, p in latest.values()]


def all_reports(out_dir: Path) -> list[Path]:
    """Every ranking report in out_dir, newest first (exported label files are not reports)."""
    found = []
    for p in out_dir.glob("ranking_*.json"):
        m = _REPORT_RE.match(p.name)
        if m and not m.group(1).startswith("labels_"):
            found.append((m.group(2) + m.group(3), p))
    return [p for _, p in sorted(found, reverse=True)]


def _read_report(path: Path) -> tuple[str, list[dict]]:
    """(run_id, items) of one report file; the run id falls back to the timestamp in the file name."""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("items", [])
    m = _REPORT_RE.match(path.name)
    run = (data.get("run_id") if isinstance(data, dict) else None) or (f"{m.group(2)}_{m.group(3)}" if m else "")
    return run, items


def _send_one(api, chat_id: str, state: dict, row: dict, source_run: str, recheck: bool) -> bool:
    text, markup = format_message(row, recheck=recheck)
    try:
        msg = api("sendMessage", {"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                                  "disable_web_page_preview": True, "reply_markup": markup})
    except TelegramError as e:
        print(f"[telegram] could not send {row.get('url')}: {e}")
        return False
    state["sent"][str(msg["message_id"])] = {"row": row, "source_run": source_run, "recheck": recheck,
                                             "text": text, "label": None, "note": ""}
    return True


def _recheck_row(g: dict) -> dict:
    # a golden row back in report-row shape, without its label, so the message cannot anchor the answer
    return {"url": g["url"], "title": g.get("title", ""), "origin": g.get("origin", ""), "group": g.get("group", ""),
            "read_status": g.get("read_status", ""), "content": g.get("content", ""),
            "picked": g.get("ranker_picked"), "score": g.get("ranker_score")}


def send(report_paths: list[Path], api, chat_id: str, state_path: Path, golden_path: Path, today: str) -> int:
    """Post each report's selection (and at most one re-check) as rating messages. Returns the number sent."""
    golden = _load_golden(golden_path)
    skip = {g["url"] for g in golden if g.get("url")}
    n = 0
    with _locked(state_path):
        state = load_state(state_path)
        for path in report_paths:
            run, items = _read_report(path)
            already = {e["row"]["url"] for e in state["sent"].values() if e["source_run"] == run}
            for row in telegram_select.select(items, skip | already, settings.RANK_MIN_SCORE, settings.TELEGRAM_EXTRA):
                skip.add(row["url"])  # the same article in two profiles' reports is asked once
                if _send_one(api, chat_id, state, row, run, recheck=False):
                    n += 1
                    save_state(state_path, state)
        g = telegram_select.pick_recheck(golden, state.get("last_recheck"), today,
                                         settings.TELEGRAM_RECHECK_EVERY_DAYS) if report_paths else None
        if g and _send_one(api, chat_id, state, _recheck_row(g), g.get("source_run", ""), recheck=True):
            n += 1
            state["last_recheck"] = today
            save_state(state_path, state)
    return n


def send_today(today: str | None = None) -> int:
    """send() for today's reports with the token and chat id from the environment; no token: one line, nothing sent."""
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[telegram] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; nothing sent for rating")
        return 0
    today = today or str(date.today())
    reports = todays_reports(OUTPUT_DIR, today)
    n = send(reports, make_api(token), chat_id, STATE_PATH, GOLDEN_RANKING_PATH, today)
    print(f"[telegram] {n} article(s) sent for rating from {len(reports)} report(s)")
    return n


def poll(api, chat_id: str | None, state_path: Path, paths: dict, once: bool = False, sleep=time.sleep) -> None:
    """Long-poll getUpdates and handle each update; never exits on an API error (once=True: one successful batch)."""
    failures = 0
    while True:
        try:
            updates = api("getUpdates", {"offset": load_state(state_path)["offset"], "timeout": POLL_TIMEOUT,
                                         "allowed_updates": ["message", "callback_query"]})
        except TelegramError as e:
            failures += 1
            print(f"[telegram] {e}")
            sleep(30 if failures >= 5 else 5)
            continue
        failures = 0
        with _locked(state_path):
            state = load_state(state_path)
            for update in updates:
                try:
                    handle_update(update, api, chat_id, state, paths, str(date.today()))
                except Exception:
                    # one bad update must not block the queue: it is logged and skipped
                    print(f"[telegram] update {update.get('update_id')} failed:\n{traceback.format_exc()}")
                state["offset"] = update["update_id"] + 1
                save_state(state_path, state)
        if once:
            return


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m evals.telegram_rater", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["send", "poll", "setup"])
    cmd = ap.parse_args(argv).command
    if cmd == "send":
        send_today()
        return 0
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID") or None
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set: create a bot with @BotFather and put the token in .env")
        return 1
    api = make_api(token)
    if cmd == "setup":
        print("Press Start (or send /start) in the bot chat in Telegram; the bot replies with your chat id.")
        offset = load_state(STATE_PATH)["offset"]
        for _ in range(6):  # about five minutes of long polls
            poll(api, None, STATE_PATH, PATHS, once=True)
            if load_state(STATE_PATH)["offset"] != offset:
                return 0
        print("No message reached the bot; run setup again.")
        return 1
    if chat_id is None:
        print("[telegram] TELEGRAM_CHAT_ID is not set: only /start is answered (run setup)")
    poll(api, chat_id, STATE_PATH, PATHS)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
