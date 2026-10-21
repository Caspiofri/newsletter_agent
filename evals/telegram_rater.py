"""Telegram rating bot: one tap on an article message labels it, and the label lands in evals/ranking_golden.json.

State (evals/telegram_state.json) maps each sent message_id to the full report row, so a tap carries the same
fields the labeling hub exports. Every tap is appended to evals/telegram_labels.jsonl (the raw log, never rewritten)
and merged into the golden set through evals.import_labels.merge. Re-check taps go to evals/telegram_recheck.jsonl
and never touch the golden set.
"""
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from config.settings import EVALS_DIR
from evals.import_labels import GOLDEN_RANKING_PATH, VALID_LABELS, merge
from evals.label_hub import DEFAULT_TARGET
from evals.telegram_api import TelegramError
from evals.telegram_select import BUTTONS, rated_markup_text, rating_markup

STATE_PATH = EVALS_DIR / "telegram_state.json"
LOG_PATH = EVALS_DIR / "telegram_labels.jsonl"
RECHECK_PATH = EVALS_DIR / "telegram_recheck.jsonl"
PATHS = {"golden": GOLDEN_RANKING_PATH, "log": LOG_PATH, "recheck": RECHECK_PATH}
HELP = "Tap a button under an article to label it; reply to an article to add a note; /progress for the counts."


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


def _record(message_id: int, entry: dict, paths: dict, today: str) -> None:
    """Log the entry's current label (and note) and, unless it is a re-check, merge it into the golden set."""
    line = {"at": datetime.now().isoformat(timespec="seconds"), "message_id": message_id, "url": entry["row"]["url"],
            "label": entry["label"], "note": entry.get("note", ""), "source_run": entry["source_run"]}
    if entry.get("recheck"):
        _append(paths["recheck"], line)
        return
    _append(paths["log"], line)
    golden = _load_golden(paths["golden"])
    merged, _, _ = merge(golden, {"run_id": entry["source_run"], "labels": [golden_row(entry, entry["label"])]}, today)
    _write_atomic(paths["golden"], merged)


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
    label = (q.get("data") or "").removeprefix("rate:")
    if entry is None or label not in VALID_LABELS:
        _call(api, "answerCallbackQuery", {"callback_query_id": q["id"], "text": "unknown article"})
        return
    entry["label"] = label
    _record(message_id, entry, paths, today)
    _call(api, "answerCallbackQuery", {"callback_query_id": q["id"], "text": f"Saved: {BUTTONS[label]}"})
    _call(api, "editMessageText", {"chat_id": message["chat"]["id"], "message_id": message_id,
                                   "text": rated_markup_text(entry["text"], label), "parse_mode": "HTML",
                                   "disable_web_page_preview": True, "reply_markup": rating_markup()})


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
