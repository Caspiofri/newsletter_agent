"""What the Telegram rating bot sends each day, and how one article message looks.

select() takes the digest picks of a ranking report plus a few non-picks that fill the gaps in the golden set:
a dropped article near the score cutoff, a low-scored drop, and one from the random sample the ranker never saw.
select_more() serves the bot's /more command with the same slots, drawn from all reports, newest first.
Pure functions only; the bot (evals.telegram_rater) does the sending.
"""
import html
import random
from datetime import date

from bs4 import BeautifulSoup

# label -> button text; the order is the order of the buttons under each message
BUTTONS = {"must": "⭐ Must-read", "worth": "👍 Worth it", "skip": "👎 Skip"}
NOT_IN_DIGEST = "🔍 not in today's digest"
RECHECK = "🔁 re-check"
RATED = "✅"
CHANGE = "✏️ Change"
COLLAPSED_TITLE_CHARS = 80
TEASER_CHARS = 280
TITLE_CHARS = 300
MAX_TEXT = 4096  # Telegram's limit for one message
MORE_SLOTS = ("picks", "near cutoff", "low score", "never ranked")  # /more rotates through these, in order


def _is_drop(row: dict) -> bool:
    return row.get("group") == "shortlisted" and not row.get("picked") and row.get("score") is not None


def _nonpick_slots(rest: list[dict], min_score: float, age=lambda r: 0) -> list[list[dict]]:
    """Non-picks by slot: near the cutoff (highest first), low-scored drops (lowest first), never ranked.
    `age` (0 = newest report) sorts ahead of the score, so newer reports come first within a slot."""
    drops = [r for r in rest if _is_drop(r)]
    return [
        sorted((r for r in drops if min_score - 1 <= r["score"] < min_score), key=lambda r: (age(r), -r["score"])),
        sorted(drops, key=lambda r: (age(r), r["score"])),
        sorted((r for r in rest if r.get("group") == "sample"), key=age),
    ]


def select(items: list[dict], labeled_urls: set[str], min_score: float, extra: int = 3) -> list[dict]:
    """All picks (report order), then up to `extra` non-picks by slot: near the cutoff, low-scored, never ranked."""
    seen = set(labeled_urls)
    rows = []
    for r in items:
        if r.get("url") and r["url"] not in seen:
            seen.add(r["url"])
            rows.append(r)
    picks = [r for r in rows if r.get("picked")]
    rest = [r for r in rows if not r.get("picked")]
    slots = _nonpick_slots(rest, min_score)
    chosen: list[dict] = []
    for i in range(extra):
        # an empty slot is filled from the next slot type, then from any remaining non-pick
        k = i % len(slots)
        for candidates in slots[k:] + [rest]:
            row = next((r for r in candidates if all(r is not c for c in chosen)), None)
            if row is not None:
                chosen.append(row)
                break
    return picks + chosen


def select_more(reports: list[tuple[str, list[dict]]], skip_urls: set[str], min_score: float,
                n: int) -> list[tuple[str, dict, str]]:
    """Up to n (slot, row, run_id) from (run_id, items) reports given newest first, one per slot per round
    in MORE_SLOTS order; an empty slot is skipped. A URL in several reports counts once, from the newest."""
    seen = set(skip_urls)
    rows, run_of, age = [], {}, {}
    for i, (run, items) in enumerate(reports):
        for r in items:
            if r.get("url") and r["url"] not in seen:
                seen.add(r["url"])
                rows.append(r)
                run_of[r["url"]], age[r["url"]] = run, i
    picks = [r for r in rows if r.get("picked")]
    slots = [picks] + _nonpick_slots([r for r in rows if not r.get("picked")], min_score, lambda r: age[r["url"]])
    chosen: list[tuple[str, dict, str]] = []
    taken: set[str] = set()
    while len(chosen) < n:
        added = False
        for name, candidates in zip(MORE_SLOTS, slots):
            if len(chosen) == n:
                break
            # a near-cutoff drop is also a low-scored drop: whichever slot reaches it first takes it
            row = next((r for r in candidates if r["url"] not in taken), None)
            if row is not None:
                taken.add(row["url"])
                chosen.append((name, row, run_of[row["url"]]))
                added = True
        if not added:
            break
    return chosen


def select_low(reports: list[tuple[str, list[dict]]], skip_urls: set[str], n: int) -> list[tuple[str, dict, str]]:
    """/more low: up to n ("low score", row, run_id) drops from all reports, lowest score first.
    A URL in several reports counts once, from the newest."""
    seen = set(skip_urls)
    drops = []
    for run, items in reports:
        for r in items:
            if r.get("url") and r["url"] not in seen:
                seen.add(r["url"])
                if not r.get("picked") and _is_drop(r):
                    drops.append((r, run))
    drops.sort(key=lambda d: d[0]["score"])
    return [("low score", r, run) for r, run in drops[:n]]


def pick_recheck(golden: list[dict], last_recheck: str | None, today: str, every_days: int = 7,
                 rng=random) -> dict | None:
    """One random already-labeled article to ask again, at most once every `every_days` days."""
    if not golden:
        return None
    if last_recheck and (date.fromisoformat(today) - date.fromisoformat(last_recheck)).days < every_days:
        return None
    return rng.choice(golden)


def _teaser(content: str, limit: int = TEASER_CHARS) -> str:
    text = " ".join(BeautifulSoup(content or "", "html.parser").get_text(" ").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return (cut.rsplit(" ", 1)[0] if " " in cut else cut) + "…"


def rating_markup() -> dict:
    """One row of the three rating buttons."""
    return {"inline_keyboard": [[{"text": t, "callback_data": f"rate:{label}"} for label, t in BUTTONS.items()]]}


def format_message(row: dict, recheck: bool = False) -> tuple[str, dict]:
    """(HTML text, reply_markup) for one article message."""
    title = row.get("title") or row.get("url", "")
    if len(title) > TITLE_CHARS:
        title = title[:TITLE_CHARS] + "…"
    lines = [f"<b>{html.escape(title, quote=False)}</b>", html.escape(row.get("origin") or "", quote=False)]
    if recheck:
        lines.append(RECHECK)
    elif not row.get("picked"):
        lines.append(NOT_IN_DIGEST)
    teaser = _teaser(row.get("content", ""))
    if teaser:
        lines += ["", html.escape(teaser, quote=False)]
    link = f'<a href="{html.escape(row.get("url", ""))}">Read article</a>'
    text = "\n".join(lines + ["", link])
    if len(text) > MAX_TEXT - 40:  # an absurdly long URL: drop the link rather than fail the send
        text = "\n".join(lines)
    return text, rating_markup()


def rated_markup_text(text: str, label: str) -> str:
    """The message text with a final '✅ <button>' line, replacing the line of an earlier tap."""
    base = text.rsplit(f"\n\n{RATED} ", 1)[0]
    return f"{base}\n\n{RATED} {BUTTONS[label]}"


def collapsed_text(row: dict, label: str) -> str:
    """A rated message shrunk to one line: '✅ <button> · <linked title>'."""
    title = row.get("title") or row.get("url", "")
    if len(title) > COLLAPSED_TITLE_CHARS:
        title = title[:COLLAPSED_TITLE_CHARS].rstrip() + "…"
    link = f'<a href="{html.escape(row.get("url", ""))}">{html.escape(title, quote=False)}</a>'
    return f"{RATED} {BUTTONS[label]} · {link}"


def change_markup() -> dict:
    """The single button under a collapsed message; it brings back the full card and the rating buttons."""
    return {"inline_keyboard": [[{"text": CHANGE, "callback_data": "edit"}]]}
