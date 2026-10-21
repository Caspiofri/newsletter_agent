"""What the Telegram rating bot sends each day, and how one article message looks.

select() takes the digest picks of a ranking report plus a few non-picks that fill the gaps in the golden set:
a dropped article near the score cutoff, a low-scored drop, and one from the random sample the ranker never saw.
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
TEASER_CHARS = 280
TITLE_CHARS = 300
MAX_TEXT = 4096  # Telegram's limit for one message


def _is_drop(row: dict) -> bool:
    return row.get("group") == "shortlisted" and not row.get("picked") and row.get("score") is not None


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
    drops = [r for r in rest if _is_drop(r)]
    slots = [
        sorted((r for r in drops if min_score - 1 <= r["score"] < min_score), key=lambda r: -r["score"]),
        sorted(drops, key=lambda r: r["score"]),
        [r for r in rest if r.get("group") == "sample"],
    ]
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
    markup = {"inline_keyboard": [[{"text": t, "callback_data": f"rate:{label}"} for label, t in BUTTONS.items()]]}
    return text, markup


def rated_markup_text(text: str, label: str) -> str:
    """The message text with a final '✅ <button>' line, replacing the line of an earlier tap."""
    base = text.rsplit(f"\n\n{RATED} ", 1)[0]
    return f"{base}\n\n{RATED} {BUTTONS[label]}"
