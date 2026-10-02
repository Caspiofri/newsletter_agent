import asyncio
import os
import traceback
import config.settings  # noqa: F401  (loads .env)
from modules.graph import graph
import requests

from modules import article_store, trace_judge

GRAPH_TIMEOUT = 600
HEALTHCHECK_PING_URL = os.getenv("HEALTHCHECK_PING_URL")


def _ping_healthcheck(ok: bool, message: str = ""):
    if not HEALTHCHECK_PING_URL:
        return
    url = HEALTHCHECK_PING_URL if ok else f"{HEALTHCHECK_PING_URL}/fail"
    try:
        requests.post(url, data=message.encode("utf-8")[:10000], timeout=10)
    except requests.RequestException as e:
        print(f"[warn] Failed to ping healthcheck: {e}")


async def run_profile(profile: str) -> str:
    state = {
        "gmail_label": os.getenv(f"{profile}_GMAIL_LABEL_ID"),
        "subject": os.getenv(f"{profile}_SUBJECT"),
        "target_audience": os.getenv(f"{profile}_AUDIENCE"),
        "experience_level": os.getenv(f"{profile}_EXPERIENCE"),
        "digest_name": os.getenv(f"{profile}_DIGEST_NAME"),
        "recipient": os.getenv(f"{profile}_RECIPIENT"),
        "sender": os.getenv("NO_REPLAY_EMAIL"),
        "top_k": os.getenv("TOP_K"),
        "articles": [],
        "top_articles": [],
        "newsletter_data": None,
        "summary": "",
        "email_status": "",
        "tries": 0,
        "max_tries": 3,
        "days_back": 1,
        "max_days_back": 7,
    }
    db_profile = os.getenv(f"{profile}_DIGEST_NAME", profile).lower().replace(" ", "_")
    purged = article_store.purge_old(db_profile)
    if purged:
        print(f"[{profile}] Purged {purged} article(s) older than {article_store.RETENTION_DAYS} days.")

    try:
        result = await asyncio.wait_for(
            graph.ainvoke(state, config={"configurable": {"thread_id": "1"}}),
            timeout=GRAPH_TIMEOUT,
        )
    except asyncio.TimeoutError:
        print(f"\n[{profile}] Graph timed out after {GRAPH_TIMEOUT}s — no email sent.")
        return "timeout"

    data = result.get("newsletter_data")
    if data:
        print(f"\nEmail status: {result['email_status']}")
        print(f"Articles fetched: {len(result['articles'])}")
        print(f"Top articles: {[a.name for a in result['top_articles']]}")
        print(f"Full cards ({len(data.cards)}):")
        for c in data.cards:
            print(f"  - {c.title}")
        print(f"Brief items ({len(data.brief_items)}):")
        for b in data.brief_items:
            print(f"  - {b.summary[:80]}")
        print(f"Actions:")
        for a in data.actions:
            print(f"  → {a}")
    else:
        print(f"\nNo articles found. Email status: {result['email_status']}")

    trace_judge.judge_run(result, profile)
    return result["email_status"]


async def main():
    failures = []
    #for profile in ["AI", "MEDICAL"]:
    for profile in ["AI"]:
        try:
            status = await run_profile(profile)
        except Exception:
            print(f"\n[{profile}] Crashed:\n{traceback.format_exc()}")
            failures.append(f"[{profile}] crashed:\n{traceback.format_exc()}")
            continue
        if status in ("failed", "timeout"):
            failures.append(f"[{profile}] email_status={status}")

    if failures:
        _ping_healthcheck(ok=False, message="\n\n".join(failures))
    else:
        _ping_healthcheck(ok=True)


asyncio.run(main())
