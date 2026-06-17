import asyncio
from dotenv import load_dotenv
from graph import graph
import os.path

import gmail_client

GRAPH_TIMEOUT = 60


async def run_profile(profile: str):
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
        "max_tries": 5,
        "article_count_last": -1,
    }
    try:
        result = await asyncio.wait_for(
            graph.ainvoke(state, config={"configurable": {"thread_id": "1"}}),
            timeout=GRAPH_TIMEOUT,
        )
    except asyncio.TimeoutError:
        print(f"\n[{profile}] Graph timed out after {GRAPH_TIMEOUT}s — no email sent.")
        return

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


async def main():
    #for profile in ["AI", "MEDICAL"]:
    for profile in ["AI"]:
        await run_profile(profile)


asyncio.run(main())
