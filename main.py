from dotenv import load_dotenv
from state import DigestState
from graph import graph
import os.path

import gmail_client
print(dir(gmail_client.GmailClient))

#for profile in ["AI", "MEDICAL"]:
for profile in ["AI"]:
 graph.invoke(
    {
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
        "summary": "",
        "email_status": "",
        "tries": 0,
        "max_tries": 5
    },
    config={"configurable": {"thread_id": "1"}}
)
