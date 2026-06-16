# Newsletter Agent

An AI-powered newsletter automation pipeline that ingests emails from Gmail, filters and summarizes them using Google Gemini, and delivers a formatted HTML digest back to a recipient.

## How It Works

The agent is built as a stateful graph using **LangGraph**. Each run flows through the following nodes:

```
fetch → [check results] → filter → summarize → send_email
              ↓
        expand_search (retry if no articles found)
```

1. **Fetch** — pulls emails from a configured Gmail label using the Gmail API
2. **Filter** — uses Gemini to rank and select the top-k most relevant articles by subject
3. **Summarize** — generates a professional, mobile-responsive Hebrew newsletter in HTML via Gemini
4. **Send** — delivers the digest via Gmail to the configured recipient

## Tech Stack

- **Python**
- **LangGraph** — stateful agent graph with conditional retry logic
- **Google Gemini API** (`gemini-2.5-flash` / `gemini-2.5-pro`) — article ranking and newsletter generation
- **Gmail API** — OAuth 2.0 authenticated email ingestion and delivery
- **Pydantic** — data modeling

## Setup

### 1. Prerequisites

- Python 3.10+
- A Google Cloud project with the Gmail API enabled
- A `credentials.json` file from the Google Cloud Console (OAuth 2.0 Desktop App)
- A Google Gemini API key

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key

GMAIL_LABEL=your_gmail_label_id   # e.g. Label_123456

DIGEST_SUBJECT=AI Engineering
DIGEST_AUDIENCE=Junior Software Engineers
DIGEST_EXPERIENCE=Junior
DIGEST_NAME=The AI Digest

NO_REPLAY_EMAIL=you@example.com
RECIPIENT_EMAIL=recipient@example.com

TOP_K=5
```

### 4. Authenticate with Gmail

On the first run the browser will open for OAuth consent. A `token.json` file is saved locally for subsequent runs.

### 5. Run

```bash
python main.py
```

## Reliability

Production fault tolerance is built into the pipeline:

- **LLM retries** — every Gemini API call retries up to 3 times with exponential backoff (2s → 4s → 8s) using `tenacity`. Handles transient network errors, rate limits, and 5xx responses without crashing.
- **Graph timeout** — the full pipeline execution is wrapped in a 60-second async timeout. If Gmail or Gemini hangs, the run exits cleanly with a logged message instead of blocking indefinitely.

## Project Structure

```
├── main.py          # Entry point — invokes the compiled graph
├── graph.py         # LangGraph node wiring and conditional edges
├── nodes.py         # Node implementations (fetch, filter, summarize, send)
├── state.py         # Shared DigestState TypedDict
├── models.py        # Article Pydantic model
├── gmail_client.py  # Gmail API wrapper (auth, read, send)
└── requirements.txt
```
