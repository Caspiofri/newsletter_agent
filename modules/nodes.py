import os
import re
import base64
import json
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

from datetime import date
from email.utils import parsedate_to_datetime
from googleapiclient.errors import HttpError
import google.auth
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

from modules.state import DigestState
from modules.models import Article, NewsletterData
from modules import gmail_client, article_store, link_utils

import config.settings  # noqa: F401  (loads .env)

_credentials, _project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
client = genai.Client(vertexai=True, project=_project, location="us-central1", credentials=_credentials)
gmail = gmail_client.GmailClient()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_gemini(contents: str, config: dict | None = None) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config or {},
    )
    return response.text

def fetch_articles(state: DigestState) -> dict:
    label = state["gmail_label"]
    messages = gmail.gmail_read_messages(label, days_back=state.get("days_back", 1))
    articles = []
    for message in messages:
        payload = message['payload']
        subject = next((h['value'] for h in payload['headers'] if h['name'] == 'Subject'), '')
        author = next((h['value'] for h in payload['headers'] if h['name'] == 'From'), '')
        date_str = next((h['value'] for h in payload['headers'] if h['name'] == 'Date'), '')
        url = next((h['value'] for h in payload['headers'] if h['name'] == 'Archived-At'), '')
        published_at = parsedate_to_datetime(date_str).date()
        if 'parts' not in payload:
            content = payload.get('body', {}).get('data', '')
            if content:
                decoded_content = base64.urlsafe_b64decode(content + '==').decode('utf-8')
                articles.append(Article(
                    name=subject,
                    subject=state["subject"],
                    author=author,
                    published_at=published_at,
                    content=decoded_content,
                    url=url
                ))
        else:
            for part in payload['parts']:
                if part['mimeType'] == 'text/html':
                    content = part['body']['data']
                    decoded_content = base64.urlsafe_b64decode(content + '==').decode('utf-8')
                    articles.append(Article(
                        name=subject,
                        subject=state["subject"],
                        author=author,
                        published_at=published_at,
                        content=decoded_content,
                        url=url
                    ))
                    break  # one HTML body per email is enough
    return {"articles": articles}


def fillter_articles(state: DigestState) -> dict:
    articles = state["articles"]
    if not articles:
        return {"top_articles": []}

    top_k = int(state["top_k"])
    subject = state["subject"]
    # Prefer stories with a working link; fall back to the full pool only if there are too few.
    good = [a for a in articles if a.link_status == "ok"]
    if len(good) >= top_k:
        articles = good
    articles_meta = [
        {"name": a.name, "author": a.author, "published_at": str(a.published_at)}
        for a in articles
    ]
    prompt = f"""You are a newsletter writer for a digest about {subject}.
    From these articles {json.dumps(articles_meta, ensure_ascii=False)}, select up to {top_k}
    that are GENUINELY on the subject of {subject}.
    EXCLUDE any article that is off-topic or only tangentially related — it is better to
    return fewer articles (or none at all) than to include irrelevant ones.
    If NONE of the articles are relevant to {subject}, return an empty array [].
    please return ONLY a JSON array of the selected article names, no extra text:
    ["article name 1", "article name 2", "article name 3"]
    """
    text = _call_gemini(prompt)
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in filter response: {text!r}")
    selected_names = json.loads(match.group())
    top_articles = [a for a in articles if a.name in selected_names]
    return {"top_articles": top_articles}


_NEWSLETTER_CSS = """\
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #f0f4f8; font-family: 'Segoe UI', Arial, sans-serif; direction: rtl; }
  .wrapper { max-width: 680px; margin: 0 auto; background: #ffffff; }
  .header { background: linear-gradient(135deg, #2c3e50 0%, #3d5a80 100%); padding: 32px 24px; text-align: center; }
  .header h1 { color: #ffffff; font-size: 26px; font-weight: 700; letter-spacing: 0.5px; }
  .header .date { color: #a8c6e0; font-size: 14px; margin-top: 6px; }
  .section-label { background: #2c3e50; color: #ffffff; font-size: 13px; font-weight: 600;
    padding: 6px 16px; margin: 24px 24px 0 24px; border-radius: 4px 4px 0 0;
    display: inline-block; }
  .card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 0 8px 8px 8px;
    margin: 0 24px 20px 24px; padding: 20px 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .card-title { font-size: 18px; font-weight: 700; color: #1a202c; margin-bottom: 8px; }
  .card-source { font-size: 12px; color: #718096; margin-bottom: 14px; }
  .card-label { font-size: 12px; font-weight: 700; color: #2c3e50; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 4px; }
  .card-body { font-size: 14px; color: #4a5568; line-height: 1.65; margin-bottom: 14px; }
  .card-so-what { background: #eef2ff; border-right: 4px solid #3d5a80; padding: 10px 14px;
    border-radius: 0 6px 6px 0; margin-bottom: 16px; }
  .card-so-what .card-label { color: #3d5a80; }
  .read-more { display: inline-block; background: #2c3e50; color: #ffffff; font-size: 13px;
    font-weight: 600; padding: 8px 20px; border-radius: 6px; text-decoration: none; }
  .read-more.disabled { background: #cbd5e0; color: #718096; cursor: not-allowed; }
  .link-disabled { color: #a0aec0; font-size: 12px; }
  .brief-section { margin: 0 24px 20px 24px; padding: 16px 20px;
    border: 1px solid #e2e8f0; border-radius: 8px; background: #f8fafc; }
  .brief-section h2 { font-size: 15px; font-weight: 700; color: #2c3e50; margin-bottom: 12px; }
  .brief-item { font-size: 13px; color: #4a5568; line-height: 1.6; margin-bottom: 8px;
    padding-bottom: 8px; border-bottom: 1px dashed #e2e8f0; }
  .brief-item:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
  .brief-item a { color: #3d5a80; text-decoration: none; font-weight: 600; }
  .actions-section { background: #1a202c; margin: 0 24px 28px 24px; padding: 20px 22px;
    border-radius: 8px; }
  .actions-section h2 { font-size: 16px; font-weight: 700; color: #fbbf24; margin-bottom: 12px; }
  .actions-section li { font-size: 14px; color: #e2e8f0; line-height: 1.6; margin-bottom: 6px;
    padding-right: 16px; list-style: none; }
  .actions-section li::before { content: "→ "; color: #fbbf24; font-weight: 700; }
  .footer { text-align: center; padding: 16px; font-size: 12px; color: #a0aec0; }
  @media (max-width: 600px) {
    .card, .brief-section, .actions-section, .section-label { margin-right: 12px; margin-left: 12px; }
    .header h1 { font-size: 20px; }
  }
</style>"""


def _read_more(url: str, bad_urls) -> str:
    if url in bad_urls:
        return '<span class="read-more disabled">הקישור אינו זמין</span>'
    return f'<a href="{url}" class="read-more">קרא עוד &larr;</a>'


def _brief_link(url: str, bad_urls) -> str:
    if url in bad_urls:
        return '<span class="link-disabled">(הקישור אינו זמין)</span>'
    return f'<a href="{url}">קרא עוד</a>'


def _render_html(data: NewsletterData, title: str, today: str, bad_urls: frozenset = frozenset()) -> str:
    cards_html = ""
    if data.cards:
        cards_html += '  <div class="section-label" dir="rtl" style="text-align:right">🔥 עיקרי הגיליון</div>\n'
        for card in data.cards:
            cards_html += (
                f'  <div class="card" dir="rtl" style="text-align:right">\n'
                f'    <div class="card-title" dir="rtl" style="text-align:right">{card.title}</div>\n'
                f'    <div class="card-source" dir="rtl" style="text-align:right">מקור: {card.source}</div>\n'
                f'    <div class="card-label" dir="rtl" style="text-align:right">השורה התחתונה</div>\n'
                f'    <div class="card-body" dir="rtl" style="text-align:right">{card.brief}</div>\n'
                f'    <div class="card-so-what" dir="rtl" style="text-align:right">\n'
                f'      <div class="card-label" dir="rtl" style="text-align:right">איך זה פוגש אותך</div>\n'
                f'      <div class="card-body" dir="rtl" style="text-align:right">{card.personalization}</div>\n'
                f'    </div>\n'
                f'    {_read_more(card.url, bad_urls)}\n'
                f'  </div>\n'
            )

    brief_html = ""
    if data.brief_items:
        items = "\n".join(
            f'    <div class="brief-item" dir="rtl" style="text-align:right">'
            f'{item.summary} — {_brief_link(item.url, bad_urls)}</div>'
            for item in data.brief_items
        )
        brief_html = (
            f'  <div class="brief-section" dir="rtl" style="text-align:right">\n'
            f'    <h2 dir="rtl" style="text-align:right">⚡ עוד כותרות בקצר</h2>\n'
            f'{items}\n'
            f'  </div>\n'
        )

    actions_html = ""
    if data.actions:
        items = "\n".join(
            f'      <li dir="rtl" style="text-align:right">{action}</li>'
            for action in data.actions
        )
        actions_html = (
            f'  <div class="actions-section" dir="rtl" style="text-align:right">\n'
            f'    <h2 dir="rtl" style="text-align:right">⚡ המלצות לביצוע</h2>\n'
            f'    <ul dir="rtl" style="text-align:right">\n'
            f'{items}\n'
            f'    </ul>\n'
            f'  </div>\n'
        )

    return (
        f'<!DOCTYPE html>\n'
        f'<html lang="he" dir="rtl">\n'
        f'<head>\n'
        f'  <meta charset="UTF-8">\n'
        f'  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f'  <title>{title}</title>\n'
        f'  {_NEWSLETTER_CSS}\n'
        f'</head>\n'
        f'<body dir="rtl" style="text-align:right">\n'
        f'<div class="wrapper" dir="rtl" style="text-align:right">\n\n'
        f'  <div class="header">\n'
        f'    <h1 dir="rtl">{title}</h1>\n'
        f'    <div class="date">{today}</div>\n'
        f'  </div>\n\n'
        f'{cards_html}\n'
        f'{brief_html}\n'
        f'{actions_html}\n'
        f'  <div class="footer">Newsletter generated automatically</div>\n'
        f'</div>\n'
        f'</body>\n'
        f'</html>'
    )


def summraize(state: DigestState):
    subject = state["subject"]
    target_audience = state["target_audience"]
    top_k = state["top_k"]
    articles = state["top_articles"]
    experience_level = state["experience_level"]
    newsletter_name = state["digest_name"]

    articles_plain = [
        a.model_copy(update={
            "content": BeautifulSoup(a.content, "html.parser").get_text(separator=" ", strip=True)[:2000]
        })
        for a in articles
    ]

    prompt = f"""You are a Senior Technical Newsletter Editor specializing in {subject}.
Your reader: {target_audience} ({experience_level}) — a Junior Software Engineer transitioning to an AI Engineer role, focused on practical implementation, RAG architectures, and Multi-Agent Systems.

TASK: Analyze the articles and return a JSON object with newsletter content. Return ONLY valid JSON — no markdown wrappers, no commentary.

STEP 1 — SCORE each article internally (0-10) for relevance to: AI engineering, Multi-Agent pipelines, RAG, practical coding tools. Do not output scores.
STEP 2 — CATEGORIZE (all {top_k} articles must appear):
  - Score ≥ 6 → full entry in "cards"
  - Score < 6 → one-line entry in "brief_items"
STEP 3 — WRITE each card:
  - "brief": 1-2 sentences. Factual. What happened / what was launched. Zero fluff, no "game-changer".
  - "personalization": 2-3 sentences. Concrete. Connect to LangGraph, RAG, or daily dev workflow. Coach the reader directly.
STEP 4 — WRITE "actions": 1-2 specific actions grounded in today's articles.

RULES:
- Hebrew prose only. Technical terms (LangGraph, RAG, LLM, API, etc.) stay in English.
- All {top_k} articles must appear (either as a card or brief_item).
- Provide 1-2 items in "actions".

OUTPUT FORMAT:
{{
  "cards": [
    {{
      "title": "[EMOJI] [HEBREW TITLE]",
      "source": "[SOURCE NAME]",
      "brief": "[1-2 sentence factual summary in Hebrew]",
      "personalization": "[2-3 sentences connecting to the reader's context in Hebrew]",
      "url": "[URL]"
    }}
  ],
  "brief_items": [
    {{
      "summary": "[1-sentence Hebrew summary]",
      "url": "[URL]"
    }}
  ],
  "actions": [
    "[Specific action 1 in Hebrew]",
    "[Specific action 2 in Hebrew]"
  ]
}}

INPUT ARTICLES:
{articles_plain}"""

    text = _call_gemini(prompt, config={
        "max_output_tokens": 8000,
        "thinking_config": {"thinking_budget": 4000},
    })

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in newsletter response: {text!r}")
    data = NewsletterData(**json.loads(match.group()))

    today = date.today().strftime('%d.%m.%Y')
    title = f"{newsletter_name} - {today}"
    # A URL is "bad" if its source link was dead/unverifiable, or if the LLM returned one that was not in the input.
    good_urls = {a.url for a in articles if a.link_status == "ok"}
    bad_urls = frozenset(
        u for u in [c.url for c in data.cards] + [b.url for b in data.brief_items] if u not in good_urls
    )
    html = _render_html(data, title, today, bad_urls)

    return {"summary": html, "newsletter_data": data}


def send_email(state: DigestState) -> dict:
    html_content = state["summary"]
    subject = state['subject']
    sender = state['sender']
    to = state['recipient']
    try:
        gmail.gmail_send_message(to, sender, subject, html_content)
    except HttpError:
        return {"email_status": "failed"}

    profile = state["digest_name"].lower().replace(" ", "_")
    try:
        article_store.store_articles(state["top_articles"], profile)
    except Exception as e:
        print(f"[warn] Failed to store articles in ChromaDB: {e}")

    return {"email_status": "success"}


def dedupe_articles(state: DigestState) -> dict:
    profile = state["digest_name"].lower().replace(" ", "_")
    unseen = article_store.filter_seen(state["top_articles"], profile)
    dropped = len(state["top_articles"]) - len(unseen)
    if dropped:
        print(f"[dedupe] Dropped {dropped} article(s) seen in the last {article_store.DEDUPE_DAYS} days.")
    return {"top_articles": unseen}


def _map_selection_to_articles(items: list, candidates: list[dict], email_article: Article) -> list[Article]:
    """Turn the LLM's picks (candidate ids) into Articles. URLs come only from the email's real links."""
    by_id = {c["id"]: c for c in candidates}
    results, seen_urls = [], set()
    for item in items:
        if not isinstance(item, dict):
            continue
        cand = by_id.get(item.get("id"))
        title = str(item.get("title", "")).strip()
        if cand is None or not title or cand["url"] in seen_urls:
            continue
        seen_urls.add(cand["url"])
        results.append(Article(
            name=title,
            subject=email_article.subject,
            author=email_article.author,
            published_at=email_article.published_at,
            content=str(item.get("snippet", "")).strip(),
            url=cand["url"],
        ))
    return results


def _extract_from_email(email_article: Article) -> list[Article]:
    candidates = link_utils.extract_candidate_links(email_article.content)
    if not candidates:
        return []
    body_text = link_utils.email_plain_text(email_article.content)
    candidates_json = json.dumps(
        [{"id": c["id"], "text": c["text"], "context": c["context"]} for c in candidates],
        ensure_ascii=False,
    )
    prompt = f"""You are parsing a newsletter email. Identify every individual article or story it links to.

You are given the email's plain text and a numbered list of the links found in it. Choose ONLY from this list.
For each article return:
- "id": the id of the link that leads to that article (an integer from the list)
- "title": the article headline (string)
- "snippet": 2-3 sentence plain-text summary, based only on the email text and link context

Skip links that are ads, sponsors, unsubscribe, social, "read in app" duplicates of the same article, or navigation.
Never invent an article that is not in the email. Return ONLY a valid JSON array, no markdown:
[{{"id": 0, "title": "...", "snippet": "..."}}]
If there are no real articles, return [].

EMAIL TEXT:
{body_text}

LINKS:
{candidates_json}"""

    text = _call_gemini(prompt)
    if not text:
        return []
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    return _map_selection_to_articles(items, candidates, email_article)


def extract_sub_articles(state: DigestState) -> dict:
    """Explode each newsletter email into individual Article objects with real, live URLs."""
    emails = state["articles"]
    all_sub_articles: list[Article] = []

    with ThreadPoolExecutor(max_workers=len(emails)) as pool:
        futures = {pool.submit(_extract_from_email, e): e for e in emails}
        for future in as_completed(futures):
            all_sub_articles.extend(future.result())

    status = link_utils.classify_links([a.url for a in all_sub_articles])
    checked = [a.model_copy(update={"link_status": status[a.url]}) for a in all_sub_articles]
    bad_before = sum(a.link_status != "ok" for a in checked)
    upgraded = article_store.upgrade_bad_links(checked)
    bad_after = sum(a.link_status != "ok" for a in upgraded)
    if bad_before:
        print(f"[extract] {bad_before} article(s) had a dead/unverifiable link; "
              f"{bad_before - bad_after} replaced by a direct link from another newsletter, {bad_after} remain.")
    return {"articles": upgraded}


def expand_search(state: DigestState) -> dict:
    return {
        "days_back": state.get("days_back", 1) + 1,
    }


def no_content(state: DigestState) -> dict:
    return {"email_status": "no_content"}
