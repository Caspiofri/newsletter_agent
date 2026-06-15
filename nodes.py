import os
import base64
import json
from bs4 import BeautifulSoup

from datetime import date
from email.utils import parsedate_to_datetime
from googleapiclient.errors import HttpError
from google import genai
from dotenv import load_dotenv

from state import DigestState
from models import Article
import gmail_client

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
gmail = gmail_client.GmailClient()

def fetch_articles(state: DigestState) -> dict:
    label = os.getenv("GMAIL_LABEL", "INBOX")
    messages = gmail.gmail_read_messages(label, days_back=1)
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
    return {"articles": articles}


def fillter_articles(state: DigestState) -> dict:
    articles = state["articles"]
    top_k = state["top_k"]
    if len(articles) <= int(top_k):
        return {"top_articles": articles}

    subject = state["subject"]
    articles_meta = [
        {"name": a.name, "author": a.author, "published_at": str(a.published_at)}
        for a in articles
    ]
    prompt = f"""You are a newsletter writer, you have these articles {json.dumps(articles_meta, ensure_ascii=False)} to
    review, choose the top {top_k} articles for your newsletter writing, the articles
    should be on the subject of {subject}.
    please return ONLY a JSON array of the selected article names, no extra text:
    ["article name 1", "article name 2", "article name 3"]
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    clean_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
    selected_names = json.loads(clean_text)
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

_HTML_SKELETON = """\
<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  {css}
</head>
<body dir="rtl" style="text-align:right">
<div class="wrapper" dir="rtl" style="text-align:right">

  <div class="header">
    <h1 dir="rtl">{title}</h1>
    <div class="date">{today}</div>
  </div>

  <!-- PASTE ARTICLE CARDS HERE (one block per top-scored article):
  <div class="section-label" dir="rtl" style="text-align:right">🔥 עיקרי הגיליון</div>
  <div class="card" dir="rtl" style="text-align:right">
    <div class="card-title" dir="rtl" style="text-align:right">[EMOJI] [HEBREW TITLE]</div>
    <div class="card-source" dir="rtl" style="text-align:right">מקור: [SOURCE NAME]</div>
    <div class="card-label" dir="rtl" style="text-align:right">השורה התחתונה</div>
    <div class="card-body" dir="rtl" style="text-align:right">[1-2 sentence factual summary in Hebrew]</div>
    <div class="card-so-what" dir="rtl" style="text-align:right">
      <div class="card-label" dir="rtl" style="text-align:right">איך זה פוגש אותך</div>
      <div class="card-body" dir="rtl" style="text-align:right">[personalized angle for junior dev transitioning to AI]</div>
    </div>
    <a href="[URL]" class="read-more">קרא עוד &larr;</a>
  </div>
  -->

  <!-- PASTE IN-BRIEF SECTION HERE (for score 3-5 articles):
  <div class="brief-section" dir="rtl" style="text-align:right">
    <h2 dir="rtl" style="text-align:right">⚡ עוד כותרות בקצר</h2>
    <div class="brief-item" dir="rtl" style="text-align:right">[1-sentence summary] — <a href="[URL]">קרא עוד</a></div>
  </div>
  -->

  <!-- PASTE DAILY ACTIONS HERE (always last):
  <div class="actions-section" dir="rtl" style="text-align:right">
    <h2 dir="rtl" style="text-align:right">⚡ המלצות לביצוע</h2>
    <ul dir="rtl" style="text-align:right">
      <li dir="rtl" style="text-align:right">[concrete action 1]</li>
      <li dir="rtl" style="text-align:right">[concrete action 2]</li>
    </ul>
  </div>
  -->

  <div class="footer">Newsletter generated automatically</div>
</div>
</body>
</html>"""


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

    today = date.today().strftime('%d.%m.%Y')
    title = f"{newsletter_name} - {today}"
    skeleton = _HTML_SKELETON.format(title=title, today=today, css=_NEWSLETTER_CSS)

    prompt = f"""You are a Senior Technical Newsletter Editor specializing in {subject}.
Your reader: {target_audience} ({experience_level}) — a Junior Software Engineer transitioning to an AI Engineer role, focused on practical implementation, RAG architectures, and Multi-Agent Systems.

TASK: Fill in the HTML skeleton below with real content. Return ONLY the completed HTML — no markdown wrappers, no commentary.

STEP 1 — SCORE each article internally (0–10) for relevance to: AI engineering, Multi-Agent pipelines, RAG, practical coding tools. Do not output scores.
STEP 2 — CATEGORIZE:
  - Score ≥ 6 → full article card (main section)
  - Score 3–5 → one-line entry in "עוד כותרות בקצר"
  - Score < 3 → exclude entirely
STEP 3 — WRITE each card:
  - "השורה התחתונה": 1-2 sentences. Factual. What happened / what was launched. Zero fluff, no "game-changer".
  - "איך זה פוגש אותך": 2-3 sentences. Concrete. Connect to LangGraph, RAG, or daily dev workflow. Coach the reader directly.
STEP 4 — WRITE "המלצות לביצוע": 1-2 specific actions grounded in today's articles (e.g. "Clone X repo and run the quickstart", "Add Y as a tool node in your LangGraph agent").

RULES:
- Hebrew prose only. Technical terms (LangGraph, RAG, LLM, API, etc.) stay in English.
- Every HTML element must carry dir="rtl" and style="text-align:right" as inline attributes.
- Use ONLY the CSS classes already defined in the skeleton — do not add new <style> blocks.
- ALL HTML comments must be in English.

HTML SKELETON:
{skeleton}

INPUT ARTICLES:
{articles_plain}"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "max_output_tokens": 12000,
            "thinking_config": {"thinking_budget": 4000},
        }
    )
    newsletter_html = response.text.strip().removeprefix("```html").removesuffix("```").strip()
    newsletter_html = newsletter_html.replace('\n', '')
    return {"summary": newsletter_html}


def send_email(state: DigestState) -> dict:
    html_content = state["summary"]
    subject = state['subject']
    sender = state['sender']
    to = state['recipient']
    try:
        gmail.gmail_send_message(to, sender, subject, html_content)
        return {"email_status": "success"}
    except HttpError:
        return {"email_status": "failed"}


def expand_search(state: DigestState) -> dict:
    # TODO: implement when sources list is ready
    return {"tries": state["tries"] + 1}
