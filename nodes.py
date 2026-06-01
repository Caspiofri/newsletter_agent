import os
import base64
import json

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


def summraize(state: DigestState):
    subject = state["subject"]
    target_audience = state["target_audience"]
    top_k = state["top_k"]
    articles = state["articles"]
    experience_level = state["experience_level"]
    newsletter_name = state["digest_name"]

    prompt = f"""
        Role: You are an expert Content Curator and Senior Technical Newsletter Editor specializing in {subject}. Your goal is to filter, score, and summarize the top {top_k} most critical updates tailored specifically for: {target_audience} ({experience_level}) - with a specific focus on a Junior Software Engineer transitioning to an AI Engineer role, who is interested in practical implementation, RAG architectures, and Multi-Agent Systems.

        Objective: Generate a professional, highly scannable Hebrew newsletter in HTML format based ONLY on the provided input articles and their metadata.

        Tone & Style:
        - Professional, direct, and factual. Zero fluff (do NOT use phrases like "game-changer" or generic marketing text). Let the facts speak for themselves.
        - Language: Clean, native Hebrew for prose. Technical concepts, frameworks, and architecture terms MUST remain in English.
        - Formatting: No duplicate titles or mixed languages in headings. Use clear emojis as visual anchors.

        Relevance Scoring & Filtering:
        Analyze the provided articles. Score them based on relevance to AI engineering, Multi-Agent pipelines, RAG architecture, and practical coding tools.
        - Top items: Feature them fully using the "Main Article Structure" below.
        - Low-score but notable items: Place them at the bottom under a category named "עוד כותרות בקצר" (In Brief), with just a 1-sentence summary and a link. Exclude completely irrelevant articles.

        Main Article Structure (For top-scored items):
        Use a strict, scannable format for each article card:
        - [Emoji] [Sharp Title in Hebrew] (Use distinct emojis like 🛠️ for tools, 🔒 for security, 🚀 for releases, 🧠 for models)
        - **מקור:** [Extract and inject the exact source name from the article's metadata]
        - **השורה התחתונה:** (The What) A 1-2 sentence bottom-line explaining exactly what happened or was launched.
        - **איך זה פוגש אותך:** (The So What) A personalized angle explaining why this matters for a junior developer transitioning to AI. Focus on practical implementation, how it connects to architectures like LangGraph/RAG, or its impact on daily development workflows.
        - Include a "קרא עוד" button linking to the original URL.

        Mandatory Final Category:
        ⚡ המלצות לביצוע (Daily Actions): Suggest 1-2 concrete, practical coding, learning, or integration steps based on today's news that the reader can implement immediately to advance their transition to AI engineering.

        Newsletter Title: {newsletter_name} - {date.today().strftime('%d.%m.%Y')}

        Design & HTML Requirements:
        - Rich, modern newsletter design with cards for each article.
        - Use shadows, borders, and spacing to create a clear visual hierarchy.
        - Color theme: Professional slate-blue (#2c3e50) with clean accent colors.
        - CRITICAL RTL: Add dir="rtl" and style="text-align:right" as inline attributes on EVERY html element: table, td, div, p, h1, h2, h3, li
        - Must be fully mobile-responsive.
        - ALL HTML/CSS comments within the code MUST be in English.
        - Return ONLY the raw HTML code block starting with <!DOCTYPE html>. Do NOT include any markdown wrappers.

        Input Articles & Metadata:
        {articles}
        """
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config={"max_output_tokens": 8192}
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
