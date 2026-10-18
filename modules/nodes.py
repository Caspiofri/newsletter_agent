import os
import re
import base64
import json
from concurrent.futures import as_completed

from langsmith.utils import ContextThreadPoolExecutor as ThreadPoolExecutor  # copies trace context into workers
from datetime import date
from email.utils import parsedate_to_datetime
from googleapiclient.errors import HttpError
import google.auth
from google import genai
from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential

from modules.state import DigestState
from config import settings
from modules.collectors import hackernews, medium
from modules.models import Article, NewsletterData
from modules.article_text import article_text
from modules.card_resolver import resolve_cards
from modules import gmail_client, article_store, link_utils, ranking_report, reader


_credentials, _project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
client = genai.Client(vertexai=True, project=_project, location="us-central1", credentials=_credentials)
gmail = gmail_client.GmailClient()


@traceable(run_type="llm", name="gemini")
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


def _pick_by_ids(items: list, pool: list[Article], limit: int) -> list[Article]:
    """Map the LLM's ids to articles: ints in range only, LLM order, no duplicates, at most `limit`."""
    picked, seen = [], set()
    for i in items:
        if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(pool) and i not in seen:
            seen.add(i)
            picked.append(pool[i])
    return picked[:limit]


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
        {"id": i, "name": a.name, "author": a.author, "published_at": str(a.published_at)}
        for i, a in enumerate(articles)
    ]
    prompt = f"""You are a newsletter writer for a digest about {subject}.
    From these articles {json.dumps(articles_meta, ensure_ascii=False)}, select up to {top_k}
    that are GENUINELY on the subject of {subject}.
    EXCLUDE any article that is off-topic or only tangentially related — it is better to
    return fewer articles (or none at all) than to include irrelevant ones.
    If NONE of the articles are relevant to {subject}, return an empty array [].
    please return ONLY a JSON array of the selected article ids, no extra text,
    for example [2, 0], or [] if none:
    """
    text = _call_gemini(prompt)
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in filter response: {text!r}")
    top_articles = _pick_by_ids(json.loads(match.group()), articles, top_k)
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


def _sanitize_ids(parsed: dict) -> dict:
    """Null any card/brief id that is not a real int (str, bool, float, missing) so resolve_cards drops it."""
    for key in ("cards", "brief_items"):
        for item in parsed.get(key) or []:
            if isinstance(item, dict) and (isinstance(item.get("id"), bool) or not isinstance(item.get("id"), int)):
                item["id"] = None
    return parsed


def summraize(state: DigestState):
    subject = state["subject"]
    target_audience = state["target_audience"]
    top_k = state["top_k"]
    articles = state["top_articles"]
    experience_level = state["experience_level"]
    newsletter_name = state["digest_name"]

    # The model refers to articles by list index; resolve_cards maps each id back to the real URL.
    articles_for_prompt = [
        {"id": i, "title": a.name, "author": a.author, "signal": a.signal, "origin": a.origin, "read_status": a.read_status, "text": article_text(a)}
        for i, a in enumerate(articles)
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
      "id": <int, the "id" of the input article>
    }}
  ],
  "brief_items": [
    {{
      "summary": "[1-sentence Hebrew summary]",
      "id": <int, the "id" of the input article>
    }}
  ],
  "actions": [
    "[Specific action 1 in Hebrew]",
    "[Specific action 2 in Hebrew]"
  ]
}}

INPUT ARTICLES (use each article's "id" exactly as given):
{json.dumps(articles_for_prompt, ensure_ascii=False)}"""

    text = _call_gemini(prompt, config={
        "max_output_tokens": 8000,
        "thinking_config": {"thinking_budget": 4000},
    })

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in newsletter response: {text!r}")
    data = resolve_cards(NewsletterData(**_sanitize_ids(json.loads(match.group()))), articles)

    today = date.today().strftime('%d.%m.%Y')
    title = f"{newsletter_name} - {today}"
    # A URL is "bad" (rendered without a link) if its source link was dead/unverifiable or missing.
    bad_urls = frozenset(a.url or "" for a in articles if a.link_status != "ok" or not a.url)
    html = _render_html(data, title, today, bad_urls)

    stats = {**state.get("run_stats", {}), "cards": len(data.cards), "brief_items": len(data.brief_items)}
    return {"summary": html, "newsletter_data": data, "run_stats": stats}


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
    """Drop stories already sent in the last few days, before they can take a shortlist slot."""
    profile = state["digest_name"].lower().replace(" ", "_")
    unseen, matches = article_store.filter_seen_detailed(state["articles"], profile)
    dropped = len(state["articles"]) - len(unseen)
    if dropped:
        print(f"[dedupe] Dropped {dropped} article(s) seen in the last {article_store.DEDUPE_DAYS} days.")
    stats = {**state.get("run_stats", {}), "after_dedupe": len(unseen), "dedupe_dropped": dropped}
    return {"articles": unseen, "dedupe_drops": matches, "run_stats": stats}


def shortlist_articles(state: DigestState) -> dict:
    """Title screen run separately per source, so one noisy source cannot crowd out the others."""
    articles = state["articles"]
    origins = sorted({a.origin for a in articles})

    def screen(origin: str) -> list[Article]:
        group = [a for a in articles if a.origin == origin]
        quota = settings.SHORTLIST_QUOTA.get(origin, 5)
        return fillter_articles({**state, "articles": group, "top_k": quota})["top_articles"]

    with ThreadPoolExecutor(max_workers=max(len(origins), 1)) as pool:
        candidates = [a for group in pool.map(screen, origins) for a in group]
    print(f"[shortlist] {len(candidates)} of {len(articles)} candidates: "
          f"{ {o: sum(c.origin == o for c in candidates) for o in origins} }")
    return {"candidates": candidates, "run_stats": {**state.get("run_stats", {}), "shortlisted": len(candidates)}}


def read_articles(state: DigestState) -> dict:
    candidates = reader.read_articles(state["candidates"], max_chars=settings.READ_MAX_CHARS)
    stats = {s: sum(a.read_status == s for a in candidates) for s in ("full", "partial", "snippet")}
    print(f"[reader] read status: {stats}")
    run_stats = {**state.get("run_stats", {}), **{f"read_{s}": n for s, n in stats.items()}}
    return {"candidates": candidates, "run_stats": run_stats}


_SOURCE_DESCRIPTIONS = {
    "inbox": "newsletter email curated by an editor",
    "medium": "Medium tag feed — unvetted, anyone can publish; often beginner tutorials or self-promotion",
    "hackernews": "Hacker News front-page style submission voted by engineers",
}


def _final_scores(scored: list, candidates: list[Article], snippet_cap: int) -> dict[int, float]:
    """id -> score. Unknown ids and malformed rows are ignored; snippet-only articles are capped."""
    best: dict[int, float] = {}
    for row in scored:
        if not isinstance(row, dict) or not isinstance(row.get("id"), int):
            continue
        if 0 <= row["id"] < len(candidates) and isinstance(row.get("score"), (int, float)):
            best[row["id"]] = max(best.get(row["id"], 0), row["score"])
    return {i: min(sc, snippet_cap) if candidates[i].read_status == "snippet" else sc for i, sc in best.items()}


def _select_top(scored: list, candidates: list[Article], top_k: int, min_score: int,
                snippet_cap: int = 10) -> list[Article]:
    """Best-first articles with final score >= min_score."""
    final = _final_scores(scored, candidates, snippet_cap)
    ranked = sorted((i for i, sc in final.items() if sc >= min_score), key=lambda i: -final[i])
    return [candidates[i] for i in ranked[:top_k]]


def rank_articles(state: DigestState) -> dict:
    """One LLM call scores every candidate on its content and source; the best top_k become top_articles."""
    candidates = state["candidates"]
    if not candidates:
        return {"top_articles": [], "rank_scores": [],
                "run_stats": {**state.get("run_stats", {}), "ranked": 0, "picked": 0}}
    top_k = int(state["top_k"])
    items = [
        {
            "id": i,
            "title": a.name,
            "source": _SOURCE_DESCRIPTIONS.get(a.origin, a.origin) + (f" ({a.signal})" if a.signal else ""),
            "text_available": a.read_status,
            "text": a.content[:2500],
        }
        for i, a in enumerate(candidates)
    ]
    prompt = f"""You are the editor of a daily digest about {state["subject"]}.
Reader: {state["target_audience"]} ({state["experience_level"]}).

Score each candidate article 0-10 for how much this reader would value reading it, judging the TEXT, not just the title.
Reward: concrete engineering substance, primary sources (releases, papers, engineering blogs), novel results, practical techniques for RAG / agents / LLM systems.
Penalize: beginner tutorials, listicles, self-promotion, vague hype, thin teasers with no substance, off-topic items.
"text_available" tells how much text you have: "full", "partial", or "snippet" (teaser only). Do not reward a good title when the text is empty of substance.
Calibration: 9-10 = must-read; 7-8 = clearly worth the reader's time; 5-6 = acceptable filler; 3-4 = skip; 0-2 = off-topic or junk.
Be strict: on a typical day only about {top_k} candidates deserve 7 or above, and most should land in 3-6.

Return ONLY a JSON array, no commentary: [{{"id": 0, "score": 7, "reason": "short reason"}}]

CANDIDATES:
{json.dumps(items, ensure_ascii=False)}"""

    text = _call_gemini(prompt)
    match = re.search(r'\[.*\]', text or "", re.DOTALL)
    try:
        scored = json.loads(match.group()) if match else []
    except json.JSONDecodeError:
        scored = []
    if not scored:
        raise ValueError(f"No usable scores in ranker response: {text!r}")
    final = _final_scores(scored, candidates, settings.SNIPPET_SCORE_CAP)
    top = _select_top(scored, candidates, top_k, settings.RANK_MIN_SCORE, settings.SNIPPET_SCORE_CAP)
    reasons = {r["id"]: r.get("reason", "") for r in scored if isinstance(r, dict) and isinstance(r.get("id"), int)}
    picked = {id(a) for a in top}
    for i in sorted(final, key=lambda i: -final[i]):
        a = candidates[i]
        mark = "*" if id(a) in picked else " "
        print(f"[rank]{mark}{final[i]:>4}  {a.origin:<10} {a.read_status:<7} {a.name[:55]} — {reasons.get(i, '')[:60]}")
    try:  # the labeling report is a by-product; it must never break a run
        profile = state["digest_name"].lower().replace(" ", "_")
        rows = ranking_report.build_rows(candidates, final, reasons, top, state["articles"], seed=str(date.today()))
        print(f"[rank] Ranking report for labeling: {ranking_report.save(profile, rows)}")
    except Exception as e:
        print(f"[rank] could not write ranking report: {e}")
    rank_scores = [
        {"title": candidates[i].name, "origin": candidates[i].origin, "read_status": candidates[i].read_status,
         "score": final[i], "reason": reasons.get(i, ""), "picked": id(candidates[i]) in picked}
        for i in sorted(final, key=lambda i: -final[i])
    ]
    stats = {**state.get("run_stats", {}), "ranked": len(final), "picked": len(top)}
    return {"top_articles": top, "rank_scores": rank_scores, "run_stats": stats}


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
    """Explode each newsletter email into individual Article objects with real URLs."""
    emails = state["articles"]
    if not emails:
        return {"articles": []}
    all_sub_articles: list[Article] = []

    with ThreadPoolExecutor(max_workers=len(emails)) as pool:
        futures = {pool.submit(_extract_from_email, e): e for e in emails}
        for future in as_completed(futures):
            all_sub_articles.extend(future.result())

    return {"articles": all_sub_articles}


def collect_medium(state: DigestState) -> dict:
    return {"extra_articles": medium.collect(
        state["medium_tags"], state["subject"], settings.COLLECTOR_MAX_AGE_DAYS)}


def collect_hackernews(state: DigestState) -> dict:
    return {"extra_articles": hackernews.collect(
        state["hn_queries"], state["subject"], settings.COLLECTOR_MAX_AGE_DAYS, settings.HN_MIN_POINTS)}


def merge_sources(state: DigestState) -> dict:
    """Join point for all collectors: de-duplicate by URL, check every link, swap bad links for direct ones."""
    pool = list(state["articles"]) + list(state["extra_articles"])
    by_url = {}
    for a in pool:
        by_url.setdefault(a.url, a)
    articles = list(by_url.values())

    status = link_utils.classify_links([a.url for a in articles])
    checked = [a.model_copy(update={"link_status": status[a.url]}) for a in articles]
    bad_before = sum(a.link_status != "ok" for a in checked)
    upgraded = article_store.upgrade_bad_links(checked)
    bad_after = sum(a.link_status != "ok" for a in upgraded)

    per_source = {o: sum(a.origin == o for a in upgraded) for o in sorted({a.origin for a in upgraded})}
    print(f"[merge] {len(upgraded)} articles by source: {per_source}")
    if bad_before:
        print(f"[merge] {bad_before} article(s) had a dead/unverifiable link; "
              f"{bad_before - bad_after} replaced by a direct link, {bad_after} remain.")
    return {"articles": upgraded, "run_stats": {"merged": len(upgraded)}}


def expand_search(state: DigestState) -> dict:
    return {
        "days_back": state.get("days_back", 1) + 1,
    }


def no_content(state: DigestState) -> dict:
    return {"email_status": "no_content"}
