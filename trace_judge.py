"""
trace_judge.py — Per-run LLM-as-judge for the newsletter pipeline.

Complements eval.py (which tests nodes against fixtures offline) by judging
a real pipeline run — evaluating loop efficiency, filter decisions, and
content faithfulness on live data.

Judges three stages:
  1. fetch_health     — loop efficiency (tries, convergence, article yield)
  2. filter_quality   — did the LLM pick the right articles from what was fetched?
  3. content_quality  — faithfulness of the newsletter to the source articles

Scores: 1–5 per stage. Saves report to logs/trace_<timestamp>.json.

Usage:
  from trace_judge import judge_run
  report = judge_run(result, profile="AI")
"""

import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime

import google.auth
from dotenv import load_dotenv
from google import genai

from models import Article, NewsletterData

load_dotenv()

_credentials, _project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
_client = genai.Client(vertexai=True, project=_project, location="us-central1", credentials=_credentials)

SKIPPED = "skipped"


@dataclass
class StageScore:
    stage: str
    score: int | str  # 1–5, or SKIPPED
    reasoning: str
    details: dict


@dataclass
class TraceReport:
    profile: str
    timestamp: str
    email_status: str
    stages: list[StageScore]
    overall: float  # average of numeric stage scores


# ---------------------------------------------------------------------------
# LLM judge helper
# ---------------------------------------------------------------------------

def _llm_judge(criterion: str, rubric: str, content: str) -> dict:
    """Return {"score": int 1-5, "reasoning": str}."""
    prompt = f"""You are a strict pipeline quality evaluator.
Score the following criterion from 1 to 5 and give a one-sentence justification.
Return ONLY valid JSON with no extra text: {{"score": <int 1-5>, "reasoning": "<one sentence>"}}

Criterion: {criterion}

Rubric:
{rubric}

Content to evaluate:
{content}"""

    response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"max_output_tokens": 2048},
    )
    text = response.text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in judge response: {text!r}")
    return json.loads(match.group())


# ---------------------------------------------------------------------------
# Stage 1: fetch_health — rule-based, no LLM
# ---------------------------------------------------------------------------

def _judge_fetch_health(
    articles: list[Article],
    tries: int,
    max_tries: int,
    article_count_last: int,
    email_status: str,
) -> StageScore:
    n = len(articles)

    if email_status == "no_content" and n == 0:
        score, reasoning = 1, "Pipeline exited with no articles after exhausting retries."
    elif tries > 0 and n == article_count_last:
        score, reasoning = 2, f"Convergence detected after {tries} tries — expand_search produced no new articles."
    elif tries >= max_tries and n == 0:
        score, reasoning = 1, f"Hard limit of {max_tries} tries reached with zero articles."
    elif tries == 0:
        score, reasoning = 5, f"Fetched {n} articles on the first attempt — no retries needed."
    elif tries == 1:
        score, reasoning = 4, f"Fetched {n} articles after 1 retry."
    else:
        score, reasoning = 3, f"Fetched {n} articles after {tries} retries."

    return StageScore(
        stage="fetch_health",
        score=score,
        reasoning=reasoning,
        details={"articles_fetched": n, "tries": tries, "max_tries": max_tries},
    )


# ---------------------------------------------------------------------------
# Stage 2: filter_quality — LLM judge
# ---------------------------------------------------------------------------

def _judge_filter_quality(
    articles: list[Article],
    top_articles: list[Article],
    subject: str,
) -> StageScore:
    if not top_articles:
        return StageScore(
            stage="filter_quality",
            score=SKIPPED,
            reasoning="No top articles — filter stage did not run.",
            details={},
        )

    all_names = [a.name for a in articles]
    selected_names = [a.name for a in top_articles]

    result = _llm_judge(
        criterion="Filter selection relevance",
        rubric=(
            "5 = selected articles are clearly the most relevant to the stated subject; "
            "off-topic articles are excluded.\n"
            "3 = mix of relevant and tangentially relevant articles selected.\n"
            "1 = selected articles are mostly off-topic or miss obvious relevant picks."
        ),
        content=(
            f"Subject: {subject}\n"
            f"All fetched articles ({len(all_names)}): {json.dumps(all_names, ensure_ascii=False)}\n"
            f"Selected ({len(selected_names)}): {json.dumps(selected_names, ensure_ascii=False)}"
        ),
    )

    return StageScore(
        stage="filter_quality",
        score=result["score"],
        reasoning=result["reasoning"],
        details={"total_articles": len(all_names), "selected": len(selected_names)},
    )


# ---------------------------------------------------------------------------
# Stage 3: content_faithfulness — LLM judge
# ---------------------------------------------------------------------------

def _judge_content_faithfulness(
    top_articles: list[Article],
    newsletter_data: NewsletterData,
) -> StageScore:
    if not newsletter_data or not newsletter_data.cards:
        return StageScore(
            stage="content_faithfulness",
            score=SKIPPED,
            reasoning="No newsletter content — summarize stage did not run.",
            details={},
        )

    source_summaries = "\n\n".join(
        f"[{a.name}]: {a.content[:600]}" for a in top_articles
    )
    card_briefs = "\n\n".join(
        f"Card '{c.title}': {c.brief}" for c in newsletter_data.cards
    )

    result = _llm_judge(
        criterion="Content faithfulness",
        rubric=(
            "5 = every factual claim in the card briefs is directly supported by the source articles; "
            "no invented details.\n"
            "3 = most claims are supported, with minor embellishments.\n"
            "1 = multiple invented facts not present in the source articles."
        ),
        content=f"Source articles:\n{source_summaries}\n\nGenerated card briefs:\n{card_briefs}",
    )

    return StageScore(
        stage="content_faithfulness",
        score=result["score"],
        reasoning=result["reasoning"],
        details={"cards": len(newsletter_data.cards), "brief_items": len(newsletter_data.brief_items)},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def judge_run(result: dict, profile: str) -> TraceReport:
    """
    Judge a completed pipeline run. Pass the dict returned by graph.ainvoke().
    Saves a JSON report to logs/ and returns a TraceReport.
    """
    stages = [
        _judge_fetch_health(
            articles=result.get("articles", []),
            tries=result.get("tries", 0),
            max_tries=result.get("max_tries", 5),
            article_count_last=result.get("article_count_last", -1),
            email_status=result.get("email_status", ""),
        ),
        _judge_filter_quality(
            articles=result.get("articles", []),
            top_articles=result.get("top_articles", []),
            subject=result.get("subject", ""),
        ),
        _judge_content_faithfulness(
            top_articles=result.get("top_articles", []),
            newsletter_data=result.get("newsletter_data"),
        ),
    ]

    numeric_scores = [s.score for s in stages if isinstance(s.score, int)]
    overall = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0

    report = TraceReport(
        profile=profile,
        timestamp=datetime.now().isoformat(),
        email_status=result.get("email_status", ""),
        stages=stages,
        overall=overall,
    )

    _save(report)
    _print(report)
    return report


def _save(report: TraceReport) -> None:
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"logs/trace_{report.profile.lower()}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2, default=str)
    print(f"  Trace report saved → {path}")


def _print(report: TraceReport) -> None:
    print(f"\n{'=' * 56}")
    print(f"  TRACE JUDGE — {report.profile}  ({report.email_status})")
    print(f"{'=' * 56}")
    for s in report.stages:
        bar = ("█" * s.score + "░" * (5 - s.score)) if isinstance(s.score, int) else "  (skipped)  "
        score_str = f"{s.score}/5" if isinstance(s.score, int) else SKIPPED
        print(f"  {s.stage:<24} {bar}  {score_str}")
        print(f"    → {s.reasoning}")
    print(f"\n  Overall: {report.overall:.1f}/5")
    print(f"{'=' * 56}\n")
