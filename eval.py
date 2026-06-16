#!/usr/bin/env python3
"""
eval.py — LLM evaluation suite for the newsletter agent.

Evaluates two LLM-driven nodes without touching Gmail:
  1. fillter_articles — filter relevance + completeness
  2. summraize        — faithfulness, Hebrew quality, personalization,
                        actionability, and HTML structure

Run with:  python eval.py
Results are printed as a table and saved to logs/eval_<timestamp>.json
"""

import json
import os
import re
from datetime import date, datetime

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai

load_dotenv()

from models import Article, NewsletterData
from nodes import fillter_articles, summraize

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------------------------------------------------------------------
# Fixture articles: 6 AI-relevant, 2 off-topic (to test filter discrimination)
# ---------------------------------------------------------------------------
FIXTURE_ARTICLES = [
    Article(
        name="LangGraph 0.3 ships persistent memory and multi-agent handoffs",
        subject="AI Engineering",
        author="LangChain Blog",
        published_at=date.today(),
        content="""LangGraph 0.3 introduces built-in persistent memory stores and first-class
        multi-agent handoff primitives. Engineers can now share state across agent runs without
        external databases. The new HandoffTool lets a supervisor agent delegate tasks to
        subagents with typed inputs and structured outputs. Persistent memory uses a pluggable
        backend (SQLite by default). Migration from 0.2 requires renaming add_conditional_edges
        to add_conditional_edge (singular). Full changelog at github.com/langchain-ai/langgraph.""",
        url="https://blog.langchain.dev/langgraph-0-3",
    ),
    Article(
        name="RAG vs Fine-tuning: A practical benchmark on domain Q&A",
        subject="AI Engineering",
        author="Towards Data Science",
        published_at=date.today(),
        content="""A team at Cohere ran head-to-head tests of RAG pipelines vs fine-tuned models
        on 10 domain-specific Q&A datasets. RAG scored 12% higher on factual accuracy and was
        3x cheaper to update when new documents arrived. Fine-tuning won on latency (40ms vs
        220ms) and on style consistency tasks. The study recommends RAG for knowledge-intensive
        tasks and fine-tuning for tone/format alignment. All evaluation code is open-sourced at
        github.com/cohere-ai/rag-vs-finetune-bench.""",
        url="https://towardsdatascience.com/rag-vs-finetune",
    ),
    Article(
        name="OpenAI releases o3-mini with function-calling support",
        subject="AI Engineering",
        author="OpenAI Blog",
        published_at=date.today(),
        content="""OpenAI's o3-mini model now supports structured function calling and JSON mode
        at 60% lower cost than o3. The model scores 87.3 on AIME and 79.1 on SWE-bench-verified.
        API access is rolling out to Tier 3+ users this week. Rate limits start at 200k tokens/min.
        The model supports up to 128k context. System prompts are now cached automatically in
        the API, reducing latency on repeated calls.""",
        url="https://openai.com/blog/o3-mini",
    ),
    Article(
        name="Anthropic publishes Constitutional AI v2 research paper",
        subject="AI Engineering",
        author="Anthropic",
        published_at=date.today(),
        content="""Anthropic's Constitutional AI v2 paper describes a self-supervised technique
        where models critique and revise their own outputs according to a set of principles
        without human labeling. The method reduces harmful outputs by 34% while maintaining
        helpfulness scores. The paper includes ablation studies on constitution size and
        revision iterations. Code and evaluation harnesses are available on the Anthropic
        GitHub. The technique is model-agnostic and has been tested on Llama and Mistral.""",
        url="https://anthropic.com/research/constitutional-ai-v2",
    ),
    Article(
        name="Vector database shootout: Pinecone vs Weaviate vs Qdrant in 2025",
        subject="AI Engineering",
        author="The New Stack",
        published_at=date.today(),
        content="""An independent benchmark compared Pinecone, Weaviate, and Qdrant on 1M vector
        datasets. Qdrant led on recall@10 (0.987) and throughput (12k QPS single node). Pinecone
        had the lowest p99 latency (18ms) and best managed-service developer experience. Weaviate
        offered the richest hybrid search (BM25 + dense vectors) out of the box. For RAG
        applications under 10M vectors, Qdrant's free tier covers most use cases. Full benchmark
        at newstack.io/vdb-bench-2025.""",
        url="https://thenewstack.io/vdb-benchmark-2025",
    ),
    Article(
        name="Cursor IDE adds multi-file agent mode with LLM routing",
        subject="AI Engineering",
        author="Cursor Blog",
        published_at=date.today(),
        content="""Cursor's new Agent Mode lets developers define multi-step coding tasks in
        natural language. The IDE automatically routes sub-tasks to different models: Claude
        for reasoning, GPT-4o for code generation, and a local model for file search. Users
        can inspect and override routing decisions. Early beta users report a 30% reduction
        in back-and-forth prompting. The feature integrates with existing .cursorrules configs.""",
        url="https://cursor.sh/blog/agent-mode",
    ),
    # Off-topic — the filter should deprioritize these
    Article(
        name="Israel's tech startup funding drops 18% in Q1 2026",
        subject="Business",
        author="Calcalist",
        published_at=date.today(),
        content="""Israeli tech startups raised $1.2B in Q1 2026, down 18% from Q4 2025.
        Cybersecurity and defense-tech accounted for 41% of total funding. Late-stage rounds
        shrank while seed and Series A activity stayed stable. Analysts attribute the slowdown
        to global VC caution and rising interest rates.""",
        url="https://calcalist.co.il/funding-q1-2026",
    ),
    Article(
        name="New study links ultra-processed food to cognitive decline",
        subject="Health",
        author="Nature Medicine",
        published_at=date.today(),
        content="""A longitudinal study tracking 72,000 adults over 10 years found that
        consuming 4+ servings of ultra-processed foods daily was associated with a 28% higher
        risk of cognitive decline. The association held after controlling for exercise, BMI,
        and socioeconomic status. The study did not establish causality.""",
        url="https://nature.com/articles/cognitive-decline-upf",
    ),
]

BASE_STATE = {
    "gmail_label": "test",
    "subject": "AI Engineering",
    "target_audience": "Junior Software Engineers transitioning to AI Engineering",
    "experience_level": "Junior",
    "digest_name": "AI Digest",
    "recipient": "test@example.com",
    "sender": "noreply@example.com",
    "top_k": "5",
    "articles": FIXTURE_ARTICLES,
    "top_articles": [],
    "newsletter_data": None,
    "summary": "",
    "email_status": "",
    "tries": 0,
    "max_tries": 5,
}


# ---------------------------------------------------------------------------
# Rule-based structural checks
# ---------------------------------------------------------------------------

def check_structure(html: str) -> dict[str, bool]:
    try:
        soup = BeautifulSoup(html, "html.parser")
        parseable = True
    except Exception:
        return {k: False for k in ["parseable", "has_card", "has_brief_section",
                                   "has_actions_section", "has_header", "no_ltr_violations"]}

    content_tags = [
        el for el in soup.find_all(True)
        if el.name not in ("style", "script", "head", "html")
    ]
    ltr_violations = [el.name for el in content_tags if el.get("dir") == "ltr"]

    return {
        "parseable": parseable,
        "has_header": bool(soup.find(class_="header")),
        "has_card": bool(soup.find(class_="card")),
        "has_brief_section": bool(soup.find(class_="brief-section")),
        "has_actions_section": bool(soup.find(class_="actions-section")),
        "no_ltr_violations": len(ltr_violations) == 0,
    }


# ---------------------------------------------------------------------------
# LLM-as-a-judge
# ---------------------------------------------------------------------------

def judge(criterion: str, rubric: str, content: str) -> dict:
    """Return {"score": int 1-5, "reasoning": str}."""
    prompt = f"""You are a strict newsletter quality evaluator.
Score the following criterion from 1 to 5 and give a one-sentence justification.
Return ONLY valid JSON with no extra text: {{"score": <int 1-5>, "reasoning": "<one sentence>"}}

Criterion: {criterion}

Rubric:
{rubric}

Content to evaluate:
{content}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"max_output_tokens": 2048},
    )
    text = response.text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text!r}")
    return json.loads(match.group())


# ---------------------------------------------------------------------------
# Eval 1: filter node
# ---------------------------------------------------------------------------

def eval_filter() -> dict:
    top_k = int(BASE_STATE["top_k"])
    state = {**BASE_STATE}
    result = fillter_articles(state)
    selected: list[Article] = result["top_articles"]

    completeness_pass = len(selected) == top_k
    selected_names = [a.name for a in selected]

    all_names = [a.name for a in FIXTURE_ARTICLES]
    relevance = judge(
        criterion="Filter relevance",
        rubric=(
            "5 = the selected articles are clearly the most relevant to 'AI Engineering'; "
            "off-topic articles (business funding, food/health) are excluded.\n"
            "3 = mix of relevant and tangentially relevant articles selected.\n"
            "1 = selected articles are mostly off-topic."
        ),
        content=(
            f"Subject: AI Engineering\n"
            f"All available articles: {json.dumps(all_names, ensure_ascii=False)}\n"
            f"Selected top-{top_k}: {json.dumps(selected_names, ensure_ascii=False)}"
        ),
    )

    return {
        "selected_articles": selected_names,
        "completeness_pass": completeness_pass,
        "expected_count": top_k,
        "actual_count": len(selected),
        "relevance_score": relevance["score"],
        "relevance_reasoning": relevance["reasoning"],
        # Return selected Article objects for the next stage
        "_selected_objects": selected,
    }


# ---------------------------------------------------------------------------
# Eval 2: newsletter node
# ---------------------------------------------------------------------------

def eval_newsletter(top_articles: list[Article]) -> dict:
    state = {**BASE_STATE, "top_articles": top_articles}
    result = summraize(state)
    html: str = result["summary"]
    data: NewsletterData = result["newsletter_data"]

    structure = check_structure(html)

    article_summaries = "\n\n".join(
        f"Article: {a.name}\nContent: {a.content[:500]}"
        for a in top_articles
    )
    card_briefs = "\n\n".join(
        f"Card: {c.title}\nBrief: {c.brief}"
        for c in data.cards
    )
    personalization_texts = "\n\n".join(
        f"Card: {c.title}\nPersonalization: {c.personalization}"
        for c in data.cards
    )
    actions_text = "\n".join(f"- {a}" for a in data.actions)
    all_prose = "\n\n".join([card_briefs, personalization_texts, actions_text])

    faithfulness = judge(
        criterion="Faithfulness",
        rubric=(
            "5 = every factual claim in the card briefs is directly supported by the "
            "source articles; no invented details.\n"
            "3 = most claims are supported, with minor embellishments.\n"
            "1 = multiple invented facts not present in the source articles."
        ),
        content=f"Source articles:\n{article_summaries}\n\nCard briefs:\n{card_briefs}",
    )

    hebrew_quality = judge(
        criterion="Hebrew prose quality",
        rubric=(
            "5 = natural, fluent Hebrew prose; technical terms (LangGraph, RAG, LLM, API, etc.) "
            "are correctly kept in English.\n"
            "3 = mostly Hebrew but with awkward phrasing or incorrectly translated technical terms.\n"
            "1 = prose is not in Hebrew, or key technical terms are wrongly translated."
        ),
        content=all_prose,
    )

    personalization = judge(
        criterion="Audience personalization",
        rubric=(
            "5 = the personalization sections give concrete, specific advice for a junior "
            "dev transitioning to AI engineering, referencing LangGraph/RAG/multi-agent workflows.\n"
            "3 = advice is relevant but generic — could apply to any developer.\n"
            "1 = no connection to the stated audience."
        ),
        content=personalization_texts,
    )

    actionability = judge(
        criterion="Actionability of recommendations",
        rubric=(
            "5 = 1-2 specific, immediately executable actions tied to today's articles "
            "(e.g. a specific repo name, a concrete command).\n"
            "3 = actions are somewhat related to the articles but vague.\n"
            "1 = generic advice with no connection to today's articles."
        ),
        content=actions_text,
    )

    return {
        "structure": structure,
        "card_count": len(data.cards),
        "brief_item_count": len(data.brief_items),
        "action_count": len(data.actions),
        "faithfulness_score": faithfulness["score"],
        "faithfulness_reasoning": faithfulness["reasoning"],
        "hebrew_quality_score": hebrew_quality["score"],
        "hebrew_quality_reasoning": hebrew_quality["reasoning"],
        "personalization_score": personalization["score"],
        "personalization_reasoning": personalization["reasoning"],
        "actionability_score": actionability["score"],
        "actionability_reasoning": actionability["reasoning"],
        "html_length": len(html),
    }


# ---------------------------------------------------------------------------
# Report + entrypoint
# ---------------------------------------------------------------------------

def _pass(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def print_report(filter_r: dict, newsletter_r: dict) -> None:
    print("\n" + "=" * 62)
    print("  NEWSLETTER AGENT EVAL REPORT")
    print("=" * 62)

    print("\n[1] FILTER NODE")
    print(f"  Completeness ({filter_r['actual_count']}/{filter_r['expected_count']} articles): "
          f"{_pass(filter_r['completeness_pass'])}")
    print(f"  Relevance score: {filter_r['relevance_score']}/5")
    print(f"    → {filter_r['relevance_reasoning']}")
    print(f"  Selected: {filter_r['selected_articles']}")

    print(f"\n[2] NEWSLETTER NODE — structured output")
    print(f"  Full cards:  {newsletter_r['card_count']}")
    print(f"  Brief items: {newsletter_r['brief_item_count']}")
    print(f"  Actions:     {newsletter_r['action_count']}")

    print("\n[2] NEWSLETTER NODE — structural checks")
    s = newsletter_r["structure"]
    for key, label in [
        ("parseable",          "HTML parseable     "),
        ("has_header",         "has header         "),
        ("has_card",           "has full card      "),
        ("has_brief_section",  "has brief section  "),
        ("has_actions_section","has actions section"),
        ("no_ltr_violations",  "no dir=ltr elements"),
    ]:
        print(f"    {label}: {_pass(s[key])}")

    print("\n[2] NEWSLETTER NODE — LLM-as-a-judge scores")
    criteria = [
        ("faithfulness",   "Faithfulness      "),
        ("hebrew_quality", "Hebrew quality    "),
        ("personalization","Personalization   "),
        ("actionability",  "Actionability     "),
    ]
    scores = []
    for key, label in criteria:
        score = newsletter_r[f"{key}_score"]
        reason = newsletter_r[f"{key}_reasoning"]
        bar = "█" * score + "░" * (5 - score)
        print(f"    {label} {bar} {score}/5")
        print(f"      → {reason}")
        scores.append(score)

    avg = sum(scores) / len(scores)
    print(f"\n  Average LLM score: {avg:.1f}/5")
    print(f"  HTML length: {newsletter_r['html_length']} chars")
    print("=" * 62 + "\n")


def main() -> None:
    print("Running newsletter agent eval...")

    print("\n[1/2] Filter node...")
    filter_results = eval_filter()
    top_articles = filter_results.pop("_selected_objects")

    print("[2/2] Newsletter node...")
    newsletter_results = eval_newsletter(top_articles)

    print_report(filter_results, newsletter_results)

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/eval_{timestamp}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {"filter": filter_results, "newsletter": newsletter_results},
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"Results saved to {log_path}")


if __name__ == "__main__":
    main()
