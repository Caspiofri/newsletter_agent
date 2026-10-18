#!/usr/bin/env python3
"""
eval_golden.py — Regression test for trace_judge's content_faithfulness judge
against a hand-labeled golden set of real production cases.

Unlike eval.py (which tests pipeline nodes on synthetic fixtures), this
replays trace_judge._judge_content_faithfulness — the actual judge used in
production — against real (source articles, generated cards) pairs a human
has verified and scored. If the judge's score drifts from the human label,
that's a signal the prompt, rubric, or underlying model changed behavior.

Golden set: golden_set.json (built from output/trace_ai_*.json + Chroma +
Gmail Sent, hand-labeled — see golden_set_candidates.md for provenance).

Run with:  python -m evals.eval_golden [--model <id>|eval|daily]
Results are printed as a table and saved to output/eval_golden_<model>_<timestamp>.json
"""

import argparse
import json
import re
from datetime import date, datetime

from config.settings import GOLDEN_SET_PATH, JUDGE_MODEL_DAILY, JUDGE_MODEL_EVAL, OUTPUT_DIR
from modules.models import Article, ArticleCard, NewsletterData
from modules.trace_judge import _judge_content_faithfulness

AGREEMENT_TOLERANCE = 1  # judge score within ±1 of human label counts as agreement


def _load_golden_set() -> list[dict]:
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def _to_top_articles(sources: list[dict]) -> list[Article]:
    return [
        Article(
            subject="AI Engineering",
            name=s["name"],
            published_at=date.today(),
            content=s["content"],
        )
        for s in sources
    ]


def _to_newsletter_data(cards: list[dict]) -> NewsletterData:
    return NewsletterData(
        cards=[
            ArticleCard(
                title=c["title"],
                source=c["source"],
                brief=c["brief"],
                personalization=c["personalization"],
                url="",
            )
            for c in cards
        ],
        brief_items=[],
        actions=[],
    )


def _resolve_model(model: str | None) -> str:
    if model is None or model == "daily":
        return JUDGE_MODEL_DAILY
    if model == "eval":
        return JUDGE_MODEL_EVAL
    return model


def run_golden_eval(model: str | None = None) -> dict:
    model = _resolve_model(model)
    cases = _load_golden_set()
    results = []

    for case in cases:
        top_articles = _to_top_articles(case["sources"])
        newsletter_data = _to_newsletter_data(case["cards"])

        judged = _judge_content_faithfulness(top_articles, newsletter_data, model=model)
        human_score = case["human_label"]["score"]
        judge_score = judged.score

        agree = isinstance(judge_score, int) and abs(judge_score - human_score) <= AGREEMENT_TOLERANCE

        results.append({
            "run_ts": case["run_ts"],
            "human_score": human_score,
            "human_note": case["human_label"]["note"],
            "judge_score": judge_score,
            "original_judge_score": case.get("judge_original_score"),
            "judge_reasoning": judged.reasoning,
            "agree": agree,
        })

    agreement_count = sum(1 for r in results if r["agree"])
    return {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "tolerance": AGREEMENT_TOLERANCE,
        "cases": results,
        "agreement_rate": agreement_count / len(results) if results else 0.0,
    }


def print_report(report: dict) -> None:
    print("\n" + "=" * 62)
    print("  GOLDEN SET EVAL — content_faithfulness judge vs. human labels")
    print(f"  model: {report['model']}")
    print("=" * 62)
    for r in report["cases"]:
        status = "AGREE" if r["agree"] else "DRIFT"
        print(f"\n  [{status}] {r['run_ts']}   human={r['human_score']}/5  judge={r['judge_score']}/5  original={r['original_judge_score']}/5")
        print(f"    human note:  {r['human_note'][:120]}")
        print(f"    judge said:  {r['judge_reasoning'][:120]}")
    print(f"\n  Agreement rate (±{report['tolerance']}): {report['agreement_rate']*100:.0f}% "
          f"({sum(1 for r in report['cases'] if r['agree'])}/{len(report['cases'])})")
    print("=" * 62 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Golden set eval for the content_faithfulness judge")
    parser.add_argument("--model", default=None, help="judge model id, or 'eval' / 'daily' (default: daily)")
    args = parser.parse_args()

    model = _resolve_model(args.model)
    print(f"Running golden set eval (content_faithfulness judge, model={model})...")
    report = run_golden_eval(model)
    print_report(report)

    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    path = OUTPUT_DIR / f"eval_golden_{safe_model}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"Report saved → {path}")


if __name__ == "__main__":
    main()
