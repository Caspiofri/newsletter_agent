"""LangSmith helpers: judge scores and stage counts as feedback on the digest_run root run."""
from modules import trace_judge
from modules.trace_judge import SKIPPED, TraceReport


def feedback_items(report: TraceReport) -> list[dict]:
    items = [
        {"key": f"judge.{s.stage}", "score": s.score, "comment": s.reasoning}
        for s in report.stages
        if s.score != SKIPPED
    ]
    items.append({"key": "judge.overall", "score": report.overall})
    return items


def run_summary(result: dict) -> dict:
    return {
        "stats": dict(result.get("run_stats", {})),
        "picked": [a.name for a in result.get("top_articles", [])],
        "dedupe_drops": result.get("dedupe_drops", []),
        "rank_scores": result.get("rank_scores", []),
        "email_status": result.get("email_status", ""),
    }


# run_stats key -> LangSmith feedback key. Counts get "count.", similarity measurements "metric.",
# so they are not read as judge scores. An unmapped run_stats key posts as "count.<key>".
FEEDBACK_KEY_NAMES: dict[str, str] = {
    "merged": "count.articles_collected_merged",
    "after_dedupe": "count.articles_after_dedupe",
    "dedupe_dropped": "count.articles_dedupe_dropped",
    "dedupe_matches": "count.dedupe_matches",
    "dedupe_near_misses": "count.dedupe_near_misses",
    "dedupe_batch_duplicates": "count.dedupe_batch_duplicates",
    "dedupe_max_similarity": "metric.dedupe_max_similarity",
    "shortlisted": "count.articles_shortlisted",
    "read_full": "count.articles_read_full",
    "read_partial": "count.articles_read_partial",
    "read_snippet": "count.articles_read_snippet",
    "ranked": "count.articles_ranked",
    "picked": "count.final_picked_articles",
    "cards": "count.cards",
    "brief_items": "count.brief_items",
}


def summary_feedback_items(summary: dict) -> list[dict]:
    return [{"key": FEEDBACK_KEY_NAMES.get(k, f"count.{k}"), "score": v}
            for k, v in summary.get("stats", {}).items()]


def _post(items: list[dict], run_id, client) -> None:
    if run_id is None:
        return
    try:
        if client is None:
            import langsmith
            client = langsmith.Client()
        for item in items:
            client.create_feedback(run_id, key=item["key"], score=item["score"], comment=item.get("comment"))
    except Exception as e:
        print(f"[warn] Failed to post LangSmith feedback: {e}")


def post_feedback(report: TraceReport, run_id, client=None) -> None:
    _post(feedback_items(report), run_id, client)


def post_summary(summary: dict, run_id, client=None) -> None:
    _post(summary_feedback_items(summary), run_id, client)


def judge_and_post(result: dict, profile: str, run_tree) -> None:
    """Attach the run summary, run the judge, post feedback. Never raises."""
    try:
        run_id = run_tree.id if run_tree is not None else None
        summary = run_summary(result)
        try:
            if run_tree is not None:
                run_tree.add_metadata({"run_summary": summary})
        except Exception as e:
            print(f"[warn] Failed to add run summary metadata: {e}")
        post_summary(summary, run_id)
        try:
            report = trace_judge.judge_run(result, profile)
            post_feedback(report, run_id)
        except Exception as e:
            print(f"[warn] Judge failed: {e}")
    except Exception as e:
        print(f"[warn] judge_and_post failed: {e}")
