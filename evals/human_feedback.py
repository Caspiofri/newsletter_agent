"""Your verdict on the digest picks as LangSmith feedback on the run's digest_run root run.

Usage (from the repo root; LANGSMITH_API_KEY and LANGSMITH_PROJECT in .env):
  python -m evals.human_feedback   # backfill: post for every run in evals/ranking_golden.json with rated picks

The Telegram bot posts the same score after every rating tap on a digest pick. Posting is idempotent: each
(run, key) has a fixed feedback id, so a later tap or a backfill updates the score in place.
"""
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

from evals.import_labels import GOLDEN_RANKING_PATH, VALID_LABELS

PRECISION_KEY = "human.picks_precision"
COUNT_KEY = "count.human_picks_rated"
GOOD_LABELS = {"must", "worth"}
RUN_WINDOW = timedelta(minutes=60)  # the report is saved by the rank step, a few minutes after the run starts
FEEDBACK_NAMESPACE = uuid.UUID("5d7c3c0e-0b8f-4c47-9a2f-6a1f3f4f2b11")


def picks_precision(golden: list[dict], source_run: str) -> tuple[float | None, int]:
    """(share of the run's rated picks labeled must or worth, number rated); None when none is rated."""
    rated = [g for g in golden if g.get("source_run") == source_run and g.get("ranker_picked")
             and g.get("label") in VALID_LABELS]
    if not rated:
        return None, 0
    return sum(g["label"] in GOOD_LABELS for g in rated) / len(rated), len(rated)


def _utc(dt: datetime) -> datetime:
    # LangSmith start times are UTC; older SDKs return them naive
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def find_digest_run(source_run: str, client):
    """The latest digest_run root run that started at or before the report time (Pi local) and within RUN_WINDOW."""
    try:
        saved = datetime.strptime(source_run, "%Y%m%d_%H%M%S").astimezone(timezone.utc)  # naive = system local
    except ValueError:
        return None
    runs = client.list_runs(project_name=os.getenv("LANGSMITH_PROJECT", "default"), is_root=True,
                            filter='eq(name, "digest_run")', start_time=saved - RUN_WINDOW)
    inside = [r for r in runs if r.name == "digest_run" and saved - RUN_WINDOW <= _utc(r.start_time) <= saved]
    return max(inside, key=lambda r: _utc(r.start_time), default=None)


def _upsert(client, run_id, key: str, score, comment: str | None) -> None:
    feedback_id = uuid.uuid5(FEEDBACK_NAMESPACE, f"{run_id}:{key}")
    try:
        client.create_feedback(run_id, key=key, score=score, comment=comment, feedback_id=feedback_id)
    except Exception as e:
        # the fixed id already exists (409): this run was scored before, so the score is replaced
        if "409" not in str(e) and "onflict" not in type(e).__name__ + str(e) and "exist" not in str(e):
            raise
        client.update_feedback(feedback_id, score=score, comment=comment)


def post_picks_precision(golden: list[dict], source_run: str, client=None) -> bool:
    """Post the run's picks precision and rated count. False when skipped or failed; never raises."""
    if not os.getenv("LANGSMITH_API_KEY"):
        return False
    precision, n = picks_precision(golden, source_run)
    if precision is None:
        return False
    try:
        if client is None:
            import langsmith
            client = langsmith.Client()
        run = find_digest_run(source_run, client)
        if run is None:
            return False
        good = round(precision * n)
        _upsert(client, run.id, PRECISION_KEY, precision, f"{good} of {n} rated picks must/worth")
        _upsert(client, run.id, COUNT_KEY, n, None)
        return True
    except Exception as e:
        print(f"[warn] Failed to post human picks score for {source_run}: {e}")
        return False


def backfill(golden: list[dict], client=None) -> None:
    """post_picks_precision for every run with rated picks, oldest first, one line per run."""
    if client is None and os.getenv("LANGSMITH_API_KEY"):
        import langsmith
        client = langsmith.Client()
    runs = sorted({g.get("source_run", "") for g in golden if g.get("source_run")})
    for run in runs:
        precision, n = picks_precision(golden, run)
        if precision is None:
            continue
        status = "posted" if post_picks_precision(golden, run, client) else "not found"
        print(f"{run}  precision={precision:.2f}  n={n}  {status}")


if __name__ == "__main__":
    from config import settings  # noqa: F401  loads .env
    if not os.getenv("LANGSMITH_API_KEY"):
        sys.exit("LANGSMITH_API_KEY is not set: nothing posted")
    backfill(json.loads(GOLDEN_RANKING_PATH.read_text(encoding="utf-8")) if GOLDEN_RANKING_PATH.exists() else [])
