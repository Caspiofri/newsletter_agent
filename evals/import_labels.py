"""Merge labels exported from a ranking report page into the golden set.

Usage (from the repo root):  python -m evals.import_labels output/ranking_labels_ai_<run>.json

Labels are keyed by URL; importing a URL again replaces its earlier label. The golden set keeps the text snapshot
of each article, so it stays usable after the link rots.
"""
import json
import sys
from datetime import date

from config.settings import EVALS_DIR

GOLDEN_RANKING_PATH = EVALS_DIR / "ranking_golden.json"
VALID_LABELS = {"must", "worth", "skip"}


def merge(golden: list[dict], exported: dict, today: str) -> tuple[list[dict], int, int]:
    """Return (merged golden set, number added, number updated)."""
    by_url = {g["url"]: g for g in golden}
    added = updated = 0
    for row in exported.get("labels", []):
        if row.get("label") not in VALID_LABELS or not row.get("url"):
            continue
        entry = {**row, "labeled_on": today, "source_run": exported.get("run_id", "")}
        if row["url"] in by_url:
            updated += 1
        else:
            added += 1
        by_url[row["url"]] = entry
    return list(by_url.values()), added, updated


def main(path: str) -> None:
    exported = json.loads(open(path, encoding="utf-8").read())
    golden = json.loads(GOLDEN_RANKING_PATH.read_text(encoding="utf-8")) if GOLDEN_RANKING_PATH.exists() else []
    merged, added, updated = merge(golden, exported, str(date.today()))
    GOLDEN_RANKING_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {k: sum(g["label"] == k for g in merged) for k in sorted(VALID_LABELS)}
    print(f"{added} added, {updated} updated -> {GOLDEN_RANKING_PATH}  (total {len(merged)}: {counts})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
