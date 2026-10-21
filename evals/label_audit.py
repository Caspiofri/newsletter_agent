"""Label audit of the ranking golden set, and a self-consistency check against a later relabel.

Usage (from the repo root):  python -m evals.label_audit [--golden evals/ranking_golden.json] [--relabel <export.json>]

The audit counts the labels by label, source, group, provenance and report, shows the train/test split, finds the
same article labeled twice (same normalized title and source) and warns about a set too thin to measure with.
--relabel compares a second labeling of some of the same articles (a label hub export, or a golden-style list).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from evals import rank_split
from evals.import_labels import GOLDEN_RANKING_PATH, VALID_LABELS

DEFAULT_TARGET = 80
MIN_SKIP_SHARE = 0.15
MIN_SAMPLE_SHARE = 0.25
MIN_REPORTS = 4
MIN_ARTICLE_SHARE = 0.5
MIN_AGREEMENT_N = 5
LABEL_ORDER = ("must", "worth", "skip")


def _labeled(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("label") in VALID_LABELS]


def _count(rows: list[dict], field: str) -> dict:
    return dict(sorted(Counter(str(r.get(field) or "") for r in rows).items()))


def audit(rows: list[dict], target: int = DEFAULT_TARGET) -> dict:
    rows = _labeled(rows)
    n = len(rows)
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(rank_split.split_key(r), []).append(r)
    duplicates = [{"key": key, "urls": [r.get("url") for r in g]} for key, g in groups.items() if len(g) > 1]
    out = {"total": n, "target": target}
    for field in ("label", "origin", "group", "labeled_with", "read_status", "source_run"):
        out[f"by_{field}"] = _count(rows, field)
    out["split"] = rank_split.split_counts(rows)
    out["duplicates"] = duplicates

    share = lambda count: count / n if n else 0.0
    warnings = []
    if n < target:
        warnings.append(f"only {n} of the {target} labels aimed for")
    if share(out["by_label"].get("skip", 0)) < MIN_SKIP_SHARE:
        warnings.append(f"skip labels are under {MIN_SKIP_SHARE:.0%}: precision will look better than it is")
    if share(out["by_group"].get("sample", 0)) < MIN_SAMPLE_SHARE:
        warnings.append(f"random-sample rows are under {MIN_SAMPLE_SHARE:.0%}: recall misses stories the ranker never saw")
    if len(out["by_source_run"]) < MIN_REPORTS:
        warnings.append(f"labels come from {len(out['by_source_run'])} report(s), fewer than {MIN_REPORTS}")
    if share(out["by_labeled_with"].get("article", 0)) < MIN_ARTICLE_SHARE:
        warnings.append("fewer than half the labels were judged from the full article (labeled_with=article)")
    if duplicates:
        warnings.append(f"duplicates: {len(duplicates)} article(s) labeled more than once under different addresses")
    out["warnings"] = warnings
    return out


def agreement(original: list[dict], relabel: list[dict]) -> dict:
    """Share of the articles in both sets with the same label (matched by URL, else by split key)."""
    by_url = {r["url"]: r for r in _labeled(original) if r.get("url")}
    by_key = {rank_split.split_key(r): r for r in _labeled(original)}
    confusion = {a: {b: 0 for b in LABEL_ORDER} for a in LABEL_ORDER}
    seen, same = set(), 0
    for r in _labeled(relabel):
        o = by_url.get(r.get("url")) or by_key.get(rank_split.split_key(r))
        if o is None or id(o) in seen:
            continue
        seen.add(id(o))
        confusion[o["label"]][r["label"]] += 1
        same += o["label"] == r["label"]
    n = len(seen)
    return {"n_compared": n, "agreement": same / n if n >= MIN_AGREEMENT_N else None, "confusion": confusion}


def _load(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("labels", []) if isinstance(data, dict) else data


def format_audit(out: dict) -> str:
    lines = [f"{out['total']} labels (target {out['target']}), split {out['split']}"]
    for field in ("label", "origin", "group", "labeled_with", "read_status", "source_run"):
        lines.append(f"  {field + ':':<14} {out[f'by_{field}']}")
    for d in out["duplicates"]:
        lines.append(f"  duplicate {d['key']!r}: {', '.join(map(str, d['urls']))}")
    lines += [f"WARNING: {w}" for w in out["warnings"]] or ["no warnings"]
    return "\n".join(lines)


def format_agreement(out: dict) -> str:
    value = "n/a (fewer than 5 compared)" if out["agreement"] is None else f"{out['agreement']:.0%}"
    lines = [f"agreement with the relabel: {value} over {out['n_compared']} articles (aim for at least 80 percent)",
             "  " + "original / relabel".ljust(20) + "".join(f"{b:>7}" for b in LABEL_ORDER)]
    for a in LABEL_ORDER:
        lines.append(f"  {a:<20}" + "".join(f"{out['confusion'][a][b]:>7}" for b in LABEL_ORDER))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--golden", type=Path, default=GOLDEN_RANKING_PATH)
    ap.add_argument("--relabel", type=Path, help="a later labeling of some of the same articles")
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET)
    args = ap.parse_args(argv)
    golden = _load(args.golden)
    print(format_audit(audit(golden, args.target)))
    if args.relabel:
        print(format_agreement(agreement(golden, _load(args.relabel))))


if __name__ == "__main__":
    main()
