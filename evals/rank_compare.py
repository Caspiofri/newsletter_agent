"""Compare two evals.rank_eval results: a baseline against a change.

Usage (from the repo root):  python -m evals.rank_compare <baseline.json> <candidate.json>

A difference is "clear" only when it is larger than the larger spread (std. deviation over repeats) of the two
runs; otherwise it is "within noise". Results scored on a different split, k or golden set are not comparable.
"""
import json
import sys
from pathlib import Path

from evals.rank_metrics import METRICS, POOLED

MUST_MATCH = ("split", "k", "golden_rows")
NOT_SETTINGS = ("date",)  # differs on every run, says nothing about the change


def compare(baseline: dict, candidate: dict) -> dict:
    """Raises ValueError naming the setting when the two results are not comparable."""
    bs, cs = baseline.get("settings", {}), candidate.get("settings", {})
    for key in MUST_MATCH:
        if bs.get(key) != cs.get(key):
            raise ValueError(f"not comparable: {key} differs ({bs.get(key)} vs {cs.get(key)})")
    changed = [f"{key}: {bs.get(key)} vs {cs.get(key)}" for key in sorted(set(bs) | set(cs))
               if key not in NOT_SETTINGS and bs.get(key) != cs.get(key)]
    metrics = {}
    # pooled metrics sit under overall.pooled; results written before they existed show them as n/a
    sides = [(r["overall"], r["overall"].get("pooled") or {"mean": {}, "spread": {}}) for r in (baseline, candidate)]
    for m in METRICS + POOLED:
        (bo, bp), (co, cp) = sides
        b_part, c_part = (bo, co) if m in METRICS else (bp, cp)
        b, c = b_part["mean"].get(m), c_part["mean"].get(m)
        noise = max(b_part["spread"].get(m) or 0.0, c_part["spread"].get(m) or 0.0)
        diff = None if b is None or c is None else c - b
        verdict = "n/a" if diff is None else "clear" if abs(diff) > noise else "within noise"
        metrics[m] = {"baseline": b, "candidate": c, "diff": diff, "noise": noise, "verdict": verdict}
    return {"metrics": metrics, "changed_settings": changed}


def _fmt(value, sign: bool = False) -> str:
    return "n/a" if value is None else f"{value:+.2f}" if sign else f"{value:.2f}"


def format_comparison(out: dict) -> str:
    lines = [f"{'metric':<13} {'baseline':>8} {'candidate':>9} {'diff':>6} {'noise':>6}  verdict"]
    for m, v in out["metrics"].items():
        lines.append(f"{m:<13} {_fmt(v['baseline']):>8} {_fmt(v['candidate']):>9} {_fmt(v['diff'], True):>6} "
                     f"{_fmt(v['noise']):>6}  {v['verdict']}")
    lines.append("changed settings: " + (", ".join(out["changed_settings"]) or "none"))
    lines.append("verdict: " + "; ".join(f"{m} {_fmt(v['diff'], True)} ({v['verdict']})"
                                         for m, v in out["metrics"].items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        sys.exit(__doc__)
    baseline, candidate = (json.loads(Path(p).read_text(encoding="utf-8")) for p in argv)
    try:
        out = compare(baseline, candidate)
    except ValueError as e:
        sys.exit(str(e))
    print(format_comparison(out))


if __name__ == "__main__":
    main()
