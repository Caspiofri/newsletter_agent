"""Ranker replay: re-score the labeled articles of each ranking report with the production ranker prompt, several
times, and measure precision@k and capped recall@k against the labels (evals.rank_metrics).

Usage (from the repo root):
  python -m evals.rank_eval [--split train|test|all] [--k 5] [--repeats 3] [--text-chars 6000]
                            [--golden evals/ranking_golden.json] [--profile AI] [--run]

Without --run it only prints the plan (reports, articles, model calls, rough input size) and makes no model call.
With --run it calls Gemini once per report per repeat and writes output/rank_eval_<timestamp>.json.
The test split is meant to be scored once per tuning round, never to guide edits.
"""
import argparse
import json
import os
import statistics
from datetime import date, datetime
from pathlib import Path

from config import settings
from config.settings import OUTPUT_DIR
from evals import rank_metrics, rank_split
from evals.import_labels import GOLDEN_RANKING_PATH, VALID_LABELS
from modules.article_text import article_text
from modules.models import Article

PROMPT_OVERHEAD_CHARS = 1500  # the fixed part of the ranker prompt
ITEM_OVERHEAD_CHARS = 200  # id, title, source description per candidate


def golden_to_candidates(rows: list[dict], subject: str = "") -> list[Article]:
    """One Article per golden row, built from the stored snapshot (the link may be dead by now)."""
    out = []
    for r in rows:
        try:
            published = date.fromisoformat(str(r.get("published_at") or "")[:10])
        except ValueError:
            published = date.today()
        out.append(Article(subject=subject, name=r.get("title") or "", published_at=published,
                           content=r.get("content") or "", url=r.get("url"), origin=r.get("origin") or "inbox",
                           read_status=r.get("read_status") or "snippet", signal=r.get("signal") or ""))
    return out


def _labeled(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("label") in VALID_LABELS and r.get("url")]


def _group_by_report(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in _labeled(rows):
        groups.setdefault(r.get("source_run") or "", []).append(r)
    return dict(sorted(groups.items()))


def _mean_and_spread(repeats: list[dict]) -> tuple[dict, dict]:
    """Mean and population standard deviation of each metric over the repeats where it is defined."""
    mean, spread = {}, {}
    for m in rank_metrics.METRICS:
        values = [r[m] for r in repeats if "failed" not in r and r.get(m) is not None]
        mean[m] = statistics.fmean(values) if values else None
        spread[m] = statistics.pstdev(values) if values else None
    return mean, spread


def replay(rows: list[dict], k: int, repeats: int, text_chars: int, score_fn=None, subject: str = "",
           target_audience: str = "", experience_level: str = "") -> dict:
    """Score every report's labeled articles `repeats` times and measure each repeat against the labels.

    score_fn(candidates, subject=, target_audience=, experience_level=, top_k=, text_chars=) -> {index: score};
    it defaults to the production modules.nodes.score_candidates. A repeat whose scoring raises, or that scores
    fewer articles than it was given, is recorded as {"failed": reason} and left out of the means.
    """
    if score_fn is None:
        from modules.nodes import score_candidates as score_fn  # imported late: it sets up Google clients
    reports, failed = [], 0
    by_origin: dict[str, list[int]] = {}  # origin -> [good labeled picks, labeled picks]
    for run, group in _group_by_report(rows).items():
        candidates = golden_to_candidates(group, subject)
        labels = {r["url"]: r["label"] for r in group}
        origin = {r["url"]: r.get("origin") or "" for r in group}
        results = []
        for _ in range(repeats):
            try:
                scores = score_fn(candidates, subject=subject, target_audience=target_audience,
                                  experience_level=experience_level, top_k=k, text_chars=text_chars)
                if len(scores) < len(candidates):
                    raise ValueError(f"ranker scored {len(scores)} of {len(candidates)} articles")
            except Exception as e:
                failed += 1
                results.append({"failed": f"{type(e).__name__}: {e}"[:300]})
                continue
            metrics = rank_metrics.evaluate_report({group[i]["url"]: s for i, s in scores.items()}, labels, k)
            results.append(metrics)
            for p in metrics["picks"]:
                tally = by_origin.setdefault(origin[p], [0, 0])
                tally[0] += labels[p] in rank_metrics.GOOD_LABELS
                tally[1] += 1
        mean, spread = _mean_and_spread(results)
        reports.append({"source_run": run, "n": len(group), "repeats": results, "mean": mean, "spread": spread})

    # overall: per repeat, the mean over the reports where the metric is defined; then mean and spread over repeats
    per_repeat = []
    for i in range(repeats):
        ok = [rep["repeats"][i] for rep in reports if "failed" not in rep["repeats"][i]]
        if ok:
            per_repeat.append(rank_metrics.aggregate(ok))
    mean, spread = _mean_and_spread(per_repeat)
    defined = rank_metrics.aggregate([rep["mean"] for rep in reports])
    overall = {"mean": mean, "spread": spread, "n_reports": len(reports), "n_defined": defined["n_defined"],
               "by_origin": {o: {"precision": good / n if n else None, "labeled_picks": n}
                             for o, (good, n) in sorted(by_origin.items())}}
    return {"settings": {"k": k, "repeats": repeats, "text_chars": text_chars}, "reports": reports,
            "overall": overall, "failed_repeats": failed}


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def format_result(result: dict) -> str:
    lines = [f"{'report':<18} {'n':>3}  {'precision':>16}  {'recall':>16}  failed"]
    for rep in result["reports"]:
        cells = [f"{_fmt(rep['mean'][m])} ± {_fmt(rep['spread'][m])}" for m in rank_metrics.METRICS]
        fails = sum("failed" in r for r in rep["repeats"])
        lines.append(f"{rep['source_run'] or '(none)':<18} {rep['n']:>3}  {cells[0]:>16}  {cells[1]:>16}  {fails}")
    o = result["overall"]
    cells = [f"{_fmt(o['mean'][m])} ± {_fmt(o['spread'][m])}" for m in rank_metrics.METRICS]
    lines.append(f"{'overall':<18} {'':>3}  {cells[0]:>16}  {cells[1]:>16}  {result['failed_repeats']}")
    lines.append(f"defined in {o['n_defined']['precision']}/{o['n_reports']} reports (precision), "
                 f"{o['n_defined']['recall']}/{o['n_reports']} (recall); spread = std. deviation over repeats")
    for origin, v in o["by_origin"].items():
        lines.append(f"  precision of picks from {origin}: {_fmt(v['precision'])} ({v['labeled_picks']} labeled picks)")
    return "\n".join(lines)


def _plan(rows: list[dict], k: int, repeats: int, text_chars: int) -> str:
    groups = _group_by_report(rows)
    articles = sum(len(g) for g in groups.values())
    text = sum(len(article_text(a, text_chars)) for g in groups.values() for a in golden_to_candidates(g))
    per_round = text + articles * ITEM_OVERHEAD_CHARS + len(groups) * PROMPT_OVERHEAD_CHARS
    return (f"{len(groups)} reports, {articles} labeled articles, k={k}, text_chars={text_chars}\n"
            f"{len(groups) * repeats} model calls ({len(groups)} reports x {repeats} repeats), "
            f"about {per_round * repeats:,} input characters in total\n"
            f"No model was called. Add --run to score them with Gemini.")


def main(argv: list[str] | None = None, output_dir: Path = OUTPUT_DIR, score_fn=None) -> Path | None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=rank_split.SPLITS, default="train")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--text-chars", type=int, default=settings.RANK_TEXT_CHARS)
    ap.add_argument("--golden", type=Path, default=GOLDEN_RANKING_PATH)
    ap.add_argument("--profile", default="AI", help="reads <PROFILE>_SUBJECT, _AUDIENCE, _EXPERIENCE from the env")
    ap.add_argument("--run", action="store_true", help="call Gemini (costs money); without it only the plan prints")
    args = ap.parse_args(argv)

    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    rows = rank_split.split_rows(golden, args.split)
    print(f"golden: {len(golden)} rows, split {args.split}: {len(rows)} rows {rank_split.split_counts(golden)}")
    if args.split == "test":
        print("Reminder: the test split is meant to be scored once per tuning round, never to guide edits.")
    if not args.run:
        print(_plan(rows, args.k, args.repeats, args.text_chars))
        return None

    p = args.profile.upper()
    subject, audience, level = (os.getenv(f"{p}_{n}", "") for n in ("SUBJECT", "AUDIENCE", "EXPERIENCE"))
    if score_fn is None:
        if not subject:
            raise SystemExit(f"{p}_SUBJECT is not set; pass --profile for a configured profile")
        from modules.nodes import GEMINI_MODEL as model
    else:
        model = getattr(score_fn, "model", "injected")
    result = replay(rows, args.k, args.repeats, args.text_chars, score_fn=score_fn, subject=subject,
                    target_audience=audience, experience_level=level)
    result["settings"].update(split=args.split, model=model, golden_rows=len(golden), golden=str(args.golden),
                              profile=p, snippet_cap=settings.SNIPPET_SCORE_CAP, date=str(date.today()))
    print(format_result(result))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"rank_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {path}")
    return path


if __name__ == "__main__":
    main()
