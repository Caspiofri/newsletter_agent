"""Ranking metrics against the user's labels (pure functions, no model calls).

precision@k = picks labeled must or worth / labeled picks in the top k.
recall@k    = must-reads in the top k / min(k, total must-reads)  (capped: a perfect ranker scores 1.0 even when
              there are more must-reads than k).
A metric is None (never 0) when it is undefined: no labeled picks, or no must-reads.
"""
METRICS = ("precision", "recall")
GOOD_LABELS = {"must", "worth"}


def evaluate_report(scores: dict[str, float], labels: dict[str, str], k: int) -> dict:
    # highest score first, ties broken by key so the same scores always give the same picks
    picks = sorted(scores, key=lambda key: (-scores[key], key))[:min(k, len(scores))]
    labeled = [p for p in picks if p in labels]
    must_total = sum(v == "must" for v in labels.values())
    must_in_top = sum(labels.get(p) == "must" for p in picks)
    # k larger than the pool means all of it was picked, so the cap is the number of picks
    cap = min(len(picks), must_total)
    return {
        "k": k,
        "picks": picks,
        "precision": sum(labels[p] in GOOD_LABELS for p in labeled) / len(labeled) if labeled else None,
        "recall": must_in_top / cap if cap else None,
        "labeled_picks": len(labeled),
        "must_total": must_total,
        "must_in_top": must_in_top,
    }


def aggregate(per_report: list[dict]) -> dict:
    """Mean of each metric over the reports where it is defined; n_defined says how many that was."""
    out = {"n_reports": len(per_report), "n_defined": {}}
    for m in METRICS:
        values = [r[m] for r in per_report if r.get(m) is not None]
        out[m] = sum(values) / len(values) if values else None
        out["n_defined"][m] = len(values)
    return out
