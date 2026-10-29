"""Ranking metrics against the user's labels (pure functions, no model calls).

precision@k = picks labeled must or worth / labeled picks in the top k.
recall@k    = must-reads in the top k / min(k, total must-reads)  (capped: a perfect ranker scores 1.0 even when
              there are more must-reads than k).
A metric is None (never 0) when it is undefined: no labeled picks, or no must-reads.

Pooled over every labeled article, not only the top k (pooled()):
pairwise   = share of pairs with different labels, within one report, where the better label scored higher
             (a tie counts half). A random ranker scores about 0.5. pairwise_read leaves out teaser-only articles,
             whose score the snippet cap holds down whatever the prompt says.
mean_<label>         = average score of the articles with that label, teaser-only ones left out;
teaser_mean_<label>  = the same for the teaser-only articles.
"""
METRICS = ("precision", "recall")
POOLED = ("pairwise", "pairwise_read", "mean_must", "mean_worth", "mean_skip")
GOOD_LABELS = {"must", "worth"}
LABEL_RANK = {"skip": 0, "worth": 1, "must": 2}


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


def pair_counts(scores: dict[str, float], labels: dict[str, str], exclude=frozenset()) -> tuple[float, int]:
    """(correctly ordered pairs, pairs) over the scored, labeled articles not in exclude; a tie counts half."""
    keys = sorted(k for k in scores if labels.get(k) in LABEL_RANK and k not in exclude)
    right, total = 0.0, 0
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ra, rb = LABEL_RANK[labels[a]], LABEL_RANK[labels[b]]
            if ra == rb:
                continue
            hi, lo = (a, b) if ra > rb else (b, a)
            total += 1
            right += 1.0 if scores[hi] > scores[lo] else 0.5 if scores[hi] == scores[lo] else 0.0
    return right, total


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pooled(reports: list[tuple[dict, dict, set]]) -> dict:
    """Pairwise accuracy and mean score per label over (scores, labels, teaser keys) of several reports.

    Pairs are only formed within a report: that is the choice the ranker faces on one day.
    """
    right = total = right_read = total_read = 0
    read, teaser = {lab: [] for lab in LABEL_RANK}, {lab: [] for lab in LABEL_RANK}
    for scores, labels, teasers in reports:
        r, t = pair_counts(scores, labels)
        rr, tr = pair_counts(scores, labels, exclude=teasers)
        right, total, right_read, total_read = right + r, total + t, right_read + rr, total_read + tr
        for key, score in scores.items():
            if labels.get(key) in LABEL_RANK:
                (teaser if key in teasers else read)[labels[key]].append(score)
    out = {"pairwise": right / total if total else None, "pairs": total,
           "pairwise_read": right_read / total_read if total_read else None, "pairs_read": total_read}
    for lab in ("must", "worth", "skip"):
        out[f"mean_{lab}"] = _mean(read[lab])
        out[f"teaser_mean_{lab}"] = _mean(teaser[lab])
    return out


def aggregate(per_report: list[dict]) -> dict:
    """Mean of each metric over the reports where it is defined; n_defined says how many that was."""
    out = {"n_reports": len(per_report), "n_defined": {}}
    for m in METRICS:
        values = [r[m] for r in per_report if r.get(m) is not None]
        out[m] = sum(values) / len(values) if values else None
        out["n_defined"][m] = len(values)
    return out
