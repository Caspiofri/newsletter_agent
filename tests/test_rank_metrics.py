"""Offline tests for the ranking metrics (precision@k, capped recall@k). Run: python -m tests.test_rank_metrics"""
from evals import rank_metrics


def test_precision_counts_must_and_worth():
    scores = {"a": 9, "b": 8, "c": 7, "d": 1}
    labels = {"a": "must", "b": "worth", "c": "skip", "d": "must"}
    out = rank_metrics.evaluate_report(scores, labels, k=3)
    assert out["picks"] == ["a", "b", "c"]
    assert out["labeled_picks"] == 3
    assert abs(out["precision"] - 2 / 3) < 1e-9


def test_recall_is_capped_by_k():
    keys = [f"m{i:02d}" for i in range(17)] + ["w1", "w2"]
    labels = {k: "must" if k.startswith("m") else "worth" for k in keys}
    scores = {k: (10 if k.startswith("m") else 1) for k in keys}  # a perfect ranker
    out = rank_metrics.evaluate_report(scores, labels, k=5)
    assert out["must_total"] == 17 and out["must_in_top"] == 5
    assert out["recall"] == 1.0  # uncapped it would be 5/17


def test_recall_with_fewer_must_reads_than_k():
    labels = {"m1": "must", "m2": "must", "a": "worth", "b": "worth", "c": "skip", "d": "skip"}
    scores = {"m1": 9, "a": 8, "b": 7, "c": 6, "d": 5, "m2": 1}
    out = rank_metrics.evaluate_report(scores, labels, k=5)
    assert out["must_in_top"] == 1 and out["must_total"] == 2
    assert out["recall"] == 0.5


def test_no_must_reads_gives_none_recall():
    out = rank_metrics.evaluate_report({"a": 5, "b": 3}, {"a": "worth", "b": "skip"}, k=1)
    assert out["recall"] is None
    assert out["precision"] == 1.0


def test_no_labeled_picks_gives_none_precision():
    # the top pick is not labeled (scores may hold more articles than labels)
    out = rank_metrics.evaluate_report({"x": 9, "a": 1}, {"a": "must"}, k=1)
    assert out["picks"] == ["x"] and out["labeled_picks"] == 0
    assert out["precision"] is None
    assert out["recall"] == 0.0
    assert rank_metrics.evaluate_report({}, {}, k=5)["precision"] is None


def test_k_larger_than_pool_uses_all():
    out = rank_metrics.evaluate_report({"a": 2, "b": 1}, {"a": "must", "b": "skip"}, k=10)
    assert out["picks"] == ["a", "b"]
    assert out["precision"] == 0.5 and out["recall"] == 1.0


def test_ties_broken_by_key_deterministically():
    scores = {"c": 7, "a": 7, "b": 7, "d": 9}
    labels = {k: "worth" for k in scores}
    assert rank_metrics.evaluate_report(scores, labels, k=3)["picks"] == ["d", "a", "b"]
    assert rank_metrics.evaluate_report(dict(reversed(scores.items())), labels, k=3)["picks"] == ["d", "a", "b"]


def test_aggregate_skips_undefined_and_counts_defined():
    reports = [{"precision": 1.0, "recall": None}, {"precision": 0.5, "recall": 0.5}, {"precision": None, "recall": 1.0}]
    out = rank_metrics.aggregate(reports)
    assert out["n_reports"] == 3
    assert out["precision"] == 0.75 and out["recall"] == 0.75
    assert out["n_defined"] == {"precision": 2, "recall": 2}
    empty = rank_metrics.aggregate([{"precision": None, "recall": None}])
    assert empty["precision"] is None and empty["recall"] is None and empty["n_defined"] == {"precision": 0, "recall": 0}


def test_pair_counts_order_ties_and_same_label_pairs():
    labels = {"m": "must", "w": "worth", "s": "skip", "s2": "skip"}
    scores = {"m": 9, "w": 5, "s": 5, "s2": 7}
    # m>w ok, m>s ok, m>s2 ok, w>s tie (half), w>s2 wrong; s vs s2 is not a pair
    assert rank_metrics.pair_counts(scores, labels) == (3.5, 5)
    assert rank_metrics.pair_counts(scores, labels, exclude={"m"}) == (0.5, 2)
    assert rank_metrics.pair_counts({"a": 1}, {"a": "must"}) == (0, 0)


def test_pooled_counts_pairs_only_within_a_report_and_means_by_label():
    r1 = ({"a": 9, "b": 2}, {"a": "must", "b": "skip"}, set())
    r2 = ({"c": 3, "d": 8, "e": 4}, {"c": "must", "d": "skip", "e": "worth"}, {"e"})
    out = rank_metrics.pooled([r1, r2])
    # r1: 1 of 1 right; r2: must<skip wrong, must<worth wrong, worth<skip wrong -> 1 of 4 overall
    assert out["pairwise"] == 0.25 and out["pairs"] == 4
    # without teaser e: r1 1/1, r2 c vs d wrong -> 1 of 2
    assert out["pairwise_read"] == 0.5 and out["pairs_read"] == 2
    # means leave teasers out: must (9+3)/2, skip (2+8)/2, worth has no read article
    assert out["mean_must"] == 6.0 and out["mean_skip"] == 5.0 and out["mean_worth"] is None
    assert out["teaser_mean_worth"] == 4.0 and out["teaser_mean_must"] is None
    empty = rank_metrics.pooled([])
    assert empty["pairwise"] is None and empty["pairs"] == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
