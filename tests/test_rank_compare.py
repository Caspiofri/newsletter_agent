"""Offline tests for comparing two rank_eval results. Run from the repo root: python -m tests.test_rank_compare"""
import contextlib
import io
import json
import tempfile
from pathlib import Path

from evals import rank_compare


def _result(precision, recall, p_spread=0.05, r_spread=0.05, **settings):
    s = {"k": 5, "repeats": 3, "text_chars": 6000, "split": "train", "model": "gemini-2.5-flash",
         "golden_rows": 60, "date": "2026-10-20", **settings}
    return {"settings": s, "overall": {"mean": {"precision": precision, "recall": recall},
                                       "spread": {"precision": p_spread, "recall": r_spread}}}


def test_compare_reports_difference_and_noise_flag():
    out = rank_compare.compare(_result(0.6, 0.5), _result(0.8, 0.53, p_spread=0.1))
    p, r = out["metrics"]["precision"], out["metrics"]["recall"]
    assert abs(p["diff"] - 0.2) < 1e-9 and p["verdict"] == "clear"  # 0.2 > max(0.05, 0.1)
    assert abs(r["diff"] - 0.03) < 1e-9 and r["verdict"] == "within noise"  # 0.03 <= 0.05
    assert p["baseline"] == 0.6 and p["candidate"] == 0.8 and p["noise"] == 0.1


def test_compare_includes_pooled_metrics_when_both_have_them():
    base, cand = _result(0.6, 0.5), _result(0.6, 0.5)
    base["overall"]["pooled"] = {"mean": {"pairwise": 0.60, "mean_must": 6.0}, "spread": {"pairwise": 0.02, "mean_must": 0.1}}
    cand["overall"]["pooled"] = {"mean": {"pairwise": 0.70, "mean_must": 6.05}, "spread": {"pairwise": 0.03, "mean_must": 0.2}}
    out = rank_compare.compare(base, cand)
    assert out["metrics"]["pairwise"]["verdict"] == "clear" and abs(out["metrics"]["pairwise"]["diff"] - 0.1) < 1e-9
    assert out["metrics"]["mean_must"]["verdict"] == "within noise"
    assert "pairwise" in rank_compare.format_comparison(out)
    # an older baseline without pooled metrics shows them as n/a instead of failing
    out = rank_compare.compare(_result(0.6, 0.5), cand)
    assert out["metrics"]["pairwise"]["verdict"] == "n/a"


def test_compare_refuses_different_split_or_k():
    for changed in ({"split": "test"}, {"k": 3}, {"golden_rows": 61}):
        try:
            rank_compare.compare(_result(0.6, 0.5), _result(0.6, 0.5, **changed))
            assert False, changed
        except ValueError as e:
            assert list(changed)[0] in str(e)


def test_compare_lists_changed_settings():
    out = rank_compare.compare(_result(0.6, 0.5, text_chars=2500), _result(0.7, 0.5, date="2026-10-21"))
    assert out["changed_settings"] == ["text_chars: 2500 vs 6000"]  # the run date is not a setting


def test_compare_handles_none_metrics():
    out = rank_compare.compare(_result(None, 0.5, p_spread=None), _result(0.7, 0.5))
    p = out["metrics"]["precision"]
    assert p["diff"] is None and p["verdict"] == "n/a"
    assert "n/a" in rank_compare.format_comparison(out)


def test_cli_prints_table_and_refusal():
    with tempfile.TemporaryDirectory() as tmp:
        a, b, c = Path(tmp) / "a.json", Path(tmp) / "b.json", Path(tmp) / "c.json"
        a.write_text(json.dumps(_result(0.6, 0.5, text_chars=2500)))
        b.write_text(json.dumps(_result(0.8, 0.5)))
        c.write_text(json.dumps(_result(0.8, 0.5, k=3)))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rank_compare.main([str(a), str(b)])
        assert "precision" in buf.getvalue() and "text_chars: 2500 vs 6000" in buf.getvalue()
        try:
            rank_compare.main([str(a), str(c)])
            assert False
        except SystemExit as e:
            assert "k" in str(e.code)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
