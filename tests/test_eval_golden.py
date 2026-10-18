"""Offline tests for the golden eval model parameter. Run from the repo root: python -m tests.test_eval_golden"""
from types import SimpleNamespace
from unittest.mock import patch

from config import settings
from evals import eval_golden


def _run(model=None, fake_score=3):
    calls = []

    def fake_judge(top_articles, newsletter_data, model=None):
        calls.append(model)
        return SimpleNamespace(score=fake_score, reasoning="ok")

    with patch("evals.eval_golden._judge_content_faithfulness", fake_judge):
        report = eval_golden.run_golden_eval(model) if model is not None else eval_golden.run_golden_eval()
    return report, calls


def test_model_passes_through_to_judge():
    report, calls = _run("some-model")
    assert calls and all(c == "some-model" for c in calls), calls
    assert report["model"] == "some-model"


def test_default_resolves_to_daily_model():
    report, _ = _run()
    assert report["model"] == "claude-haiku-4-5", report["model"]


def test_eval_shortcut_resolves_to_eval_model():
    report, calls = _run("eval")
    assert report["model"] == "claude-sonnet-5-5"
    assert all(c == "claude-sonnet-5-5" for c in calls), calls


def test_report_has_original_score_per_case():
    report, _ = _run()
    assert report["cases"]
    assert all("original_judge_score" in c for c in report["cases"])
    assert [c["original_judge_score"] for c in report["cases"]] == [3, 3, 3, 3, 5, 5]


def test_score_within_tolerance_counts_as_agreement():
    cases = eval_golden._load_golden_set()
    human = cases[0]["human_label"]["score"]
    near, _ = _run(fake_score=human + eval_golden.AGREEMENT_TOLERANCE)
    far, _ = _run(fake_score=human + eval_golden.AGREEMENT_TOLERANCE + 1)
    assert near["cases"][0]["agree"] is True
    assert far["cases"][0]["agree"] is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
