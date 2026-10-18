"""Offline tests for LangSmith feedback and run summary. Run from the repo root: python -m tests.test_tracing"""
from unittest.mock import MagicMock, patch

from modules import tracing
from modules.trace_judge import SKIPPED, StageScore, TraceReport


def _report():
    return TraceReport(
        profile="AI", timestamp="t", email_status="sent",
        stages=[
            StageScore("collect", 4, "good", {}),
            StageScore("dedupe", SKIPPED, "n/a", {}),
            StageScore("summarize", 5, "faithful", {}),
        ],
        overall=4.5,
    )


def test_feedback_items_skip_skipped_stages():
    keys = [i["key"] for i in tracing.feedback_items(_report())]
    assert "judge.dedupe" not in keys
    assert "judge.collect" in keys and "judge.summarize" in keys


def test_feedback_items_include_overall():
    items = tracing.feedback_items(_report())
    assert items[-1] == {"key": "judge.overall", "score": 4.5}
    assert {"key": "judge.collect", "score": 4, "comment": "good"} in items


def test_post_feedback_calls_client_per_item():
    client = MagicMock()
    tracing.post_feedback(_report(), "run-1", client=client)
    assert client.create_feedback.call_count == 3  # collect, summarize, overall
    client.create_feedback.assert_any_call("run-1", key="judge.overall", score=4.5, comment=None)


def test_post_feedback_swallows_client_errors():
    client = MagicMock()
    client.create_feedback.side_effect = RuntimeError("boom")
    tracing.post_feedback(_report(), "run-1", client=client)  # must not raise


def test_post_is_noop_without_run_id():
    client = MagicMock()
    tracing.post_feedback(_report(), None, client=client)
    tracing.post_summary({"stats": {"merged": 3}}, None, client=client)
    assert client.create_feedback.call_count == 0


class _Art:
    def __init__(self, name):
        self.name = name


def test_run_summary_reads_final_state_and_tolerates_missing_keys():
    result = {
        "run_stats": {"merged": 10, "picked": 2},
        "top_articles": [_Art("A"), _Art("B")],
        "dedupe_drops": [{"title": "x", "similarity": 0.9}],
        "rank_scores": [{"title": "A", "score": 8}],
        "email_status": "sent",
    }
    s = tracing.run_summary(result)
    assert s == {"stats": {"merged": 10, "picked": 2}, "picked": ["A", "B"],
                 "dedupe_drops": result["dedupe_drops"], "rank_scores": result["rank_scores"],
                 "email_status": "sent"}
    empty = tracing.run_summary({})
    assert empty == {"stats": {}, "picked": [], "dedupe_drops": [], "rank_scores": [], "email_status": ""}


def test_summary_feedback_items_one_per_stat():
    items = tracing.summary_feedback_items({"stats": {"merged": 10, "picked": 2}})
    assert items == [{"key": "stat.merged", "score": 10}, {"key": "stat.picked", "score": 2}]


def test_summary_feedback_items_accepts_float_stats():
    items = tracing.summary_feedback_items({"stats": {"dedupe_max_similarity": 0.934, "picked": 2}})
    assert items == [{"key": "stat.dedupe_max_similarity", "score": 0.934}, {"key": "stat.picked", "score": 2}]


def test_judge_and_post_survives_judge_failure():
    rt = MagicMock()
    rt.id = "run-1"
    with patch("modules.trace_judge.judge_run", side_effect=RuntimeError("judge down")):
        with patch("modules.tracing.post_summary") as ps:
            tracing.judge_and_post({"email_status": "sent"}, "AI", rt)
    rt.add_metadata.assert_called_once()
    ps.assert_called_once()


def test_judge_and_post_without_run_tree():
    with patch("modules.trace_judge.judge_run", side_effect=RuntimeError("judge down")):
        tracing.judge_and_post({"email_status": "sent"}, "AI", None)  # must not raise


def test_judge_and_post_never_raises_when_summary_fails():
    rt = MagicMock()
    with patch.object(tracing, "run_summary", side_effect=RuntimeError("boom")):
        tracing.judge_and_post({"email_status": "sent"}, "AI", rt)  # must not raise


def test_tests_package_disables_tracing():
    import os
    from langsmith.utils import get_env_var, tracing_is_enabled
    old = os.environ.get("LANGCHAIN_TRACING")
    os.environ["LANGCHAIN_TRACING"] = "true"
    get_env_var.cache_clear()
    try:
        assert os.environ["LANGSMITH_TRACING"] == "false"
        assert not tracing_is_enabled()
    finally:
        if old is None:
            os.environ.pop("LANGCHAIN_TRACING", None)
        else:
            os.environ["LANGCHAIN_TRACING"] = old
        get_env_var.cache_clear()


def test_context_thread_pool_copies_context_to_workers():
    import contextvars
    from langsmith.utils import ContextThreadPoolExecutor
    from modules import nodes
    assert nodes.ThreadPoolExecutor is ContextThreadPoolExecutor
    var = contextvars.ContextVar("v", default=None)
    var.set("parent")
    with nodes.ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: var.get(), [1, 2])) == ["parent", "parent"]
        assert pool.submit(var.get).result() == "parent"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
