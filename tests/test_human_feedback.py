"""Offline tests for the human picks score posted to LangSmith. Run from the repo root: python -m tests.test_human_feedback
No network: the LangSmith client is a fake passed in."""
import io
import os
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from langsmith.utils import LangSmithConflictError

from evals import human_feedback

RUN = "20261021_071000"
SAVED = datetime.strptime(RUN, "%Y%m%d_%H%M%S").astimezone(timezone.utc)  # report time, local -> UTC
KEY_ENV = {"LANGSMITH_API_KEY": "test-key", "LANGSMITH_PROJECT": "test-project"}


def _g(url, label, picked=True, run=RUN):
    return {"url": url, "label": label, "ranker_picked": picked, "source_run": run}


def _run(minutes_before, run_id=None):
    return SimpleNamespace(id=run_id or uuid.uuid4(), name="digest_run", start_time=SAVED - timedelta(minutes=minutes_before))


class FakeClient:
    """Keeps feedback by id; a second create with the same id raises a conflict, like the API."""
    def __init__(self, runs=(), fail=None):
        self.runs, self.fail = list(runs), fail
        self.feedback, self.calls = {}, []

    def list_runs(self, **kw):
        self.calls.append(("list_runs", kw))
        if self.fail:
            raise self.fail
        return iter(self.runs)

    def create_feedback(self, run_id, key, *, score=None, comment=None, feedback_id=None, **kw):
        self.calls.append(("create_feedback", key))
        if feedback_id in self.feedback:
            raise LangSmithConflictError("409 Conflict: feedback already exists")
        self.feedback[feedback_id] = {"run_id": run_id, "key": key, "score": score, "comment": comment}

    def update_feedback(self, feedback_id, *, score=None, comment=None, **kw):
        self.calls.append(("update_feedback", feedback_id))
        self.feedback[feedback_id].update(score=score, comment=comment)


# ---- metric

def test_precision_counts_rated_picks_only():
    golden = [_g("a", "must"), _g("b", "worth"), _g("c", "skip"), _g("d", "must"), _g("e", "skip"),
              _g("f", "must", picked=False),       # not a pick
              _g("g", "must", run="20261020_070000"),  # another run
              _g("h", "", picked=True), _g("i", "maybe")]  # no valid label
    p, n = human_feedback.picks_precision(golden, RUN)
    assert n == 5 and abs(p - 0.6) < 1e-9


def test_precision_none_when_nothing_rated():
    assert human_feedback.picks_precision([], RUN) == (None, 0)
    assert human_feedback.picks_precision([_g("a", "must", picked=False), _g("b", "")], RUN) == (None, 0)
    assert human_feedback.picks_precision([_g("a", "skip")], RUN) == (0.0, 1)


# ---- run lookup

def test_find_run_latest_inside_window():
    inside, older = _run(5), _run(50)
    client = FakeClient([older, _run(61), _run(-1), inside])
    with patch.dict(os.environ, KEY_ENV):
        assert human_feedback.find_digest_run(RUN, client) is inside
    [(_, kw)] = client.calls
    assert kw["project_name"] == "test-project" and kw["is_root"] is True and "digest_run" in kw["filter"]


def test_find_run_at_report_time_and_naive_utc():
    exact = SimpleNamespace(id=uuid.uuid4(), name="digest_run", start_time=SAVED.replace(tzinfo=None))
    with patch.dict(os.environ, KEY_ENV):
        assert human_feedback.find_digest_run(RUN, FakeClient([exact])) is exact


def test_find_run_too_early_after_or_none():
    with patch.dict(os.environ, KEY_ENV):
        assert human_feedback.find_digest_run(RUN, FakeClient([_run(61), _run(-1)])) is None
        assert human_feedback.find_digest_run(RUN, FakeClient([])) is None
        assert human_feedback.find_digest_run("not-a-run", FakeClient([_run(5)])) is None


def test_find_run_ignores_other_names():
    other = SimpleNamespace(id=uuid.uuid4(), name="rank", start_time=SAVED - timedelta(minutes=2))
    with patch.dict(os.environ, KEY_ENV):
        assert human_feedback.find_digest_run(RUN, FakeClient([other])) is None


# ---- posting

GOLDEN = [_g("a", "must"), _g("b", "worth"), _g("c", "skip"), _g("d", "must"), _g("e", "worth")]


def test_post_creates_both_keys():
    root = _run(5)
    client = FakeClient([root])
    with patch.dict(os.environ, KEY_ENV):
        assert human_feedback.post_picks_precision(GOLDEN, RUN, client) is True
    by_key = {f["key"]: f for f in client.feedback.values()}
    assert set(by_key) == {"human.picks_precision", "count.human_picks_rated"}
    assert by_key["human.picks_precision"]["score"] == 0.8 and by_key["human.picks_precision"]["run_id"] == root.id
    assert by_key["human.picks_precision"]["comment"] == "4 of 5 rated picks must/worth"
    assert by_key["count.human_picks_rated"]["score"] == 5
    assert uuid.uuid5(human_feedback.FEEDBACK_NAMESPACE, f"{root.id}:human.picks_precision") in client.feedback


def test_post_again_updates_same_feedback():
    client = FakeClient([_run(5)])
    with patch.dict(os.environ, KEY_ENV):
        human_feedback.post_picks_precision(GOLDEN, RUN, client)
        assert human_feedback.post_picks_precision(GOLDEN[:3], RUN, client) is True
    assert len(client.feedback) == 2  # same ids, updated in place
    assert [c[0] for c in client.calls].count("update_feedback") == 2
    by_key = {f["key"]: f for f in client.feedback.values()}
    assert abs(by_key["human.picks_precision"]["score"] - 2 / 3) < 1e-9 and by_key["count.human_picks_rated"]["score"] == 3
    assert by_key["human.picks_precision"]["comment"] == "2 of 3 rated picks must/worth"


def test_post_skips_without_key_run_or_rating():
    client = FakeClient([_run(5)])
    env = {k: v for k, v in os.environ.items() if k != "LANGSMITH_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        assert human_feedback.post_picks_precision(GOLDEN, RUN, client) is False
    assert client.calls == []
    with patch.dict(os.environ, KEY_ENV):
        assert human_feedback.post_picks_precision([], RUN, client) is False  # nothing rated: no lookup
        assert client.calls == []
        assert human_feedback.post_picks_precision(GOLDEN, RUN, FakeClient([_run(90)])) is False
    assert client.feedback == {}


def test_post_swallows_langsmith_errors():
    out = io.StringIO()
    with patch.dict(os.environ, KEY_ENV), redirect_stdout(out):
        assert human_feedback.post_picks_precision(GOLDEN, RUN, FakeClient(fail=RuntimeError("503 down"))) is False
    assert "503 down" in out.getvalue()


# ---- backfill

def test_backfill_one_line_per_rated_run():
    other = "20261020_070000"
    golden = GOLDEN + [_g("x", "skip", run=other), _g("y", "must", picked=False, run="20261019_070000")]
    client = FakeClient([_run(5)])  # only RUN has a root run
    out = io.StringIO()
    with patch.dict(os.environ, KEY_ENV), redirect_stdout(out):
        human_feedback.backfill(golden, client)
    lines = out.getvalue().splitlines()
    assert len(lines) == 2  # 20261019 has no rated picks
    assert lines[0].startswith(other) and "0.00" in lines[0] and "n=1" in lines[0] and "not found" in lines[0]
    assert lines[1].startswith(RUN) and "0.80" in lines[1] and "n=5" in lines[1] and "posted" in lines[1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
