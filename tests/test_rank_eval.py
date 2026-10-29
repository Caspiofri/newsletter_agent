"""Offline tests for the ranker replay tool (evals.rank_eval) and the ranker prompt/scoring extracted from
rank_articles. No test calls a model. Run from the repo root: python -m tests.test_rank_eval"""
import contextlib
import io
import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

from config import settings
from evals import rank_eval
from modules import nodes, ranking_report
from modules.models import Article


def _art(name, origin="inbox", status="full", content="body text"):
    return Article(subject="AI", name=name, published_at=date(2026, 10, 18), content=content,
                   url=f"https://x.example/{name}", origin=origin, read_status=status)


def _rank_state(cands):
    return {"candidates": cands, "top_k": 2, "subject": "AI", "target_audience": "devs",
            "experience_level": "mid", "digest_name": "d", "articles": cands, "run_stats": {}}


def _run_rank_articles(cands, response):
    seen = []

    def fake(prompt):
        seen.append(prompt)
        return response

    with patch.object(nodes, "_call_gemini", side_effect=fake), \
            patch.object(ranking_report, "build_rows", return_value=[]), \
            patch.object(ranking_report, "save", return_value="x"), \
            contextlib.redirect_stdout(io.StringIO()):
        out = nodes.rank_articles(_rank_state(cands))
    return out, seen


def test_build_rank_prompt_matches_what_rank_articles_sends():
    cands = [_art("A", content="<p>alpha " * 50 + "</p>"), _art("B", "hackernews", "snippet")]
    cands[1] = cands[1].model_copy(update={"signal": "120 points"})
    _, seen = _run_rank_articles(cands, '[{"id": 0, "score": 7, "reason": "ok"}]')
    built = nodes.build_rank_prompt(cands, "AI", "devs", "mid", 2, settings.RANK_TEXT_CHARS)
    assert seen == [built]
    assert nodes.build_rank_prompt(cands, "AI", "devs", "mid", 2, 10) != built  # text_chars is honored


def test_rank_articles_behavior_unchanged():
    cands = [_art("A"), _art("B", status="snippet"), _art("C")]
    resp = json.dumps([{"id": 0, "score": 6, "reason": "ok"}, {"id": 1, "score": 9, "reason": "teaser"},
                       {"id": 2, "score": 8, "reason": "great"}])
    out, _ = _run_rank_articles(cands, resp)
    assert [a.name for a in out["top_articles"]] == ["C", "A"]  # A and B tie at 6 after the cap: input order
    assert {r["title"]: r["score"] for r in out["rank_scores"]} == {"A": 6, "B": 6, "C": 8}  # B capped
    assert out["run_stats"] == {"ranked": 3, "picked": 2}
    # score_candidates gives the same final scores from the same reply, through the same prompt
    with patch.object(nodes, "_call_gemini", return_value=resp):
        assert nodes.score_candidates(cands, "AI", "devs", "mid", 2, settings.RANK_TEXT_CHARS) == {0: 6, 1: 6, 2: 8}
    for bad in ("no json here", "[]", '[{"id": "x"}]'):
        try:
            nodes.score_candidates(cands, "AI", "devs", "mid", 2, 100, call=lambda p, bad=bad: bad)
            assert False, bad
        except ValueError:
            pass


def _row(title, label, run="r1", origin="inbox"):
    return {"url": f"https://x.example/{run}/{title}", "title": title, "label": label, "origin": origin,
            "source_run": run, "read_status": "full", "content": f"text of {title}", "group": "shortlisted"}


def _score_fn(table):
    """Fake ranker: scores each candidate by its title from a dict."""
    calls = []

    def fn(candidates, **kw):
        calls.append([c.name for c in candidates])
        return {i: table[c.name] for i, c in enumerate(candidates)}
    fn.calls = calls
    return fn


ROWS = [_row("a", "must"), _row("b", "skip"), _row("c", "worth"),
        _row("d", "must", "r2"), _row("e", "worth", "r2"), _row("f", "must", "r2")]
SCORES = {"a": 9, "b": 8, "c": 1, "d": 9, "e": 1, "f": 2}


def test_replay_groups_by_report_and_averages():
    fn = _score_fn(SCORES)
    out = rank_eval.replay(ROWS, k=1, repeats=2, text_chars=100, score_fn=fn)
    assert sorted(fn.calls) == sorted([["a", "b", "c"]] * 2 + [["d", "e", "f"]] * 2)
    by = {r["source_run"]: r for r in out["reports"]}
    assert by["r1"]["n"] == 3 and len(by["r1"]["repeats"]) == 2
    assert by["r1"]["mean"] == {"precision": 1.0, "recall": 1.0}
    assert by["r2"]["mean"] == {"precision": 1.0, "recall": 1.0}
    out = rank_eval.replay(ROWS, k=2, repeats=1, text_chars=100, score_fn=_score_fn(SCORES))
    by = {r["source_run"]: r for r in out["reports"]}
    assert by["r1"]["mean"]["precision"] == 0.5 and by["r1"]["mean"]["recall"] == 1.0
    assert by["r2"]["mean"]["precision"] == 1.0 and by["r2"]["mean"]["recall"] == 1.0
    assert out["overall"]["mean"] == {"precision": 0.75, "recall": 1.0}
    assert out["settings"]["k"] == 2 and out["settings"]["repeats"] == 1 and out["settings"]["text_chars"] == 100
    assert out["failed_repeats"] == 0


def test_replay_failed_repeat_is_reported_not_raised():
    table, n = dict(SCORES), [0]

    def flaky(candidates, **kw):
        n[0] += 1
        if n[0] == 2:
            raise ValueError("No usable scores in ranker response: 'oops'")
        if n[0] == 3:
            return {0: 5}  # fewer rows than candidates
        return {i: table[c.name] for i, c in enumerate(candidates)}

    out = rank_eval.replay(ROWS, k=1, repeats=2, text_chars=100, score_fn=flaky)
    assert out["failed_repeats"] == 2
    failed = [r for rep in out["reports"] for r in rep["repeats"] if "failed" in r]
    assert len(failed) == 2 and any("No usable" in f["failed"] for f in failed)
    assert any("1 of 3" in f["failed"] for f in failed)
    for rep in out["reports"]:
        assert rep["mean"]["precision"] == 1.0  # the failed repeat is excluded, not counted as 0


def test_replay_single_repeat_has_zero_spread():
    out = rank_eval.replay(ROWS, k=1, repeats=1, text_chars=100, score_fn=_score_fn(SCORES))
    for rep in out["reports"]:
        assert rep["spread"] == {"precision": 0.0, "recall": 0.0}
    assert out["overall"]["spread"] == {"precision": 0.0, "recall": 0.0}


def test_replay_spread_across_repeats():
    n = [0]

    def alternating(candidates, **kw):
        n[0] += 1
        good = {"a": 9, "b": 1, "c": 1} if n[0] % 2 else {"a": 1, "b": 9, "c": 1}
        return {i: good[c.name] for i, c in enumerate(candidates)}

    out = rank_eval.replay(ROWS[:3], k=1, repeats=2, text_chars=100, score_fn=alternating)
    rep = out["reports"][0]
    assert rep["mean"]["precision"] == 0.5 and rep["spread"]["precision"] == 0.5


def test_replay_report_without_must_reads_is_n_a_and_skipped_in_overall():
    rows = ROWS[:3] + [_row("x", "worth", "r3"), _row("y", "skip", "r3")]
    out = rank_eval.replay(rows, k=1, repeats=1, text_chars=100, score_fn=_score_fn({**SCORES, "x": 2, "y": 9}))
    by = {r["source_run"]: r for r in out["reports"]}
    assert by["r3"]["mean"]["recall"] is None and by["r3"]["spread"]["recall"] is None
    assert out["overall"]["mean"]["recall"] == 1.0  # r1 only, r3 skipped rather than counted as 0
    assert out["overall"]["n_defined"]["recall"] == 1 and out["overall"]["n_reports"] == 2
    assert "n/a" in rank_eval.format_result(out)


def test_replay_keeps_every_score_and_pools_pairwise():
    rows = ROWS[:3] + [{**_row("t", "must"), "read_status": "snippet"}]
    out = rank_eval.replay(rows, k=1, repeats=2, text_chars=100, score_fn=_score_fn({**SCORES, "t": 6}))
    rep = out["reports"][0]["repeats"][0]
    assert rep["scores"] == {f"https://x.example/r1/{t}": s for t, s in (("a", 9), ("b", 8), ("c", 1), ("t", 6))}
    pooled = out["overall"]["pooled"]
    # a(must)9 b(skip)8 c(worth)1 t(must, teaser)6: pairs a>b ok, a>c ok, c<b wrong, t<b wrong, t>c ok -> 3/5
    assert pooled["mean"]["pairwise"] == 0.6 and pooled["spread"]["pairwise"] == 0.0
    assert pooled["mean"]["pairwise_read"] == 2 / 3  # t left out
    assert pooled["mean"]["mean_must"] == 9 and pooled["mean"]["teaser_mean_must"] == 6
    assert pooled["pairs"] == 5 and pooled["pairs_read"] == 3
    text = rank_eval.format_result(out)
    assert "pairwise" in text and "0.60" in text and "must 9.0" in text


def test_replay_pooled_skips_failed_repeats():
    n = [0]

    def flaky(candidates, **kw):
        n[0] += 1
        if n[0] == 1:
            raise ValueError("bad reply")
        return {i: SCORES[c.name] for i, c in enumerate(candidates)}

    out = rank_eval.replay(ROWS[:3], k=1, repeats=2, text_chars=100, score_fn=flaky)
    assert out["overall"]["pooled"]["mean"]["pairwise"] == 2 / 3  # a>b ok, a>c ok, c<b wrong


def test_by_origin_precision():
    rows = [_row("a", "must", origin="inbox"), _row("b", "skip", origin="medium"),
            _row("c", "worth", origin="hackernews"), _row("d", "skip", origin="inbox")]
    out = rank_eval.replay(rows, k=3, repeats=1, text_chars=100,
                           score_fn=_score_fn({"a": 9, "b": 8, "c": 7, "d": 1}))
    by = out["overall"]["by_origin"]
    assert by["inbox"] == {"precision": 1.0, "labeled_picks": 1}
    assert by["medium"] == {"precision": 0.0, "labeled_picks": 1}
    assert by["hackernews"] == {"precision": 1.0, "labeled_picks": 1}


def test_golden_to_candidates():
    rows = [{**_row("a", "must", origin="hackernews"), "signal": "99 points", "published_at": "2026-10-01"},
            {**_row("b", "skip"), "read_status": "snippet"}]
    cands = rank_eval.golden_to_candidates(rows, subject="AI")
    assert [c.name for c in cands] == ["a", "b"]
    assert cands[0].origin == "hackernews" and cands[0].signal == "99 points"
    assert cands[0].published_at == date(2026, 10, 1) and cands[1].published_at == date.today()
    assert cands[1].read_status == "snippet" and cands[0].content == "text of a" and cands[0].subject == "AI"


def _golden_file(tmp, rows):
    path = Path(tmp) / "golden.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_cli_without_run_makes_no_model_call():
    with tempfile.TemporaryDirectory() as tmp:
        golden = _golden_file(tmp, ROWS)
        buf = io.StringIO()
        with patch.object(nodes, "_call_gemini", side_effect=AssertionError("model called")), \
                contextlib.redirect_stdout(buf):
            rank_eval.main(["--split", "train", "--golden", str(golden)], output_dir=Path(tmp))
        text = buf.getvalue()
        assert "model calls" in text and "--run" in text
        assert list(Path(tmp).glob("rank_eval_*.json")) == []
        buf = io.StringIO()
        with patch.object(nodes, "_call_gemini", side_effect=AssertionError("model called")), \
                contextlib.redirect_stdout(buf):
            rank_eval.main(["--split", "test", "--golden", str(golden)], output_dir=Path(tmp))
        assert "once per tuning round" in buf.getvalue()


def test_cli_run_writes_result_with_settings():
    with tempfile.TemporaryDirectory() as tmp:
        golden = _golden_file(tmp, ROWS)
        with contextlib.redirect_stdout(io.StringIO()):
            path = rank_eval.main(["--split", "all", "--golden", str(golden), "--k", "1", "--repeats", "2",
                                   "--text-chars", "2500", "--run"],
                                  output_dir=Path(tmp), score_fn=_score_fn(SCORES))
        assert path.parent == Path(tmp) and path.name.startswith("rank_eval_")
        result = json.loads(path.read_text(encoding="utf-8"))
        s = result["settings"]
        assert s["text_chars"] == 2500 and s["k"] == 1 and s["repeats"] == 2 and s["split"] == "all"
        assert s["golden_rows"] == len(ROWS) and s["model"] and s["date"]
        assert result["overall"]["mean"]["precision"] == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
