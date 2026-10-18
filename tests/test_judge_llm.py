"""Offline tests for the Claude judge transport. Run from the repo root: python -m tests.test_judge_llm"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modules import judge_llm as jl
from modules import trace_judge
from modules.models import Article, ArticleCard, NewsletterData
from datetime import date


class _FakeClient:
    def __init__(self, blocks):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self._blocks = blocks

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self._blocks)


def _text(t):
    return SimpleNamespace(type="text", text=t)


def test_parse_json_plain():
    assert jl.parse_judge_json('{"score": 4, "reasoning": "ok"}') == {"score": 4, "reasoning": "ok"}


def test_parse_json_inside_code_fence_with_prose():
    text = 'Here you go:\n```json\n{"score": 2, "reasoning": "weak"}\n```\nThanks.'
    assert jl.parse_judge_json(text) == {"score": 2, "reasoning": "weak"}


def test_parse_json_missing_raises():
    try:
        jl.parse_judge_json("no json here")
    except ValueError as e:
        assert "No JSON in judge response" in str(e) and "no json here" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_judge_llm_sends_prompt_and_returns_text():
    client = _FakeClient([SimpleNamespace(type="thinking", text="skip"), _text("ab"), _text("cd")])
    with patch.object(jl, "_get_client", return_value=client):
        out = jl.judge_llm("hello", model="m-1", max_tokens=77)
    assert out == "abcd"
    assert client.calls == [{"model": "m-1", "max_tokens": 77, "messages": [{"role": "user", "content": "hello"}]}]


def test_hebrew_prompt_not_escaped():
    prompt = "שלום עולם"
    client = _FakeClient([_text("{}")])
    with patch.object(jl, "_get_client", return_value=client):
        jl.judge_llm(prompt)
    assert client.calls[0]["messages"][0]["content"] == prompt


def test_real_client_builds_offline_with_messages():
    import os
    import anthropic
    old_key = os.environ.get("ANTHROPIC_API_KEY")
    old_cache = jl._client
    os.environ["ANTHROPIC_API_KEY"] = "dummy-key-for-offline-test"
    jl._client = None
    try:
        client = jl._get_client()
        assert isinstance(client, anthropic.Anthropic)
        assert hasattr(client, "messages")
    finally:
        jl._client = old_cache
        if old_key is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = old_key


def test_judge_llm_tags_current_run_with_model():
    rt = MagicMock()
    client = _FakeClient([_text("ok")])
    with patch.object(jl, "_get_client", return_value=client), \
            patch("langsmith.run_helpers.get_current_run_tree", return_value=rt):
        jl.judge_llm.__wrapped__("p", model="m-9")  # bypass the traceable wrapper; patched run tree is a mock
    rt.add_tags.assert_called_once_with(["m-9"])
    rt.add_metadata.assert_called_once_with({"ls_provider": "anthropic", "ls_model_name": "m-9"})


def test_llm_judge_passes_model_to_judge_llm():
    seen = {}

    def fake(prompt, model):
        seen["model"] = model
        return '{"score": 3, "reasoning": "r"}'

    with patch.object(trace_judge, "judge_llm", fake):
        out = trace_judge._llm_judge("c", "r", "x", model="custom-model")
    assert seen["model"] == "custom-model" and out["score"] == 3


def test_faithfulness_forwards_model():
    seen = {}

    def fake(criterion, rubric, content, model):
        seen["model"] = model
        return {"score": 4, "reasoning": "ok"}

    art = Article(subject="s", name="n", url="https://n.example/a", content="c", published_at=date(2026, 1, 1))
    data = NewsletterData(cards=[ArticleCard(id=0, title="t", source="s", brief="b", personalization="p")],
                          brief_items=[], actions=[])
    with patch.object(trace_judge, "_llm_judge", fake):
        trace_judge._judge_content_faithfulness([art], data, model="x")
    assert seen["model"] == "x"


def test_parse_ignores_text_and_braces_after_the_json():
    text = '{"score": 4, "reasoning": "ok"}\n\nNote: {this} is extra }'
    assert jl.parse_judge_json(text) == {"score": 4, "reasoning": "ok"}


def test_parse_first_json_object_wins_when_two_are_present():
    text = '{"score": 1, "reasoning": "a"} then {"score": 5, "reasoning": "b"}'
    assert jl.parse_judge_json(text)["score"] == 1


def test_parse_skips_a_leading_brace_that_is_not_json():
    text = 'Note {see below} {"score": 4, "reasoning": "x"}'
    assert jl.parse_judge_json(text) == {"score": 4, "reasoning": "x"}


def test_parse_nested_braces_inside_reasoning_string():
    text = '{"score": 3, "reasoning": "uses {braces} and } inside"} trailing'
    assert jl.parse_judge_json(text)["reasoning"] == "uses {braces} and } inside"


def test_parse_error_message_keeps_the_reply():
    reply = "x{" * 400
    try:
        jl.parse_judge_json(reply)
    except ValueError as e:
        assert "No JSON in judge response" in str(e) and repr(reply[:500]) in str(e)
        assert repr(reply) not in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_verdict_coerces_digit_string_and_float():
    assert trace_judge._parse_verdict({"score": " 4 ", "reasoning": "r"}) == {"score": 4, "reasoning": "r"}
    out = trace_judge._parse_verdict({"score": 4.0, "reasoning": "r"})
    assert out["score"] == 4 and isinstance(out["score"], int)
    assert trace_judge._parse_verdict({"score": 5, "reasoning": "r"})["score"] == 5


def test_verdict_rejects_missing_out_of_range_and_non_string_reasoning():
    bad = [
        {"reasoning": "r"},
        {"score": 0, "reasoning": "r"},
        {"score": 6, "reasoning": "r"},
        {"score": 4.5, "reasoning": "r"},
        {"score": True, "reasoning": "r"},
        {"score": "high", "reasoning": "r"},
        {"score": 3},
        {"score": 3, "reasoning": 7},
    ]
    for data in bad:
        try:
            trace_judge._parse_verdict(data)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {data!r}")


def test_llm_judge_retries_once_then_succeeds():
    replies = iter(["sorry, no json", '{"score": 2, "reasoning": "r"}'])
    calls = []

    def fake(prompt, model):
        calls.append(prompt)
        return next(replies)

    with patch.object(trace_judge, "judge_llm", fake):
        out = trace_judge._llm_judge("c", "r", "x")
    assert out == {"score": 2, "reasoning": "r"}
    assert len(calls) == 2 and calls[0] == calls[1]


def test_llm_judge_raises_after_second_failure():
    calls = []

    def fake(prompt, model):
        calls.append(prompt)
        return '{"score": 9, "reasoning": "r"}'

    with patch.object(trace_judge, "judge_llm", fake):
        try:
            trace_judge._llm_judge("c", "r", "x")
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    assert len(calls) == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
