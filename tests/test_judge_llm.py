"""Offline tests for the Claude judge transport. Run from the repo root: python -m tests.test_judge_llm"""
from types import SimpleNamespace
from unittest.mock import patch

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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
