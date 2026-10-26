"""Token usage of the Gemini and Claude calls is reported to their LangSmith llm runs, so the dashboard can sum it."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modules import judge_llm as jl
from modules import nodes


def _gemini_response(prompt=1200, candidates=300, thoughts=500, total=2000):
    usage = SimpleNamespace(prompt_token_count=prompt, candidates_token_count=candidates,
                            thoughts_token_count=thoughts, total_token_count=total)
    return SimpleNamespace(text="ok", usage_metadata=usage)


def test_gemini_usage_counts_thinking_as_output():
    assert nodes._gemini_usage(_gemini_response()) == {
        "input_tokens": 1200, "output_tokens": 800, "total_tokens": 2000,
        "output_token_details": {"reasoning": 500}}


def test_gemini_usage_tolerates_missing_counts():
    resp = _gemini_response(candidates=None, thoughts=None, total=None)
    assert nodes._gemini_usage(resp) == {"input_tokens": 1200, "output_tokens": 0, "total_tokens": 1200,
                                         "output_token_details": {"reasoning": 0}}
    assert nodes._gemini_usage(SimpleNamespace(text="ok", usage_metadata=None)) is None


def test_call_gemini_reports_model_and_usage_to_its_run():
    rt = MagicMock()
    fake = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: _gemini_response()))
    with patch.object(nodes, "client", fake), \
            patch("langsmith.run_helpers.get_current_run_tree", return_value=rt):
        assert nodes._call_gemini.__wrapped__("p") == "ok"  # bypass the traceable wrapper; the run tree is a mock
    rt.add_metadata.assert_called_once_with({"ls_provider": "google_genai", "ls_model_name": nodes.GEMINI_MODEL})
    rt.set.assert_called_once_with(usage_metadata=nodes._gemini_usage(_gemini_response()))


def test_call_gemini_without_tracing_still_returns_text():
    fake = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: _gemini_response()))
    with patch.object(nodes, "client", fake), patch("langsmith.run_helpers.get_current_run_tree", return_value=None):
        assert nodes._call_gemini.__wrapped__("p") == "ok"


def test_judge_llm_reports_usage_to_its_run():
    rt = MagicMock()
    resp = SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")],
                           usage=SimpleNamespace(input_tokens=900, output_tokens=120))
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resp))
    with patch.object(jl, "_get_client", return_value=client), \
            patch("langsmith.run_helpers.get_current_run_tree", return_value=rt):
        assert jl.judge_llm.__wrapped__("p", model="m-9") == "ok"
    rt.set.assert_called_once_with(usage_metadata={"input_tokens": 900, "output_tokens": 120, "total_tokens": 1020})


def test_judge_llm_without_usage_sets_nothing():
    rt = MagicMock()
    resp = SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resp))
    with patch.object(jl, "_get_client", return_value=client), \
            patch("langsmith.run_helpers.get_current_run_tree", return_value=rt):
        jl.judge_llm.__wrapped__("p", model="m-9")
    rt.set.assert_not_called()
