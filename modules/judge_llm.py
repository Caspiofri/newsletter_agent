"""
judge_llm.py — Claude transport for the trace judge.

The judge runs on a different model family than the ranker and writer so it
does not grade its own homework. The client is built lazily, so importing
this module needs no API key.
"""

import json

from langsmith import run_helpers, traceable

from config.settings import JUDGE_MODEL_DAILY

_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


@traceable(run_type="llm", name="judge_llm")
def judge_llm(prompt: str, model: str = JUDGE_MODEL_DAILY, max_tokens: int = 1024) -> str:
    """Send one user prompt to Claude and return the text of its reply."""
    rt = run_helpers.get_current_run_tree()
    if rt is not None:
        rt.add_tags([model])
        rt.add_metadata({"ls_provider": "anthropic", "ls_model_name": model})
    response = _get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = getattr(response, "usage", None)
    if rt is not None and usage is not None:
        rt.set(usage_metadata={"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                               "total_tokens": usage.input_tokens + usage.output_tokens})
    return "".join(b.text for b in response.content if b.type == "text")


def parse_judge_json(text: str) -> dict:
    """Return the first JSON object in a judge reply; text around or after it is ignored."""
    decoder = json.JSONDecoder()
    pos = text.find("{")
    while pos != -1:
        try:
            obj, _ = decoder.raw_decode(text[pos:])
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            return obj
        pos = text.find("{", pos + 1)
    raise ValueError(f"No JSON in judge response: {text[:500]!r}")
