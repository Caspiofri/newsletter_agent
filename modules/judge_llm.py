"""
judge_llm.py — Claude transport for the trace judge.

The judge runs on a different model family than the ranker and writer so it
does not grade its own homework. The client is built lazily, so importing
this module needs no API key.
"""

import json
import re

from config.settings import JUDGE_MODEL_DAILY

_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        from langsmith.wrappers import wrap_anthropic
        _client = wrap_anthropic(anthropic.Anthropic())
    return _client


def judge_llm(prompt: str, model: str = JUDGE_MODEL_DAILY, max_tokens: int = 1024) -> str:
    """Send one user prompt to Claude and return the text of its reply."""
    response = _get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def parse_judge_json(text: str) -> dict:
    """Extract the first {...} span from a judge reply and parse it."""
    text = text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in judge response: {text!r}")
    return json.loads(match.group())
