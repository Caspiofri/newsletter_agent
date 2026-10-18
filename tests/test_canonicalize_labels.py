"""Offline tests for the label URL canonicaliser. Run from the repo root: python -m tests.test_canonicalize_labels"""
import copy

from evals import canonicalize_labels

SUB = "https://substack.com/redirect/2/abc?j=1"


def test_canonicalize_labels_rewrites_urls_and_counts():
    exported = {"run_id": "r1", "labels": [
        {"url": SUB, "label": "must", "title": "A"},
        {"url": "https://b.example/x", "label": "skip", "title": "B"},
        {"label": "worth", "title": "no url"}]}
    before = copy.deepcopy(exported)
    resolver = lambda u: ("ok", "https://real.example/a?utm_source=e" if u == SUB else u)
    out, changed = canonicalize_labels.canonicalize(exported, resolver)
    assert changed == 1
    assert [l.get("url") for l in out["labels"]] == ["https://real.example/a", "https://b.example/x", None]
    assert out["run_id"] == "r1" and out["labels"][0]["label"] == "must"
    assert exported == before  # input not mutated


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
