"""Offline tests for the label audit and the relabel self-consistency check. Run: python -m tests.test_label_audit"""
import contextlib
import io
import json
import socket
import tempfile
from pathlib import Path
from unittest.mock import patch

from evals import label_audit

LABELS = ("must", "worth", "skip")
ORIGINS = ("inbox", "medium", "hackernews")


def _row(i, label="must", origin="inbox", group="shortlisted", labeled_with="article", run="r1", title=None):
    return {"url": f"https://x.example/{i}", "title": title or f"Article {i}", "label": label, "origin": origin,
            "group": group, "labeled_with": labeled_with, "read_status": "full", "source_run": run}


def _healthy(n=80):
    return [_row(i, label=LABELS[i % 3], origin=ORIGINS[i % 3], group="sample" if i % 3 == 0 else "shortlisted",
                 labeled_with="article" if i % 4 else "snapshot", run=f"r{i % 4}") for i in range(n)]


def test_audit_counts_and_warnings():
    rows = [_row(i, label="skip" if i == 0 else "must", group="sample" if i == 1 else "shortlisted",
                 labeled_with="snapshot" if i < 7 else "article") for i in range(10)]
    out = label_audit.audit(rows, target=80)
    assert out["total"] == 10
    assert out["by_label"] == {"must": 9, "skip": 1}
    assert out["by_group"] == {"sample": 1, "shortlisted": 9}
    assert out["by_labeled_with"] == {"article": 3, "snapshot": 7}
    assert out["by_source_run"] == {"r1": 10} and out["by_origin"] == {"inbox": 10}
    assert sum(out["split"].values()) == 10
    w = " ".join(out["warnings"])
    for phrase in ("skip", "sample", "report", "article", "80"):
        assert phrase in w, phrase
    assert "duplicate" not in w


def test_audit_flags_duplicates_by_split_key():
    rows = _healthy()
    rows.append(_row(999, title="  ARTICLE 5 ", origin=ORIGINS[5 % 3], run="r9"))  # same story, new address
    out = label_audit.audit(rows)
    assert len(out["duplicates"]) == 1
    assert sorted(out["duplicates"][0]["urls"]) == ["https://x.example/5", "https://x.example/999"]
    assert any("duplicate" in w for w in out["warnings"])


def test_audit_no_warnings_on_a_healthy_set():
    out = label_audit.audit(_healthy())
    assert out["warnings"] == [], out["warnings"]


def test_audit_ignores_unlabeled_rows():
    out = label_audit.audit(_healthy() + [{**_row(500), "label": ""}])
    assert out["total"] == 80


def test_agreement_matches_and_confusion():
    original = [_row(i, label=LABELS[i % 3]) for i in range(6)]
    relabel = [dict(r) for r in original]
    relabel[0]["label"] = "worth"  # must -> worth
    relabel[5] = {**relabel[5], "url": "https://other.example/5"}  # same title and origin, new address
    relabel.append(_row(77, label="skip"))  # not in the original
    out = label_audit.agreement(original, relabel)
    assert out["n_compared"] == 6
    assert abs(out["agreement"] - 5 / 6) < 1e-9
    assert out["confusion"]["must"] == {"must": 1, "worth": 1, "skip": 0}
    assert out["confusion"]["skip"]["skip"] == 2


def test_agreement_none_below_five_compared():
    rows = [_row(i) for i in range(4)]
    out = label_audit.agreement(rows, rows)
    assert out["n_compared"] == 4 and out["agreement"] is None


def test_cli_prints_without_touching_the_network():
    with tempfile.TemporaryDirectory() as tmp:
        golden, relabel = Path(tmp) / "golden.json", Path(tmp) / "relabel.json"
        rows = _healthy()
        golden.write_text(json.dumps(rows))
        relabel.write_text(json.dumps({"run_id": "hub", "labels": rows[:10]}))  # a hub export
        buf = io.StringIO()
        with patch.object(socket.socket, "connect", side_effect=AssertionError("network used")), \
                contextlib.redirect_stdout(buf):
            label_audit.main(["--golden", str(golden), "--relabel", str(relabel)])
        text = buf.getvalue()
        assert "80 labels" in text and "agreement" in text and "80 percent" in text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
