"""Offline tests for the labeling hub. Run from the repo root: python -m tests.test_label_hub
The page logic is tested through node (pure functions in the page's #logic script); without node those tests skip."""
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from evals import import_labels, label_hub


def _row(url, title=None, origin="Src", group="shortlisted", content="text", **kw):
    return {"title": title or f"title of {url}", "url": url, "origin": origin, "author": "", "published_at": "2026-10-01",
            "read_status": "full", "link_status": "ok", "signal": "", "group": group, "score": 5.0,
            "reason": "", "picked": False, "content": content, **kw}


def _write(out, ts, rows, as_list=False, profile="ai_engineering_daily"):
    data = rows if as_list else {"profile": profile, "run_id": ts, "items": rows}
    (out / f"ranking_{profile}_{ts}.json").write_text(json.dumps(data), encoding="utf-8")


def _golden(path, entries):
    path.write_text(json.dumps(entries), encoding="utf-8")


def test_collect_items_merges_reports_and_dedupes_by_url_keeping_newest():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        _write(out, "20261001_070000", [_row("https://a", title="old A"), _row("https://b", title="B")])
        _write(out, "20261003_070000", [_row("https://a", title="new A")])
        items = label_hub.collect_items(out, out / "none.json")
        by_url = {i["url"]: i for i in items}
        assert len(items) == 2
        assert by_url["https://a"]["title"] == "new A" and by_url["https://a"]["source_run"] == "20261003_070000"
        assert by_url["https://b"]["source_run"] == "20261001_070000"


def test_collect_items_accepts_both_json_shapes():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        _write(out, "20261001_070000", [_row("https://a")], as_list=True)
        _write(out, "20261002_070000", [_row("https://b")])
        items = label_hub.collect_items(out, out / "none.json")
        assert {i["url"] for i in items} == {"https://a", "https://b"}


def test_collect_items_attaches_existing_labels_and_orders_unlabeled_first():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        _write(out, "20261001_070000", [_row("https://old-todo")])
        _write(out, "20261002_070000", [_row("https://done"), _row("https://new-todo")])
        golden = out / "golden.json"
        _golden(golden, [{"url": "https://done", "label": "must", "tag": "beginner", "note": "n",
                          "labeled_with": "article"}])
        items = label_hub.collect_items(out, golden)
        assert [i["url"] for i in items] == ["https://new-todo", "https://old-todo", "https://done"]
        done = items[-1]
        assert (done["label"], done["tag"], done["note"], done["labeled_with"]) == ("must", "beginner", "n", "article")
        assert items[0]["label"] == "" and items[0]["tag"] == ""


def test_collect_items_title_origin_fallback_merges_redirect_variants():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        _write(out, "20261001_070000", [
            _row("https://substack.com/redirect/1", title="  Same Title ", origin="Seq"),
            _row("https://x/other", title="Same Title", origin="Different")])
        _write(out, "20261002_070000", [_row("https://seq.com/p/real", title="same title", origin="Seq")])
        golden = out / "golden.json"
        _golden(golden, [{"url": "https://substack.com/redirect/1", "label": "skip"}])
        items = label_hub.collect_items(out, golden)
        urls = sorted(i["url"] for i in items)
        assert urls == ["https://seq.com/p/real", "https://x/other"], urls
        kept = next(i for i in items if i["url"] == "https://seq.com/p/real")
        assert kept["label"] == "skip" and kept["source_run"] == "20261002_070000"


def test_title_origin_fallback_needs_the_same_date():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        _write(out, "20261001_070000", [_row("https://a", title="Weekly", origin="inbox", published_at="2026-10-01"),
                                        _row("https://c", title="Weekly", origin="inbox", published_at="2026-10-01")])
        _write(out, "20261002_070000", [_row("https://b", title="Weekly", origin="inbox", published_at="2026-10-02"),
                                        _row("https://c2", title="weekly", origin="inbox", published_at="2026-10-01")])
        urls = sorted(i["url"] for i in label_hub.collect_items(out, out / "none.json"))
        assert urls == ["https://b", "https://c2"], urls   # same title+origin+date merge; another date stays apart


def test_collect_items_keeps_the_key_the_label_was_found_under():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        _write(out, "20261001_070000", [_row("https://old", title="Same", origin="Seq")])
        _write(out, "20261002_070000", [_row("https://new", title="Same", origin="Seq")])
        golden = out / "g.json"
        _golden(golden, [{"url": "https://old", "label": "skip"}])
        (item,) = label_hub.collect_items(out, golden)
        assert item["url"] == "https://new" and item["label_url"] == "https://old"


def test_item_text_is_html_stripped_and_capped():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        _write(out, "20261001_070000", [
            _row("https://a", content="<p>Hello   <b>big</b>\n world</p><script>x()</script>"),
            _row("https://b", content="y" * 9000)])
        by_url = {i["url"]: i for i in label_hub.collect_items(out, out / "none.json")}
        assert by_url["https://a"]["content"].startswith("Hello big world")
        assert "<" not in by_url["https://a"]["content"]
        assert len(by_url["https://b"]["content"]) == 6000


def _html(items=None):
    return label_hub.render_html(items if items is not None else [_row("https://a")], target=80)


def test_hub_html_embeds_items_and_escapes_script_end():
    item = _row("https://a", content="evil </script><b>x</b>")
    html = _html([item])
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    assert m and "</" not in m.group(1)
    data = json.loads(m.group(1).replace("<\\/", "</"))
    assert data["items"][0]["content"] == item["content"] and data["target"] == 80
    assert data["fields"] == label_hub.LABEL_FIELDS
    assert "http://" not in re.sub(r'<script id="data".*?</script>', "", html, flags=re.S)


def test_hub_html_has_the_controls():
    html = _html()
    for needle in ['id="btn-must"', 'id="btn-worth"', 'id="btn-skip"', 'id="tag"', 'id="note"', 'id="read-full"',
                   'id="export"', 'id="load"', 'id="load-file"', 'id="target"', 'id="reveal"',
                   'id="f-status"', 'id="f-label"', 'id="f-origin"', 'id="f-group"', 'id="f-read"',
                   'full text mattered', 'great primary source', 'label_hub:v1', 'FileReader', 'Blob',
                   'not exported yet', 'show ranker score and picks', "onclick", "function setLabel"]:
        assert needle in html, needle


# ---- page logic through node ----

def _node_missing():
    if shutil.which("node") is None:
        print("SKIP (node not on PATH)")
        return True
    return False


def _logic(html):
    return re.search(r'<script id="logic">(.*?)</script>', html, re.S).group(1)


def _run(expr, args):
    """Run `expr` (using A = args and the page's pure functions) in node; return the JSON result."""
    prog = ("const A = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n" + _logic(_html())
            + f"\nconsole.log(JSON.stringify({expr}));")
    r = subprocess.run(["node", "-e", prog], input=json.dumps(args), capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def _items(n, **over):
    return [{"url": f"u{i}", "title": f"t{i}", "origin": "A" if i % 2 else "B", "group": "sample" if i % 3 == 0 else "shortlisted",
             "read_status": "full" if i < 3 else "snippet", "source_run": "r1", "author": "", "published_at": "",
             "signal": "", "picked": False, "score": 1, "content": "c", **over} for i in range(n)]


def test_js_apply_filters():
    if _node_missing():
        return
    items = _items(6)
    labels = {"u0": {"label": "must"}, "u1": {"label": "skip"}}
    f = lambda **kw: [i["url"] for i in _run("applyFilters(A.items, A.labels, A.f)", {"items": items, "labels": labels, "f": kw})]
    assert f() == [f"u{i}" for i in range(6)]
    assert f(status="unlabeled") == ["u2", "u3", "u4", "u5"]
    assert f(status="labeled") == ["u0", "u1"]
    assert f(label="skip") == ["u1"]
    assert f(origin="A") == ["u1", "u3", "u5"]
    assert f(group="sample") == ["u0", "u3"]
    assert f(read="snippet") == ["u3", "u4", "u5"]
    assert f(status="unlabeled", origin="A", read="snippet") == ["u3", "u5"]


def test_js_progress_stats_counts_and_warnings():
    if _node_missing():
        return
    items = _items(30, group="shortlisted")
    stats = lambda labels, target=80: _run("progressStats(A.items, A.labels, A.target)",
                                           {"items": items, "labels": labels, "target": target})
    few = stats({f"u{i}": {"label": "must"} for i in range(5)})
    assert few["labeled"] == 5 and few["warnings"] == []          # under 10 labels: no warnings
    labels = {f"u{i}": {"label": "must"} for i in range(10)}
    s = stats(labels)
    assert s["byLabel"]["must"] == 10 and s["byGroup"]["shortlisted"] == 10
    assert len(s["warnings"]) == 2                                 # no skips, no sample items
    ok = {}
    for i in range(30):
        ok[f"u{i}"] = {"label": "skip" if i < 5 else "worth"}
    items[:] = _items(30)                                          # mixed groups: every third is a sample
    s = stats(ok)
    assert s["labeled"] == 30 and s["byLabel"]["skip"] == 5 and s["warnings"] == []
    assert sum(s["byOrigin"].values()) == 30 and s["target"] == 80
    low_skip = {f"u{i}": {"label": "skip" if i == 1 else "worth"} for i in range(20)}  # 1/20 = 5% skips
    w = stats(low_skip)["warnings"]
    assert any("skip" in x.lower() for x in w)


def test_js_build_export_has_exactly_label_fields_and_only_labeled():
    if _node_missing():
        return
    items = _items(3)
    labels = {"u1": {"label": "must", "tag": "beginner", "note": "n", "labeled_with": "article"},
              "u2": {"tag": "promo"}}                               # tag without a label: not exported
    out = _run("buildExport(A.items, A.labels, 'hub_x')", {"items": items, "labels": labels})
    assert out["run_id"] == "hub_x" and len(out["labels"]) == 1
    row = out["labels"][0]
    assert list(row) == label_hub.LABEL_FIELDS + ["at"]   # `at` lets a loaded file be compared with local labels
    assert isinstance(row["at"], (int, float)) and row["at"] > 0
    assert (row["url"], row["label"], row["tag"], row["labeled_with"], row["source_run"]) == ("u1", "must", "beginner", "article", "r1")
    assert row["ranker_score"] == 1 and row["ranker_picked"] is False
    default = _run("buildExport(A.items, A.labels, 'h').labels[0].labeled_with", {"items": items, "labels": {"u0": {"label": "skip"}}})
    assert default == "snapshot"


def test_js_merge_loaded_keeps_newer_local_label():
    if _node_missing():
        return
    local = {"u1": {"label": "must", "at": 200}, "u2": {"label": "skip", "at": 100},
             "u5": {"label": "skip", "at": 0},                      # seeded from the golden set
             "u6": {"label": "must", "at": 500}, "u7": {"label": "worth"}}  # u7: no real timestamp
    loaded = {"labels": [{"url": "u1", "label": "skip", "at": 150}, {"url": "u2", "label": "worth", "at": 300},
                         {"url": "u3", "label": "worth", "tag": "hype", "note": "x", "at": 50},
                         {"url": "u4", "label": "bogus"}, {"url": "u5", "label": "must", "at": 90},
                         {"url": "u6", "label": "skip", "at": 500}, {"url": "u7", "label": "skip"}]}
    out = _run("mergeLoaded(A.labels, A.loaded)", {"labels": local, "loaded": loaded})
    assert out["u1"]["label"] == "must"            # local is newer than the file
    assert out["u2"]["label"] == "worth"           # file is newer
    assert out["u3"]["label"] == "worth" and out["u3"]["tag"] == "hype"   # only in the file: added
    assert "u4" not in out
    assert out["u5"]["label"] == "must"            # a golden-seeded label is not a newer local label
    assert out["u6"]["label"] == "must"            # equal timestamps: local stays
    assert out["u7"]["label"] == "skip"            # local without a real timestamp loses to the file
    kept = _run("mergeLoaded(A.labels, {labels: []})", {"labels": local})
    assert kept == local                            # only local: kept


def test_js_build_export_has_replaces_url_only_when_different():
    if _node_missing():
        return
    items = _items(3)
    items[0]["label_url"] = "old-redirect"
    items[1]["label_url"] = "u1"                       # same as url: no replaces_url
    labels = {f"u{i}": {"label": "must"} for i in range(3)}
    rows = _run("buildExport(A.items, A.labels, 'h').labels", {"items": items, "labels": labels})
    assert rows[0]["replaces_url"] == "old-redirect"
    assert "replaces_url" not in rows[1] and "replaces_url" not in rows[2]


def test_js_unmatched_labels():
    if _node_missing():
        return
    items = _items(3)
    labels = {"u0": {"label": "must"}, "gone": {"label": "skip"}, "tagonly": {"tag": "promo"}, "u9": {"label": "worth"}}
    out = _run("unmatchedLabels(A.items, A.labels)", {"items": items, "labels": labels})
    assert sorted(out) == ["gone", "u9"]


def test_hub_html_warns_about_unmatched_labels_and_resets_file_input():
    html = _html()
    assert "match no article in this page" in html and "python -m evals.canonicalize_labels" in html
    assert "value = ''" in html  # file input reset, so loading the same file twice fires onchange


def test_js_next_unlabeled():
    if _node_missing():
        return
    items = _items(5)
    labels = {"u1": {"label": "must"}, "u2": {"label": "must"}}
    nxt = lambda frm, lab=labels: _run("nextUnlabeled(A.items, A.labels, A.from)", {"items": items, "labels": lab, "from": frm})
    assert nxt(0) == 3 and nxt(3) == 4 and nxt(4) == 0     # wraps around
    assert nxt(0, {f"u{i}": {"label": "skip"} for i in range(5)}) == -1


def test_label_fields_row_is_accepted_by_import_merge():
    row = {f: "x" for f in label_hub.LABEL_FIELDS}
    row.update(url="https://a", label="worth", source_run="20261001_070000")
    merged, added, updated = import_labels.merge([], {"run_id": "hub_1", "labels": [row]}, "2026-10-18")
    assert (added, updated) == (1, 0)
    assert merged[0]["source_run"] == "20261001_070000" and merged[0]["labeled_on"] == "2026-10-18"


def test_import_drops_the_hub_timestamp():
    exported = {"run_id": "hub_1", "labels": [{"url": "https://a", "label": "must", "at": 123}]}
    merged, _, _ = import_labels.merge([], exported, "2026-10-18")
    assert "at" not in merged[0]


def test_import_keeps_an_unchanged_row_entirely():
    golden = [{"url": "https://a", "label": "must", "tag": "t", "note": "n", "labeled_with": "article",
               "labeled_on": "2026-09-01", "source_run": "r0", "content": "old snapshot"}]
    row = {"url": "https://a", "label": "must", "tag": "t", "note": "n", "labeled_with": "article",
           "source_run": "r9", "content": "new text", "at": 5}
    merged, added, updated = import_labels.merge(golden, {"run_id": "hub_1", "labels": [row]}, "2026-10-18")
    assert (added, updated) == (0, 0) and merged == golden
    changed = {**row, "label": "skip"}
    merged, added, updated = import_labels.merge(golden, {"run_id": "hub_1", "labels": [changed]}, "2026-10-18")
    assert (added, updated) == (0, 1)
    assert merged[0]["label"] == "skip" and merged[0]["labeled_on"] == "2026-10-18" and "at" not in merged[0]


def test_import_replaces_url_retires_the_old_key():
    golden = [{"url": "https://old", "label": "skip", "labeled_on": "2026-09-01"},
              {"url": "https://other", "label": "must", "labeled_on": "2026-09-01"}]
    row = {"url": "https://new", "label": "skip", "replaces_url": "https://old", "at": 5}
    merged, added, updated = import_labels.merge(golden, {"run_id": "hub_1", "labels": [row]}, "2026-10-18")
    assert sorted(g["url"] for g in merged) == ["https://new", "https://other"]
    assert (added, updated) == (0, 1)
    assert all("replaces_url" not in g and "at" not in g for g in merged)


def test_hub_round_trip_does_not_change_the_golden_row_count():
    if _node_missing():
        return
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        _write(out, "20261001_070000", [_row("https://old", title="Same", origin="Seq", published_at="2026-10-01"),
                                        _row("https://b", title="B")])
        _write(out, "20261002_070000", [_row("https://new", title="Same", origin="Seq", published_at="2026-10-01")])
        golden = [{"url": "https://old", "label": "skip", "labeled_on": "2026-09-01"},
                  {"url": "https://b", "label": "must", "labeled_on": "2026-09-01"}]
        gp = out / "g.json"
        _golden(gp, golden)
        items = label_hub.collect_items(out, gp)
        labels = {i["url"]: {"label": i["label"]} for i in items if i["label"]}
        exported = _run("buildExport(A.items, A.labels, 'h')", {"items": items, "labels": labels})
        merged, _, _ = import_labels.merge(golden, exported, "2026-10-18")
        assert len(merged) == len(golden) == 2


def test_import_keeps_row_source_run():
    exported = {"run_id": "hub_9", "labels": [{"url": "https://a", "label": "must", "source_run": "r1"},
                                              {"url": "https://b", "label": "skip"}]}
    merged, _, _ = import_labels.merge([], exported, "2026-10-18")
    assert {g["url"]: g["source_run"] for g in merged} == {"https://a": "r1", "https://b": "hub_9"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
