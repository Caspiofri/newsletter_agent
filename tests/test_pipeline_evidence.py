"""Offline tests for the pipeline evidence (run_stats, dedupe_drops, rank_scores).
Run from the repo root: python -m tests.test_pipeline_evidence"""
import json
from datetime import date
from unittest.mock import patch

from modules import article_store, nodes
from modules.models import Article


def _art(name, origin="inbox", status="full"):
    return Article(subject="s", name=name, published_at=date.today(), content="body",
                   url=f"https://x.example/{name}", origin=origin, read_status=status)


class FakeCol:
    """Stands in for a Chroma collection: one nearest neighbour per query text."""
    def __init__(self, neighbours):
        self.neighbours = neighbours  # article name -> (distance, matched name, matched date int) or None

    def count(self):
        return 3

    def query(self, query_texts, n_results, where, include):
        d, m = [], []
        for text in query_texts:
            hit = next((v for k, v in self.neighbours.items() if text.startswith(k + ".")), None)
            d.append([hit[0]] if hit else [])
            m.append([{"name": hit[1], "date": hit[2]}] if hit else [])
        return {"distances": d, "metadatas": m}


def _detailed(articles, neighbours):
    with patch.object(article_store, "_collection", return_value=FakeCol(neighbours)), \
            patch.object(article_store, "_dedup_batch", side_effect=lambda a: a):
        return article_store.filter_seen_detailed(articles, "p")


NEIGHBOURS = {
    "Dup": (0.10, "Old dup title", 20261016),
    "Near": (0.25, "Old near title", 20261017),
    "Far": (0.50, "Old far title", 20261017),
}


def test_filter_seen_detailed_reports_drop_and_near_miss():
    arts = [_art("Dup"), _art("Near"), _art("Far"), _art("New")]
    kept, drops = _detailed(arts, NEIGHBOURS)
    assert [a.name for a in kept] == ["Near", "Far", "New"]
    assert [d["title"] for d in drops] == ["Dup", "Near"], drops
    assert drops[0] == {"title": "Dup", "url": "https://x.example/Dup", "similarity": 0.9,
                        "matched_title": "Old dup title", "matched_sent_on": "2026-10-16", "dropped": True}
    assert drops[1]["similarity"] == 0.75 and drops[1]["dropped"] is False


def test_filter_seen_still_returns_survivors():
    arts = [_art("Dup"), _art("Near"), _art("New")]
    with patch.object(article_store, "_collection", return_value=FakeCol(NEIGHBOURS)), \
            patch.object(article_store, "_dedup_batch", side_effect=lambda a: a):
        kept = article_store.filter_seen(arts, "p")
    assert [a.name for a in kept] == ["Near", "New"]


def test_dedupe_node_records_drops_and_stats():
    arts = [_art("Dup"), _art("Near"), _art("New")]
    drops = [{"title": "Dup", "similarity": 0.9, "dropped": True}, {"title": "Near", "similarity": 0.78, "dropped": False}]
    state = {"digest_name": "My Digest", "articles": arts, "run_stats": {"merged": 3}}
    with patch.object(article_store, "filter_seen_detailed", return_value=(arts[1:], drops)):
        out = nodes.dedupe_articles(state)
    assert out["articles"] == arts[1:] and out["dedupe_drops"] == drops
    assert out["run_stats"] == {"merged": 3, "after_dedupe": 2, "dedupe_dropped": 1, "dedupe_matches": 1,
                                "dedupe_near_misses": 1, "dedupe_batch_duplicates": 0,
                                "dedupe_max_similarity": 0.9}
    # partial state without run_stats must not break
    with patch.object(article_store, "filter_seen_detailed", return_value=(arts, [])):
        out = nodes.dedupe_articles({"digest_name": "d", "articles": arts})
    assert out["run_stats"] == {"after_dedupe": 3, "dedupe_dropped": 0, "dedupe_matches": 0,
                                "dedupe_near_misses": 0, "dedupe_batch_duplicates": 0,
                                "dedupe_max_similarity": 0.0}


def _match(title, similarity, dropped):
    return {"title": title, "url": "u", "similarity": similarity, "matched_title": f"old {title}",
            "matched_sent_on": "2026-10-17", "dropped": dropped}


def test_dedupe_node_records_match_near_miss_batch_and_max():
    arts = [_art(f"A{i}") for i in range(6)]
    matches = [_match("A0", 0.91, True), _match("A1", 0.97, True), _match("A2", 0.85, True),
               _match("A3", 0.78, False)]
    with patch.object(article_store, "filter_seen_detailed", return_value=(arts[:2], matches)):
        out = nodes.dedupe_articles({"digest_name": "d", "articles": arts})
    stats = out["run_stats"]
    assert stats["dedupe_dropped"] == 4
    assert stats["dedupe_matches"] == 3
    assert stats["dedupe_near_misses"] == 1
    assert stats["dedupe_batch_duplicates"] == 1
    assert stats["dedupe_max_similarity"] == 0.97


def test_dedupe_max_similarity_zero_without_matches():
    arts = [_art("A"), _art("B")]
    near = [_match("A", 0.79, False)]
    with patch.object(article_store, "filter_seen_detailed", return_value=(arts, near)):
        out = nodes.dedupe_articles({"digest_name": "d", "articles": arts})
    assert out["run_stats"]["dedupe_max_similarity"] == 0.0
    assert out["run_stats"]["dedupe_near_misses"] == 1


def test_dedupe_trace_inputs_lists_titles_not_articles():
    out = article_store._dedupe_trace_inputs({"articles": [_art("A"), _art("B")], "profile": "p", "days": 5})
    assert out == {"queries": ["A", "B"], "window_days": 5}
    default = article_store._dedupe_trace_inputs({"articles": [_art("A")], "profile": "p"})
    assert default == {"queries": ["A"], "window_days": article_store.DEDUPE_DAYS}
    assert article_store._dedupe_trace_inputs({}) == {"queries": [], "window_days": article_store.DEDUPE_DAYS}


def test_dedupe_trace_outputs_shape():
    matches = [_match("A", 0.9, True), _match("B", 0.8, False)]
    out = article_store._dedupe_trace_outputs(([], matches))
    assert out == {"documents": [
        {"page_content": "old A", "metadata": {"query_title": "A", "similarity": 0.9,
                                                "matched_sent_on": "2026-10-17", "dropped": True}},
        {"page_content": "old B", "metadata": {"query_title": "B", "similarity": 0.8,
                                                "matched_sent_on": "2026-10-17", "dropped": False}},
    ]}
    assert article_store._dedupe_trace_outputs(([], [])) == {"documents": []}


def _rank_state(cands, run_stats=None):
    return {"candidates": cands, "top_k": 1, "subject": "s", "target_audience": "t",
            "experience_level": "e", "digest_name": "d", "articles": cands, "run_stats": run_stats or {}}


def _rank(cands, response, run_stats=None):
    with patch.object(nodes, "_call_gemini", return_value=response), \
            patch.object(nodes.ranking_report, "build_rows", return_value=[]), \
            patch.object(nodes.ranking_report, "save", return_value="x"):
        return nodes.rank_articles(_rank_state(cands, run_stats))


def test_rank_scores_cover_all_candidates_and_mark_picked():
    cands = [_art("A"), _art("B", status="snippet"), _art("C")]
    resp = json.dumps([{"id": 0, "score": 6, "reason": "ok"}, {"id": 1, "score": 9, "reason": "teaser"},
                       {"id": 2, "score": 8, "reason": "great"}])
    out = _rank(cands, resp, {"shortlisted": 3})
    rows = out["rank_scores"]
    assert rows[0]["title"] == "C"
    assert len(rows) == 3
    by = {r["title"]: r for r in rows}
    assert by["B"]["score"] == 6 and by["B"]["read_status"] == "snippet"  # capped
    assert by["C"]["picked"] is True and by["A"]["picked"] is False
    assert by["C"]["reason"] == "great" and by["C"]["origin"] == "inbox"
    assert [r["score"] for r in rows] == sorted((r["score"] for r in rows), reverse=True)
    assert out["run_stats"] == {"shortlisted": 3, "ranked": 3, "picked": 1}


def test_rank_without_candidates_still_records_stats():
    out = nodes.rank_articles({"candidates": [], "run_stats": {"merged": 1}})
    assert out["top_articles"] == [] and out["rank_scores"] == []
    assert out["run_stats"] == {"merged": 1, "ranked": 0, "picked": 0}


def test_run_stats_accumulate_across_nodes():
    arts = [_art("A"), _art("B", "medium"), _art("C", "hackernews")]
    state = {"digest_name": "d", "articles": arts, "extra_articles": [], "run_stats": {}}
    with patch.object(article_store, "filter_seen_detailed", return_value=(arts, [])):
        state.update(nodes.dedupe_articles(state))
    with patch.object(nodes, "fillter_articles", side_effect=lambda s: {"top_articles": s["articles"]}):
        state.update(nodes.shortlist_articles(state))
    with patch.object(nodes.reader, "read_articles", side_effect=lambda c, max_chars: c):
        state.update(nodes.read_articles(state))
    resp = json.dumps([{"id": i, "score": 7 - i, "reason": "r"} for i in range(3)])
    state.update(_rank(state["candidates"], resp, state["run_stats"]))
    assert state["run_stats"] == {"after_dedupe": 3, "dedupe_dropped": 0, "dedupe_matches": 0,
                                  "dedupe_near_misses": 0, "dedupe_batch_duplicates": 0,
                                  "dedupe_max_similarity": 0.0, "shortlisted": 3,
                                  "read_full": 3, "read_partial": 0, "read_snippet": 0,
                                  "ranked": 3, "picked": 1}, state["run_stats"]


def test_merge_starts_fresh_run_stats():
    arts = [_art("A"), _art("A"), _art("B")]
    with patch.object(nodes.link_utils, "classify_links", side_effect=lambda urls: {u: "ok" for u in urls}), \
            patch.object(nodes.article_store, "upgrade_bad_links", side_effect=lambda a: a):
        out = nodes.merge_sources({"articles": arts[:1], "extra_articles": arts[1:], "run_stats": {"stale": 9}})
    assert out["run_stats"] == {"merged": 2}


def test_summarize_records_card_counts():
    arts = [_art("A"), _art("B")]
    payload = {"cards": [{"id": 0, "title": "t", "source": "s", "brief": "b", "personalization": "p"}],
               "brief_items": [{"id": 1, "summary": "x"}], "actions": ["a"]}
    state = {"subject": "AI", "target_audience": "dev", "top_k": 2, "top_articles": arts,
             "experience_level": "junior", "digest_name": "D", "run_stats": {"picked": 2}}
    with patch.object(nodes, "_call_gemini", return_value=json.dumps(payload)):
        out = nodes.summraize(state)
    assert out["run_stats"] == {"picked": 2, "cards": 1, "brief_items": 1}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
