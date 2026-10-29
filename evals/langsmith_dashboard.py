"""Create (or rebuild) the LangSmith dashboard of the daily digest runs.

Usage (from the repo root):  python -m evals.langsmith_dashboard

Every chart reads the root `digest_run` runs of LANGSMITH_PROJECT, one point per day, from the feedback keys that
modules.tracing posts (judge.* scores, count.* counts, metric.* measurements) and the built-in run metrics.
Running it again deletes the dashboard of the same title and creates it afresh, so edit CHARTS and re-run.
"""
import os

import requests

from config import settings  # noqa: F401  (loads .env)

API = "https://api.smith.langchain.com/api/v1"
TITLE = "Newsletter agent: daily digest"
ROOT_RUNS = 'and(eq(is_root, true), eq(name, "digest_run"))'
# every model call in the project (digest runs and evals alike); only llm runs, since a root run already sums its
# children and counting both would count each call twice
LLM_RUNS = 'eq(run_type, "llm")'



def _score(key: str) -> dict:
    return {"type": "avg", "field": "feedback_score", "params": {"feedback_key": key}}


def _field(agg: str, field: str) -> dict:
    return {"type": agg, "field": field}


def _percentile(field: str, p: float) -> dict:
    return {"type": "percentile", "field": field, "params": {"p": p}}


RUN_COUNT = {"type": "count"}
ERROR_RATE = {"type": "ratio", "numerator": {"type": "count", "filter": 'eq(status, "error")'},
              "denominator": {"type": "count"}}

# (title, chart type, description, [(series name, LangSmith v2 metric definition)][, run filter]);
# without a run filter a chart reads the digest_run root runs (ROOT_RUNS)
CHARTS = [
    ("Digest quality (judge scores, 1-5)", "line", "Average Claude-judge score per stage; a drop means a regression.",
     [("overall", _score("judge.overall")),
      ("content faithfulness", _score("judge.content_faithfulness")),
      ("filter quality", _score("judge.filter_quality")),
      ("fetch health", _score("judge.fetch_health"))]),
    ("Your verdict on the picks (Telegram)", "line",
     "Share of the digest picks you rated must-read or worth it. Fills in once the bot posts human.picks_precision.",
     [("picks precision", _score("human.picks_precision"))]),
    # the funnel is split in two so the small selection counts are not squashed under the 200+ intake on one axis
    ("Pipeline intake (articles collected)", "line", "How many articles came in, and how many dedupe removed.",
     [("collected", _score("count.articles_collected_merged")),
      ("after dedupe", _score("count.articles_after_dedupe"))]),
    ("Pipeline selection (articles per stage)", "line", "How many made it through the shortlist, ranking and picks.",
     [("shortlisted", _score("count.articles_shortlisted")),
      ("ranked", _score("count.articles_ranked")),
      ("picked", _score("count.final_picked_articles"))]),
    ("Reading quality (shortlisted articles)", "bar", "Full text vs partial vs teaser-only; more full is better.",
     [("full", _score("count.articles_read_full")),
      ("partial", _score("count.articles_read_partial")),
      ("snippet", _score("count.articles_read_snippet"))]),
    ("Dedupe activity", "line", "Many near misses means the 0.82 threshold is borderline.",
     [("dropped", _score("count.articles_dedupe_dropped")),
      ("history matches", _score("count.dedupe_matches")),
      ("same-URL matches", _score("count.dedupe_url_matches")),
      ("near misses", _score("count.dedupe_near_misses")),
      ("in-batch duplicates", _score("count.dedupe_batch_duplicates"))]),
    ("Dedupe max similarity", "line", "Highest similarity among the dropped articles (0-1).",
     [("max similarity", _score("metric.dedupe_max_similarity"))]),
    ("Digest size", "bar", "Full cards and one-line brief items sent.",
     [("cards", _score("count.cards")), ("brief items", _score("count.brief_items"))]),
    ("Runs and errors", "bar", "Did it run, and did it fail.",
     [("runs", RUN_COUNT), ("error rate", ERROR_RATE)]),
    ("Run latency (s)", "line", "End-to-end time of one digest run.",
     [("p50", _percentile("latency_seconds", 0.5)), ("p99", _percentile("latency_seconds", 0.99))]),
    ("Tokens and cost", "line", "Gemini and Claude tokens per day, and their cost as LangSmith prices them. Runs before Oct 27 show 0.",
     [("tokens", _field("sum", "total_tokens")), ("cost (USD)", _field("sum", "total_cost"))]),
    ("Model cost, all runs", "line",
     "Every Gemini and Claude call in the project per day: the daily digest plus evals such as rank_eval.",
     [("tokens", _field("sum", "total_tokens")), ("cost (USD)", _field("sum", "total_cost")),
      ("model calls", RUN_COUNT)], LLM_RUNS),
]


def _request(method: str, path: str, **kw) -> dict | list | None:
    resp = requests.request(method, API + path, headers={"x-api-key": os.environ["LANGSMITH_API_KEY"]},
                            timeout=60, **kw)
    if resp.status_code >= 400:
        raise RuntimeError(f"{method} {path}: {resp.status_code} {resp.text[:300]}")
    return resp.json() if resp.content else None


def _project_id(name: str) -> str:
    projects = _request("GET", "/sessions", params={"name": name})
    if not projects:
        raise SystemExit(f"LangSmith project {name!r} not found")
    return projects[0]["id"]


def main() -> None:
    project = os.getenv("LANGSMITH_PROJECT", "default")
    project_ids = [_project_id(project)]
    for section in _request("GET", "/charts/section") or []:
        if section.get("title") == TITLE:
            _request("DELETE", f"/charts/section/{section['id']}")
    section = _request("POST", "/charts/section",
                       json={"title": TITLE, "description": f"Root digest_run runs of project {project}, per day."})
    for i, (title, chart_type, description, series, *run_filter) in enumerate(CHARTS):
        source = {"source_type": "tracing_project", "project_ids": project_ids,
                  "run_filter": run_filter[0] if run_filter else ROOT_RUNS}
        _request("POST", "/charts/create", json={
            "title": title, "description": description, "chart_type": chart_type, "index": i,
            "section_id": section["id"],
            "series": [{"name": name, "metric_definition": metric, "filter_definition": source}
                       for name, metric in series]})
    print(f"Dashboard {TITLE!r}: {len(CHARTS)} charts. Open LangSmith -> Monitoring -> Dashboards.")


if __name__ == "__main__":
    main()
