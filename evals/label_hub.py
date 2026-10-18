"""Labeling hub: one self-contained HTML page to grade articles from every ranking report and build the golden set.

Usage (from the repo root):  python -m evals.label_hub [--target 80] [--out output/label_hub.html]

Gathers every output/ranking_*.json (newest report wins on a duplicate article) plus the labels already in
evals/ranking_golden.json. "Export labels" on the page downloads a file for:  python -m evals.import_labels <file>
"""
import argparse
import json
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

from config.settings import OUTPUT_DIR
from evals.import_labels import GOLDEN_RANKING_PATH

# the fields the page exports per label; import_labels.merge keeps them all
LABEL_FIELDS = ["url", "title", "label", "tag", "note", "labeled_with", "source_run", "origin", "author",
                "published_at", "read_status", "signal", "group", "ranker_picked", "ranker_score", "content"]
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)  # teaser text that is just a URL
CONTENT_CHARS = 6000
DEFAULT_TARGET = 80
DEFAULT_OUT = OUTPUT_DIR / "label_hub.html"
_TEMPLATE_PATH = Path(__file__).with_name("label_hub_template.html")
_RUN_RE = re.compile(r"(\d{8}_\d{6})\.json$")


def _text(content: str) -> str:
    # same cleaning as modules.article_text.article_text, which needs an Article object we do not have per row
    return " ".join(BeautifulSoup(content or "", "html.parser").get_text(" ").split())[:CONTENT_CHARS]


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _read_report(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("items", [])


def collect_items(output_dir: Path, golden_path: Path) -> list[dict]:
    """Every article from every ranking report, de-duplicated, with the labels already in the golden set."""
    reports = sorted((m.group(1), p) for p in output_dir.glob("ranking_*.json") if (m := _RUN_RE.search(p.name)))
    by_url: dict[str, dict] = {}
    for run, path in reports:  # oldest first, so a later report overwrites an earlier copy of the same URL
        for row in _read_report(path):
            if row.get("url"):
                by_url[row["url"]] = {**row, "source_run": run}
    golden = {}
    if golden_path.exists():
        golden = {g["url"]: g for g in json.loads(golden_path.read_text(encoding="utf-8")) if g.get("url")}

    # the same article under two URLs (e.g. a Substack redirect and the real post): keep the newest copy
    # (origin is "inbox" for every newsletter story, so the date has to match too)
    groups: dict[tuple, list[dict]] = {}
    for it in by_url.values():
        key = (_norm(it.get("title")), _norm(it.get("origin")), str(it.get("published_at", "")))
        groups.setdefault(key, []).append(it)
    items = []
    for (title, _, _), copies in groups.items():
        if not title or len(copies) == 1:
            items.extend(copies)
            continue
        keep = max(copies, key=lambda c: c["source_run"])  # max keeps the first of equal runs
        labeled = next((c for c in [keep, *copies] if c["url"] in golden), keep)
        items.append({**keep, "_label_url": labeled["url"]})

    for it in items:
        label_url = it.pop("_label_url", it["url"])
        g = golden.get(label_url, {})
        it.update(label_url=label_url if g else "", label=g.get("label", ""), tag=g.get("tag", ""), note=g.get("note", ""),
                  labeled_with=g.get("labeled_with", ""), content=_text(it.get("content", "")))
    items.sort(key=lambda i: i["source_run"], reverse=True)
    items.sort(key=lambda i: bool(i["label"]))  # stable: unlabeled first, newest report first within each
    return items


def render_html(items: list[dict], target: int = DEFAULT_TARGET) -> str:
    # "</" inside an inline <script> would end it early
    data = json.dumps({"items": items, "target": target, "fields": LABEL_FIELDS},
                      ensure_ascii=False).replace("</", "<\\/")
    return _TEMPLATE_PATH.read_text(encoding="utf-8").replace("__DATA__", data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET, help="number of labels to aim for")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    items = collect_items(OUTPUT_DIR, GOLDEN_RANKING_PATH)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(items, args.target), encoding="utf-8")
    print(f"{len(items)} items ({sum(bool(i['label']) for i in items)} already labeled) -> {args.out}")


if __name__ == "__main__":
    main()
