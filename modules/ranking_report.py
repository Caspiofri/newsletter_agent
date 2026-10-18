"""Ranking report: every candidate the ranker saw, plus a random sample it never saw, as JSON and a self-contained
HTML page for hand-labeling. Labels exported from the page become the ground truth in evals/ranking_golden.json."""
import json
import random
from datetime import datetime
from pathlib import Path

from config.settings import OUTPUT_DIR
from modules.models import Article

SAMPLE_SIZE = 10
CONTENT_CHARS = 6000  # text snapshot kept per article, so labels stay usable after the link rots


def _row(a: Article, group: str, score=None, reason: str = "", picked: bool = False) -> dict:
    return {
        "title": a.name, "url": a.url, "origin": a.origin, "author": a.author or "",
        "published_at": str(a.published_at), "read_status": a.read_status, "link_status": a.link_status,
        "signal": a.signal, "group": group, "score": score, "reason": reason, "picked": picked,
        "content": a.content[:CONTENT_CHARS],
    }


def build_rows(candidates: list[Article], final: dict[int, float], reasons: dict[int, str],
               picked: list[Article], pool: list[Article], sample_size: int = SAMPLE_SIZE,
               seed: str = "", read_sample=None) -> list[dict]:
    """One row per shortlisted candidate (with the ranker's score) plus a seeded random sample of the rest.

    The sample is what makes false negatives visible: stories the ranker never even saw. read_sample, when given,
    fetches the full text of just the sampled articles so they can be judged from the article, not a teaser;
    it only feeds this report, never the ranker.
    """
    picked_ids = {id(a) for a in picked}
    rows = [
        _row(a, "shortlisted", final.get(i), reasons.get(i, ""), id(a) in picked_ids)
        for i, a in enumerate(candidates)
    ]
    seen_urls = {a.url for a in candidates}
    rest = [a for a in pool if a.url not in seen_urls]
    sample = random.Random(seed).sample(rest, min(sample_size, len(rest)))
    if read_sample is not None:
        try:
            read = read_sample(sample)
            if len(read) != len(sample):
                raise ValueError(f"read {len(read)} of {len(sample)} articles")
            sample = read
        except Exception as e:
            print(f"[rank] could not read sample text: {e}")
    return rows + [_row(a, "sample") for a in sample]


def render_html(rows: list[dict], profile: str, run_id: str) -> str:
    # "</" inside an inline <script> would end it early
    data = json.dumps({"profile": profile, "run_id": run_id, "items": rows}, ensure_ascii=False).replace("</", "<\\/")
    return _TEMPLATE.replace("__TITLE__", f"Ranking labels — {profile} — {run_id}").replace("__DATA__", data)


def save(profile: str, rows: list[dict], out_dir: Path = OUTPUT_DIR) -> Path:
    out_dir.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"ranking_{profile.lower()}_{run_id}"
    (out_dir / f"{stem}.json").write_text(
        json.dumps({"profile": profile, "run_id": run_id, "items": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    html_path = out_dir / f"{stem}.html"
    html_path.write_text(render_html(rows, profile, run_id), encoding="utf-8")
    return html_path


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --text:#1a202c; --muted:#64748b; --line:#e2e8f0; --accent:#3d5a80;
          --must:#15803d; --worth:#b45309; --skip:#b91c1c; --chip:#eef2f7; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0f1419; --card:#18202a; --text:#e5e9f0; --muted:#94a3b8; --line:#2a3544; --accent:#7aa2d6;
            --must:#4ade80; --worth:#fbbf24; --skip:#f87171; --chip:#222d3a; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }
  header { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line); padding:12px 16px; }
  h1 { font-size:17px; margin:0 0 8px; }
  .bar { display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; font-size:13px; color:var(--muted); }
  .bar select, .bar button { font:inherit; padding:4px 8px; border:1px solid var(--line); border-radius:6px;
                             background:var(--card); color:var(--text); cursor:pointer; }
  .bar button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
  main { max-width:860px; margin:0 auto; padding:12px 16px 60px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:12px 0; }
  .card.done { border-left:4px solid var(--accent); }
  .title { font-weight:600; font-size:16px; }
  .title a { color:inherit; text-decoration:none; } .title a:hover { text-decoration:underline; }
  .meta { display:flex; flex-wrap:wrap; gap:6px; margin:6px 0; }
  .chip { background:var(--chip); color:var(--muted); border-radius:999px; padding:1px 9px; font-size:12px; }
  .chip.bad { color:var(--skip); }
  .ranker { margin:6px 0; font-size:13px; padding:6px 10px; border-radius:6px; background:var(--chip); }
  details { margin:6px 0; } summary { cursor:pointer; color:var(--muted); font-size:13px; }
  pre { white-space:pre-wrap; word-break:break-word; font:13px/1.55 inherit; margin:8px 0 0; color:var(--muted); }
  .labels { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:10px; }
  .labels button { font:inherit; padding:6px 14px; border-radius:8px; border:1px solid var(--line);
                   background:var(--card); color:var(--text); cursor:pointer; }
  .labels button.on[data-l="must"] { background:var(--must); border-color:var(--must); color:#fff; }
  .labels button.on[data-l="worth"] { background:var(--worth); border-color:var(--worth); color:#fff; }
  .labels button.on[data-l="skip"] { background:var(--skip); border-color:var(--skip); color:#fff; }
  .labels select { font:inherit; padding:5px 8px; border:1px solid var(--line); border-radius:8px;
                   background:var(--card); color:var(--text); }
  .hint { color:var(--muted); font-size:12px; margin:8px 0 0; }
</style>
</head>
<body>
<header>
  <h1 id="heading"></h1>
  <div class="bar">
    <span id="progress"></span>
    <label>Show <select id="filter">
      <option value="all">all</option><option value="todo">not labeled yet</option>
      <option value="shortlisted">shortlisted by the ranker</option><option value="sample">random sample (never ranked)</option>
    </select></label>
    <label>Order <select id="order">
      <option value="blind">shuffled (blind)</option><option value="score">by ranker score</option>
    </select></label>
    <label><input type="checkbox" id="reveal"> show ranker score and picks</label>
    <button class="primary" id="export">Export labels (JSON)</button>
  </div>
</header>
<main id="list"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const KEY = 'ranking-labels:' + D.run_id;
const TAGS = ['', 'great primary source', 'practical / useful', 'beginner', 'promo', 'hype', 'off-topic', 'too thin', 'full text mattered', 'other'];
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} };
const $ = id => document.getElementById(id);
const hash = s => { let h = 0; for (const c of s) h = (h * 31 + c.charCodeAt(0)) | 0; return h; };
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

function render() {
  const reveal = $('reveal').checked, filter = $('filter').value;
  let items = D.items.filter(it => filter === 'all' || (filter === 'todo' ? !labels[it.url] : it.group === filter));
  items.sort($('order').value === 'score' ? (a, b) => (b.score ?? -1) - (a.score ?? -1) : (a, b) => hash(a.url) - hash(b.url));
  const list = $('list'); list.replaceChildren();
  for (const it of items) {
    const lab = labels[it.url] || {};
    const card = el('div', 'card' + (lab.label ? ' done' : ''));
    const t = el('div', 'title'); t.dir = 'auto';
    const a = el('a', null, it.title); a.href = it.url; a.target = '_blank'; a.rel = 'noopener'; t.append(a); card.append(t);
    const meta = el('div', 'meta');
    for (const [txt, bad] of [[it.origin], [it.author], [it.published_at], ['text: ' + it.read_status, it.read_status === 'snippet'],
                              [it.signal], ['link: ' + it.link_status, it.link_status !== 'ok']])
      if (txt && txt !== 'link: ok') meta.append(el('span', 'chip' + (bad ? ' bad' : ''), txt));
    card.append(meta);
    if (reveal) {
      const r = el('div', 'ranker');
      r.textContent = it.group === 'sample' ? 'Not seen by the ranker (random sample)'
        : `${it.picked ? '★ picked · ' : ''}ranker score ${it.score ?? '–'} — ${it.reason}`;
      card.append(r);
    }
    const d = el('details'); d.append(el('summary', null, 'Text the ranker saw (' + it.content.length + ' chars)'));
    const pre = el('pre', null, it.content || '(none)'); pre.dir = 'auto'; d.append(pre); card.append(d);
    const row = el('div', 'labels');
    for (const [k, name] of [['must', 'Must-read'], ['worth', 'Worth it'], ['skip', 'Skip']]) {
      const b = el('button', lab.label === k ? 'on' : '', name); b.dataset.l = k;
      b.onclick = () => { labels[it.url] = { ...labels[it.url], label: lab.label === k ? undefined : k }; clean(it.url); save(); render(); };
      row.append(b);
    }
    const sel = el('select'); sel.title = 'optional reason';
    for (const tg of TAGS) { const o = el('option', null, tg || 'reason (optional)'); o.value = tg; if (lab.tag === tg) o.selected = true; sel.append(o); }
    sel.onchange = () => { labels[it.url] = { ...labels[it.url], tag: sel.value || undefined }; clean(it.url); save(); updateProgress(); };
    row.append(sel); card.append(row); list.append(card);
  }
  updateProgress();
}
function clean(url) { const l = labels[url]; if (l && !l.label && !l.tag) delete labels[url]; }
function updateProgress() {
  const n = D.items.filter(it => labels[it.url] && labels[it.url].label).length;
  $('progress').textContent = `${n} / ${D.items.length} labeled`;
}
$('export').onclick = () => {
  const out = D.items.filter(it => labels[it.url] && labels[it.url].label).map(it => ({
    url: it.url, title: it.title, origin: it.origin, author: it.author, published_at: it.published_at,
    read_status: it.read_status, signal: it.signal, content: it.content,
    label: labels[it.url].label, tag: labels[it.url].tag || '',
    ranker_score: it.score, ranker_picked: it.picked, group: it.group }));
  const blob = new Blob([JSON.stringify({ profile: D.profile, run_id: D.run_id, labels: out }, null, 2)], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = `ranking_labels_${D.profile.toLowerCase()}_${D.run_id}.json`; a.click();
};
['filter', 'order', 'reveal'].forEach(id => $(id).onchange = render);
$('heading').textContent = `Label articles — ${D.profile} — ${D.run_id}`;
render();
</script>
</body>
</html>
"""
