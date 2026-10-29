# Dashboard metrics: "Newsletter agent: daily digest"

The LangSmith dashboard (Monitoring → Dashboards) has 11 charts. Every chart reads the root `digest_run` run of the
`newsletter-agent` project, so there is **one point per day** (the 07:00 cron run; a manual run adds an extra point).
The values come from feedback that `modules/tracing.py` posts on that run after it finishes, plus LangSmith's own
run fields. The charts are defined in `evals/langsmith_dashboard.py`; edit `CHARTS` and re-run
`.venv/bin/python -m evals.langsmith_dashboard` to rebuild.

Feedback keys use three prefixes:

| Prefix | Meaning | Posted by |
|---|---|---|
| `judge.*` | a 1–5 quality score | `modules/trace_judge.py`, after every run |
| `count.*` | a number of articles or items | the pipeline nodes' `run_stats` |
| `metric.*` | a measurement that is not a count | the pipeline nodes' `run_stats` |
| `human.*` | your own ratings | the Telegram bot (`evals/human_feedback.py`) |

---

## 1. Digest quality (judge scores, 1–5)

Automatic grades for each run. **A drop means something regressed.**

| Series | Key | How it is scored |
|---|---|---|
| fetch health | `judge.fetch_health` | **Rule-based, no model call.** 5 = articles found in a 1-day window; 4 = had to widen to 2 days; 3 = 3+ days; 1 = no articles at all. |
| filter quality | `judge.filter_quality` | Claude judge. It sees the **titles** of every fetched article and the titles picked, and grades whether the picks are the most relevant to the digest's subject. 5 = clearly the best picks, 3 = mixed, 1 = mostly off-topic. |
| content faithfulness | `judge.content_faithfulness` | Claude judge. It pairs each card's summary with the text the summarizer read and checks that every claim is supported. 5 = nothing invented, 3 = minor embellishment, 1 = several invented facts. |
| overall | `judge.overall` | Average of the stages that were scored (a skipped stage is left out). |

**Limits to keep in mind**
- Fetch health is close to always 5 on a normal day, so it only catches outages.
- Filter quality judges from titles only, and the judge has no idea what *you* like. Your own verdict (chart 2) is the
  better signal for ranking.
- Faithfulness is checked against the text the summarizer saw. If that text was only a teaser (see chart 5), the
  card can be faithful to the teaser and still miss the article. The judge's own catch rate has not been measured yet.
- With one run a day, a single 4 among 5s is noise. Look for a run of low points.

## 2. Your verdict on the picks (Telegram)

| Series | Key | Meaning |
|---|---|---|
| picks precision | `human.picks_precision` | Of the day's digest picks you rated in Telegram, the share you rated ⭐ must-read or 👍 worth it. |

- Posted (and re-posted) every time you rate a digest pick, from the daily batch or from `/more`, so an old day's
  point can change when you rate more of its picks.
- The number of picks behind each point is posted too, as `count.human_picks_rated`. It's not on the chart, but you
  can see it in the run's feedback.
- **This is the closest thing to "is the digest good for me".** With 1–5 rated picks a day it is noisy: 1 of 1 = 1.00.
  Days where you rated all 5 picks are the reliable ones.

## 3–4. Pipeline intake and Pipeline selection

How many articles are left after each step. **Together they show where articles are lost.** The funnel is split
into two charts because the intake (200+) would squash the selection counts (5–22) into flat lines on a shared axis.

**Pipeline intake (articles collected)**

| Series | Key | Step |
|---|---|---|
| collected | `count.articles_collected_merged` | All articles from the collectors (inbox newsletters, Hacker News, Medium), merged. |
| after dedupe | `count.articles_after_dedupe` | After removing stories already sent (same URL, or a similar story in the last 7 days) and duplicates within today's batch. |

The inbox drives most of the intake: each run reads email from yesterday 00:00 until 07:00, and one newsletter
holds many links. Expect a dip on the Monday run, which covers Sunday.

**Pipeline selection (articles per stage)**

| Series | Key | Step |
|---|---|---|
| shortlisted | `count.articles_shortlisted` | After the title screen, run per source (quota: inbox 10, Hacker News 6, Medium 6). |
| ranked | `count.articles_ranked` | Shortlisted articles the ranker actually scored. |
| picked | `count.final_picked_articles` | Articles that scored at least `RANK_MIN_SCORE` (5) and made the top k. |

**What to watch**
- *collected* falls sharply: a collector broke (Gmail auth, an RSS feed, Medium).
- big gap between *collected* and *after dedupe*: dedupe is dropping a lot; see charts 6–7.
- *ranked* below *shortlisted*: some articles got no score (a parse failure in the ranker's reply).
- *picked* below k: few articles reached the minimum score, so it was a thin day or the ranker was harsh.

## 5. Reading quality (shortlisted articles)

How much of each shortlisted article the pipeline could actually read before ranking and summarizing.

| Series | Key | Meaning |
|---|---|---|
| full | `count.articles_read_full` | At least 1,500 characters of article text fetched. |
| partial | `count.articles_read_partial` | Some text fetched, but under 1,500 characters. |
| snippet | `count.articles_read_snippet` | The fetch failed, so only the newsletter's teaser was kept. |

**More "full" is better.** Snippet-only articles are ranked and summarized from a few lines, which hurts both ranking
and faithfulness. The Medium full-text fix (Oct 21) should push "full" up from that day on.

## 6. Dedupe activity

The dedupe step first drops any article whose exact URL was already sent (the store keeps 14 days). It then compares
the rest to the articles sent in the last 7 days, using embedding similarity. Two articles count as the same story at
a similarity of **0.82** or more.

| Series | Key | Meaning |
|---|---|---|
| dropped | `count.articles_dedupe_dropped` | Total articles removed by dedupe. |
| history matches | `count.dedupe_matches` | Removed because they matched something already sent (by URL or by similarity). |
| same-URL matches | `count.dedupe_url_matches` | The history matches caught by exact URL. Articles sent before Oct 26 have no stored URL, so this starts at 0. |
| in-batch duplicates | `count.dedupe_batch_duplicates` | Removed because the same story appeared twice in today's batch. |
| near misses | `count.dedupe_near_misses` | Similar to a past story (0.70–0.82) but **kept**. |

**Many near misses means the 0.82 threshold is borderline.** Some of those might be repeats that slipped through, or
the threshold might be dropping distinct stories. The threshold has not been validated with labels yet (that's the
planned "same story?" rating in Telegram).

## 7. Dedupe max similarity

| Series | Key | Meaning |
|---|---|---|
| max similarity | `metric.dedupe_max_similarity` | The highest similarity among the articles dropped as embedding matches (0–1). URL matches are left out. 0 when nothing was dropped. |

A value close to 1.0 is a clear repeat. Values that stay just above 0.82 are the risky ones: those drops are the
most likely to be different stories that happened to look alike.

## 8. Digest size

| Series | Key | Meaning |
|---|---|---|
| cards | `count.cards` | Full cards in the email (the "🔥 main stories" with a summary and personalization). |
| brief items | `count.brief_items` | One-line items in the brief section. |

A day with 0 cards means the summarize step produced nothing usable. Faithfulness is then skipped.

## 9. Runs and errors

| Series | Source | Meaning |
|---|---|---|
| runs | LangSmith run count | How many `digest_run` runs that day. Normally 1; 0 means the cron did not run (Pi off, cron broken). |
| error rate | runs with status `error` ÷ all runs | 0 is healthy. A run that crashed shows 1.0. |

## 10. Run latency (s)

| Series | Source | Meaning |
|---|---|---|
| p50 | median of `latency_seconds` | Typical end-to-end time of one digest run. |
| p99 | 99th percentile | The slowest run. With one run a day, p50 and p99 are the same point. |

A jump usually means slow article fetching (timeouts) or slow model replies.

## 11. Tokens and cost

| Series | Source | Meaning |
|---|---|---|
| tokens | sum of `total_tokens` | Model tokens used per day. |
| cost (USD) | sum of `total_cost` | Estimated model cost per day. |

Each `gemini` and `judge_llm` run reports its token counts (Gemini's thinking tokens count as output, since they are
billed as output), and LangSmith prices them from the model name. The root run sums its model calls. Runs before
Oct 27 had no usage data, so they show 0.

---

## Quick health check

| Look at | Healthy | Investigate when |
|---|---|---|
| Runs and errors | 1 run, error rate 0 | 0 runs or any error |
| Pipeline intake: collected | steady from day to day | sudden drop |
| Reading quality | "full" is the largest bar | "snippet" grows |
| Digest quality: overall | 4–5 | several days below 4 |
| Your verdict | trending up, or flat near 1.0 | trending down |
| Dedupe near misses | a few | many, or max similarity hovering near 0.82 |
