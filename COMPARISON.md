# Negative Sentiment Comparison: Female vs Male Athletes

**Research question:** Do female Dutch speed skating athletes receive more negative attention than male athletes in Reddit comments?

## Dataset

`comments_all.csv` — 29,220 Reddit comments about Dutch athletes across 36 subreddits. After filtering (see below): **8,113 comments** used for analysis.

| Gender | Comments |
|---|---|
| vrouw (female) | 5,545 |
| man (male) | 2,568 |

---

## Approaches Tried

### 1. NLI Zero-shot Classification (abandoned)

**Script:** `zeroshot/classify_sentiment.py`

Used `cross-encoder/nli-deberta-v3-small` with `multi_label=True` to classify three dimensions at once:
- `speedskating`: is this about speed skating?
- `performance`: does this discuss results, race times, medals?
- `negative`: is the tone negative toward the athlete?

**Why it failed:** The model scored "Bitterballen for everybody!" at 0.999 for "discusses athletic results, race times, medals" — it doesn't understand content, it only checks whether a hypothesis is plausible in context. Short comments in a sports subreddit all score near-certain entailment for sports-related labels regardless of actual content. Dutch text made this worse.

**Tried a larger model** (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`) — same problem, some cases got worse. The issue is fundamental to how NLI works, not model size.

---

### 2. Hybrid: Subreddit Filter + Multilingual Sentiment Model (implemented)

**Script:** `zeroshot/classify_hybrid.py`

**Pipeline:**
1. Apply subreddit allowlist + name-mention filter → 8,113 rows
2. Run `cardiffnlp/twitter-xlm-roberta-base-sentiment` on each comment
3. Classify as `negative=True` if sentiment label is "negative"

The sentiment model is multilingual (trained on tweets in many languages including Dutch) and much better at tone detection than NLI. It does not need a relevance pre-filter — the subreddit filter handles that.

**Runtime:** ~2.5 minutes for 8,113 comments.

**Results:**

| Gender | Negative | Total | Rate |
|---|---|---|---|
| man | 975 | 2,568 | 38.0% |
| vrouw | 808 | 1,568 | 41.7% |

> Note: The sentiment model classifies the overall tone of the comment, not whether the negativity is directed specifically at the athlete. A comment criticising media coverage while praising the athlete would still flag as negative.

---

### 3. Llama 3.1:8b Three-label Classification (primary approach)

**Script:** `llama3.1-8b/classify_sentiment.py`

**Pipeline:**
1. Apply subreddit allowlist + name-mention filter → 8,113 rows
2. For each comment, send a prompt to `llama3.1:8b` via Ollama asking three explicit questions:
   - `speedskating`: is this comment about speed skating?
   - `performance`: does this discuss results, technique, race times, medals, records, or competitive achievements?
   - `negative`: is the tone negative or critical toward the athlete or their performance?
3. Parse the JSON response, store all three boolean labels

The prompt includes subreddit and thread title as context to help the model distinguish a speed skating thread from an off-topic one.

**Runtime:** ~1h40m for 8,113 comments at 4 parallel workers.

**Results — speedskating + negative:**

| Gender | Negative | Total | Rate |
|---|---|---|---|
| man | 257 | 2,568 | 10.0% |
| vrouw | 861 | 5,545 | 15.5% |

**Results — speedskating + performance + negative** (most precise filter):

| Gender | Negative | Total | Share of negatives |
|---|---|---|---|
| man | 26 | 2,568 | 26% |
| vrouw | 75 | 5,545 | 74% |

Output files:
- `llama3.1-8b/negative_speedskating.csv` — 1,118 rows (speedskating + negative)
- `llama3.1-8b/negative_performance.csv` — 101 rows (speedskating + performance + negative)

---

## Key Technical Fix: Non-unique comment_ids

During development, all checkpoint/resume logic was broken because `comment_id` is **not unique** in `comments_all.csv` — IDs are sequential per thread and reset across threads (29,220 rows, only 5,590 unique IDs). This caused massive row duplication in output files (e.g. 6,384 saved rows representing only 1,248 actual comments).

**Fix:** Added a `uid` column to `comments_all.csv` — a 16-character MD5 hash of the composite key `(comment_id, thread_url, persoon)`, which is unique across all 29,220 rows. All scripts now track progress using `uid`.

---

## Comparison Summary

| Approach | Signal | Female rate | Male rate | Gap |
|---|---|---|---|---|
| Hybrid (sentiment model) | Negative tone | 41.7% | 38.0% | +3.7pp |
| Llama (speedskating + negative) | Negative skating comment | 15.5% | 10.0% | +5.5pp |
| Llama (speedskating + performance + negative) | Negative performance comment | 74% of total | 26% of total | ~3× more |

All three approaches agree: **female athletes receive more negative attention than male athletes.** The llama approach with the performance filter gives the clearest signal, isolating negativity that is specifically directed at athletic achievement rather than appearance, media coverage, or other context.
