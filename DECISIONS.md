# Decisions

## Relevance scoring model too small

`llama3.2:1b` is not capable of reliably scoring comment relevance. Testing showed it outputs the same score (2) for every comment regardless of content — both clearly relevant and clearly irrelevant comments received identical scores.

### Options considered

**Option 1: Pull a larger model**
Use `llama3.1:8b` instead. Handles instruction following correctly and can infer relevance even when the person's name isn't explicitly mentioned (e.g. "he skated a world record" in a thread about Sven). Costs ~5GB disk and significantly more inference time.

**Option 2: Name-based matching**
Skip the LLM entirely. Check whether the `persoon` value appears as a substring in the comment text. Instant, fully reliable for explicit mentions, zero compute cost. Misses implicit references.

### Decision

Went with **Option 1: `llama3.1:8b`**. Validated on a 30-row sample (3 name mentions + 3 non-mentions per person across sven, jutta, kjeld, femke, ireen). Results:

| Person | Name mentioned (avg score) | Not mentioned (avg score) |
|---|---|---|
| ireen | 9.3 | 0.3 |
| femke | 8.3 | 3.3 |
| sven | 9.3 | 4.7 |
| jutta | 6.3 | 5.0 |
| kjeld | 6.0 | 7.0 |

Strong separation for ireen, femke, and sven. Jutta and kjeld show weaker separation, but false positives are thematically reasonable (speed skating threads are inherently relevant). Model correctly handles implicit relevance without requiring an explicit name mention.

Full run started on `comments_all.csv` with results going to `llama3.1-8b/comments_scored.csv`.
