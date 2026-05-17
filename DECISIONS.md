# Decisions

## Relevance scoring model too small

`llama3.2:1b` is not capable of reliably scoring comment relevance. Testing showed it outputs the same score (2) for every comment regardless of content — both clearly relevant and clearly irrelevant comments received identical scores.

### Options considered

**Option 1: Pull a larger model**
Use `llama3.1:8b` instead. Handles instruction following correctly and can infer relevance even when the person's name isn't explicitly mentioned (e.g. "he skated a world record" in a thread about Sven). Costs ~5GB disk and significantly more inference time.

**Option 2: Name-based matching**
Skip the LLM entirely. Check whether the `persoon` value appears as a substring in the comment text. Instant, fully reliable for explicit mentions, zero compute cost. Misses implicit references.

### Status

Pending decision — chose to document before proceeding.
