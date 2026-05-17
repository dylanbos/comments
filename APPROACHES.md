# Approaches for Comment Relevance Scoring

Six approaches considered for scoring whether a Reddit comment is about a specific person.

---

## 1. Local LLM via Ollama (local)

Run a generative LLM locally using Ollama and prompt it to return a relevance score. Flexible — the scoring criteria can be adjusted by changing the prompt.

**Models tried:** `llama3.2:1b` (failed), `llama3.1:8b` (works)
**Output:** integer score 0–10
**Speed:** slow — ~12 rows/second with 4 workers on `llama3.1:8b`; estimated 8–12 hours for 41K rows
**Accuracy:** `llama3.2:1b` too small — outputs constant scores regardless of input. `llama3.1:8b` shows good separation between relevant and irrelevant comments.
**Cost:** free; requires ~5GB disk for the 8b model
**Setup:** install [Ollama](https://ollama.com), `ollama pull llama3.1:8b`, `pip install requests`

**Best for:** flexible scoring criteria that are hard to express as a classification label, or when you want natural language control over what "relevant" means.

---

## 2. HuggingFace Zero-Shot Classification (local)

Frames relevance as a natural language inference task. Given a comment and labels like `"about sven"` / `"not about sven"`, the model returns a probability. No prompt engineering, no fine-tuning needed.

**Model used:** `cross-encoder/nli-deberta-v3-small` (~180MB)
**Output:** probability 0–1
**Speed:** fast, batched, fully local
**Accuracy:** good — handles implicit references, struggles with ambiguous pronouns
**Cost:** free
**Setup:** `pip install transformers torch`

**Best for:** local, offline use where you want a meaningful confidence score without running a full generative LLM.

---

## 3. Sentence Transformers / Embeddings (local)

Embeds the comment and a reference sentence (e.g. `"sven kramer the speed skater"`) into vector space, then computes cosine similarity. Higher similarity = more relevant.

**Model:** `sentence-transformers/all-MiniLM-L6-v2` (~80MB)
**Output:** cosine similarity 0–1
**Speed:** very fast — thousands of rows per second
**Accuracy:** weaker on implicit or indirect references; purely semantic
**Cost:** free
**Setup:** `pip install sentence-transformers`

**Best for:** very large datasets where speed matters more than nuance, or as a pre-filter before a slower method.

---

## 4. HuggingFace Inference API (cloud)

Same models as approach 1, but hosted by HuggingFace. No local compute needed — send HTTP requests to their API.

**Output:** probability 0–1
**Speed:** moderate, rate-limited on free tier
**Accuracy:** same as local zero-shot
**Cost:** free tier available; paid plans for higher throughput
**Setup:** `pip install huggingface_hub` + API token

**Best for:** testing or one-off runs without wanting to download models locally.

---

## 5. OpenAI / Claude API (cloud)

Send each comment to a hosted LLM (GPT-4o-mini, Claude Haiku) with a scoring prompt. Best reasoning, handles nuance, sarcasm, and implicit references well.

**Output:** integer score 0–10 (prompt-defined)
**Speed:** slow for 41K rows; parallelisable with async calls
**Accuracy:** highest of all approaches
**Cost:** ~$2–5 for 41K short comments with GPT-4o-mini or Claude Haiku
**Setup:** `pip install openai` or `pip install anthropic` + API key

**Best for:** when accuracy is critical and a small cost is acceptable. Also the easiest to prompt in natural language.

---

## 6. spaCy Named Entity Recognition (local)

Uses NER to detect whether a person's name appears in the comment. Fast and deterministic — no model uncertainty.

**Output:** boolean (name mentioned / not mentioned)
**Speed:** very fast
**Accuracy:** only catches explicit name mentions; misses pronouns, nicknames, implicit references
**Cost:** free
**Setup:** `pip install spacy && python -m spacy download en_core_web_sm`

**Best for:** a cheap first-pass filter to remove comments that definitely don't mention a person, before running a slower method on what's left.

---

## Comparison

| Approach | Local | Cost | Speed | Handles implicit refs | Output |
|---|---|---|---|---|---|
| Local LLM via Ollama | Yes | Free | Slow | Yes | 0–10 score |
| Zero-shot classification | Yes | Free | Fast | Yes | 0–1 score |
| Sentence transformers | Yes | Free | Very fast | Partial | 0–1 score |
| HuggingFace Inference API | No | Free tier | Moderate | Yes | 0–1 score |
| OpenAI / Claude API | No | ~$2–5 | Slow | Best | 0–10 score |
| spaCy NER | Yes | Free | Very fast | No | Boolean |

## What we tried

- `llama3.2:1b` via Ollama — failed, model too small to follow instructions reliably
- `llama3.1:8b` via Ollama — works, currently running full dataset
- `cross-encoder/nli-deberta-v3-small` zero-shot — works well, validated on 30-row sample
