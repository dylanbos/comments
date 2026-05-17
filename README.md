# Comment Relevance Scorer

Scores ~41,000 Reddit comments for relevance to a specific person using local AI models. Each comment gets a relevance score based on how much it is about the person tagged in the `persoon` column.

## Dataset

`comments_all.csv` contains Reddit comments across threads about Dutch athletes: **sven** (9,917), **jutta** (3,580), **kjeld** (1,036), **thomas** (646), **femke** (579), **ireen** (372). Many comments are off-topic — general reactions, jokes, or tangents. Relevance scoring filters these out before further analysis.

## Approaches

Six approaches were evaluated for this task. See [APPROACHES.md](APPROACHES.md) for a full comparison. Two are implemented:

| Approach | Folder | Output | Speed |
|---|---|---|---|
| Local LLM via Ollama (`llama3.1:8b`) | `llama3.1-8b/` | 0–10 score | ~12 rows/sec |
| Zero-shot classification (`deberta-v3-small`) | `zeroshot/` | 0–1 probability | faster |

The original `llama3.2:1b` model was tried and abandoned — too small to follow instructions reliably. See [DECISIONS.md](DECISIONS.md).

## Setup

```bash
python3 -m venv .venv

# For Ollama approach
ollama pull llama3.1:8b
.venv/bin/pip install pandas requests tqdm

# For zero-shot approach
.venv/bin/pip install transformers torch pandas tqdm
```

## Scoring

**Ollama (0–10 integer score):**
```bash
cd llama3.1-8b
../.venv/bin/python3 score_relevance.py
# output: llama3.1-8b/comments_scored.csv
```

**Zero-shot (0–1 probability):**
```bash
cd zeroshot
../.venv/bin/python3 score_relevance.py
# output: zeroshot/comments_scored.csv
```

Both scripts checkpoint every 200 rows and resume automatically if interrupted — just rerun the same command.

## Filtering

```bash
# Ollama scores (0–10 integer, default threshold 7)
.venv/bin/python3 filter_by_score.py --min 7 --persoon sven

# Zero-shot scores (0–1 probability, use decimal threshold)
.venv/bin/python3 filter_by_score.py --input zeroshot/comments_scored.csv --min 0.7 --persoon sven
```

| Argument | Default | Description |
|---|---|---|
| `--min` | `7` | Minimum score to include |
| `--persoon` | *(all)* | Filter to a specific person |
| `--input` | `comments_scored.csv` | Input file |
| `--output` | `comments_filtered_min{N}.csv` | Output file |

## Docs

| File | Description |
|---|---|
| [APPROACHES.md](APPROACHES.md) | All six approaches compared — local LLM, zero-shot, embeddings, HuggingFace API, OpenAI/Claude API, spaCy NER |
| [DECISIONS.md](DECISIONS.md) | Why `llama3.2:1b` was dropped and `llama3.1:8b` chosen, with validation results |
| [zeroshot/MODEL_CHOICE.md](zeroshot/MODEL_CHOICE.md) | Why `cross-encoder/nli-deberta-v3-small` was chosen for zero-shot, with validation results |
