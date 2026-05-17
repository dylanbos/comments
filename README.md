# Comment Relevance Scorer

Scores ~41,000 Reddit comments for relevance to a specific person using a local Llama model via Ollama. Each comment is scored 0–10 based on how much it is about the person tagged in the `persoon` column.

## Why

`comments_all.csv` contains comments from multiple Reddit threads covering several Dutch athletes (Sven, Jutta, Kjeld, etc.). Many comments are off-topic — general reactions, jokes, or tangents. The relevance score lets you filter down to comments that are actually about the person in question before doing further analysis.

## Requirements

- [Ollama](https://ollama.com) installed and running with `llama3.2:1b` pulled
- Python 3 with a virtual environment

## Setup

```bash
# Pull the model if you haven't yet
ollama pull llama3.2:1b

# Create the virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install pandas requests tqdm
```

## Usage

```bash
.venv/bin/python3 score_relevance.py
```

Output is written to `comments_scored.csv` — the original CSV with an added `relevance_score` column (0–10). A score of `-1` means the model returned an unparseable response.

## Resume

If the script is interrupted, rerun the same command. It reads `comments_scored.csv` on startup and skips already-scored rows, so no work is lost. Progress is also checkpointed to disk every 200 rows during the run.

## Performance

With 4 concurrent workers and `llama3.2:1b`, expect **3–6 hours** for the full dataset. To speed it up, increase `MAX_WORKERS` at the top of `score_relevance.py`. For better accuracy at the cost of speed, switch `MODEL` to a larger model like `llama3.1:8b` (requires `ollama pull llama3.1:8b`).

## Output

| Column | Description |
|---|---|
| *(all original columns)* | Unchanged from `comments_all.csv` |
| `relevance_score` | 0–10, higher = more about the person in `persoon` |

## Filtering

Once scoring is complete (or partially complete), use `filter_by_score.py` to filter down to relevant comments.

```bash
# Keep comments with score >= 7 (default)
.venv/bin/python3 filter_by_score.py

# Custom minimum score
.venv/bin/python3 filter_by_score.py --min 5

# Filter to a specific person and score
.venv/bin/python3 filter_by_score.py --min 6 --persoon sven

# Custom output file
.venv/bin/python3 filter_by_score.py --min 7 --persoon sven --output sven_relevant.csv
```

| Argument | Default | Description |
|---|---|---|
| `--min` | `7` | Minimum relevance score to include |
| `--persoon` | *(all)* | Filter to a specific person |
| `--input` | `comments_scored.csv` | Input file |
| `--output` | `comments_filtered_min{N}.csv` | Output file |
