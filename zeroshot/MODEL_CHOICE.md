# Model Choice: Zero-Shot Classification

## Chosen model

`cross-encoder/nli-deberta-v3-small` via HuggingFace `transformers`

## Why this approach

Zero-shot classification frames relevance scoring as a natural language inference (NLI) task. Given a comment and two candidate labels — `"about {person}"` and `"not about {person}"` — the model returns a probability for each. This is purpose-built for the task, unlike a generative LLM which has to be coaxed into returning a number.

## Why this model

| Model | Size | Speed | Notes |
|---|---|---|---|
| `cross-encoder/nli-deberta-v3-small` | ~180MB | Fast | Good accuracy, chosen |
| `facebook/bart-large-mnli` | ~1.6GB | Moderate | Classic choice, larger |
| `MoritzLaurer/deberta-v3-large-zeroshot-v2` | ~900MB | Slow | Highest accuracy, overkill here |

Small footprint (~180MB vs 5GB for llama3.1:8b) and fast inference made it the practical choice for 41K rows.

## Validation results (30-row varied sample)

Tested on 3 name-mention + 3 non-mention comments per person:

| Person | Name mentioned (avg score) | Not mentioned (avg score) |
|---|---|---|
| ireen | 0.88 | 0.56 |
| femke | 0.80 | 0.62 |
| sven | 0.96 | 0.38 |
| jutta | 0.90 | 0.41 |
| kjeld | 0.96 | 0.29 |

Strong separation for sven, jutta, kjeld. Ireen and femke show higher false positive rates, likely because comments in those threads use gendered pronouns ("she", "her") that the model associates with a specific person even without a name.

## Known limitations

- Ambiguous pronoun references score higher than expected (e.g. "she's big mad" in a Jutta thread scores 0.85)
- General sports/context comments can score high when thematically adjacent to the person
- Recommended threshold: **0.7** to balance precision and recall

## Output

Scores are probabilities between 0 and 1. Use `filter_by_score.py` with `--min` adjusted accordingly (e.g. `--min 0.7`).
