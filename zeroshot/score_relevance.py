import pandas as pd
import os
from transformers import pipeline
from tqdm import tqdm

MODEL = "cross-encoder/nli-deberta-v3-small"
INPUT_CSV = "../comments_all.csv"
OUTPUT_CSV = "comments_scored.csv"
BATCH_SIZE = 32
CHECKPOINT_EVERY = 200


def main():
    print(f"Loading model: {MODEL}")
    classifier = pipeline("zero-shot-classification", model=MODEL)

    df = pd.read_csv(INPUT_CSV)

    if os.path.exists(OUTPUT_CSV):
        done_df = pd.read_csv(OUTPUT_CSV)
        done_ids = set(done_df["comment_id"].astype(str))
        print(f"Resuming: {len(done_ids)} done, {len(df) - len(done_ids)} remaining.")
    else:
        done_df = pd.DataFrame()
        done_ids = set()

    remaining = df[~df["comment_id"].astype(str).isin(done_ids)].copy()

    if remaining.empty:
        print("All comments already scored.")
        return

    results = []
    batch_comments = []
    batch_rows = []

    def flush_batch():
        if not batch_rows:
            return
        persons = [str(r["persoon"]) for r in batch_rows]
        labels = [[f"about {p}", f"not about {p}"] for p in persons]

        for i, (comment, row, candidate_labels) in enumerate(zip(batch_comments, batch_rows, labels)):
            out = classifier(comment[:512], candidate_labels=candidate_labels)
            # score = probability of the "about {person}" label
            score = round(out["scores"][out["labels"].index(candidate_labels[0])], 4)
            results.append({"comment_id": row["comment_id"], "relevance_score": score})

        batch_comments.clear()
        batch_rows.clear()

        if len(results) % CHECKPOINT_EVERY == 0:
            save_checkpoint()

    def save_checkpoint():
        scores_df = pd.DataFrame(results)
        scored_ids = set(scores_df["comment_id"].astype(str))
        batch = remaining[remaining["comment_id"].astype(str).isin(scored_ids)].copy()
        batch["relevance_score"] = batch["comment_id"].astype(str).map(
            scores_df.set_index("comment_id")["relevance_score"].to_dict()
        )
        combined = pd.concat([done_df, batch], ignore_index=True)
        combined.to_csv(OUTPUT_CSV, index=False)

    for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Scoring"):
        batch_comments.append(str(row["comment"]))
        batch_rows.append(row)
        if len(batch_rows) >= BATCH_SIZE:
            flush_batch()

    flush_batch()

    scores_df = pd.DataFrame(results)
    remaining["relevance_score"] = remaining["comment_id"].astype(str).map(
        scores_df.set_index("comment_id")["relevance_score"].to_dict()
    )
    final = pd.concat([done_df, remaining], ignore_index=True)
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Saved to {OUTPUT_CSV}")
    print(f"\nScore distribution (binned):")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    print(pd.cut(final["relevance_score"], bins=bins).value_counts().sort_index())


if __name__ == "__main__":
    main()
