import pandas as pd
import requests
import re
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
INPUT_CSV = "comments_all.csv"
OUTPUT_CSV = "llama3.1-8b/comments_scored.csv"
MAX_WORKERS = 4
CHECKPOINT_EVERY = 200


def score_comment(args):
    idx, comment, persoon = args
    prompt = (
        f"How relevant is this comment to {persoon}? "
        f"Score 0-10 where 0=not about {persoon} at all, 10=entirely about {persoon}. "
        f"Reply with a single integer only.\n\nComment: {comment[:400]}"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0, "num_predict": 5}},
            timeout=30,
        )
        text = resp.json()["response"].strip()
        match = re.search(r"\b(\d+)\b", text)
        if match:
            return idx, min(10, max(0, int(match.group(1))))
        return idx, -1
    except Exception:
        return idx, -1


def main():
    df = pd.read_csv(INPUT_CSV)

    # Resume from previous run
    if os.path.exists(OUTPUT_CSV):
        done_df = pd.read_csv(OUTPUT_CSV)
        done_ids = set(done_df["comment_id"].astype(str))
        print(f"Resuming: {len(done_ids)} already scored, {len(df) - len(done_ids)} remaining.")
    else:
        done_df = pd.DataFrame()
        done_ids = set()

    remaining = df[~df["comment_id"].astype(str).isin(done_ids)].copy()

    if remaining.empty:
        print("All comments already scored.")
        return

    args_list = [
        (row["comment_id"], str(row["comment"]), str(row["persoon"]))
        for _, row in remaining.iterrows()
    ]

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(score_comment, a): a for a in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
            comment_id, score = future.result()
            results.append({"comment_id": comment_id, "relevance_score": score})

            if len(results) % CHECKPOINT_EVERY == 0:
                scores_df = pd.DataFrame(results)
                batch = remaining[remaining["comment_id"].astype(str).isin(
                    scores_df["comment_id"].astype(str)
                )].copy()
                batch["relevance_score"] = batch["comment_id"].astype(str).map(
                    scores_df.set_index("comment_id")["relevance_score"].to_dict()
                )
                combined = pd.concat([done_df, batch], ignore_index=True)
                combined.to_csv(OUTPUT_CSV, index=False)

    # Final save
    scores_df = pd.DataFrame(results)
    remaining["relevance_score"] = remaining["comment_id"].astype(str).map(
        scores_df.set_index("comment_id")["relevance_score"].to_dict()
    )
    final = pd.concat([done_df, remaining], ignore_index=True)
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Saved to {OUTPUT_CSV}")
    print(f"Score distribution:\n{final['relevance_score'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
