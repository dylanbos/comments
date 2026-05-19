import pandas as pd
import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
INPUT_CSV = "negative_speedskating.csv"
OUTPUT_CSV = "negative_speedskating_categorized.csv"
MAX_WORKERS = 4
CHECKPOINT_EVERY = 50

CATEGORIES = ("personal", "performance", "mixed", "other")


def categorize_comment(args):
    uid, comment, persoon, subreddit, thread_title = args
    prompt = (
        "Categorize the negativity in this Reddit comment about a Dutch speed skater.\n\n"
        f"Subreddit: r/{subreddit}\n"
        f"Thread: {thread_title}\n\n"
        "Choose exactly one category:\n"
        "- personal: negativity about the athlete as a person (appearance, personality, personal life, character)\n"
        "- performance: negativity about their athletic results, technique, race times, or competitive achievements\n"
        "- mixed: negativity about both the person and their performance\n"
        "- other: negative but about something else (media coverage, event, audience, etc.)\n\n"
        "Reply with only valid JSON, no explanation: "
        "{\"category\": \"personal|performance|mixed|other\"}\n\n"
        f"Comment: {comment[:400]}"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0, "num_predict": 30}},
            timeout=30,
        )
        raw = resp.json()["response"].strip()
        parsed = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        category = parsed.get("category", "").lower()
        if category not in CATEGORIES:
            category = None
        return uid, category
    except Exception:
        return uid, None


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Input rows: {len(df)}")

    if os.path.exists(OUTPUT_CSV):
        done_df = pd.read_csv(OUTPUT_CSV)
        done_ids = set(done_df["uid"].astype(str))
        print(f"Resuming: {len(done_ids)} done, {len(df) - len(done_ids)} remaining.")
    else:
        done_df = pd.DataFrame()
        done_ids = set()

    remaining = df[~df["uid"].astype(str).isin(done_ids)].copy()

    if remaining.empty:
        print("All comments already categorized.")
        return

    args_list = [
        (row["uid"], str(row["comment"]), str(row["persoon"]),
         str(row["subreddit"]), str(row["thread_title"]))
        for _, row in remaining.iterrows()
    ]

    results = []

    def checkpoint():
        scores_df = pd.DataFrame(results).drop_duplicates(subset="uid", keep="last")
        scored_ids = set(scores_df["uid"].astype(str))
        batch = remaining[remaining["uid"].astype(str).isin(scored_ids)].copy()
        mapping = scores_df.set_index("uid")["category"].to_dict()
        batch["category"] = batch["uid"].astype(str).map(mapping)
        pd.concat([done_df, batch], ignore_index=True).to_csv(OUTPUT_CSV, index=False)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(categorize_comment, a): a for a in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Categorizing"):
            uid, category = future.result()
            results.append({"uid": uid, "category": category})
            if len(results) % CHECKPOINT_EVERY == 0:
                checkpoint()

    scores_df = pd.DataFrame(results).drop_duplicates(subset="uid", keep="last")
    mapping = scores_df.set_index("uid")["category"].to_dict()
    remaining["category"] = remaining["uid"].astype(str).map(mapping)

    final = pd.concat([done_df, remaining], ignore_index=True)
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Saved to {OUTPUT_CSV}")

    print("\nCategory breakdown:")
    print(final["category"].value_counts())
    print("\nBy gender:")
    print(final.groupby(["gender", "category"]).size().unstack(fill_value=0))
    print("\nBy gender (%):")
    counts = final.groupby(["gender", "category"]).size().unstack(fill_value=0)
    print(counts.div(counts.sum(axis=1), axis=0).round(3) * 100)


if __name__ == "__main__":
    main()
