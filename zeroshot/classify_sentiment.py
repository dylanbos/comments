import pandas as pd
import os
from transformers import pipeline
from tqdm import tqdm

MODEL = "cross-encoder/nli-deberta-v3-small"
INPUT_CSV = "../comments_all.csv"
OUTPUT_CSV = "comments_classified.csv"
CHECKPOINT_EVERY = 50
ROW_START = None
ROW_END = None

SUBREDDIT_ALLOWLIST = {
    # Speed skating & Olympics
    "Speedskating", "speedskatebabes", "JuttaLeerdam", "WinterOlympics2026",
    "MilanOlympics", "OlympicsV2", "olympics", "OlympicBornToday",
    "BeautifulOlympians", "HottestWinterAthletes", "dutchwomanathletes",
    "sports", "nextfuckinglevel", "Damnthatsinteresting", "interestingasfuck",
    "gifs", "coolguides", "Infographics",
    # Dutch media & culture
    "NUjijDiscussies", "Nederland", "thenetherlands", "Netherlands",
    "Politiek", "PolitiekeMemes", "cirkeltrek", "taalfout", "DeStagiair",
    "tokkiefeesboek", "papgrappen", "HLNFails", "NLCelebs",
    # Grey area: gossip & attractiveness
    "LAinfluencersnark", "sportsgossips", "ladyladyboners",
    "HottestFemaleAthletes", "ik_ihe", "pics",
}

GENERAL_SUBREDDITS = {
    "coolguides", "Infographics", "Damnthatsinteresting",
    "interestingasfuck", "nextfuckinglevel", "gifs", "sports",
}

LABELS = [
    "about speed skating",
    "discusses athletic results, race times, medals, or competitive performance",
    "negative or critical toward the athlete or their performance",
]

THRESHOLD = 0.5


def classify_comment(classifier, comment_id, comment, subreddit, thread_title):
    text = f"Subreddit: r/{subreddit}. Thread: {thread_title}. Comment: {comment[:400]}"
    try:
        out = classifier(text, candidate_labels=LABELS, multi_label=True)
        scores = dict(zip(out["labels"], out["scores"]))
        return (
            comment_id,
            scores[LABELS[0]] >= THRESHOLD,
            scores[LABELS[1]] >= THRESHOLD,
            scores[LABELS[2]] >= THRESHOLD,
        )
    except Exception:
        return comment_id, None, None, None


def main():
    print(f"Loading model: {MODEL}")
    classifier = pipeline("zero-shot-classification", model=MODEL)

    df = pd.read_csv(INPUT_CSV)
    df = df[df["subreddit"].isin(SUBREDDIT_ALLOWLIST)]

    general_mask = df["subreddit"].isin(GENERAL_SUBREDDITS)
    name_mentioned = df.apply(
        lambda r: str(r["persoon"]).lower() in str(r["comment"]).lower(), axis=1
    )
    df = df[~general_mask | name_mentioned]

    if ROW_START is not None or ROW_END is not None:
        df = df.iloc[ROW_START:ROW_END]

    print(f"Rows after subreddit filter: {len(df)}")

    if os.path.exists(OUTPUT_CSV):
        done_df = pd.read_csv(OUTPUT_CSV)
        done_ids = set(done_df["comment_id"].astype(str))
        print(f"Resuming: {len(done_ids)} done, {len(df) - len(done_ids)} remaining.")
    else:
        done_df = pd.DataFrame()
        done_ids = set()

    remaining = df[~df["comment_id"].astype(str).isin(done_ids)].copy()

    if remaining.empty:
        print("All comments already classified.")
        return

    results = []

    def checkpoint():
        scores_df = pd.DataFrame(results).drop_duplicates(subset="comment_id", keep="last")
        scored_ids = set(scores_df["comment_id"].astype(str))
        batch = remaining[remaining["comment_id"].astype(str).isin(scored_ids)].copy()
        mapping = scores_df.set_index("comment_id")[["speedskating", "performance", "negative"]].to_dict("index")
        batch[["speedskating", "performance", "negative"]] = batch["comment_id"].astype(str).map(mapping).apply(pd.Series)
        combined = pd.concat([done_df, batch], ignore_index=True)
        combined.to_csv(OUTPUT_CSV, index=False)

    for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Classifying"):
        comment_id, speedskating, performance, negative = classify_comment(
            classifier,
            row["comment_id"],
            str(row["comment"]),
            str(row["subreddit"]),
            str(row["thread_title"]),
        )
        results.append({
            "comment_id": comment_id,
            "speedskating": speedskating,
            "performance": performance,
            "negative": negative,
        })
        if len(results) % CHECKPOINT_EVERY == 0:
            checkpoint()

    scores_df = pd.DataFrame(results).drop_duplicates(subset="comment_id", keep="last")
    mapping = scores_df.set_index("comment_id")[["speedskating", "performance", "negative"]].to_dict("index")
    remaining[["speedskating", "performance", "negative"]] = remaining["comment_id"].astype(str).map(mapping).apply(pd.Series)

    final = pd.concat([done_df, remaining], ignore_index=True)
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Saved to {OUTPUT_CSV}")

    relevant = final[
        (final["speedskating"] == True) &
        (final["performance"] == True) &
        (final["negative"] == True)
    ]
    print(f"\nNegative speed skating performance comments: {len(relevant)}")
    print("\nBy gender:")
    print(relevant["gender"].value_counts())


if __name__ == "__main__":
    main()
