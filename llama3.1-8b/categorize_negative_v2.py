import json
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"
INPUT_CSV = "negative_speedskating.csv"
OUTPUT_CSV = "negative_speedskating_categorized_v2.csv"
MAX_WORKERS = 4
CHECKPOINT_EVERY = 50
MAX_RETRIES = 2
REQUEST_TIMEOUT = 120

SKATER_FULL = {
    "sven":   "Sven Kramer (Dutch long-track speed skater)",
    "ireen":  "Ireen Wüst (Dutch long-track speed skater)",
    "jutta":  "Jutta Leerdam (Dutch long-track speed skater)",
    "kjeld":  "Kjeld Nuis (Dutch long-track speed skater)",
    "thomas": "Thomas Krol (Dutch long-track speed skater) — NOT Thomas Müller the footballer",
    "femke":  "Femke Kok (Dutch long-track speed skater)",
}

SYSTEM = """You are a research assistant categorising negative Reddit comments about Dutch speed skaters for an academic thesis.

Each comment has already been confirmed as negative toward the speed skater. Your task is to categorise WHAT the negativity is directed at.

Categories:
- personal: negativity about the skater as a person — their appearance, personality, private life, relationships, character, or behaviour off the ice
- performance: negativity about their athletic results, race times, technique, training, competitive achievements, or career
- mixed: negativity directed at BOTH the person AND their performance
- other: negative in tone but directed at something else entirely — media coverage, event organisation, the audience, another commenter, or unclear

Output ONLY a JSON object, no other text:
{"category": "personal" or "performance" or "mixed" or "other", "confidence": "high" or "medium" or "low", "reason": "short phrase, max 12 words"}
"""

FEW_SHOT = """Here are examples of correct categorisation.

Example 1 (about appearance → personal):
Skater: Jutta Leerdam (Dutch long-track speed skater)
Thread: Jutta Leerdam — Dutch speed skater
Subreddit: r/pics
Comment: She looks like a man honestly
JSON: {"category": "personal", "confidence": "high", "reason": "negative comment about physical appearance"}

Example 2 (about race result → performance):
Skater: Sven Kramer (Dutch long-track speed skater)
Thread: 2022 Winter Olympics — Men's 5000m Speed Skating
Subreddit: r/olympics
Comment: Kramer is way past his peak, should have retired two years ago
JSON: {"category": "performance", "confidence": "high", "reason": "criticises declining athletic performance and career timing"}

Example 3 (appearance + bad result combined → mixed):
Skater: Femke Kok (Dutch long-track speed skater)
Thread: Femke Kok finishes 4th in 500m final
Subreddit: r/Speedskating
Comment: She choked again. Also that outfit is ridiculous.
JSON: {"category": "mixed", "confidence": "high", "reason": "criticises both race result and personal appearance"}

Example 4 (attacking another commenter, not the skater → other):
Skater: Ireen Wüst (Dutch long-track speed skater)
Thread: Ireen Wüst wins gold at fifth consecutive Olympics
Subreddit: r/olympics
Comment: You clearly have no idea what you are talking about, go watch football
JSON: {"category": "other", "confidence": "high", "reason": "bickering with another user, not about the skater"}

Example 5 (criticising media coverage → other):
Skater: Jutta Leerdam (Dutch long-track speed skater)
Thread: Jutta Leerdam is Olympisch kampioen
Subreddit: r/Nederland
Comment: NOS heeft het bevel gekregen de vrouwensport te hypen? Jeetje
JSON: {"category": "other", "confidence": "high", "reason": "criticises media coverage, not the skater personally"}

Now categorise the next comment.
"""

USER_TEMPLATE = """Skater: {skater}
Thread: {thread}
Subreddit: r/{subreddit}
Comment: {comment}

JSON:"""


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": SYSTEM,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 120},
        "format": "json",
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()["response"]


def parse_response(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from: {text[:200]}")


def categorize_comment(args):
    uid, comment, persoon, subreddit, thread_title = args
    skater = SKATER_FULL.get(str(persoon).lower(), persoon)
    prompt = FEW_SHOT + USER_TEMPLATE.format(
        skater=skater,
        thread=str(thread_title)[:300],
        subreddit=subreddit,
        comment=str(comment)[:1500],
    )

    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            raw = call_ollama(prompt)
            parsed = parse_response(raw)
            category = str(parsed.get("category", "")).lower().strip()
            if category not in ("personal", "performance", "mixed", "other"):
                raise ValueError(f"Bad category: {category!r}")
            return {
                "uid": uid,
                "category": category,
                "confidence": str(parsed.get("confidence", "")).lower(),
                "reason": str(parsed.get("reason", ""))[:200],
            }
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))

    return {"uid": uid, "category": "error", "confidence": "", "reason": str(last_err)[:200]}


def main():
    try:
        requests.get("http://localhost:11434/", timeout=3)
    except Exception:
        raise SystemExit("Ollama not reachable. Run `ollama serve` first.")

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
        print("All comments already categorised.")
        return

    args_list = [
        (row["uid"], row["comment"], row["persoon"], row["subreddit"], row["thread_title"])
        for _, row in remaining.iterrows()
    ]

    results = []

    def checkpoint():
        scores_df = pd.DataFrame(results).drop_duplicates(subset="uid", keep="last")
        scored_ids = set(scores_df["uid"].astype(str))
        batch = remaining[remaining["uid"].astype(str).isin(scored_ids)].copy()
        mapping = scores_df.set_index("uid")[["category", "confidence", "reason"]].to_dict("index")
        batch[["category", "confidence", "reason"]] = (
            batch["uid"].astype(str).map(mapping).apply(pd.Series)
        )
        pd.concat([done_df, batch], ignore_index=True).to_csv(OUTPUT_CSV, index=False)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(categorize_comment, a): a for a in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Categorising"):
            results.append(future.result())
            if len(results) % CHECKPOINT_EVERY == 0:
                checkpoint()

    scores_df = pd.DataFrame(results).drop_duplicates(subset="uid", keep="last")
    mapping = scores_df.set_index("uid")[["category", "confidence", "reason"]].to_dict("index")
    remaining[["category", "confidence", "reason"]] = (
        remaining["uid"].astype(str).map(mapping).apply(pd.Series)
    )

    final = pd.concat([done_df, remaining], ignore_index=True)
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Saved to {OUTPUT_CSV}")

    print("\nCategory breakdown:")
    print(final["category"].value_counts())
    print("\nBy gender:")
    counts = final.groupby(["gender", "category"]).size().unstack(fill_value=0)
    print(counts)
    print("\nBy gender (%):")
    print((counts.div(counts.sum(axis=1), axis=0) * 100).round(1))
    print("\nConfidence breakdown:")
    print(final["confidence"].value_counts())


if __name__ == "__main__":
    main()
