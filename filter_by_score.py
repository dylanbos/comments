import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Filter scored comments by relevance score.")
parser.add_argument("--min", type=int, default=7, help="Minimum relevance score (default: 7)")
parser.add_argument("--persoon", type=str, default=None, help="Filter by a specific persoon (optional)")
parser.add_argument("--input", type=str, default="comments_scored.csv")
parser.add_argument("--output", type=str, default=None)
args = parser.parse_args()

df = pd.read_csv(args.input)

filtered = df[df["relevance_score"] >= args.min]
if args.persoon:
    filtered = filtered[filtered["persoon"].str.lower() == args.persoon.lower()]

output_file = args.output or f"comments_filtered_min{args.min}.csv"
filtered.to_csv(output_file, index=False)

print(f"Total comments:   {len(df)}")
print(f"After filtering:  {len(filtered)}")
print(f"Saved to:         {output_file}")
if args.persoon:
    print(f"Persoon filter:   {args.persoon}")
print(f"Min score:        {args.min}")
