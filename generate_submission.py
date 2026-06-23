import json
import pandas as pd
import textwrap
import os
import argparse
from app import score_candidates, load_model # Re-use your existing logic

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for RedRob Hackathon")
    parser.add_argument('--input', type=str, default='candidates.jsonl', help='Path to candidates.jsonl')
    parser.add_argument('--output', type=str, default='team_submission.csv', help='Output CSV path')
    args = parser.parse_args()

    print(f"Loading candidates from {args.input}...")
    candidates = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))

    print("Loading LightGBM model...")
    model = load_model()
    if model is None:
        print("WARNING: Model not found. Running in fallback mode.")

    print("Scoring candidates...")
    df = score_candidates(candidates, model)

    # STRICT SORTING: Score descending, candidate_id ascending (for deterministic tie-breaks)
    df = df.sort_values(by=['final_score', 'candidate_id'], ascending=[False, True]).reset_index(drop=True)

    # EXACTLY 100 ROWS
    top_100 = df.head(100).copy()
    top_100['rank'] = range(1, 101)

    # FORMAT REASONING (Fixing the slice bug)
    top_100['reasoning'] = top_100['reasoning'].apply(
        lambda x: textwrap.shorten(str(x), width=190, placeholder="...")
    )

    # EXPORT STRICT COLUMNS
    submission_df = top_100[['candidate_id', 'rank', 'final_score', 'reasoning']].copy()
    submission_df.rename(columns={'final_score': 'score'}, inplace=True)
    
    submission_df.to_csv(args.output, index=False, encoding='utf-8')
    print(f"✅ Submission generated successfully at {args.output}")

if __name__ == "__main__":
    main()