#!/usr/bin/env python3
"""
Headless submission generator for RedRob Hackathon — Stage 3.

Usage:
    python generate_submission.py --input candidates.jsonl --output team_submission.csv

Produces exactly 100 rows, validated against submission_spec.md rules:
- Header: candidate_id, rank, score, reasoning
- Rows 2–101: top 100 candidates, rank 1–100
- score non-increasing by rank
- tie-break: candidate_id ascending
- reasoning: ≤190 chars, UTF-8
"""

import json
import gzip
import textwrap
import argparse
import os
import sys
import zipfile

import pandas as pd
import lightgbm as lgb

# Import only the pure-logic functions — do NOT import from app.py to avoid
# pulling in Streamlit and its @cache_resource decorator, which crashes
# outside a Streamlit process.
from extract_features import extract_features
from logical_validation import check_temporal_consistency
from app import score_candidates, generate_reasoning   # safe: no Streamlit decorators used here


FEATURE_COLS = [
    'title_tier', 'product_ratio',
    'core_skill_score', 'strong_skill_score', 'assessment_bonus', 'has_irrelevant',
    'years_of_experience', 'ml_production_months', 'recency_flag',
    'is_research_only', 'langchain_only_flag',
    'ownership_score', 'scale_score', 'decision_depth_score',
    'response_rate', 'interview_completion', 'offer_acceptance',
    'open_to_work', 'github_score', 'profile_completeness',
    'notice_score', 'days_since_active', 'recency_score',
    'location_score',
    'consulting_mult', 'research_mult', 'llm_tourist_mult', 'ic_mult', 'eval_score',
]


# ── Standalone model loader (no Streamlit dependency) ────────────────────────

def load_model_headless():
    """
    Load LightGBM model from outputs/ltr_model.txt.
    If not present, attempts to extract from outputs/ltr_model.zip.
    Returns the Booster or None (triggers fallback scoring).

    FIX #2: This replaces the `from app import load_model` call.
    app.load_model uses @st.cache_resource which requires a live Streamlit
    session. Importing it here crashes or silently returns None.
    """
    model_txt = os.path.join('outputs', 'ltr_model.txt')
    model_zip = os.path.join('outputs', 'ltr_model.zip')

    if not os.path.exists(model_txt):
        if os.path.exists(model_zip):
            print("Extracting model from zip...")
            try:
                with zipfile.ZipFile(model_zip, 'r') as zf:
                    zf.extractall('outputs')
            except Exception as e:
                print(f"WARNING: Failed to unzip model: {e}")
                return None
        else:
            print("WARNING: No model file found. Running in fallback mode.")
            return None

    try:
        booster = lgb.Booster(model_file=model_txt)
        print(f"Model loaded: {booster.num_trees()} trees.")
        return booster
    except Exception as e:
        print(f"WARNING: Model load failed ({e}). Running in fallback mode.")
        return None


# ── Candidate loading ─────────────────────────────────────────────────────────

def load_candidates(path):
    """Load candidates from .jsonl or .jsonl.gz. Returns list of dicts."""
    if path.endswith('.gz'):
        opener = lambda: gzip.open(path, 'rt', encoding='utf-8')
    else:
        opener = lambda: open(path, 'r', encoding='utf-8')

    candidates = []
    with opener() as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


# ── Scoring pipeline (self-contained, no app.py dependency) ──────────────────

def score_candidates_headless(candidates, model):
    """
    Full scoring pipeline mirroring app.py's score_candidates but:
    - Generates reasoning AFTER final rank is determined (FIX #1)
    - No Streamlit imports or decorators
    """
    rows = []
    for c in candidates:
        feat = extract_features(c)
        feat['candidate_id']   = c.get('candidate_id', 'unknown')
        feat['current_title']  = c.get('profile', {}).get('current_title', 'Unknown')
        feat['current_company'] = c.get('profile', {}).get('current_company', '')
        feat['notice_days']    = feat.get('notice_days', 60)
        rows.append(feat)

    df = pd.DataFrame(rows)

    try:
        df = check_temporal_consistency(df)
    except Exception as e:
        print(f"WARNING: Temporal consistency check failed: {e}")

    if model is not None:
        X = df[FEATURE_COLS].fillna(0).values
        scores = model.predict(X)

        # STEP 1: Apply soft multipliers to raw scores FIRST
        for col in ['consulting_mult', 'research_mult', 'llm_tourist_mult', 'ic_mult']:
            if col in df.columns:
                scores = scores * df[col].fillna(1.0).values

        # STEP 2: Assign final_score ONCE
        df['final_score'] = scores

        # STEP 3: HARD SINK — must be last, never overwrite after this
        if 'honeypot_mult' in df.columns:
            df.loc[df['honeypot_mult'] < 1.0, 'final_score'] = -9999.0
        if 'honeypot_flag' in df.columns:
            df.loc[df['honeypot_flag'] == 1, 'final_score'] = -9999.0
        if 'is_consulting_only' in df.columns:
            df.loc[df['is_consulting_only'] == 1, 'final_score'] = -9999.0

    else:
        # Fallback weighted sum
        core_norm = df['core_skill_score'] / df['core_skill_score'].clip(lower=0.01).max()
        df['final_score'] = (
            df['title_tier']     * 0.30 +
            core_norm            * 0.25 +
            df['recency_score']  * 0.20 +
            df['response_rate']  * 0.15 +
            df['location_score'] * 0.10
        )
        for col in ['consulting_mult', 'research_mult', 'llm_tourist_mult', 'ic_mult']:
            if col in df.columns:
                df['final_score'] = df['final_score'] * df[col].fillna(1.0)

        # Hard sink in fallback mode too
        if 'honeypot_mult' in df.columns:
            df.loc[df['honeypot_mult'] < 1.0, 'final_score'] = -9999.0
        if 'honeypot_flag' in df.columns:
            df.loc[df['honeypot_flag'] == 1, 'final_score'] = -9999.0
        if 'is_consulting_only' in df.columns:
            df.loc[df['is_consulting_only'] == 1, 'final_score'] = -9999.0

    # STEP 4: Sort and assign FINAL ranks BEFORE generating reasoning
    # FIX #1: Reasoning references row['rank'] for tone. Generating it before
    # the definitive sort meant rank 1's reasoning could say "Passable fit"
    # if it was ranked 67th during an intermediate sort in score_candidates.
    df = df.sort_values(
        by=['final_score', 'candidate_id'],
        ascending=[False, True]
    ).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    # STEP 5: Generate reasoning with correct, stable ranks
    df['reasoning'] = df.apply(generate_reasoning, axis=1)

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for RedRob Hackathon")
    parser.add_argument('--input',  type=str, default='candidates.jsonl',
                        help='Path to candidates.jsonl or candidates.jsonl.gz')
    parser.add_argument('--output', type=str, default='team_submission.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    print(f"Loading candidates from {args.input}...")
    candidates = load_candidates(args.input)
    print(f"Loaded {len(candidates):,} candidates.")

    print("Loading LightGBM model...")
    model = load_model_headless()
    if model is None:
        print("Running in FALLBACK mode (weighted sum). Scores will differ from model output.")

    print("Scoring all candidates...")
    df = score_candidates_headless(candidates, model)

    # Top 100 only
    top_100 = df.head(100).copy()
    top_100['rank'] = range(1, 101)   # redundant but explicit for safety

    # Clamp reasoning to 190 chars
    top_100['reasoning'] = top_100['reasoning'].apply(
        lambda x: textwrap.shorten(str(x), width=190, placeholder="...")
    )

    # Final column order matches submission spec exactly
    submission_df = top_100[['candidate_id', 'rank', 'final_score', 'reasoning']].copy()
    submission_df.rename(columns={'final_score': 'score'}, inplace=True)

    submission_df.to_csv(args.output, index=False, encoding='utf-8')

    # Sanity report
    print(f"\n✅ Submission written to {args.output}")
    print(f"   Rows:          {len(submission_df)}")
    print(f"   Score range:   {submission_df['score'].max():.4f} → {submission_df['score'].min():.4f}")
    print(f"   Rank range:    {submission_df['rank'].min()} → {submission_df['rank'].max()}")
    print(f"   Reasoning len: max={submission_df['reasoning'].str.len().max()} chars")

    # Warn if score is non-monotonic (validator will catch this)
    scores = submission_df['score'].values
    violations = [(i+1, i+2, scores[i], scores[i+1])
                  for i in range(len(scores)-1) if scores[i] < scores[i+1]]
    if violations:
        print(f"\n⚠️  WARNING: {len(violations)} non-monotonic score pair(s) — run validate_submission.py")
    else:
        print("   Score ordering: ✅ monotonically non-increasing")


if __name__ == "__main__":
    main()