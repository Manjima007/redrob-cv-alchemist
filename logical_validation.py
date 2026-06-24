import pandas as pd
import numpy as np
from datetime import datetime, date

REFERENCE_DATE = date(2026, 6, 20)


def check_temporal_consistency(df):
    """
    Vectorized honeypot detector.
    Returns df with 'honeypot_mult' column: 0.0 = honeypot, 1.0 = valid.

    Design contract:
    - honeypot_flag (from extract_features) handles per-candidate rule-based checks.
    - This function handles CROSS-FIELD temporal contradictions that require
      the full dataframe to detect (e.g. date arithmetic across multiple columns).
    - Do NOT duplicate threshold logic that already lives in extract_features.
    """
    df = df.copy()
    df['honeypot_mult'] = 1.0

    # Coerce temporal columns safely
    for col in ['earliest_job_start', 'company_founded_date', 'job_start_date']:
        if col in df.columns:
            df[col + '_dt'] = pd.to_datetime(df[col], errors='coerce')

    # ── Check 1: years_of_experience vs earliest job start date ──────────────
    # FIX #2: Added 1.5-year grace period to avoid false-positives.
    # Candidates often self-report YoE including internships, freelance, or
    # part-time work that pre-dates their first formal listed role.
    if 'years_of_experience' in df.columns and 'earliest_job_start_dt' in df.columns:
        mask = df['earliest_job_start_dt'].notna()
        years_since = (
            (pd.to_datetime(REFERENCE_DATE) - df.loc[mask, 'earliest_job_start_dt']).dt.days
            / 365.25
        )
        # Only flag truly impossible gaps (> 18 months over-claim), not rounding differences
        invalid = mask & (df['years_of_experience'] > years_since + 1.5)
        df.loc[invalid, 'honeypot_mult'] = 0.0

    # ── Check 2: company_founded_date > job_start_date (impossible) ───────────
    if 'company_founded_date_dt' in df.columns and 'job_start_date_dt' in df.columns:
        both      = df['company_founded_date_dt'].notna() & df['job_start_date_dt'].notna()
        impossible = df.loc[both, 'company_founded_date_dt'] > df.loc[both, 'job_start_date_dt']
        df.loc[both & impossible, 'honeypot_mult'] = 0.0

    # ── Check 3: expert_zero_exp — REMOVED ───────────────────────────────────
    # FIX #1: Threshold logic (>= 5 fake-expert skills) already enforced in
    # extract_features.py via honeypot_flag. Duplicating it here with a more
    # aggressive threshold (> 0) caused massive false-positive rates.
    # The honeypot_flag column from extract_features is applied as a hard sink
    # in score_candidates — no need to re-implement here.

    # ── Cross-field contradiction checks ─────────────────────────────────────

    # High assessment score but almost no experience.
    # skill_assessments_max is stored raw (0–100 scale).
    if 'skill_assessments_max' in df.columns and 'years_of_experience' in df.columns:
        suspect = (
            (df['skill_assessments_max'] >= 80.0) &
            (df['years_of_experience'] < 1.0)
        )
        df.loc[suspect, 'honeypot_mult'] = 0.0

    # High GitHub activity but negligible experience.
    # FIX #5: github_score is normalized 0–1 in extract_features, so threshold
    # must be 0.80 (not 80.0).
    if 'github_score' in df.columns and 'years_of_experience' in df.columns:
        gh_suspect = (
            (df['github_score'] >= 0.80) &      # was 80.0 — wrong scale
            (df['years_of_experience'] < 0.5)
        )
        df.loc[gh_suspect, 'honeypot_mult'] = 0.0

    # Implausible salary expectations: high ask, low tier, little experience.
    # Using a soft 0.2 multiplier (not hard zero) — salary alone isn't definitive.
    if ('expected_salary_min' in df.columns and
            'title_tier' in df.columns and
            'years_of_experience' in df.columns):
        salary_flag = (
            (df['expected_salary_min'] > 30.0) &
            (df['title_tier'] < 0.5) &
            (df['years_of_experience'] < 3.0)
        )
        df.loc[salary_flag, 'honeypot_mult'] = df.loc[salary_flag, 'honeypot_mult'] * 0.2

    return df