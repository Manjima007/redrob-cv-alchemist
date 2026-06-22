import pandas as pd
import numpy as np
from datetime import datetime, date

REFERENCE_DATE = date(2026, 6, 20)

def check_temporal_consistency(df):
    """Vectorized honeypot detector. Returns honeypot_multiplier column (0.0 = honeypot, 1.0 = valid)."""
    df['honeypot_mult'] = 1.0

    # Coerce temporal columns safely
    for col in ['earliest_job_start', 'company_founded_date', 'job_start_date']:
        if col in df.columns:
            df[col + '_dt'] = pd.to_datetime(df[col], errors='coerce')

    # Check 1: years_of_experience vs earliest job start date
    if 'years_of_experience' in df.columns and 'earliest_job_start_dt' in df.columns:
        mask = df['earliest_job_start_dt'].notna()
        years_since = ((pd.to_datetime(REFERENCE_DATE) - df.loc[mask, 'earliest_job_start_dt']).dt.days / 365.25)
        invalid = mask & (df['years_of_experience'] >  years_since)
        df.loc[invalid, 'honeypot_mult'] = 0.0

    # Check 2: company_founded > job_start_date (impossible)
    if 'company_founded_date_dt' in df.columns and 'job_start_date_dt' in df.columns:
        both = df['company_founded_date_dt'].notna() & df['job_start_date_dt'].notna()
        impossible = df.loc[both, 'company_founded_date_dt'] > df.loc[both, 'job_start_date_dt']
        df.loc[both & impossible, 'honeypot_mult'] = 0.0

    # Check 3: expert skills with zero duration
    if 'expert_zero_exp' in df.columns:
        df.loc[df['expert_zero_exp'].fillna(0) > 0, 'honeypot_mult'] = 0.0
    
    # Cross-field contradiction checks
    #  - Very high assessment score but almost no experience
    if 'skill_assessments_max' in df.columns and 'years_of_experience' in df.columns:
        suspect = (df['skill_assessments_max'] >= 80.0) & (df['years_of_experience'] < 1.0)
        df.loc[suspect, 'honeypot_mult'] = 0.0

    #  - High GitHub activity but negligible experience
    if 'github_score' in df.columns and 'years_of_experience' in df.columns:
        gh_suspect = (df['github_score'] >= 80.0) & (df['years_of_experience'] < 0.5)
        df.loc[gh_suspect, 'honeypot_mult'] = 0.0

    #  - Implausible salary expectations for low title tier/experience -> reduce multiplier
    if 'expected_salary_min' in df.columns and 'title_tier' in df.columns and 'years_of_experience' in df.columns:
        salary_flag = (df['expected_salary_min'] > 30.0) & (df['title_tier'] < 0.5) & (df['years_of_experience'] < 3.0)
        df.loc[salary_flag, 'honeypot_mult'] = df.loc[salary_flag, 'honeypot_mult'] * 0.2

    return df
