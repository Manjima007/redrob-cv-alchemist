import streamlit as st
import json
import pandas as pd
import lightgbm as lgb
import os
import sys
import zipfile

sys.path.append(os.path.dirname(__file__))
from extract_features import extract_features
from logical_validation import check_temporal_consistency

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


def generate_reasoning(row):
    yoe   = float(row['years_of_experience'])
    tier  = float(row['title_tier'])
    css   = float(row['core_skill_score'])
    sss   = float(row['strong_skill_score'])
    ml_m  = int(row['ml_production_months'])
    rr    = float(row['response_rate'])
    gh    = float(row['github_score'])
    loc   = float(row['location_score'])
    nd    = int(row['notice_days'])
    own   = float(row['ownership_score'])
    scl   = float(row['scale_score'])
    dd    = float(row['decision_depth_score'])
    rec   = float(row['recency_score'])
    otw   = int(row['open_to_work'])
    rf    = int(row['recency_flag'])
    score = float(row['final_score'])
    rank  = int(row['rank'])

    # Opening — who is this person
    if tier == 1.0:
        opening = f"{yoe:.0f}-year AI/ML engineer"
    elif tier == 0.75:
        opening = f"{yoe:.0f}-year software/data engineer with ML exposure"
    else:
        opening = f"{yoe:.0f}-year engineer (adjacent background)"

    # Core strength
    strength = ""
    if css > 1.0:
        strength = "strong retrieval and ranking background"
    elif css > 0.4:
        strength = "solid core AI skills (retrieval/ranking)"
    elif sss > 0.5:
        strength = "good ML fundamentals without deep retrieval specialization"
    elif ml_m > 12:
        strength = f"{ml_m} months of ML production experience"

    # Production signal
    prod = ""
    if rf and scl > 0.3:
        prod = "shipped ML systems to production at scale"
    elif rf:
        prod = "recent hands-on ML production experience"
    elif ml_m > 24:
        prod = f"{ml_m} months of ML work across career"

    # Ownership signal
    ownership = ""
    if own > 0.7 and dd > 0.4:
        ownership = "descriptions show ownership and architectural decision-making"
    elif own > 0.7:
        ownership = "ownership-oriented career narrative"
    elif dd > 0.5:
        ownership = "evidence of technical decision-making"

    # Availability
    availability = ""
    if rec > 0.8 and rr > 0.75:
        availability = f"actively engaged (response rate {rr:.0%})"
    elif otw:
        availability = "explicitly open to work"
    elif rec > 0.5:
        availability = "recently active on platform"

    # Location
    loc_str = ""
    if loc >= 1.0:
        loc_str = "Pune/Noida-based"
    elif loc >= 0.85:
        loc_str = "Tier-1 Indian city"
    elif loc >= 0.6:
        loc_str = "India-based"
    else:
        loc_str = "outside India"

    # Concerns
    concerns = []
    if nd > 90:
        concerns.append(f"long notice period ({nd} days)")
    elif nd > 30:
        concerns.append(f"notice period {nd} days")
    if rec < 0.3:
        concerns.append("inactive on platform 90+ days")
    if int(row.get('is_research_only', 0)):
        concerns.append("research-only background")
    if loc < 0.6:
        concerns.append("outside India")
    if rank > 70:
        concerns.append("adjacent fit — included on engagement signals")

    # Assemble — vary structure so strings differ
    parts = [p for p in [opening, strength, prod, ownership, availability, loc_str] if p]

    result = '; '.join(parts)
    if concerns:
        result += '. Concern: ' + ', '.join(concerns)

    # Trim to 200 chars, ensure uniqueness via score suffix if needed
    result = result[:190]
    if len(result) < 15:
        result = f"{yoe:.0f}yr engineer; score {score:.4f}"

    return result


@st.cache_resource
def load_model():
    model_txt = 'outputs/ltr_model.txt'
    model_zip = 'outputs/ltr_model.zip'
    
    # If the text file doesn't exist, try to extract it from the zip
    if not os.path.exists(model_txt):
        if os.path.exists(model_zip):
            try:
                with zipfile.ZipFile(model_zip, 'r') as zip_ref:
                    # Extract contents into the outputs folder
                    zip_ref.extractall('outputs')
            except Exception as e:
                st.error(f"Failed to unzip model: {e}")
                return None
        else:
            return None

    # Load the model directly from the extracted text file
    try:
        return lgb.Booster(model_file=model_txt)
    except Exception as e:
        st.error(f"Failed to load LightGBM model. The file might still be corrupted: {e}")
        return None


def score_candidates(candidates, model):
    rows = []
    for c in candidates:
        feat = extract_features(c)
        feat['candidate_id'] = c.get('candidate_id', 'unknown')
        feat['current_title'] = c.get('profile', {}).get('current_title', 'Unknown')
        feat['current_company'] = c.get('profile', {}).get('current_company', '')
        feat['years_of_experience'] = feat['years_of_experience']
        rows.append(feat)

    df = pd.DataFrame(rows)

    # Run vectorized logical consistency checks (adds 'honeypot_mult')
    try:
        df = check_temporal_consistency(df)
    except Exception:
        # keep going if logical validation isn't applicable
        pass

    if model is not None:
        X = df[FEATURE_COLS].fillna(0).values
        scores = model.predict(X)
        # Apply post-prediction multipliers from disqualifier detectors and logical checks
        for col in ['honeypot_mult', 'consulting_mult', 'research_mult', 'llm_tourist_mult', 'ic_mult']:
            if col in df.columns:
                scores = scores * df[col].fillna(1.0).values
        # maintain legacy binary flags as additional safety
        if 'honeypot_flag' in df.columns:
            scores = scores * (1 - df['honeypot_flag'].fillna(0).values)
        if 'is_consulting_only' in df.columns:
            scores = scores * (1 - df['is_consulting_only'].fillna(0).values)
        df['final_score'] = scores
    else:
        # Fallback: simple weighted sum if no model trained yet
        core_norm = df['core_skill_score'] / df['core_skill_score'].clip(lower=0.01).max()
        df['final_score'] = (
            df['title_tier']    * 0.30 +
            core_norm           * 0.25 +
            df['recency_score'] * 0.20 +
            df['response_rate'] * 0.15 +
            df['location_score']* 0.10
        )
        # Apply multiplier fields if present
        for col in ['honeypot_mult', 'consulting_mult', 'research_mult', 'llm_tourist_mult', 'ic_mult']:
            if col in df.columns:
                df['final_score'] = df['final_score'] * df[col].fillna(1.0)
        if 'honeypot_flag' in df.columns:
            df['final_score'] = df['final_score'] * (1 - df['honeypot_flag'].fillna(0))
        if 'is_consulting_only' in df.columns:
            df['final_score'] = df['final_score'] * (1 - df['is_consulting_only'].fillna(0))

    df = df.sort_values('final_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    df['reasoning'] = df.apply(generate_reasoning, axis=1)
    return df


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="CV Alchemist — RedRob Ranker", layout="wide")

st.title("CV Alchemist")
st.caption("AI-powered candidate ranker for Senior AI Engineer — RedRob Hackathon")

model = load_model()
if model is None:
    st.warning("No trained model found at outputs/ltr_model.txt. Running in fallback scoring mode.")
else:
    st.success("Model loaded. Running full LightGBM LambdaRank scoring.")

st.markdown("---")

# Upload or use sample
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader(
        "Upload candidates JSON file (array of candidate objects)",
        type=['json']
    )
with col2:
    use_sample = st.button("Use sample_candidates.json")

candidates = []

if uploaded:
    try:
        candidates = json.load(uploaded)
        if isinstance(candidates, dict):
            candidates = [candidates]
        st.success(f"Loaded {len(candidates)} candidates from upload.")
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")

elif use_sample:
    sample_path = 'outputs/sample_candidates.json'
    if os.path.exists(sample_path):
        with open(sample_path) as f:
            candidates = json.load(f)
        st.success(f"Loaded {len(candidates)} candidates from sample file.")
    else:
        st.error("data/sample_candidates.json not found.")

if candidates:
    with st.spinner("Scoring candidates..."):
        df = score_candidates(candidates, model)

    top_n = min(len(df), 20)
    st.subheader(f"Top {top_n} Candidates")

    display_cols = ['rank', 'candidate_id', 'current_title', 'current_company',
                    'years_of_experience', 'final_score', 'reasoning']
    show_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(
        df[show_cols].head(top_n).style.format({'final_score': '{:.4f}'}),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Score Breakdown for Top 5")

    breakdown_cols = ['rank', 'current_title', 'title_tier', 'core_skill_score',
                      'ownership_score', 'scale_score', 'decision_depth_score',
                      'recency_score', 'response_rate', 'final_score']
    show_breakdown = [c for c in breakdown_cols if c in df.columns]
    st.dataframe(
        df[show_breakdown].head(5).style.format({c: '{:.3f}' for c in show_breakdown if c != 'rank' and c != 'current_title'}),
        use_container_width=True
    )

    st.subheader("Honeypot / Disqualified Candidates")
    dq_mask = pd.Series(False, index=df.index)
    if 'honeypot_mult' in df.columns:
        dq_mask = dq_mask | (df['honeypot_mult'] < 1.0)
    if 'consulting_mult' in df.columns:
        dq_mask = dq_mask | (df['consulting_mult'] < 1.0)
    if 'honeypot_flag' in df.columns:
        dq_mask = dq_mask | (df['honeypot_flag'] == 1)
    if 'is_consulting_only' in df.columns:
        dq_mask = dq_mask | (df['is_consulting_only'] == 1)
    disqualified = df[dq_mask]
    if len(disqualified) > 0:
        st.warning(f"{len(disqualified)} candidates disqualified (honeypot or consulting-only)")
        dq_cols = ['candidate_id', 'current_title', 'honeypot_flag', 'is_consulting_only']
        show_dq = [c for c in dq_cols if c in disqualified.columns]
        st.dataframe(disqualified[show_dq], use_container_width=True)
    else:
        st.success("No honeypots or consulting-only candidates detected.")

    # Download button
    out_df = df[['candidate_id', 'rank', 'final_score', 'reasoning']].copy()
    out_df.columns = ['candidate_id', 'rank', 'score', 'reasoning']
    csv_data = out_df.head(100).to_csv(index=False)
    st.download_button(
        label="Download submission.csv (top 100)",
        data=csv_data,
        file_name="submission.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("CV Alchemist | RedRob Data & AI Challenge | LightGBM LambdaRank + Free-text Feature Engineering")