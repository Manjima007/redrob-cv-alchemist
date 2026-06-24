import streamlit as st
import json
import pandas as pd
import lightgbm as lgb
import os
import sys
import zipfile
import textwrap

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


# ── Reasoning generator ───────────────────────────────────────────────────────

def generate_reasoning(row):
    yoe  = float(row['years_of_experience'])
    tier = float(row['title_tier'])
    css  = float(row['core_skill_score'])
    sss  = float(row['strong_skill_score'])
    loc  = float(row['location_score'])
    nd   = int(row['notice_days'])
    own  = float(row['ownership_score'])
    rec  = float(row['recency_score'])
    rank = int(row['rank'])

    # 1. Persona
    if tier == 1.0:
        persona = f"{yoe:.0f}-year AI/ML engineer"
    elif tier == 0.75:
        persona = f"{yoe:.0f}-year software/data engineer"
    else:
        persona = f"{yoe:.0f}-year engineer (adjacent background)"

    # 2. Top 1-2 strongest signals
    positives = []
    if css > 0.8:
        positives.append("deep expertise in retrieval and ranking")
    elif sss > 0.5:
        positives.append("solid ML production fundamentals")

    if own > 0.7:
        positives.append("strong architectural ownership")
    elif rec > 0.8:
        positives.append("high platform engagement")

    if not positives:
        positives.append("transferable technical skills")

    highlights = " and ".join(positives[:2])

    # 3. Tone by rank
    if rank <= 15:
        base = (f"Exceptional {persona} with {highlights}. "
                f"Highly engaged and fits the 'product over research' JD profile perfectly.")
    elif rank <= 60:
        base = (f"Strong {persona} offering {highlights}. "
                f"Good overall fit for our stack and production needs.")
    else:
        base = (f"Passable {persona} with {highlights}. "
                f"Lacks deep AI specialization but included as filler due to solid behavioral signals.")

    # 4. One honest concern
    if nd > 60:
        return textwrap.shorten(
            f"{base} Note: Long notice period ({nd} days).", width=190, placeholder="..."
        )
    elif loc < 0.6:
        return textwrap.shorten(
            f"{base} Note: Located outside target Indian hubs.", width=190, placeholder="..."
        )

    return textwrap.shorten(base, width=190, placeholder="...")


# ── Model loader ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model_txt = 'outputs/ltr_model.txt'
    model_zip = 'outputs/ltr_model.zip'

    if not os.path.exists(model_txt):
        if os.path.exists(model_zip):
            try:
                with zipfile.ZipFile(model_zip, 'r') as zip_ref:
                    zip_ref.extractall('outputs')
            except Exception as e:
                st.error(f"Failed to unzip model: {e}")
                return None
        else:
            return None

    try:
        return lgb.Booster(model_file=model_txt)
    except Exception as e:
        st.error(f"Failed to load LightGBM model: {e}")
        return None


# ── Scoring pipeline ──────────────────────────────────────────────────────────

def score_candidates(candidates, model):
    rows = []
    for c in candidates:
        feat = extract_features(c)
        feat['candidate_id']    = c.get('candidate_id', 'unknown')
        feat['current_title']   = c.get('profile', {}).get('current_title', 'Unknown')
        feat['current_company'] = c.get('profile', {}).get('current_company', '')
        rows.append(feat)

    df = pd.DataFrame(rows)

    try:
        df = check_temporal_consistency(df)
    except Exception:
        pass

    if model is not None:
        X = df[FEATURE_COLS].fillna(0).values
        scores = model.predict(X)

        # STEP 1: Apply soft multipliers to raw scores first
        for col in ['consulting_mult', 'research_mult', 'llm_tourist_mult', 'ic_mult']:
            if col in df.columns:
                scores = scores * df[col].fillna(1.0).values

        # STEP 2: Assign final_score exactly once
        df['final_score'] = scores

        # STEP 3: Hard sink — must be last, nothing overwrites after this
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

        # Hard sink in fallback too
        if 'honeypot_mult' in df.columns:
            df.loc[df['honeypot_mult'] < 1.0, 'final_score'] = -9999.0
        if 'honeypot_flag' in df.columns:
            df.loc[df['honeypot_flag'] == 1, 'final_score'] = -9999.0
        if 'is_consulting_only' in df.columns:
            df.loc[df['is_consulting_only'] == 1, 'final_score'] = -9999.0

    # Sort and assign rank BEFORE generating reasoning (rank drives tone)
    df = df.sort_values('final_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    df['reasoning'] = df.apply(generate_reasoning, axis=1)
    return df


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CV Alchemist | RedRob",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔮 CV Alchemist")
    st.caption("AI-powered candidate ranker · Senior AI Engineer")
    st.markdown("---")

    model = load_model()
    if model is None:
        st.error("⚠️ No model found. Running in fallback mode.")
    else:
        st.success("✨ LambdaRank model active")

    st.markdown("---")
    st.subheader("Data input")

    uploaded = st.file_uploader(
        "Upload candidates (JSON / JSONL)",
        type=['json', 'jsonl']
    )
    st.markdown("**or**")
    use_sample = st.button("Use sample_candidates.json", use_container_width=True)

    st.markdown("---")
    st.subheader("Display filters")
    tier1_only = st.checkbox("Tier 1 AI/ML titles only", value=False)
    india_only = st.checkbox("India-based candidates only", value=False)

    st.markdown("---")
    st.caption("RedRob Data & AI Challenge")


# ── Main canvas ───────────────────────────────────────────────────────────────

st.title("Candidate Rankings")
st.caption("Senior AI/ML Engineer · LambdaRank model · 100K candidate pool")

candidates = []

if uploaded:
    try:
        raw = uploaded.read().decode('utf-8')
        # Support both plain JSON array and JSONL
        if uploaded.name.endswith('.jsonl'):
            candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            parsed = json.loads(raw)
            candidates = parsed if isinstance(parsed, list) else [parsed]
    except Exception as e:
        st.error(f"Failed to parse file: {e}")

elif use_sample:
    sample_path = 'outputs/sample_candidates.json'
    if os.path.exists(sample_path):
        with open(sample_path) as f:
            candidates = json.load(f)
    else:
        st.error(f"{sample_path} not found.")

# ── Processing & display ──────────────────────────────────────────────────────

if candidates:
    with st.spinner("Alchemizing candidate scores..."):
        df = score_candidates(candidates, model)

    # Split into clean (qualified) and disqualified pools
    # Using -9999.0 as the definitive hard-sink threshold.
    # dq_mask based on final_score only — not multiplier columns which
    # include soft penalties (consulting_mult=0.85) that are NOT disqualifiers.
    dq_mask    = df['final_score'] <= -9999.0
    clean_df   = df[~dq_mask].reset_index(drop=True)
    dq_df      = df[dq_mask].reset_index(drop=True)
    dq_count   = len(dq_df)

    # Apply display-side filters (does not affect submission CSV)
    filtered_df = clean_df.copy()
    if tier1_only:
        filtered_df = filtered_df[filtered_df['title_tier'] == 1.0].reset_index(drop=True)
    if india_only:
        filtered_df = filtered_df[filtered_df['location_score'] >= 0.6].reset_index(drop=True)

    top_n        = min(len(filtered_df), 20)
    highest_score = clean_df['final_score'].max() if len(clean_df) > 0 else 0.0

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total processed",  f"{len(df):,}")
    m2.metric("Top score",        f"{highest_score:.3f}")
    m3.metric("Qualified",        f"{len(clean_df):,}")
    m4.metric("🚫 Disqualified",  dq_count)

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🏆 Top Candidates",
        "📊 Score Breakdown",
        "🚫 Disqualified Log"
    ])

    # ── Tab 1: Top Candidates ─────────────────────────────────────────────────
    with tab1:
        st.subheader(f"Top {top_n} qualified candidates")
        st.caption("Honeypots and consulting-only candidates are excluded from this view.")

        display_cols = [
            'rank', 'candidate_id', 'current_title', 'current_company',
            'years_of_experience', 'final_score', 'reasoning'
        ]
        show_cols = [c for c in display_cols if c in filtered_df.columns]

        st.dataframe(
            filtered_df[show_cols].head(top_n).style.format({'final_score': '{:.4f}'}),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "🟢 Tier 1 — AI/ML engineer   "
            "🔵 Tier 2 — Software/data engineer   "
            "⚪ Tier 3 — Adjacent background"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Download uses clean_df (unfiltered by display filters) for full top 100
        out_df = clean_df[['candidate_id', 'rank', 'final_score', 'reasoning']].head(100).copy()
        out_df.columns = ['candidate_id', 'rank', 'score', 'reasoning']

        st.download_button(
            label=" Download output.csv",
            data=out_df.to_csv(index=False),
            file_name="output.csv",
            mime="text/csv",
            type="primary"
        )

    # ── Tab 2: Score Breakdown ────────────────────────────────────────────────
    with tab2:
        st.subheader("Deep dive — top 10 score breakdown")
        st.caption("Key feature values driving each candidate's final score.")

        breakdown_cols = [
            'rank', 'current_title', 'title_tier',
            'core_skill_score', 'strong_skill_score',
            'ownership_score', 'scale_score', 'decision_depth_score',
            'recency_score', 'response_rate',
            'consulting_mult', 'research_mult', 'llm_tourist_mult',
            'final_score'
        ]
        show_breakdown = [c for c in breakdown_cols if c in clean_df.columns]

        st.dataframe(
            clean_df[show_breakdown].head(10).style.format({
                c: '{:.3f}' for c in show_breakdown
                if c not in ['rank', 'current_title']
            }),
            use_container_width=True,
            hide_index=True
        )

    # ── Tab 3: Disqualified Log ───────────────────────────────────────────────
    with tab3:
        st.subheader("Honeypot & consulting-only flags")

        if dq_count > 0:
            st.warning(
                f"Found **{dq_count}** candidates hard-disqualified (score forced to -9999). "
                f"These do not appear in the Top Candidates tab or the submission CSV."
            )
            dq_cols = [
                'candidate_id', 'current_title', 'current_company',
                'honeypot_flag', 'is_consulting_only',
                'honeypot_mult', 'consulting_mult', 'final_score'
            ]
            show_dq = [c for c in dq_cols if c in dq_df.columns]
            st.dataframe(dq_df[show_dq], use_container_width=True, hide_index=True)
        else:
            st.success("✅ No candidates hard-disqualified in this batch.")

else:
    st.info("👈 Upload a JSON/JSONL file or use the sample data in the sidebar to begin ranking.")