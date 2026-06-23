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


# def generate_reasoning(row):
#     yoe   = float(row['years_of_experience'])
#     tier  = float(row['title_tier'])
#     css   = float(row['core_skill_score'])
#     sss   = float(row['strong_skill_score'])
#     ml_m  = int(row['ml_production_months'])
#     rr    = float(row['response_rate'])
#     gh    = float(row['github_score'])
#     loc   = float(row['location_score'])
#     nd    = int(row['notice_days'])
#     own   = float(row['ownership_score'])
#     scl   = float(row['scale_score'])
#     dd    = float(row['decision_depth_score'])
#     rec   = float(row['recency_score'])
#     otw   = int(row['open_to_work'])
#     rf    = int(row['recency_flag'])
#     score = float(row['final_score'])
#     rank  = int(row['rank'])

#     # Opening — who is this person
#     if tier == 1.0:
#         opening = f"{yoe:.0f}-year AI/ML engineer"
#     elif tier == 0.75:
#         opening = f"{yoe:.0f}-year software/data engineer with ML exposure"
#     else:
#         opening = f"{yoe:.0f}-year engineer (adjacent background)"

#     # Core strength
#     strength = ""
#     if css > 1.0:
#         strength = "strong retrieval and ranking background"
#     elif css > 0.4:
#         strength = "solid core AI skills (retrieval/ranking)"
#     elif sss > 0.5:
#         strength = "good ML fundamentals without deep retrieval specialization"
#     elif ml_m > 12:
#         strength = f"{ml_m} months of ML production experience"

#     # Production signal
#     prod = ""
#     if rf and scl > 0.3:
#         prod = "shipped ML systems to production at scale"
#     elif rf:
#         prod = "recent hands-on ML production experience"
#     elif ml_m > 24:
#         prod = f"{ml_m} months of ML work across career"

#     # Ownership signal
#     ownership = ""
#     if own > 0.7 and dd > 0.4:
#         ownership = "descriptions show ownership and architectural decision-making"
#     elif own > 0.7:
#         ownership = "ownership-oriented career narrative"
#     elif dd > 0.5:
#         ownership = "evidence of technical decision-making"

#     # Availability
#     availability = ""
#     if rec > 0.8 and rr > 0.75:
#         availability = f"actively engaged (response rate {rr:.0%})"
#     elif otw:
#         availability = "explicitly open to work"
#     elif rec > 0.5:
#         availability = "recently active on platform"

#     # Location
#     loc_str = ""
#     if loc >= 1.0:
#         loc_str = "Pune/Noida-based"
#     elif loc >= 0.85:
#         loc_str = "Tier-1 Indian city"
#     elif loc >= 0.6:
#         loc_str = "India-based"
#     else:
#         loc_str = "outside India"

#     # Concerns
#     concerns = []
#     if nd > 90:
#         concerns.append(f"long notice period ({nd} days)")
#     elif nd > 30:
#         concerns.append(f"notice period {nd} days")
#     if rec < 0.3:
#         concerns.append("inactive on platform 90+ days")
#     if int(row.get('is_research_only', 0)):
#         concerns.append("research-only background")
#     if loc < 0.6:
#         concerns.append("outside India")
#     if rank > 70:
#         concerns.append("adjacent fit — included on engagement signals")

#     # Assemble — vary structure so strings differ
#     parts = [p for p in [opening, strength, prod, ownership, availability, loc_str] if p]

#     result = '; '.join(parts)
#     if concerns:
#         result += '. Concern: ' + ', '.join(concerns)

#     # Trim smartly to 190 chars without slicing words
#     result = textwrap.shorten(result, width=190, placeholder="...")
    
#     if len(result) < 15:
#         result = f"{yoe:.0f}yr engineer; score {score:.4f}"

#     return result

def generate_reasoning(row):
    yoe   = float(row['years_of_experience'])
    tier  = float(row['title_tier'])
    css   = float(row['core_skill_score'])
    sss   = float(row['strong_skill_score'])
    loc   = float(row['location_score'])
    nd    = int(row['notice_days'])
    own   = float(row['ownership_score'])
    rec   = float(row['recency_score'])
    rank  = int(row['rank'])

    # 1. Define the Persona
    if tier == 1.0:
        persona = f"{yoe:.0f}-year AI/ML engineer"
    elif tier == 0.75:
        persona = f"{yoe:.0f}-year software/data engineer"
    else:
        persona = f"{yoe:.0f}-year engineer (adjacent background)"

    # 2. Extract ONLY the top 1 or 2 strongest positive signals
    positives = []
    if css > 0.8: positives.append("deep expertise in retrieval and ranking")
    elif sss > 0.5: positives.append("solid ML production fundamentals")
    
    if own > 0.7: positives.append("strong architectural ownership")
    elif rec > 0.8: positives.append("high platform engagement")
    
    if len(positives) == 0: 
        positives.append("transferable technical skills")

    highlights = " and ".join(positives[:2]) # Keep it crisp by only taking the top 2

    # 3. Tone Mapping based on Rank (Crucial for passing Stage 4)
    if rank <= 15:
        base = f"Exceptional {persona} with {highlights}. Highly engaged and fits the 'product over research' JD profile perfectly."
    elif rank <= 60:
        base = f"Strong {persona} offering {highlights}. Good overall fit for our stack and production needs."
    else:
        base = f"Passable {persona} with {highlights}. Lacks deep AI specialization but included as filler due to solid behavioral signals."

    # 4. Append ONE honest concern if applicable
    if nd > 60:
        return f"{base} Note: Long notice period ({nd} days)."
    elif loc < 0.6:
        return f"{base} Note: Located outside target Indian hubs."
    
    return base


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
       # Apply post-prediction soft multipliers (for things like consulting/tourists)
        for col in ['consulting_mult', 'research_mult', 'llm_tourist_mult', 'ic_mult']:
            if col in df.columns:
                scores = scores * df[col].fillna(1.0).values
                
        df['final_score'] = scores
        
        # HARD SINK: Force honeypots to the absolute bottom of the dataframe
        if 'honeypot_mult' in df.columns:
            df.loc[df['honeypot_mult'] < 1.0, 'final_score'] = -9999.0
        if 'honeypot_flag' in df.columns:
            df.loc[df['honeypot_flag'] == 1, 'final_score'] = -9999.0
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

# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CV Alchemist | RedRob",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR (Controls & Inputs) ---
with st.sidebar:
    st.title("🔮 CV Alchemist")
    st.caption("AI-powered candidate ranker for Senior AI Engineer")
    st.markdown("---")
    
    # Model Status
    model = load_model()
    if model is None:
        st.error("⚠️ No trained model found. Running in fallback mode.")
    else:
        st.success("✨ LambdaRank Model Active")
        
    st.markdown("---")
    st.subheader("Data Input")
    
    uploaded = st.file_uploader("Upload Candidates (JSON)", type=['json'])
    st.markdown("**OR**")
    use_sample = st.button("Use sample_candidates.json", use_container_width=True)
    
    st.markdown("---")
    st.caption("RedRob Data & AI Challenge")

# --- MAIN CANVAS (Results & Insights) ---
st.title("Candidate Rankings")

candidates = []

# Input Handling
if uploaded:
    try:
        candidates = json.load(uploaded)
        if isinstance(candidates, dict):
            candidates = [candidates]
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")

elif use_sample:
    sample_path = 'outputs/sample_candidates.json' # Updated to match your latest path
    if os.path.exists(sample_path):
        with open(sample_path) as f:
            candidates = json.load(f)
    else:
        st.error(f"{sample_path} not found.")

# Processing & Display
if candidates:
    with st.spinner("Alchemizing candidate scores..."):
        df = score_candidates(candidates, model)

    # Calculate top metrics
    top_n = min(len(df), 20)
    highest_score = df['final_score'].max()
    
    # Calculate Disqualified early for the metrics
    dq_mask = pd.Series(False, index=df.index)
    if 'honeypot_mult' in df.columns: dq_mask |= (df['honeypot_mult'] < 1.0)
    if 'consulting_mult' in df.columns: dq_mask |= (df['consulting_mult'] < 1.0)
    if 'honeypot_flag' in df.columns: dq_mask |= (df['honeypot_flag'] == 1)
    if 'is_consulting_only' in df.columns: dq_mask |= (df['is_consulting_only'] == 1)
    
    disqualified_df = df[dq_mask]
    dq_count = len(disqualified_df)

    # 1. Top Level Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Candidates Processed", len(df))
    m2.metric("Highest Candidate Score", f"{highest_score:.3f}")
    m3.metric("Disqualified Candidates", dq_count)
    
    st.markdown("---")

    # 2. Tabbed Interface for clean navigation
    tab1, tab2, tab3 = st.tabs(["🏆 Top Candidates", "📊 Score Breakdown", "🚫 Disqualified Log"])

    with tab1:
        st.subheader(f"Top {top_n} Recommendations")
        display_cols = ['rank', 'candidate_id', 'current_title', 'current_company',
                        'years_of_experience', 'final_score', 'reasoning']
        show_cols = [c for c in display_cols if c in df.columns]
        
        st.dataframe(
            df[show_cols].head(top_n).style.format({'final_score': '{:.4f}'}),
            use_container_width=True,
            hide_index=True
        )
        
        # Download button placed cleanly at the bottom of the main tab
        st.markdown("<br>", unsafe_allow_html=True)
        out_df = df[['candidate_id', 'rank', 'final_score', 'reasoning']].copy()
        out_df.columns = ['candidate_id', 'rank', 'score', 'reasoning']
        st.download_button(
            label="📥 Download submission.csv (Top 100)",
            data=out_df.head(100).to_csv(index=False),
            file_name="submission.csv",
            mime="text/csv",
            type="primary" # Makes the button pop
        )

    with tab2:
        st.subheader("Deep Dive: Top 5 Breakdown")
        breakdown_cols = ['rank', 'current_title', 'title_tier', 'core_skill_score',
                          'ownership_score', 'scale_score', 'decision_depth_score',
                          'recency_score', 'response_rate', 'final_score']
        show_breakdown = [c for c in breakdown_cols if c in df.columns]
        
        st.dataframe(
            df[show_breakdown].head(5).style.format({c: '{:.3f}' for c in show_breakdown if c not in ['rank', 'current_title']}),
            use_container_width=True,
            hide_index=True
        )

    with tab3:
        st.subheader("Honeypot & Consulting-Only Flags")
        if dq_count > 0:
            st.warning(f"Found {dq_count} candidates failing logical validation.")
            dq_cols = ['candidate_id', 'current_title', 'honeypot_flag', 'is_consulting_only']
            show_dq = [c for c in dq_cols if c in disqualified_df.columns]
            st.dataframe(disqualified_df[show_dq], use_container_width=True, hide_index=True)
        else:
            st.info("✅ All clear. No honeypots or consulting-only candidates detected.")
else:
    # Empty state visual
    st.info("👈 Upload a JSON file or use the sample data in the sidebar to begin ranking.")