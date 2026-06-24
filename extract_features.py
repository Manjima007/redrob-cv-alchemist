import pandas as pd
import re
from datetime import date
from disqualifier_detection import (detect_consulting_career, detect_research_only,
                                    detect_llm_tourist, detect_non_coder,
                                    detect_eval_framework_knowledge)

CONSULTING = ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant',
              'capgemini', 'hcl', 'mindtree', 'tech mahindra']

MODERN_RETRIEVAL = ['rag', 'pinecone', 'qdrant', 'weaviate', 'milvus', 'langchain', 'llamaindex']
FUNDAMENTAL_IR   = ['elasticsearch', 'opensearch', 'bm25', 'information retrieval', 'lucene', 'solr']
CORE_SKILLS      = ['embedding', 'vector search', 'faiss', 'rerank', 'ranking', 'recommendation',
                    'sentence-transformer', 'ndcg', 'semantic search', 'retrieval'] + MODERN_RETRIEVAL + FUNDAMENTAL_IR

STRONG_SKILLS = ['nlp', 'llm', 'fine-tun', 'lora', 'qlora', 'peft', 'bert',
                 'transformer', 'python', 'xgboost', 'lightgbm', 'a/b test',
                 'sklearn', 'tensorflow', 'pytorch', 'spark', 'kafka']

IRRELEVANT = ['sales', 'figma', 'photoshop', 'seo', 'content writ',
              'hr tool', 'tally', 'excel vba']

LANGCHAIN_SKILLS = ['langchain', 'chatgpt api', 'openai api',
                    'prompt engineer', 'llm wrapper']

ML_PROD_KW = ['embed', 'vector', 'retriev', 'ranking', 'recommend',
              'train', 'model', 'inference', 'deploy', 'a/b', 'ndcg',
              'pipeline', 'feature store', 'serving', 'rerank',
              'search', 'similarity', 'index', 'embedding']

OWNER_WORDS = ['built', 'designed', 'architected', 'led', 'owned',
               'created', 'developed', 'launched', 'founded',
               'established', 'drove', 'spearheaded', 'pioneered',
               'initiated', 'shipped', 'delivered', 'responsible for']

CONTRIB_WORDS = ['assisted', 'contributed', 'helped', 'supported',
                 'collaborated', 'participated', 'worked on',
                 'involved in', 'part of']

DECISION_WORDS = ['chose', 'evaluated', 'compared', 'migrated', 'replaced',
                  'refactored', 'tradeoff', 'trade-off', 'instead of',
                  'versus', 'vs ', 'considered', 'decided', 'selected',
                  'switched', 'adopted', 'designed the']

SCALE_KW = [
    'ranking model', 'retrieval model', 'recommendation model',
    'embedding model', 'vector index', 'offline evaluation',
    'online evaluation', 'a/b test', 'ndcg', 'mrr', 'map@',
    'click-through', 'dwell time', 'engagement metric',
    'feature store', 'model serving', 'model deployment',
    'inference pipeline', 'reranking', 're-ranking',
    'candidate generation', 'two-stage', 'recall@', 'precision@',
    'lightgbm', 'xgboost', 'lambdarank', 'learning to rank',
    'faiss', 'qdrant', 'pinecone', 'weaviate', 'elasticsearch',
    'sentence-transformer', 'dense retrieval', 'sparse retrieval',
    'bm25', 'hybrid search', 'semantic search', 'vector search',
    'qps', 'latency', 'millions', 'billions', 'terabytes',
    'high-throughput', 'large-scale', 'distributed', 'concurrent',
    'scale', 'productionize', 'ab test', 'a/b testing',
]

PROFICIENCY_MULT = {'beginner': 0.4, 'intermediate': 0.7,
                    'advanced': 1.0, 'expert': 1.2}

TIER1_CITIES = ['pune', 'noida', 'delhi', 'gurgaon', 'gurugram',
                'faridabad', 'greater noida']
TIER2_CITIES = ['hyderabad', 'mumbai', 'bangalore', 'bengaluru', 'chennai']

REFERENCE_DATE = date(2026, 6, 1)
RECENCY_CUTOFF = date(2024, 12, 1)


def _title_tier(title):
    t = title.lower()
    if any(x in t for x in ['ai engineer', 'ml engineer', 'machine learning',
                              'nlp engineer', 'applied scientist',
                              'research engineer', 'ranking engineer',
                              'search engineer', 'ai/ml', 'recommendation',
                              'retrieval engineer', 'applied ml']):
        return 1.0
    if any(x in t for x in ['software engineer', 'data scientist',
                              'backend engineer', 'full stack',
                              'platform engineer', 'sde', 'swe',
                              'data science']):
        return 0.75
    if any(x in t for x in ['data engineer', 'cloud engineer',
                              'devops', 'mlops', 'site reliability']):
        return 0.45
    return 0.10


def _is_consulting(company):
    c = company.lower()
    return any(con in c for con in CONSULTING)


def _skill_score(skills, keyword_list):
    total = 0.0
    for s in skills:
        name = s.get('name', '').lower()
        if any(kw in name for kw in keyword_list):
            pm = PROFICIENCY_MULT.get(s.get('proficiency', 'intermediate'), 0.5)
            dm = min(s.get('duration_months', 0), 60) / 60
            em = min(s.get('endorsements', 0), 30) / 30
            total += pm * (0.6 * dm + 0.4 * em)
    return total


def _notice_score(days):
    if days <= 15:  return 1.0
    if days <= 30:  return 0.85
    if days <= 60:  return 0.65
    if days <= 90:  return 0.45
    return 0.25


def _location_score(city, country):
    c  = city.lower()
    co = country.lower()
    if any(t in c for t in TIER1_CITIES): return 1.0
    if any(t in c for t in TIER2_CITIES): return 0.85
    if 'india' in co:                      return 0.60
    if any(x in co for x in ['uae', 'singapore', 'malaysia', 'dubai']): return 0.35
    return 0.10


def extract_features(candidate):
    profile = candidate.get('profile', {})
    career  = candidate.get('career_history', [])
    skills  = candidate.get('skills', [])
    sig     = candidate.get('redrob_signals', {})

    # ── Title ────────────────────────────────────────────────────────────────
    title_tier = _title_tier(profile.get('current_title', ''))

    companies = [j.get('company', '') for j in career]

    # ── Consulting ───────────────────────────────────────────────────────────
    # FIX #4: Use detect_consulting_career as the single source of truth.
    # Derive is_consulting_only from it to avoid two inconsistent signals.
    consulting_mult    = detect_consulting_career(candidate)
    is_consulting_only = int(consulting_mult < 1.0)

    total_months = sum(j.get('duration_months', 0) for j in career)
    consulting_months = sum(j.get('duration_months', 0) for j in career
                            if _is_consulting(j.get('company', '')))
    product_ratio = (total_months - consulting_months) / max(total_months, 1)

    # ── Skills ───────────────────────────────────────────────────────────────
    # FIX #3: Use CORE_SKILLS (the superset) as the single score.
    # Sub-scores kept only for interpretability — not used to override.
    core_skill_score       = _skill_score(skills, CORE_SKILLS)
    modern_retrieval_score = _skill_score(skills, MODERN_RETRIEVAL)   # diagnostic only
    fundamental_ir_score   = _skill_score(skills, FUNDAMENTAL_IR)     # diagnostic only
    strong_skill_score     = _skill_score(skills, STRONG_SKILLS)
    has_irrelevant         = int(any(
        any(kw in s.get('name', '').lower() for kw in IRRELEVANT)
        for s in skills
    ))

    assessment_scores    = list(sig.get('skill_assessment_scores', {}).values())
    assessment_bonus     = (sum(assessment_scores) / 100 / max(len(assessment_scores), 1)
                            if assessment_scores else 0.0)
    skill_assessments_max = max(assessment_scores) if assessment_scores else 0.0

    # ── Experience ───────────────────────────────────────────────────────────
    yoe = float(profile.get('years_of_experience', 0))

    ml_production_months = 0
    pre_2022_ml_months   = 0
    recency_flag         = 0
    for j in career:
        desc = j.get('description', '').lower()
        if any(kw in desc for kw in ML_PROD_KW):
            m = j.get('duration_months', 0)
            ml_production_months += m
            try:
                end = date.fromisoformat(j.get('end_date', REFERENCE_DATE.isoformat()))
                if end < date(2022, 1, 1):
                    pre_2022_ml_months += m
            except Exception:
                pass
            if j.get('is_current', False) or (
                j.get('end_date') and
                date.fromisoformat(j.get('end_date')) >= RECENCY_CUTOFF
            ):
                recency_flag = 1

    research_markers = ['university', 'iit', 'iisc', 'institute',
                        'research lab', ' labs ', 'academia']
    is_research_only = int(
        bool(companies) and
        all(any(m in c.lower() for m in research_markers) for c in companies) and
        product_ratio == 0
    )

    skill_names    = [s.get('name', '').lower() for s in skills]
    has_langchain  = any(any(lw in n for lw in LANGCHAIN_SKILLS) for n in skill_names)
    has_real_ml    = core_skill_score > 0.1 or strong_skill_score > 0.2
    langchain_only_flag = int(has_langchain and not has_real_ml and yoe < 3)

    # ── Free-text features ───────────────────────────────────────────────────
    all_desc = ' '.join(j.get('description', '') for j in career).lower()
    all_text = (all_desc + ' ' +
                profile.get('headline', '').lower() + ' ' +
                profile.get('summary', '').lower())

    owner_count   = sum(all_desc.count(w) for w in OWNER_WORDS)
    contrib_count = sum(all_desc.count(w) for w in CONTRIB_WORDS)
    total_lang    = owner_count + contrib_count
    ownership_score = owner_count / total_lang if total_lang > 0 else 0.5

    scale_hits  = sum(1 for kw in SCALE_KW if kw in all_text)
    scale_score = min(scale_hits / 2.0, 1.0)

    decision_count       = sum(all_desc.count(w) for w in DECISION_WORDS)
    decision_depth_score = min(decision_count / 5.0, 1.0)

    # ── Behavioral ───────────────────────────────────────────────────────────
    response_rate        = float(sig.get('recruiter_response_rate', 0.5))
    interview_completion = float(sig.get('interview_completion_rate', 0.5))
    raw_offer            = sig.get('offer_acceptance_rate', -1)
    offer_acceptance     = float(raw_offer) if raw_offer != -1 else 0.5
    open_to_work         = int(bool(sig.get('open_to_work_flag', False)))
    raw_github           = sig.get('github_activity_score', -1)
    # NOTE: github_score is normalized 0–1 here. Checks in logical_validation
    # must use 0.0–1.0 scale, NOT 0–100.
    github_score         = float(raw_github) / 100 if raw_github != -1 else 0.0
    profile_completeness = float(sig.get('profile_completeness_score', 50)) / 100

    notice_days  = int(sig.get('notice_period_days', 60))
    notice_score = _notice_score(notice_days)

    try:
        last_active = date.fromisoformat(sig.get('last_active_date', '2020-01-01'))
        days_since  = (REFERENCE_DATE - last_active).days
    except Exception:
        days_since = 365
    recency_score = max(0.0, 1.0 - days_since / 180.0)

    # ── Location ─────────────────────────────────────────────────────────────
    location_score = _location_score(
        profile.get('location', ''),
        profile.get('country', '')
    )

    # ── Honeypot detection ───────────────────────────────────────────────────
    honeypot_flag = 0

    # Check 1: Total claimed months are physically impossible (>3x YoE in months).
    # This catches candidates claiming e.g. 36 concurrent job-months in a 1-year career.
    if total_months > (yoe * 36):
        honeypot_flag = 1

    # Check 2: Expert-proficiency skills that have 0 duration_months EXPLICITLY set.
    # Using default=-1 so skills with no duration key are NOT counted — a missing
    # key means "not provided", which is legitimate; 0 means "I know this but
    # spent zero time on it", which is the honeypot pattern.
    fake_expert_count = sum(
        1 for s in skills
        if s.get('proficiency') == 'expert'
        and s.get('duration_months', -1) == 0   # -1 default = key absent = not a trap
    )
    if fake_expert_count >= 5:
        honeypot_flag = 1

    # ── Temporal fields for logical_validation.py ────────────────────────────
    earliest_dt = None
    for j in career:
        sd = j.get('start_date')
        if not sd:
            continue
        try:
            d = date.fromisoformat(sd)
        except Exception:
            try:
                d = pd.to_datetime(sd, errors='coerce').to_pydatetime().date()
            except Exception:
                d = None
        if d and (earliest_dt is None or d < earliest_dt):
            earliest_dt = d
    earliest_job_start = earliest_dt.isoformat() if earliest_dt else None

    company_founded_date = None
    try:
        cf = profile.get('company_founded_date')
        if cf:
            company_founded_date = cf
    except Exception:
        pass

    job_start_date = earliest_job_start

    # expert_zero_exp: count expert OR advanced skills with explicitly-zero duration.
    # Threshold enforcement lives HERE, not in logical_validation (single source of truth).
    expert_zero_exp = sum(
        1 for s in skills
        if s.get('proficiency', '').lower() in ('expert', 'advanced')
        and s.get('duration_months', -1) == 0   # same -1 default as above
    )

    # ── Other disqualifier detections ────────────────────────────────────────
    research_mult  = detect_research_only(candidate)
    llm_tourist_mult = detect_llm_tourist(candidate)
    ic_mult        = detect_non_coder(candidate)
    eval_score     = detect_eval_framework_knowledge(candidate)

    return {
        'title_tier':            title_tier,
        'is_consulting_only':    is_consulting_only,
        'product_ratio':         product_ratio,
        'core_skill_score':      core_skill_score,
        'strong_skill_score':    strong_skill_score,
        'assessment_bonus':      assessment_bonus,
        'has_irrelevant':        has_irrelevant,
        'years_of_experience':   yoe,
        'ml_production_months':  ml_production_months,
        'recency_flag':          recency_flag,
        'is_research_only':      is_research_only,
        'langchain_only_flag':   langchain_only_flag,
        'ownership_score':       ownership_score,
        'scale_score':           scale_score,
        'decision_depth_score':  decision_depth_score,
        'earliest_job_start':    earliest_job_start,
        'company_founded_date':  company_founded_date,
        'job_start_date':        job_start_date,
        'expert_zero_exp':       expert_zero_exp,
        'response_rate':         response_rate,
        'interview_completion':  interview_completion,
        'offer_acceptance':      offer_acceptance,
        'open_to_work':          open_to_work,
        'github_score':          github_score,       # normalized 0–1
        'profile_completeness':  profile_completeness,
        'notice_score':          notice_score,
        'notice_days':           notice_days,
        'days_since_active':     days_since,
        'recency_score':         recency_score,
        'location_score':        location_score,
        'honeypot_flag':         honeypot_flag,
        'agg_text':              all_text,
        'skill_assessments_max': skill_assessments_max,  # raw 0–100
        'expected_salary_min':   sig.get('expected_salary_range_inr_lpa', {}).get('min', 0.0),
        'expected_salary_max':   sig.get('expected_salary_range_inr_lpa', {}).get('max', 0.0),
        'consulting_mult':       consulting_mult,
        'research_mult':         research_mult,
        'llm_tourist_mult':      llm_tourist_mult,
        'ic_mult':               ic_mult,
        'eval_score':            eval_score,
    }