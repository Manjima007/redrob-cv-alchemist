#!/usr/bin/env python3
"""
All-in-one disqualifier and signal detection for CV Alchemist.
Detects: consulting-only, researchers, LLM tourists, non-coders, eval framework knowledge.
"""

from datetime import date
import os

# Lazy-loaded embedding artifacts (produced by precompute if available)
_ANCHOR_EMB = None
_EMB_MAP    = None

CONSULTING_FIRMS = [
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'hcl',
    'mindtree', 'tech mahindra', 'deloitte', 'pwc', 'kpmg', 'persistent'
]

RESEARCH_MARKERS = [
    'university', 'iit', 'iisc', 'institute', 'research lab', 'academia',
    'phd', 'postdoc', 'research fellow'
]

LANGCHAIN_KWS = ['langchain', 'chatgpt api', 'openai api', 'prompt engineer', 'llm wrapper']
ML_PROD_KWS   = ['embed', 'vector', 'retriev', 'ranking', 'recommend', 'deploy', 'a/b', 'ndcg']
CODE_KWS      = ['built', 'implemented', 'shipped', 'developed', 'engineered', 'architected']

# FIX #4: Removed 'lead' (substring-matches 'leading', 'tech lead', etc.)
# and 'tech lead' (IC role that codes daily). Only true management titles here.
MGT_TITLES = ['manager', 'director', 'vp of', 'vice president', 'cto', 'cpo', 'head of']

EVAL_KWS = ['ndcg', 'mrr', 'map@', 'a/b test', 'offline', 'online', 'correlation', 'lambdarank']

REFERENCE_DATE = date(2026, 6, 1)


def _parse_date_safe(d):
    try:
        return date.fromisoformat(d)
    except Exception:
        return None


def _most_recent_job(career):
    """Return the most recent job entry from career history."""
    if not career:
        return None
    for j in career:
        if j.get('is_current'):
            return j
    best, best_date = None, None
    for j in career:
        end   = _parse_date_safe(j.get('end_date',   '') or '')
        start = _parse_date_safe(j.get('start_date', '') or '')
        d = end or start
        if d is None:
            continue
        if best_date is None or d > best_date:
            best_date, best = d, j
    return best or career[0]


# ── Consulting ────────────────────────────────────────────────────────────────

def detect_consulting_career(candidate):
    """
    Multiplier for consulting-heavy careers.
    Returns 0.0 (hard disqualify) → 1.0 (clean product background).
    """
    career = candidate.get('career_history', [])
    if not career:
        return 1.0

    total_m   = sum(j.get('duration_months', 0) for j in career)
    consult_m = sum(
        j.get('duration_months', 0) for j in career
        if any(con in j.get('company', '').lower() for con in CONSULTING_FIRMS)
    )

    if total_m == 0:
        return 1.0

    ratio = consult_m / total_m

    all_companies = [j.get('company', '') for j in career if j.get('company', '')]
    all_consult   = bool(all_companies) and all(
        any(con in comp.lower() for con in CONSULTING_FIRMS)
        for comp in all_companies
    )

    if all_consult:    return 0.0   # Hard disqualify — 100% consulting
    if ratio >= 0.70:  return 0.3   # Soft disqualify
    if ratio >= 0.50:  return 0.6
    if ratio > 0:      return 0.85
    return 1.0


# ── Research-only ─────────────────────────────────────────────────────────────

def detect_research_only(candidate):
    """
    Multiplier for pure-research careers.
    Returns 0.0 (hard disqualify) → 1.0 (product background).

    FIX #3: Removed the blanket `if prod_m < 12: return 0.3` clause.
    That fired for ALL candidates with sparse job description text,
    including legitimate early-career AI engineers. Now only disqualifies
    when BOTH company signals AND zero production keywords align.
    """
    career = candidate.get('career_history', []) or []
    if not career:
        return 1.0

    prod_m = sum(
        j.get('duration_months', 0) for j in career
        if any(kw in j.get('description', '').lower() for kw in ML_PROD_KWS)
    )

    all_companies          = [j.get('company', '') for j in career if j.get('company', '')]
    research_company_count = sum(
        1 for c in all_companies
        if any(m in c.lower() for m in RESEARCH_MARKERS)
    )
    company_ratio = research_company_count / max(len(all_companies), 1)

    title              = candidate.get('profile', {}).get('current_title', '').lower()
    is_research_title  = any(m in title for m in RESEARCH_MARKERS)

    # Hard disqualify: looks like a researcher AND has zero production signal
    if (company_ratio >= 0.5 or is_research_title) and prod_m == 0:
        return 0.0

    # Soft penalty: research company background but has SOME production work
    if company_ratio >= 0.5 and prod_m < 6:
        return 0.5

    return 1.0


# ── LLM Tourist ───────────────────────────────────────────────────────────────

def detect_llm_tourist(candidate):
    """
    Multiplier for candidates who only picked up LLM wrappers recently
    with no underlying ML production history.

    FIX #5: Removed the `yoe < 1: return 1.0` early exit.
    Fresh grads with only LangChain/ChatGPT API experience are exactly
    the LLM tourist profile. They now get a soft penalty (0.5) rather
    than being exempted entirely. True zero-experience candidates
    (no career history) are still passed through cleanly.
    """
    career = candidate.get('career_history', [])
    yoe    = float(candidate.get('profile', {}).get('years_of_experience', 0))

    if not career:
        return 1.0  # No history → can't assess → don't penalize

    # Recent 12 months text
    recent_desc = []
    for job in career:
        try:
            if job.get('is_current'):
                recent_desc.append(job.get('description', ''))
            elif job.get('end_date'):
                end = _parse_date_safe(job.get('end_date'))
                if end and (REFERENCE_DATE - end).days / 30 < 12:
                    recent_desc.append(job.get('description', ''))
        except Exception:
            pass

    recent_text = ' '.join(recent_desc).lower()
    if not recent_text:
        return 1.0

    langchain_hits = sum(1 for kw in LANGCHAIN_KWS if kw in recent_text)
    ml_hits        = sum(1 for kw in ML_PROD_KWS   if kw in recent_text)

    if langchain_hits >= 1 and ml_hits <= 1:
        # Check older work for redeeming ML history
        older_desc = []
        for job in career:
            try:
                if job.get('end_date'):
                    end = _parse_date_safe(job.get('end_date'))
                    if end and (REFERENCE_DATE - end).days / 30 >= 12:
                        older_desc.append(job.get('description', ''))
            except Exception:
                pass

        older_ml = sum(1 for kw in ML_PROD_KWS if kw in ' '.join(older_desc).lower())

        if older_ml == 0 and yoe > 3:
            return 0.2   # Senior but zero real ML history — strong tourist signal
        if older_ml == 0:
            return 0.5   # Junior with only LangChain — soft penalty (was 1.0, now penalized)
        if older_ml <= 1:
            return 0.5   # Some older ML but very thin

    return 1.0


# ── Non-coder ─────────────────────────────────────────────────────────────────

def detect_non_coder(candidate):
    """
    Multiplier for non-coding managers (0.0=disqualify, 1.0=ok).

    FIX #4: Removed 'lead' and 'tech lead' from management title list.
    'lead' as substring matches 'leading', 'team lead', 'tech lead' — all
    of which are IC roles that code daily. Only pure management titles remain.
    """
    career        = candidate.get('career_history', [])
    current_title = candidate.get('profile', {}).get('current_title', '') or ''

    if not career:
        return 1.0

    recent      = _most_recent_job(career) or {}
    recent_title = (recent.get('title') or '').lower()
    recent_desc  = (recent.get('description') or '').lower()

    # Only true management titles — not IC lead roles
    is_manager = any(
        m in recent_title or m in current_title.lower()
        for m in MGT_TITLES
    )

    code_hits_recent = sum(1 for kw in CODE_KWS if kw in recent_desc)

    if is_manager and code_hits_recent == 0:
        return 0.2   # Manager with zero coding evidence
    if is_manager and code_hits_recent == 1:
        return 0.6   # Manager with thin coding evidence
    return 1.0


# ── Eval framework knowledge ──────────────────────────────────────────────────

def detect_eval_framework_knowledge(candidate):
    """Score for ranking evaluation framework knowledge (0.0–1.0)."""
    career   = candidate.get('career_history', [])
    all_text = ' '.join(j.get('description', '') for j in career).lower()

    eval_hits = sum(1 for kw in EVAL_KWS if kw in all_text)

    if eval_hits >= 4: return 0.95
    if eval_hits >= 3: return 0.7
    if eval_hits >= 2: return 0.5
    if eval_hits >= 1: return 0.3
    return 0.0


# ── Engagement ceiling ────────────────────────────────────────────────────────

def engagement_ceiling(response_rate, last_active_date):
    """Hard ceiling: non-responsive AND inactive 6+ months → 0.1 multiplier."""
    try:
        days_inactive = (REFERENCE_DATE - date.fromisoformat(last_active_date)).days
    except Exception:
        return 1.0
    if response_rate < 0.20 and days_inactive > 180:
        return 0.1
    return 1.0


# ── Semantic similarity (optional, precomputed) ───────────────────────────────

def _load_anchor_and_embeddings():
    global _ANCHOR_EMB, _EMB_MAP
    if _ANCHOR_EMB is not None and _EMB_MAP is not None:
        return
    try:
        import numpy as np
        anchor_path = os.path.join('outputs', 'anchor_embeddings.npy')
        ids_path    = os.path.join('outputs', 'emb_ids.npy')
        emb_path    = os.path.join('outputs', 'embeddings.npy')
        if (os.path.exists(anchor_path) and
                os.path.exists(ids_path) and
                os.path.exists(emb_path)):
            _ANCHOR_EMB = np.load(anchor_path)
            emb  = np.load(emb_path)
            ids  = np.load(ids_path)
            _EMB_MAP = {str(cid): emb[i] for i, cid in enumerate(ids)}
    except Exception:
        _ANCHOR_EMB = None
        _EMB_MAP    = None


def semantic_scores_for_candidate(candidate_id):
    """
    Return semantic similarity scores to anchors for a precomputed candidate embedding.
    Returns dict with keys: ml_prod_sim, research_sim, llm_sim, consulting_sim in [0,1].
    Returns None if embeddings are unavailable.
    """
    _load_anchor_and_embeddings()
    if _ANCHOR_EMB is None or _EMB_MAP is None:
        return None
    try:
        import numpy as np
        vec = _EMB_MAP.get(str(candidate_id))
        if vec is None:
            return None

        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        sims = [cos(vec, _ANCHOR_EMB[i]) for i in range(_ANCHOR_EMB.shape[0])]
        # Anchor order set in precompute: [ml_prod, research, llm, consult]
        return {
            'ml_prod_sim':    max(0.0, sims[0]),
            'research_sim':   max(0.0, sims[1]),
            'llm_sim':        max(0.0, sims[2]),
            'consulting_sim': max(0.0, sims[3]),
        }
    except Exception:
        return None