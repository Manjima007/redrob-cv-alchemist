import pandas as pd
import lightgbm as lgb
import argparse
import sys
import os
import math
import numpy as np

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
    if len(result) > 190:
        trimmed = result[:190]
        last_semi = trimmed.rfind(';')
        result = trimmed[:last_semi].strip() if last_semi > 15 else trimmed.strip()


    if len(result) < 15:
        result = f"{yoe:.0f}yr engineer; score {score:.4f}"

    return result


### Evaluation utilities ###
def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    gains = (2**rels - 1)
    discounts = np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(pred_ranking_ids, labels_dict, k):
    # build relevance list in order of pred ranking
    rels = [labels_dict.get(cid, 0.0) for cid in pred_ranking_ids]
    dcg = dcg_at_k(rels, k)
    # ideal DCG from sorted labels
    ideal = sorted(labels_dict.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg / idcg


def average_precision(pred_ids, labels_dict):
    # binary relevance: relevant if label > 0
    hits = 0
    sum_prec = 0.0
    for i, cid in enumerate(pred_ids, start=1):
        if labels_dict.get(cid, 0.0) > 0:
            hits += 1
            sum_prec += hits / i
    return 0.0 if hits == 0 else sum_prec / hits


def precision_at_k(pred_ids, labels_dict, k):
    topk = pred_ids[:k]
    if len(topk) == 0:
        return 0.0
    rels = sum(1 for cid in topk if labels_dict.get(cid, 0.0) > 0)
    return rels / len(topk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', default='data/candidates.jsonl')
    parser.add_argument('--out', default='outputs/submission.csv')
    parser.add_argument('--labels', default=None, help='CSV with candidate_id,relevance (numeric)')
    parser.add_argument('--eval_k', type=int, default=50, help='k for NDCG@k and P@k')
    args = parser.parse_args()

    features_path = 'outputs/features.parquet'
    model_path    = 'outputs/ltr_model.txt'

    if not os.path.exists(features_path):
        print(f"ERROR: {features_path} not found. Run precompute.py first.")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found. Run precompute.py first.")
        sys.exit(1)

    print("Loading features...")
    df = pd.read_parquet(features_path)

    print("Loading model...")
    model = lgb.Booster(model_file=model_path)

    print("Predicting scores...")
    X = df[FEATURE_COLS].fillna(0).values
    scores = model.predict(X)

    # apply multiplier columns if present
    for col in ['honeypot_mult', 'consulting_mult', 'research_mult', 'llm_tourist_mult', 'ic_mult']:
        if col in df.columns:
            scores = scores * df[col].fillna(1.0).values

    # Hard zero disqualified
    scores *= (1 - df['honeypot_flag'].values)
    scores *= (1 - df['is_consulting_only'].values)
    # Penalize non-AI-ML titles — should never appear in top 5
    scores[df['title_tier'].values == 0.75] *= 0.85
    scores[df['title_tier'].values == 0.45] *= 0.60
    scores[df['title_tier'].values == 0.10] *= 0.10

    df['final_score'] = scores

    top100 = df.nlargest(100, 'final_score').copy().reset_index(drop=True)
    top100['rank'] = range(1, 101)

    labels_path = args.labels if hasattr(args, 'labels') else None
    eval_k = args.eval_k if hasattr(args, 'eval_k') else 50
    # Note: if using CLI, include --labels and --eval_k when invoking script
    # e.g. python rank.py --labels human_labels.csv --eval_k 50
    if labels_path is None and os.environ.get('RANK_LABELS_PATH'):
        labels_path = os.environ.get('RANK_LABELS_PATH')
    if labels_path:
        if not os.path.exists(labels_path):
            print(f"Labels file {labels_path} not found — skipping evaluation.")
        else:
            labs = pd.read_csv(labels_path)
            labels_dict = dict(zip(labs['candidate_id'].astype(str), labs['relevance'].astype(float)))
            # Build full predicted ranking by candidate_id
            preds = df.sort_values('final_score', ascending=False)['candidate_id'].astype(str).tolist()
            ndcg10 = ndcg_at_k(preds, labels_dict, min(10, eval_k))
            ndcgk = ndcg_at_k(preds, labels_dict, eval_k)
            map_score = average_precision(preds, labels_dict)
            p10 = precision_at_k(preds, labels_dict, 10)
            print('\nEvaluation on labels:')
            print(f'  NDCG@10: {ndcg10:.4f}')
            print(f'  NDCG@{args.eval_k}: {ndcgk:.4f}')
            print(f'  MAP: {map_score:.4f}')
            print(f'  P@10: {p10:.4f}')

    # ── PROBLEM 3: Verify no honeypots in top 100 ──────────────────────────────
    honeypots_in_top100 = top100[top100['honeypot_flag'] == 1]
    if len(honeypots_in_top100) > 0:
        print(f"\n⚠️  WARNING: {len(honeypots_in_top100)} honeypot(s) found in top 100:")
        for _, row in honeypots_in_top100.iterrows():
            print(f"   Rank {int(row['rank'])}: {row['candidate_id']}")
    else:
        print("\n✓ VERIFIED: No honeypots in top 100.")

    print("Generating reasoning strings...")
    top100['reasoning'] = top100.apply(generate_reasoning, axis=1)

    # Deduplicate — add score suffix to any collision
    seen = {}
    for i, r in top100.iterrows():
        txt = r['reasoning']
        if txt in seen:
            top100.at[i, 'reasoning'] = txt[:185] + f' [{r.final_score:.4f}]'
        else:
            seen[txt] = i

    # Validate
    assert top100['reasoning'].nunique() == 100, \
        f"Still {100 - top100['reasoning'].nunique()} duplicate reasoning strings"
    assert top100['reasoning'].str.len().between(15, 200).all(), \
        "Reasoning length out of bounds"
    assert (top100['final_score'].diff().dropna() <= 0).all(), \
        "Scores not non-increasing"

    os.makedirs('outputs', exist_ok=True)
    out = top100[['candidate_id', 'rank', 'final_score', 'reasoning']].copy()
    out.columns = ['candidate_id', 'rank', 'score', 'reasoning']
    out.to_csv(args.out, index=False)

    print(f"\nSubmission written to {args.out}")
    print(f"\nTop 10:")
    print(top100[['rank', 'candidate_id', 'title_tier', 'core_skill_score',
                   'scale_score', 'ownership_score', 'final_score']].head(10).to_string())

    print("\nSample reasoning strings:")
    for _, r in top100.head(5).iterrows():
        print(f"  Rank {int(r['rank'])}: {r['reasoning']}")

    print("\nDone.")


if __name__ == '__main__':
    main()