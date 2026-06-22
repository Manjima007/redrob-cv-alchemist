import json
import ijson
import numpy as np
import pandas as pd
import lightgbm as lgb
import argparse
import os

from extract_features import extract_features

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


def compute_pseudo_labels(df):
    engagement = (
        df['response_rate']        * 0.30 +
        df['interview_completion'] * 0.25 +
        df['offer_acceptance']     * 0.20 +
        df['recency_score']        * 0.15 +
        df['open_to_work']         * 0.10
    )

    core_norm = df['core_skill_score'] / df['core_skill_score'].clip(lower=0.01).max()

    jd_fit = (
        df['title_tier']           * 0.35 +
        core_norm                  * 0.30 +
        df['ownership_score']      * 0.15 +
        df['scale_score']          * 0.10 +
        df['decision_depth_score'] * 0.10
    )

    pseudo = engagement * 0.45 + jd_fit * 0.55

    pseudo *= (1 - df['is_consulting_only'])
    pseudo *= (1 - df['honeypot_flag'])
    pseudo *= (1 - df['langchain_only_flag'])
    title_penalty = (df['title_tier'] / 0.45).clip(upper=1.0) ** 2
    pseudo *= title_penalty

    max_val = pseudo.max()
    if max_val > 0:
        pseudo = pseudo / max_val

    return pseudo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', default='data/candidates.jsonl')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--skip-embeddings', action='store_true', help='Skip sentence-transformer embeddings (speeds up precompute)')
    args = parser.parse_args()

    # ── Skip re-streaming if features already exist ───────────────────────────
    features_path = 'outputs/features.parquet'
    if os.path.exists(features_path) and not args.sample:
        print("Found existing outputs/features.parquet — skipping extraction.")
        print("Delete outputs/features.parquet to re-extract from scratch.")
        df = pd.read_parquet(features_path)
    else:
        rows = []

        if args.sample:
            print("Running on sample_candidates.json...")
            with open('data/sample_candidates.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            for c in data:
                feat = extract_features(c)
                feat['candidate_id'] = c['candidate_id']
                rows.append(feat)
        else:
            print(f"Streaming {args.candidates} ...")
            with open(args.candidates, 'rb') as f:
                for i, c in enumerate(ijson.items(f, '', multiple_values=True)):
                    feat = extract_features(c)
                    feat['candidate_id'] = c['candidate_id']
                    rows.append(feat)
                    if (i + 1) % 10000 == 0:
                        print(f"  Processed {i+1} candidates...")

        print(f"Total candidates processed: {len(rows)}")
        df = pd.DataFrame(rows)

        # Run logical temporal validations before pseudo-labeling
        try:
            from logical_validation import check_temporal_consistency
            df = check_temporal_consistency(df)
        except Exception:
            pass

        # Compute pseudo labels
        df['pseudo_label'] = compute_pseudo_labels(df)

        # Force float64 before saving (exclude text-like columns)
        numeric_cols = FEATURE_COLS + ['honeypot_flag', 'is_consulting_only', 'langchain_only_flag', 'notice_days',
                                       'skill_assessments_max', 'expected_salary_min', 'expected_salary_max']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float64')

        os.makedirs('outputs', exist_ok=True)
        df.to_parquet(features_path, index=False)
        print(f"Saved {features_path}")

    # ── Optional: compute sentence-transformers embeddings if available ─────
    embeddings_path = 'outputs/embeddings.npy'
    if args.skip_embeddings:
        print("Skipping embeddings (--skip-embeddings flag set)")
    elif os.path.exists(embeddings_path) and not args.sample:
        print(f"Found existing {embeddings_path} — skipping embedding computation.")
        print("Delete outputs/embeddings.npy to recompute.")
    else:
        try:
            import time as _time
            from sentence_transformers import SentenceTransformer
            import numpy as _np

            anchor_texts = [
                "Experienced machine learning engineer with production ML systems",
                "Research scientist with published papers in top conferences",
                "Prompt engineer and LLM practitioner",
                "Consultant working on short-term client projects",
            ]

            print("sentence-transformers available — computing embeddings (this may download model weights)...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            model.max_seq_length = 256  # explicit, don't rely on the default

            # Truncate raw text BEFORE tokenization. The model truncates at 256 tokens
            # anyway, so anything beyond ~1500 characters is wasted tokenizer work — and a
            # single very long outlier can otherwise blow up padding cost for its whole batch.
            texts = df['agg_text'].fillna('').astype(str).str.slice(0, 1500).tolist()

            print(f"Encoding {len(texts)} candidate texts...")
            t0 = _time.time()
            emb = model.encode(
                texts,
                batch_size=64,               # smaller batches tend to be faster on CPU than 1024
                show_progress_bar=True,      # so you can see it moving and estimate ETA
                convert_to_numpy=True,
                normalize_embeddings=True,   # makes cosine similarity a plain dot product later
            )
            print(f"Encoded in {_time.time() - t0:.1f}s")

            os.makedirs('outputs', exist_ok=True)
            _np.save(embeddings_path, emb)
            _np.save('outputs/emb_ids.npy', df['candidate_id'].values)

            anchor_emb = model.encode(anchor_texts, normalize_embeddings=True)
            _np.save('outputs/anchor_embeddings.npy', anchor_emb)
            import json as _json
            _json.dump(anchor_texts, open('outputs/anchor_texts.json', 'w', encoding='utf-8'))
            print('Saved embeddings and anchors to outputs/')
        except Exception as e:
            print('Skipping embeddings — sentence-transformers unavailable or failed:', type(e).__name__, str(e))

    # ── Sanity check ─────────────────────────────────────────────────────────
    top10 = df.nlargest(10, 'pseudo_label')[['candidate_id', 'title_tier', 'core_skill_score', 'pseudo_label']]
    print("\nTop 10 by pseudo_label:")
    print(top10.to_string())
    bottom5 = df.nsmallest(5, 'pseudo_label')[['candidate_id', 'title_tier', 'pseudo_label']]
    print("\nBottom 5:")
    print(bottom5.to_string())

    # ── Train LightGBM LambdaRank ─────────────────────────────────────────────
    print("\nTraining LightGBM LambdaRank...")

    X = df[FEATURE_COLS].fillna(0).values
    y = df['pseudo_label'].values

    # LambdaRank requires integer labels (relevance grades 0-4)
    y_int = pd.cut(
        y,
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.001],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # LambdaRank has a 10K row limit per group — split into groups of 1000
    group_size = 1000
    n = len(df)
    group = [group_size] * (n // group_size)
    if n % group_size != 0:
        group.append(n % group_size)

    train_data = lgb.Dataset(X, label=y_int, group=group,
                             feature_name=FEATURE_COLS)

    params = {
        'objective':        'lambdarank',
        'metric':           'ndcg',
        'ndcg_eval_at':     [10, 100],
        'label_gain':       [0, 1, 3, 7, 15],
        'num_leaves':       63,
        'learning_rate':    0.05,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq':     5,
        'verbose':          -1,
    }

    model = lgb.train(params, train_data, num_boost_round=300)
    model.save_model('outputs/ltr_model.txt')
    print("Saved outputs/ltr_model.txt")

    # Feature importance
    importance = dict(zip(FEATURE_COLS, model.feature_importance(importance_type='gain')))
    print("\nTop 10 feature importances:")
    for k, v in sorted(importance.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v:.1f}")

    print("\nPrecompute complete.")


if __name__ == '__main__':
    main()