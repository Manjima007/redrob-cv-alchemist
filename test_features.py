from extract_features import extract_features

# Create minimal test candidate
test_candidate = {
    'candidate_id': 'TEST_001',
    'signature': {
        'name': 'Test User',
        'title': 'AI/ML Engineer',
        'headline': 'AI/ML',
        'experience_summary': 'test'
    },
    'experience': [],
    'skills': []
}

feat = extract_features(test_candidate)
print(f'Total features extracted: {len(feat)}')
print('\nFeatures:')
for key in sorted(feat.keys()):
    print(f'  - {key}')
    
# Check if all required features are present
required = [
    'consulting_mult', 'research_mult', 'llm_tourist_mult', 'ic_mult', 'eval_score',
    'title_tier', 'product_ratio', 'core_skill_score'
]
missing = [k for k in required if k not in feat]
if missing:
    print(f'\nMISSING FEATURES: {missing}')
else:
    print('\n✓ All required features present')
