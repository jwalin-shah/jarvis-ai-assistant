# Classifier Optimization Experiment Plan

## Goals
1. Find optimal dataset size for response classifier
2. Find optimal C value for each dataset size (C may change with size)
3. Save ALL results for reproducibility
4. Use proper train/test methodology

## Current State
- Responses: 4,865 labeled examples
- Triggers: 4,865 labeled examples
- Available in DB: ~106k unlabeled responses, ~111k unlabeled triggers
- High-confidence (>=90%) estimate: ~34k responses, ~33k triggers

## Experiment Structure

### Phase 1: Data Preparation (run once, save everything)

```
1. Auto-label ALL available data with current classifier
2. Filter to high-confidence (>=90%) only
3. Save to: experiments/data/all_high_conf_responses.jsonl
4. Create FIXED holdout test set (15% stratified)
5. Save to: experiments/data/test_set.jsonl
6. Save remaining as training pool: experiments/data/train_pool.jsonl
```

### Phase 2: Coarse Search (every 5k)

Test sizes: 3k, 5k, 10k, 15k, 20k, 25k, 30k

For EACH size:
```
1. Take top N from training pool (sorted by confidence)
2. Test C values: [1, 2, 5, 10, 20]
3. Use 5-fold cross-validation on training subset
4. Evaluate best C on FIXED test set
5. Save results to: experiments/results/size_{N}_results.json
```

Output per size:
```json
{
  "size": 10000,
  "best_c": 10,
  "cv_scores": {"c_1": 0.72, "c_2": 0.75, ...},
  "test_f1": 0.82,
  "per_class_f1": {...},
  "timestamp": "..."
}
```

### Phase 3: Fine Search (every 1k around best)

Once we find best size range (e.g., 15k-20k performs best):
```
1. Test: 15k, 16k, 17k, 18k, 19k, 20k
2. For each, test C values again
3. Save results
```

### Phase 4: Final Model

```
1. Select optimal (size, C) combination
2. Train on full training data of that size
3. Evaluate on test set (final score)
4. Save model to: experiments/models/response_classifier_v2/
5. Save config with all parameters
```

## Directory Structure

```
experiments/
├── classifier_optimization_plan.md  (this file)
├── data/
│   ├── all_high_conf_responses.jsonl
│   ├── all_high_conf_triggers.jsonl
│   ├── test_set_responses.jsonl
│   ├── test_set_triggers.jsonl
│   ├── train_pool_responses.jsonl
│   └── train_pool_triggers.jsonl
├── results/
│   ├── coarse_search_responses.json
│   ├── coarse_search_triggers.json
│   ├── fine_search_responses.json
│   └── fine_search_triggers.json
└── models/
    ├── response_classifier_v2/
    │   ├── model.pkl
    │   └── config.json
    └── trigger_classifier_v2/
        ├── model.pkl
        └── config.json
```

## Key Questions

### Does C change with dataset size?

Hypothesis: YES, because:
- Smaller datasets → lower C (avoid overfitting to limited data)
- Larger datasets → higher C (more data allows finer boundaries)

We test this by finding optimal C at EACH size.

### What's the stopping criterion?

Stop expanding when:
- Test F1 improvement < 0.5% for 2 consecutive size increases
- OR we run out of high-confidence data

### How do we ensure fair comparison?

1. SAME test set for all experiments (created once, never touched)
2. Stratified sampling (maintain label distribution)
3. Same random seed (42) everywhere
4. Save everything for reproducibility

## Performance Optimization

### Embedding Caching
- Compute embeddings ONCE for all data
- Save to: experiments/data/embeddings_cache.npy
- Reuse across all experiments

### Parallel C Search
- Test multiple C values in parallel where possible

### Early Stopping
- If CV score plateaus, skip remaining C values

## Estimated Runtime

- Phase 1 (data prep): ~10 min (embedding 100k+ texts)
- Phase 2 (coarse search): ~20 min (7 sizes × 5 C values)
- Phase 3 (fine search): ~10 min (6 sizes × 5 C values)
- Phase 4 (final): ~2 min

Total: ~45 min

## Output Summary

At the end, we'll have:
1. Complete results for every (size, C) combination tested
2. Optimal configuration identified
3. Final model saved
4. All intermediate data saved for future analysis

## Questions for Review

1. Should we test C values beyond 20? (e.g., 50, 100)
2. Should we also test gamma parameter? (currently 'scale')
3. Do you want to run responses first, then triggers? Or both in parallel?
4. Is 15% holdout appropriate? Or prefer 20%?

---

Ready to execute when you approve.
