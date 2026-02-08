# Context Backfill Results

## Summary

Backfilled context for `validation_sets_multilabel.jsonl` (combined validation set) by looking up messages in chat.db and retrieving up to 5 prior messages per example.

**Context Coverage**: 383/400 examples (95.8%) now have real context from chat history.

## Performance Comparison: Before vs After Context Backfill

### Combined Validation Set (400 examples)

#### Before Backfill (Empty Context)
From `validation_evaluation_results_v2.txt` (run with ~0% context coverage):

**LinearSVC (915 features, tuned)**
- F1 (samples): 0.4733
- F1 (macro): 0.4127
- Hamming loss: 0.1429
- Jaccard: 0.4600

**LightGBM (915 features, tuned)**
- F1 (samples): 0.7115
- F1 (macro): 0.4928
- Hamming loss: 0.1375
- Jaccard: 0.6496

#### After Backfill (95.8% Context)
From `validation_evaluation_with_context.txt`:

**LinearSVC (915 features, tuned)**
- F1 (samples): **0.5453** (+0.072, +15.2%)
- F1 (macro): **0.4899** (+0.077, +18.7%)
- Hamming loss: 0.1400 (-0.003, -2.0%)
- Jaccard: **0.5146** (+0.055, +11.9%)

**LightGBM (915 features, tuned)**
- F1 (samples): 0.7067 (-0.005, -0.7%)
- F1 (macro): 0.4747 (-0.018, -3.7%)
- Hamming loss: 0.1358 (-0.002, -1.2%)
- Jaccard: 0.6519 (+0.002, +0.4%)

## Analysis

### LinearSVC Improved Significantly
- **F1 (samples)** improved by **7.2 percentage points** (15.2% relative)
- **F1 (macro)** improved by **7.7 percentage points** (18.7% relative)
- All metrics improved with proper context

**Why?** LinearSVC relies heavily on the 384-dim context BERT embeddings. With empty context, these features were all zeros (a pattern the model never saw during training on DailyDialog/SAMSum). Backfilling context restored the proper feature distribution.

### LightGBM Performance Stayed Flat
- Essentially no change (within noise margin)
- LightGBM was already performing well without context

**Why?** LightGBM (tree-based model) is more robust to missing features than LinearSVC. It learned to route predictions through non-context features when context embeddings were all zeros. It didn't benefit much from restored context because it already adapted to the missing data pattern.

## Per-Class Breakdown: LinearSVC Before/After

| Class | Before F1 | After F1 | Change |
|-------|-----------|----------|--------|
| acknowledge | 0.3765 | **0.4112** | +0.035 |
| closing | 0.0000 | **0.1818** | +0.182 |
| emotion | 0.4265 | **0.6369** | +0.210 |
| question | 0.6000 | **0.6019** | +0.002 |
| request | 0.4407 | **0.4444** | +0.004 |
| statement | 0.6324 | **0.6632** | +0.031 |

**Biggest improvements**: `emotion` (+21.0 points), `closing` (+18.2 points)

These categories benefit most from conversational context to determine tone and conversation flow.

## Conclusion

**Context is critical for LinearSVC** - the 384-dim context BERT embeddings are a key part of the 915-feature model. Without real context, LinearSVC F1 drops by ~15-19%. With proper context backfilled, LinearSVC performance improves significantly.

**LightGBM is robust to missing context** - tree-based models handle sparse/zero features better and learned to route around missing context during training.

**Next steps**: The validation set now has proper context (95.8% coverage), giving us a realistic estimate of production performance on iMessage data. LinearSVC F1 (samples) = 0.545 is the true validation performance with context.

The remaining gap vs test set performance (LinearSVC test F1 ~0.76 on DailyDialog/SAMSum) is likely due to:
1. Domain shift: casual iMessage conversations vs curated dialog datasets
2. Label distribution: iMessage has different class balance than training data
3. Message style: texting shorthand, emojis, slang vs cleaner dialog text
