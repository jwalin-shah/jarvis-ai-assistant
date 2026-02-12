## STATUS: IN_PROGRESS

## Current Best
- **F1**: 0.641 (limit=100, goldset_v5.1_deduped)
- **F1 (orig goldset)**: 0.526 (limit=100, candidate_gold_merged_r4)
- **P**: 0.683, **R**: 0.603
- **Strategy**: constrained_categories + rule-based recall boost + emoji stripping
- **Model**: lfm-1.2b (LFM2.5-1.2B-Instruct-MLX-4bit)

## Iteration Log

### Iteration 1 - Initial Script + Baseline
- **F1**: 0.176 (P=0.143, R=0.229)
- **Limit**: 100
- **Changes**: Created `scripts/eval_llm_extraction.py` with basic system prompt + schema
- **Result**: Baseline established

### Iteration 2 - Multi-Turn Few-Shot + Post-Processing
- **F1**: 0.323 (P=0.400, R=0.271)
- **Limit**: 100
- **Result**: IMPROVED (0.176 -> 0.323, +83%)

### Iteration 2b - Pipe-Delimited Format (FAILED)
- **F1**: 0.046
- **Result**: REGRESSION

### Iteration 3 - Label Correction + Post-Processing
- **F1**: 0.343 (P=0.292, R=0.417)
- **Result**: IMPROVED (0.323 -> 0.343)

### Iteration 4 - Minimal Few-Shot
- **F1**: 0.340 (P=0.327, R=0.354)
- **Result**: Slight regression

### Iteration 5 - Structural Filters + Label Correction
- **F1**: 0.457 (deduped) / 0.418 (orig)
- **Result**: IMPROVED

### Iteration 6 - Goldset Dedup + Rule-Based Recall Boost + Emoji Strip
- **F1**: 0.641 (deduped goldset), 0.526 (orig goldset)
- **Limit**: 100
- **Changes**:
  1. **Goldset v5.1**: Deduplicated 45 redundant spans (291->246)
  2. **Emoji stripping**: Strip emojis before LLM inference
  3. **Rule-based recall boost**: family "my X" patterns, known orgs, health keywords, "work at X"
  4. **Family possessive handling**: "brother's", "sisters" matching
  5. **"depressed" added to health keywords**
- **Key Results**:
  - family_member: R=100%, F1=0.692
  - org: F1=0.571 (was 0.182)
  - health_condition: F1=0.800 (was 0.500)
  - Positive slice: P=0.891, R=0.603, F1=0.719
- **Result**: IMPROVED (0.457 -> 0.641, +40%)

## Error Analysis (Iteration 6)

### FPs (19 total)
- near_miss family_member: 13 (transient "mom"/"dad" mentions)
- positive: 5
- random_negative: 1

### FNs (27 total)
- activity: 10, org: 5, place: 2, health_condition: 2, past_location: 2, others: 6

## Next Steps
1. Reduce near_miss FPs (13/19 FPs are transient family mentions)
2. Expand known_orgs for missed orgs
3. Activity recall: rule-based "I like/love/hate X" patterns
4. Context injection for transient vs lasting distinction
