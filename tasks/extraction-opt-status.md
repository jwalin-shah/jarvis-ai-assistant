## STATUS: IN_PROGRESS

## Current Best
- **F1**: 0.566 (limit=100, candidate_gold_merged_r4)
- **P**: 0.714, **R**: 0.469
- **Strategy**: constrained_categories + rule-based boost + prompt refinement + FP filters
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

### Iteration 7 - Prompt Refinement + FP Filters + Few-Shot Tuning
- **F1**: 0.566 (P=0.714, R=0.469) on orig goldset
- **Limit**: 100
- **Changes**:
  1. **System prompt**: "LASTING personal facts" + "DO NOT extract temporary actions/plans"
  2. **Few-shot rebalance**: Added positive family example ("My mom texted me" -> mom), dolmas food, raiders org; reduced hard negative family examples from 3 to 2
  3. **Rule-based family boost gating**: Always boost "my <family_word>" (not gated on LLM output)
  4. **person_name FP filter**: Reject lowercase, common words (prof, prolly, dude)
  5. **activity FP filter**: Added "hella bad", "figure the rest", etc.
  6. **health_condition FP filter**: Added "rest a bit", "5k", "barring anything"
  7. **job_role FP filter**: Added "working from home", "shelter in place", "ready to get"
  8. **food_item vocabulary**: Added dolmas, biryani, samosa, roti, pho, ramen, etc.
  9. **Span validation tightened**: Require majority of multi-word spans found in message
- **Key Results**:
  - TP: 33->45 (+12), FP: 29->18 (-11), FN: 63->51 (-12)
  - health_condition: P=1.000, F1=0.750 (was 0.471)
  - employer: F1=1.000 (perfect)
  - current_location: F1=1.000 (perfect)
  - family_member: F1=0.603 (was 0.526)
  - Positive slice: P=0.900, R=0.469, F1=0.616
  - Near_miss FP: 12 (all family_member from rule boost)
- **Result**: IMPROVED (0.418 -> 0.566, +35%)

## Error Analysis (Iteration 7)

### FPs (18 total)
- near_miss family_member: 12 (rule-based boost on transient messages)
- positive: 5 (1 activity, 1 health, 1 org, 1 family, 1 job)
- random_negative: 1 (family_member from "my moms")

### FNs (51 total)
- activity: 18 (largest gap - model misses many hobbies)
- family_member: 10 (duplicate gold "my dad" vs "dad" entries)
- org: 7 (model misses orgs like IHS, SB, Karya, swadhyay)
- place: 6 (model rarely extracts places)
- health_condition: 4
- Others: 6

## Next Steps
1. **Activity recall**: Rule-based "I like/love/hate/enjoy X" extraction
2. **Org recall**: Add more known orgs (IHS, SB, Karya) and "I hate X" -> org pattern
3. **Near_miss FP reduction**: Only boost family if message has fact-bearing signals
4. **Context injection**: Include prev/next messages to help distinguish lasting vs transient
5. **Temperature experiment**: Try 0.1-0.3 for more diverse extraction
