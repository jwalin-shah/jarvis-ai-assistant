## STATUS: IN_PROGRESS

## Current Best
- **F1**: 0.340 (limit=100)
- **P**: 0.327, **R**: 0.354
- **Strategy**: constrained_categories (minimal few-shot + post-processing filters)
- **Model**: lfm-1.2b (LFM2.5-1.2B-Instruct-MLX-4bit)
- **Commit**: pending

## Iteration Log

### Iteration 1 - Initial Script + Baseline
- **F1**: 0.176 (P=0.143, R=0.229)
- **Limit**: 100
- **Changes**: Created `scripts/eval_llm_extraction.py` with basic system prompt + schema
- **Result**: Baseline established. Massive FP problem (132 FP vs 22 TP)
- **Key Issues**: Model outputs full sentences as spans, hallucinates on negatives

### Iteration 2 - Multi-Turn Few-Shot + Post-Processing
- **F1**: 0.323 (P=0.400, R=0.271)
- **Limit**: 100
- **Changes**:
  - Restructured to multi-turn few-shot chat format (11 examples)
  - Added hard negative examples (greetings, transient events, phone comparisons)
  - Added span trimming: `_trim_span()` for overly long spans
  - Added label-specific post-processing filters (reject common verbs as activities, lowercase as locations, etc.)
  - Added stop-word rejection list, short-message filter (<8 chars)
  - Aligned few-shot examples with goldset behavior (family mentions ARE facts)
- **Result**: IMPROVED (0.176 → 0.323, +83%)
- **TP=26, FP=39, FN=70**
- **Breakdown**:
  - family_member: F1=0.468 (best)
  - place: F1=0.444
  - past_location: F1=0.667 (small sample)
  - activity: F1=0.244 (low recall)
  - org: F1=0.200 (only 1 TP out of 9 gold)
  - health_condition: F1=0.200
  - job_role: F1=0.000

### Iteration 2b - Pipe-Delimited Format (FAILED)
- **F1**: 0.046 (P=0.091, R=0.031)
- **Limit**: 100
- **Changes**: Tried simpler pipe-delimited output format (entity|label per line)
- **Result**: REGRESSION. Model produces entities but with wrong labels. JSON format is significantly better for this model.

### Iteration 3 - Label Correction + Post-Processing Filters
- **F1**: 0.343 (P=0.292, R=0.417)
- **Limit**: 100
- **Changes**:
  - Added `_correct_label()` heuristic: family words → family_member, health keywords → health_condition, job titles → job_role
  - Improved label-specific filtering: reject common verbs as activities, multi-word health spans need medical keyword
  - Reduced few-shot to 6 examples (was 14) → faster inference (~3.6s/msg vs 5.1s/msg)
  - Simplified system prompt to be more concise
- **Result**: IMPROVED (0.326 → 0.343, +5%)
- **TP=40, FP=97, FN=56**
- **Key Improvements**:
  - health_condition recall: 0.400 → 0.600 (emergency room now correctly labeled)
  - job_role: 0.000 → 0.667 F1 (product management corrected)
  - current_location: P=1.000 R=1.000 (perfect)

## Error Analysis

### False Positive Patterns (97 FP)
1. **Activity over-extraction** (34 FP): model labels random things as activities
2. **Near-miss family** (34 FP): "mom" from transient event mentions, despite negative examples
3. **Food hallucination** (19 FP): random words labeled as food_item
4. **Health hallucination** (13 FP): "never ended", "dgaf" as health conditions

### False Negative Patterns (56 FN)
1. **Org still zero** (9 FN): model never outputs org label on its own
2. **Activity missed** (15 FN): model can't distinguish hobbies from general actions
3. **Goldset phantoms** (~5 FN): gold text not in message, impossible to extract
4. **Multi-fact messages**: model finds 1-2 of 3-4 facts

### Goldset Quality Issues
- 5/96 gold spans in first 100 records are phantom (text not in message):
  - "python", "SQL" for PM experiences message
  - "1 min walking 1 min run" for dad's exercise message
  - "depression" for mental health message
  - "CrossFit" for Delaware summer message
- These are impossible to extract and cap maximum recall at ~95%

### Iteration 4 - Minimal Few-Shot + Improved Post-Processing
- **F1**: 0.340 (P=0.327, R=0.354)
- **Limit**: 100
- **Changes**:
  - Reduced few-shot to 7 examples (5 positive + 2 negative) for less model confusion
  - Simplified labels: removed current/future/past_location from prompt, using "place" instead
  - Reduced max_tokens from 200 to 120 to prevent verbose output
  - Added stronger health_condition filter: multi-word spans need medical keyword
  - Added food_item filter for "whatever", "everything" etc
  - Added activity filter for "contact", "rest"
- **Result**: Slight regression from 0.343 but P/R balance better
- **TP=34, FP=70, FN=62**
- **Key**: Precision on positive slice improved to 0.500 (was 0.292)
- **Key**: family_member F1=0.476, place F1=0.615

## Error Analysis (Iteration 4)

### False Positive Patterns (70 FP)
1. **Near-miss family** (29 FP): "mom" from transient event mentions
2. **Activity over-extraction** (23 FP): random things labeled as activities
3. **Food hallucination** (13 FP): non-food words as food_item
4. **Job role confusion** (6 FP): non-job words as job_role

### False Negative Patterns (62 FN)
1. **Org zero** (9 FN): model never uses org label in new prompt
2. **Activity missed** (17 FN): model can't distinguish hobbies from actions
3. **Family member missed** (14 FN): "my dad", "My brother" forms missed
4. **Goldset phantoms** (~5 FN): text not in message

## Next Steps (Priority Order)
1. **Two-pass approach**: first classify if message has facts, skip negatives entirely
2. **Activity/food FP filtering**: these are the top FP sources
3. **Temperature tuning**: try 0.1-0.3
4. **Clean goldset**: remove phantom spans
5. **Ensemble with GLiNER**: combine for better coverage
6. **Context injection**: prev/next messages for disambiguation

### Review (iteration 1) - APPROVE
Reviewer: gemini
> (node:10488) [DEP0040] DeprecationWarning: The `punycode` module is deprecated. Please use a userland alternative instead.
> (Use `node --trace-deprecation ...` to show where the warning was created)
> (node:10505) [DEP0040] DeprecationWarning: The `punycode` module is deprecated. Please use a userland alternative instead.
> (Use `node --trace-deprecation ...` to show where the warning was created)
> Loaded cached credentials.

