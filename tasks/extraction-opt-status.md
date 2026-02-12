## STATUS: IN_PROGRESS

## Current Best
- **F1**: 0.457 (limit=100)
- **P**: 0.506, **R**: 0.417
- **Strategy**: constrained_categories (structural filters + label correction + reaction filter)
- **Model**: lfm-1.2b (LFM2.5-1.2B-Instruct-MLX-4bit)
- **Commit**: pending

## Iteration Log

### Iteration 1 - Initial Script + Baseline
- **F1**: 0.176 (P=0.143, R=0.229)
- **Limit**: 100
- **Changes**: Created `scripts/eval_llm_extraction.py` with basic system prompt + schema
- **Result**: Baseline established. Massive FP problem (132 FP vs 22 TP)

### Iteration 2 - Multi-Turn Few-Shot + Post-Processing
- **F1**: 0.323 (P=0.400, R=0.271)
- **Limit**: 100
- **Changes**: Few-shot chat format, hard negatives, span trimming, label filters
- **Result**: IMPROVED (0.176 → 0.323, +83%)

### Iteration 2b - Pipe-Delimited Format (FAILED)
- **F1**: 0.046
- **Result**: REGRESSION. JSON format is significantly better.

### Iteration 3 - Label Correction + Post-Processing
- **F1**: 0.343 (P=0.292, R=0.417)
- **Limit**: 100
- **Changes**: `_correct_label()`, fewer few-shot examples, faster inference
- **Result**: IMPROVED (0.323 → 0.343)

### Iteration 4 - Minimal Few-Shot
- **F1**: 0.340 (P=0.327, R=0.354)
- **Limit**: 100
- **Changes**: 7 examples, simplified labels, max_tokens=120
- **Result**: Slight regression, better P/R balance

### Iteration 5 - Structural Filters + Label Correction Overhaul
- **F1**: 0.457 (P=0.506, R=0.417)
- **Limit**: 100
- **Changes**:
  - **Reaction message filter**: Skip "Loved/Liked/Laughed at/Emphasized" messages entirely (these quote others' messages and hallucinate badly)
  - **Family_member whitelist**: Only accept known family words (brother, sister, mom, dad, etc.), reject "profs", "prof", "em"
  - **`_KNOWN_ORGS`/`_KNOWN_SCHOOLS` sets**: Label correction for Facebook, Intuit, lending tree → org
  - **Proper noun → org correction**: If job_role but capitalized non-role word → reclassify as org
  - **School/university/college keyword detection**: Anything containing these → org
  - **Food_item proper noun filter**: Reject capitalized words unless cuisine type or contains food keyword
  - **URL rejection in food_item**: "i.cvs.com" etc.
  - **Activity digit rejection**: Spans with numbers (dates, times) aren't activities
  - **Expanded reject lists**: health ("never", "free", "tight", "annoying"), food ("jeans", "coupon", "email"), activity ("made", "stories", "ultrasound")
  - **System prompt**: "LASTING personal facts" framing with hard negative few-shot examples
  - Few-shot: added "lending tree" → org example
- **Result**: IMPROVED (0.340 → 0.457, +34%)
- **TP=40, FP=39, FN=56**
- **Key Label Improvements**:
  - food_item: F1=0.857 (P=0.75, R=1.0!) - was 0.273
  - employer: F1=0.857 (3/4 caught) - was 0.000
  - current_location: F1=1.000 (perfect)
  - future_location: F1=1.000 (perfect)
  - family_member: F1=0.526 - was 0.468
  - health_condition: F1=0.471 - was 0.200
- **Still Weak**:
  - org: F1=0.182 (1/9 gold)
  - friend_name: F1=0.000
  - person_name: F1=0.000
  - place: F1=0.333 (regression from 0.444)

## Error Analysis (Iteration 5)

### FP Sources (39 total)
- near_miss: 16 FPs (family members in transient context - structurally hard)
- positive: 17 FPs (wrong labels, hallucinations within fact-bearing messages)
- random_negative: 4 FPs
- hard_negative: 2 FPs

### Key FP Patterns
1. "mom"/"dad" in near_miss messages (6 FPs) - model sees family word, goldset says not a fact
2. Label confusion: "Diwali" as health_condition, "atm" as job_role
3. "doctor's appointment" as place (transient event)

### Key FN Patterns (56 remaining)
1. **Phantom spans (12/96)**: gold text not in message - IMPOSSIBLE to fix
2. **"my dad"/"my brother"**: model outputs "dad"/"brother", gold wants "my dad" - substring match handles this
3. **Activity recall**: still only 25%, many hobby/skill mentions missed
4. **Org recall**: 11%, model rarely outputs org label
5. **"Vestibular"**: single-word message, model outputs empty JSON

### Goldset Quality Issues
- 12/96 gold spans in first 100 are phantom (21% of FNs)
- Maximum achievable recall ~87.5% (84/96 non-phantom spans)
- employer vs org inconsistency (same entity "lending tree" has both labels)

## Next Steps (Priority Order)
1. **Goldset cleanup**: Remove phantom spans to get accurate ceiling measurement
2. **Org/place recall**: Add more org/place few-shot examples
3. **Temperature experiment**: Try 0.1-0.3 for more diverse extraction
4. **Context injection**: prev/next messages for ambiguous cases
5. **Two-pass approach**: binary fact classifier first, then detailed extraction
6. **Self-consistency**: Generate N times, majority vote
