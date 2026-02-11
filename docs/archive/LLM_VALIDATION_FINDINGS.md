# LLM Category Validation Findings

## Problem Summary

The validation script (`scripts/validate_llm_categories.py`) was updated to use the new 6-category schema (closing, acknowledge, question, request, emotion, statement), but the heuristic labeling functions still used the old 5-category schema (ack, info, emotional, social, clarify), causing stratification to yield 0 examples.

## Fixes Applied

1. **Added inline simple heuristics** (`simple_heuristic_label()`) matching the 6-category schema for stratification
2. **Fixed hardcoded model** - now uses `args.model` parameter instead of hardcoded `zai-glm-4.7`
3. **Added social→statement mapping** - handles LLM responses that use old "social" category
4. **Improved prompt** - simpler rules, few-shot examples, explicit output format
5. **Reduced batch size** - from 20 to 5 examples per API call
6. **Better response parsing** - handles multiple output formats (plain, numbered, formatted)

## Test Results

### Cerebras Models (Free Tier)

| Model | Sample Size | Batch Size | Accuracy | Issue |
|-------|-------------|------------|----------|-------|
| gpt-oss-120b | 30 (5/cat) | 20 | **80.0%** | ✓ Worked once |
| gpt-oss-120b | 240 (40/cat) | 20 | 39.6% | Reverted to reasoning mode |
| gpt-oss-120b | 60 (10/cat) | 5 | 38.0% | Still defaulting to statement |
| zai-glm-4.7 | 240 (40/cat) | 20 | 18.3% | Extreme statement bias |

**Pattern**: Both models:
- Ignore "no reasoning, no explanations" instructions
- Output reasoning/explanation instead of direct labels
- Default heavily to "statement" category (100% recall, low precision)
- Show high precision (when they predict a category, it's usually correct) but terrible recall

**Root Cause**: These free reasoning models (gpt-oss-120b, zai-glm-4.7) are designed to show their work and explain reasoning, not for structured classification tasks. They treat classification as an opportunity to demonstrate understanding rather than produce clean output.

## Alternative Approaches

### Option 1: Use Paid API (Recommended if budget allows)
- **OpenAI GPT-4**: $0.03/1K tokens input, $0.06/1K tokens output
- **Anthropic Claude 4 Haiku**: $0.25/MTok input, $1.25/MTok output (cheapest)
- **Estimated cost for 17K examples**:
  - ~150 tokens per message × 17,000 = 2.55M tokens
  - Claude Haiku: ~$2-3 total
  - GPT-4 Mini: ~$5-10 total

**Pros**: High accuracy (likely 85-90%+), proven for classification
**Cons**: Costs money (~$2-10 depending on model)

### Option 2: Use Heuristics Only (No LLM)
The existing weak supervision pipeline has:
- 29 labeling functions with weighted voting
- ~94% accuracy on high-confidence examples (confidence ≥ 0.8, 2+ LF votes)
- ~11k high-confidence labels already available

**Approach**: Instead of LLM labeling, improve heuristics:
1. Add more labeling functions for under-represented categories (closing, emotion)
2. Use DailyDialog metadata (act, emotion labels) as additional LFs
3. Increase training data by lowering confidence threshold to 0.6

**Pros**: Free, deterministic, fast
**Cons**: May not reach 85% F1 target, caps out at heuristic quality

### Option 3: Local LLM via Ollama
Install Ollama and use a local model like:
- **LLaMA 3.1 8B Instruct**: Good instruction following, fits in 8GB RAM
- **Mistral 7B Instruct**: Fast, good for classification
- **Qwen 7B**: Strong on structured tasks

**Pros**: Free, no API limits, private
**Cons**: Setup required, slower than API (but still ~1-2 examples/sec)

### Option 4: Hybrid Approach
1. Use heuristics for high-confidence examples (~11k)
2. Use GPT-4o-mini or Claude Haiku for ambiguous cases (~6k)
3. Estimated cost: ~$1-3

**Pros**: Balances cost and quality
**Cons**: More complex pipeline

### Option 5: Accept Current Performance
Current SVM: 68.5% macro F1
- Test if this is actually a problem in production
- Profile real iMessage traffic to see category distribution
- Maybe 68.5% is good enough if most messages are "statement"

## Recommendation

**Short-term**: Try **Option 2 (Heuristics Only)** first:
1. Check current F1 per category with `make test`
2. Add 5-10 new labeling functions targeting weak categories
3. Lower confidence threshold from 0.8 to 0.6 for training
4. Retrain SVM, measure improvement

**If F1 < 80% after heuristic improvements**: Use **Option 4 (Hybrid)** with Claude Haiku (~$2):
- 11k heuristic-confident labels (free)
- 6k LLM-labeled ambiguous examples ($1-2)
- Total cost: ~$2 for 17k labeled examples

**Not recommended**: Continue trying to fix Cerebras free models - they're fundamentally not suited for this task.

## Next Steps

1. Decide on labeling approach (see options above)
2. If using paid API: update `evals/judge_config.py` with provider
3. If using heuristics: run `scripts/add_labeling_functions.py` to extend LF registry
4. Run full labeling pipeline
5. Train SVM with new labels
6. Verify F1 improvement with `make test`

## Files Modified

- `scripts/validate_llm_categories.py`:
  - Added `simple_heuristic_label()` for 6-category stratification
  - Fixed model parameter bug
  - Improved prompt with few-shot examples
  - Better response parsing
  - Reduced batch size to 5
  - Added debug output for first batch
