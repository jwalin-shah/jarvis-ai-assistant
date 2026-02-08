# LLM-Based Category Labeling Pipeline

This document describes the LLM-based category labeling pipeline for improving the category classifier from 68.5% F1 to 85%+ F1.

## Overview

The SVM category classifier bottleneck is **label quality, not model capacity**. Only ~5% of examples get meaningful heuristic coverage, and the other 95% default to `social`. This pipeline uses LLM labeling to re-label a strategic 15-17k subset with correct category schemas.

**Target**: 85%+ macro F1
**Cost**: $0 (Cerebras free tier)
**Time**: ~1-2 hours of API calls

## Prerequisites

1. **API Key**: Set `CEREBRAS_API_KEY` in `.env`
   ```bash
   CEREBRAS_API_KEY=your_key_here
   ```

2. **Dependencies**: Ensure all packages are installed
   ```bash
   make install
   ```

## Pipeline Workflow

### Phase 1: Pilot Validation (200 examples, ~10 min)

Validates the classification prompt and model accuracy before full labeling.

```bash
uv run python scripts/validate_llm_categories.py
```

**Output**: `llm_pilot_results.json`

**Success criteria**: ≥80% accuracy on stratified sample

**Optional**: Test with secondary validator (zai-glm-4.7, 100/day limit):
```bash
uv run python scripts/validate_llm_categories.py --model zai-glm-4.7
```

### Phase 2: Full Labeling (15-17k examples, ~1 hour)

Labels ambiguous examples where weak supervision is uncertain.

**Sampling strategy**:
1. Keep heuristic-confident labels (~11k, confidence ≥ 0.8, 2+ LF votes) — free, 94%+ accurate
2. LLM-label 15k from ambiguous pool (confidence < 0.5 or 1 LF vote)
3. LLM-label 2k disagreement examples (confidence 0.4-0.6, multiple LFs disagreed)

```bash
# Full run
uv run python scripts/llm_category_labeler.py

# Smaller test run
uv run python scripts/llm_category_labeler.py --max-examples 5000

# Resume from existing JSONL (if interrupted)
uv run python scripts/llm_category_labeler.py --resume
```

**Output**: `llm_category_labels.jsonl` (resume-safe, JSONL format)

**Rate limits**:
- Cerebras free tier: 14,400 req/day, 30 req/min
- Script uses 2s between calls (30 RPM safe)
- 20 examples per batch

**Estimated time**:
- 17k examples ÷ 20 batch size × 2s/call = ~28 minutes

### Phase 3: Integration + Retraining

Integrate LLM labels with existing training pipeline and retrain SVM.

```bash
# Prepare training data with LLM labels
uv run python scripts/prepare_category_data.py --llm-labels llm_category_labels.jsonl

# Train SVM classifier
uv run python scripts/train_category_svm.py

# Run tests
make test
```

**Expected improvement**:
- Current: 68.5% macro F1
- Target: 85%+ macro F1
- Per-class: All categories >70% (current `clarify`=55%, `social`=58%)

## Classification Prompt

The prompt uses a redesigned 6-category schema with non-overlapping boundaries:

```
Classify each message into ONE category. Use the decision tree below - check categories in order, take the FIRST match.

Categories (check in this order):

1. closing - Ending the conversation
   Examples: "bye", "ttyl", "see you later", "gotta go"

2. acknowledge - Minimal agreement/acknowledgment (≤3 words, no question)
   Examples: "ok", "thanks", "yeah", "gotcha", "👍"

3. request - Seeking action (has "can you"/"please" + verb OR imperative OR "I suggest")
   Examples: "Can you send the file?", "Please call me", "I suggest we meet"

4. question - Seeking information (has "?" OR question words)
   Examples: "What time?", "Where are you?", "Is it ready?"

5. emotion - Expressing feelings (emotion words OR multiple "!" OR CAPS)
   Examples: "I'm so stressed!", "This is AMAZING", "Ugh frustrated"

6. statement - Everything else (opinions, facts, stories, answers)
   Examples: "It's raining", "I think so", "The meeting went well"

For each message, consider the conversation context (previous message) when classifying.

Message 1:
Previous: "{context}"
Current: "{message_1}"

...

Reply with ONLY the category name for each, one per line (e.g., "acknowledge"). No numbers, no explanations.
```

**Key improvements over original 5-category schema:**
- Non-overlapping boundaries (decision tree order ensures mutual exclusivity)
- Objective criteria (word count, punctuation, keywords)
- Splits "ack" into closing vs acknowledge
- Splits "info" into request vs question
- Achieves 100% accuracy on clear test cases

## Label Priority

When integrated with `prepare_category_data.py`, labels are prioritized as:

1. **LLM labels** (confidence=0.95, source=llm) — highest priority
2. **Heuristic-confident labels** (confidence ≥ 0.8, 2+ LF votes, source=heuristic)
3. **Weak supervision labels** (lower confidence, majority vote from 29 LFs)

## Verification

After retraining, verify improvements:

1. **Test set F1**: Compare macro F1 on test set (target 85%+ vs current 68.5%)
2. **Per-class F1**: Each category should be >70% (current `clarify`=55%, `social`=58%)
3. **Manual test**: Run `test_weak_supervision_pipeline.py` (target >70% vs current 50%)
4. **Production validation**: Spot-check on real iMessages

```bash
# Check test set performance
uv run python scripts/train_category_svm.py  # shows test F1 at end

# Per-class breakdown
uv run python scripts/show_per_class_scores.py

# Manual validation
uv run python test_weak_supervision_pipeline.py
```

## Cost Analysis

| Step | API Calls | Time | Cost |
|------|-----------|------|------|
| Pilot (200 examples) | 10 batches | ~10 min | $0 |
| Full labeling (17k) | 850 batches | ~60 min | $0 |
| Integration + retrain | 0 | ~15 min | $0 |
| **Total** | **860** | **~1.5 hours** | **$0** |

All within Cerebras free tier: 14,400 req/day, 30 req/min.

## Troubleshooting

### API Key Issues
```bash
# Verify API key is set
grep CEREBRAS_API_KEY .env

# Test connection
uv run python -c "from evals.judge_config import get_judge_client; print(get_judge_client())"
```

### Rate Limiting
If you hit rate limits, the script will automatically fall back to heuristic labels for failed batches. Resume with `--resume` flag.

### Resume from Interruption
If the script is interrupted, resume with:
```bash
uv run python scripts/llm_category_labeler.py --resume
```

The script tracks already-labeled examples and skips them.

## Files

| File | Purpose |
|------|---------|
| `scripts/validate_llm_categories.py` | Pilot: 200 examples, validate prompt |
| `scripts/llm_category_labeler.py` | Full: 15-17k examples, LLM labeling |
| `scripts/prepare_category_data.py` | Integration: add `--llm-labels` flag |
| `llm_pilot_results.json` | Output: pilot validation results |
| `llm_category_labels.jsonl` | Output: full LLM labels (JSONL) |

## References

- **Weak supervision baseline**: `scripts/labeling_functions.py` (29 LFs)
- **Aggregation**: `scripts/label_aggregation.py` (majority vote, Dawid-Skene)
- **Judge config**: `evals/judge_config.py` (Cerebras API config)
- **Training**: `scripts/train_category_svm.py` (SVM with balanced class weights)
