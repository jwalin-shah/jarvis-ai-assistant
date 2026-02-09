# Context Strategy Analysis

Comprehensive analysis of different context extraction strategies for category classification on iMessage validation data.

## Experiments

We tested three context strategies:
1. **Empty context** (broken SQL join, baseline)
2. **Fixed 5-message window** (simple backfill)
3. **Semantic filtering** (cosine similarity > 0.6)

## Results Summary

| Strategy | Coverage | Avg Length | LinearSVC F1 | LightGBM F1 | Notes |
|----------|----------|------------|--------------|-------------|-------|
| Empty | ~0% | 0 msgs | 0.4733 | **0.7115** | Baseline (broken SQL) |
| Fixed 5 | 95.8% | 5.0 msgs | **0.5453** | 0.7067 | Simple backfill from chat.db |
| Semantic | 82.8% | 3.5 msgs | 0.5052 | 0.7023 | Filtered by embedding similarity |

### Per-Class Performance (Fixed 5 vs Semantic)

**LinearSVC:**
| Class | Fixed 5 | Semantic | Change |
|-------|---------|----------|--------|
| acknowledge | 0.4112 | 0.3964 | -0.015 |
| closing | 0.1818 | 0.2500 | +0.068 |
| emotion | 0.6369 | 0.5930 | -0.044 |
| question | 0.6019 | 0.5905 | -0.011 |
| request | 0.4444 | 0.4000 | -0.044 |
| statement | 0.6632 | 0.6175 | -0.046 |

**LightGBM:**
| Class | Fixed 5 | Semantic | Change |
|-------|---------|----------|--------|
| acknowledge | 0.4706 | 0.4872 | +0.017 |
| closing | 0.0000 | 0.0000 | 0.000 |
| emotion | 0.6500 | 0.6410 | -0.009 |
| question | 0.6296 | 0.6452 | +0.016 |
| request | 0.3265 | 0.3243 | -0.002 |
| statement | 0.7717 | 0.7634 | -0.008 |

## Key Findings

### 1. LightGBM Performs Best with NO Context

**Counterintuitive result:** Adding context makes LightGBM predictions WORSE.

- Empty context: F1 = 0.7115 (best)
- Fixed 5 messages: F1 = 0.7067 (-0.7%)
- Semantic filtered: F1 = 0.7023 (-1.3%)

**Why?**
- LightGBM (tree-based) is robust to missing features
- During training, it learned "if context is available, use it"
- But iMessage context follows different patterns than DailyDialog/SAMSum training data
- The model is misled by context that doesn't match training distribution
- Better to route through non-context features than use low-quality context

**Evidence:** Question and request categories got worse with context:
- Question: 0.6909 (empty) → 0.6296 (fixed) = -6.1 points
- Request: 0.3846 (empty) → 0.3265 (fixed) = -5.8 points

These categories rely on context patterns that differ between clean dialogs and real iMessage.

### 2. LinearSVC Needs Context Quantity, Not Quality

- Empty: 0.4733
- Semantic (3.5 msgs): 0.5052 (+3.2 points)
- Fixed 5 (5.0 msgs): 0.5453 (+7.2 points)

**Why?**
- LinearSVC is a linear model with fixed feature weights
- 384 context BERT features = 42% of total features
- Without context, these features are all zeros (OOD for the model)
- More context = more non-zero features = closer to training distribution
- Quality matters less than just populating the feature space

### 3. Semantic Filtering Doesn't Help

**Hypothesis:** LightGBM got worse because fixed 5-message window includes off-topic noise.

**Test:** Filter last 10 messages by cosine similarity > 0.6 to target message.

**Result:** Semantic filtering (3.5 msgs avg) still performs worse than empty context.

**Conclusion:** The problem isn't context noise - it's **fundamental domain shift**.

## Domain Shift Analysis

### Training Data (DailyDialog/SAMSum)
- Linear conversations on single topics
- Questions/requests have clear setup in prior messages
- Context is always relevant and informative
- Clean, curated language

### Production Data (iMessage)
- Non-linear, topic-jumping conversations
- Questions/requests appear out of the blue ("yo wanna hang?")
- Context often unrelated to current message
- Informal language, slang, emojis, abbreviations

### The Model Learned Wrong Patterns

**Example decision rules LightGBM might have learned:**
```
if context_mentions_availability AND message_asks_question:
    → probably "question" category

if context_describes_problem AND message_requests_action:
    → probably "request" category
```

These rules work on DailyDialog but fail on iMessage where questions/requests are often non-sequiturs.

## Recommendations

### For Production Deployment

1. **Use fixed 5-message context** for the combined classifier
   - Best overall F1: 0.545 (LinearSVC) / 0.707 (LightGBM)
   - Maintains expected feature distribution for LinearSVC
   - LightGBM degradation is acceptable (-0.7%)

2. **Re-tune decision thresholds on iMessage data**
   - Current thresholds optimized for DailyDialog/SAMSum
   - iMessage has different score distributions
   - Use validation sets 2-4 (300 examples) as dev set
   - Use validation set 5 (100 examples) as test set

3. **Consider model-specific strategies** (longer term)
   - LinearSVC: always use context (critical for performance)
   - LightGBM: could train a separate model without context features
   - Ensemble: combine both approaches

### For Future Improvement

1. **Collect more iMessage training data**
   - Label 500-1000 real iMessage examples
   - Include in training set for domain adaptation
   - Reduces gap between training and production distributions

2. **Feature engineering for iMessage**
   - Add features that capture iMessage-specific patterns
   - Topic continuity indicators
   - Conversation recency signals
   - Turn-taking patterns

3. **Context quality indicators**
   - Train model to predict whether context is useful
   - Dynamically weight context features based on quality
   - Use topic segmentation at inference time to get better context

## Files

- `scripts/backfill_validation_context.py` - Simple 5-message backfill (RECOMMENDED)
- `scripts/backfill_validation_context_semantic.py` - Semantic filtering approach
- `scripts/backfill_validation_context_topic_aware.py` - Full topic segmentation (too slow)

## Evaluation Results

- `validation_evaluation_results_v2.txt` - Empty context baseline
- `validation_evaluation_with_context.txt` - Fixed 5-message context
- `validation_evaluation_semantic_context.txt` - Semantic filtered context

## Decision

**Use fixed 5-message context for production.** It provides the best balance of:
- LinearSVC performance (needs context to avoid OOD features)
- Implementation simplicity (no complex filtering logic)
- Coverage (95.8% of messages have context available)

The LightGBM degradation (-0.7%) is acceptable given the overall system benefits.
