# DailyDialog Native Dialog Acts Classifier Experiments

## Overview

This experiment framework systematically searches for the optimal configuration to classify iMessages using DailyDialog's native dialog act labels instead of custom categories.

**Problem**: Current category classifier achieves only 0.38 F1 due to:
- Severe class imbalance (9.6x ratio)
- Subjective custom categories that don't match natural data
- Inefficient training (memory constraints)

**Solution**: Use DailyDialog's native dialog acts (ISO 24617-2 standard):
- **inform**: declarative statements, sharing information (48.4% of data)
- **question**: interrogative utterances, seeking information (25.6%)
- **directive**: commands, requests, suggestions (15.4%)
- **commissive**: commitments, promises, offers (10.6%)

**Expected improvement**: 4.6x class balance ratio (vs 9.6x), expert-annotated labels with 78.9% inter-annotator agreement, targeting 0.70-0.75 F1 (vs 0.38 baseline).

---

## Quick Start

```bash
# Step 1: Prepare data (5 minutes)
make prepare-dailydialog

# Step 2a: Quick sweep - 3-class combined only, ~1 hour, 32 experiments
make dailydialog-sweep-quick

# Step 2b: Full sweep - all configurations, 5-7 hours, 144 experiments
make dailydialog-sweep

# Step 3: Analyze results and get recommendations
make dailydialog-analyze

# Step 4: Train production model with recommended config
make train-category-svm ARGS="--data-dir data/dailydialog_native --label-map 3class"
```

---

## Experiment Dimensions

The sweep tests **144 configurations** across 5 dimensions:

### 1. Category Configs (2 variants)
- **4-class**: inform, question, directive, commissive (native labels)
- **3-class**: inform, question, action (merge directive+commissive)

### 2. Feature Sets (3 variants)
- **embeddings**: 384-dim BGE-small embeddings only
- **handcrafted**: 19-dim hand-crafted features only (length, punctuation, context, etc.)
- **combined**: 403 dims (embeddings + hand-crafted)

### 3. Class Balancing (3 strategies)
- **natural**: Use original 4.6x imbalance
- **balanced**: Downsample to minority class size (perfectly balanced)
- **moderate**: Allow max 2x imbalance (middle ground)

### 4. SVM Hyperparameters
- **C**: [0.1, 1.0, 10.0, 50.0] (regularization strength)
- **class_weight**: [balanced, None] (automatic vs manual weighting)

**Total**: 2 × 3 × 3 × 4 × 2 = **144 experiments**

---

## Memory Strategy (8GB RAM Constraint)

- Load data **once**, reuse across all experiments (no copies)
- GridSearchCV with **n_jobs=1** (avoid worker memory overhead)
- Peak memory per experiment: ~400MB (safe on 8GB)
- Each worker would use ~550MB (data + optimizer buffers)
- n_jobs=2 would cause 881MB swap → 10x slowdown

---

## Runtime Estimates

- **Data preparation**: ~5 minutes (one-time)
- **Quick sweep** (--quick flag): ~1 hour (32 experiments)
  - 3-class combined features only
  - All 3 balancing strategies
  - All C values and class weights
- **Full sweep**: 5-7 hours (144 experiments)
  - ~3 minutes per experiment (5-fold CV + test eval)
  - Progress logged every 10 experiments
  - Incremental saves (resume on crash)

---

## Output Files

```
data/dailydialog_native/
  train.npz          # 70k examples, 403 features
  test.npz           # 17k examples, 403 features
  metadata.json      # Distribution, feature dims

experiments/results/
  dailydialog_sweep.json         # Final results, ranked by CV F1
  dailydialog_sweep_partial.json # Intermediate saves (every 10 experiments)
```

---

## Verification Commands

After data preparation:
```bash
uv run python -c "
import numpy as np
import json
from pathlib import Path

data = np.load('data/dailydialog_native/train.npz')
meta = json.loads(Path('data/dailydialog_native/metadata.json').read_text())

print(f'Train shape: {data[\"X\"].shape}')
print(f'Labels: {sorted(set(data[\"y\"]))}')
print(f'Distribution: {meta[\"label_distribution_train\"]}')
print(f'Balance ratio: {meta[\"balance_ratio\"]:.1f}x')
"
```

Expected output:
```
Train shape: (60841, 403)
Labels: ['commissive', 'directive', 'inform', 'question']
Distribution: {'inform': 29452, 'question': 15579, 'directive': 9348, 'commissive': 6462}
Balance ratio: 4.6x
```

After sweep:
```bash
uv run python -c "
import json
results = json.load(open('experiments/results/dailydialog_sweep.json'))
print(f'Completed: {len(results[\"experiments\"])} / 144')
print(f'Best F1: {results[\"top_10\"][0][\"cv_mean_f1\"]:.3f}')
print(f'Config: {results[\"top_10\"][0]}')
"
```

---

## Decision Criteria (from analyze script)

1. **Category config**: Use 3-class if F1 gain > 3% over 4-class
2. **Features**: Use embeddings-only if hand-crafted add < 1% F1
3. **Balancing**: Use moderate if natural F1 < 5% lower
4. **Hyperparameters**: Select C that maximizes CV F1
5. **Generalization**: Test F1 within 2% of CV F1 (no overfitting)

If F1 < 65%: Add SAMSum back, try bge-large embedder, add more features

---

## Next Steps After Optimal Config Found

1. **Error Analysis**
   ```bash
   uv run python experiments/scripts/analyze_errors.py \
     --model-path ~/.jarvis/embeddings/bge-small/category_classifier_model \
     --test-data data/dailydialog_native/test.npz
   ```
   - Inspect misclassified examples
   - Find common failure patterns
   - Identify underrepresented scenarios

2. **Integration Testing**
   - Update `jarvis/classifiers/category_classifier.py` to use new labels
   - Map dialog acts to reply strategies in `jarvis/prompts.py`
   - Run tests: `make test`

3. **A/B Testing**
   - Compare old 4-category vs new dialog act classifier on real iMessages
   - Sample 100 recent messages, get human labels, measure F1 on production data

4. **Synthetic Data Augmentation**
   - Generate synthetic "clarify" examples (under-represented in training)
   - Use LLM to create ambiguous messages
   - Retrain with augmented data

5. **Production Monitoring**
   - Track per-class F1 on real iMessages
   - Detect distribution drift
   - Retrain if F1 drops > 5%

---

## Alternative Experiments (Future Work)

### 1. Different Classifiers
Replace LinearSVC with:
- Logistic Regression (faster, probabilistic)
- Random Forest (handles imbalance better)
- XGBoost (best for structured data)
- Neural network (2-layer MLP)

### 2. Different Embedders
Replace BGE-small (384d) with:
- BGE-large (1024d, higher quality but slower)
- Arctic-m (256d, specialized for short text)
- Arctic-l (1024d, best quality)

### 3. Cost-Sensitive Learning
Custom class weights based on production distribution:
```python
# If clarify is 30% of production but 5% of training
class_weight = {
    "inform": 1.0,
    "question": 1.0,
    "directive": 1.0,
    "commissive": 6.0,  # 30% prod / 5% train = 6x
}
```

### 4. Multi-Task Learning
Predict both dialog act AND emotion simultaneously:
- Joint model learns shared representations
- Emotion signals can help disambiguate dialog acts
- Requires multi-output classifier

### 5. Ensemble Methods
Combine multiple classifiers:
- 3-class SVM + 4-class SVM + centroid fallback
- Vote or average probabilities
- Usually 2-3% F1 improvement

### 6. Synthetic Clarify Examples
Generate training data for under-represented "clarify" category:
```bash
uv run python scripts/generate_synthetic_clarify.py
```
- LLM generates ambiguous messages
- Heuristic rules for context gaps
- Deictic pronouns with thin context

---

## Files Created

- `scripts/prepare_dailydialog_data.py` - Data preparation
- `experiments/scripts/dailydialog_sweep.py` - Experiment sweep
- `experiments/scripts/analyze_dailydialog_results.py` - Results analysis
- Modified: `scripts/train_category_svm.py` - Added --label-map support

## Makefile Targets Added

- `make prepare-dailydialog` - Extract DailyDialog data
- `make dailydialog-sweep` - Run full 144-experiment sweep
- `make dailydialog-sweep-quick` - Quick 32-experiment sweep (~1 hour)
- `make dailydialog-analyze` - Analyze results
- `make train-category-svm` - Train production model

---

## Troubleshooting

**Out of memory during sweep**:
- Already using n_jobs=1 (correct)
- Check Activity Monitor for swap usage
- If swap > 500MB: reduce batch size in code
- Kill other apps to free RAM

**Sweep taking too long**:
- Use `--quick` flag for 1-hour sweep
- Reduce CV folds from 5 to 3 (edit code)
- Use smaller data subset for initial exploration

**Poor F1 results (< 65%)**:
- Check data quality: `head -20 data/dailydialog_native/train.npz`
- Verify label distribution matches expected
- Try different embedder (bge-large)
- Add SAMSum back to training data
- Consider synthetic data augmentation

**Results file not found**:
- Ensure sweep completed successfully
- Check `experiments/results/dailydialog_sweep_partial.json` for progress
- Re-run sweep from last checkpoint (auto-resume)

---

## Expected Outcomes

**Baseline**: Human inter-annotator agreement is 78.9%, expect classifier F1 ~70-75%

**Distribution** (DailyDialog natural):
- inform: 48.4% (majority)
- question: 25.6%
- directive: 15.4%
- commissive: 10.6% (minority)

**Balance**: 4.6x ratio (vs current 9.6x - major improvement!)

**Runtime**:
- Quick sweep: ~1 hour (good for prototyping)
- Full sweep: 5-7 hours (overnight run)

**Memory**: Peak 400MB per experiment (safe on 8GB with n_jobs=1)

**Improvement**: Targeting 0.70-0.75 F1 (vs 0.38 baseline) = **85-97% relative improvement**
