# GLiNER Fact Extraction - Quick Start

## 1. Install GLiNER

```bash
# In your JARVIS venv
pip install gliner

# Verify installation
python -c "from gliner import GLiNER; print('GLiNER installed')"
```

## 2. Run Comparison on Your Messages

```bash
# Compare current regex vs GLiNER on 100 real iMessages
python scripts/compare_extraction_approaches.py --imessage --limit 100
```

This will show:
- Side-by-side extraction results
- Precision/recall metrics
- Specific examples where GLiNER wins

## 3. Try GLiNER Directly

```python
from gliner import GLiNER

# Load model (350MB, ~5 seconds first time)
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# Define custom entity labels for personal facts
labels = [
    "food_preference",
    "disliked_food",
    "current_location",
    "future_location",
    "employer",
    "family_member",
    "allergy",
]

# Test on messy iMessage text
text = "omg obsessed with this new ramen place in sf"

entities = model.predict_entities(text, labels, threshold=0.5)
for ent in entities:
    print(f"{ent['text']} -> {ent['label']} (confidence: {ent['score']:.2f})")

# Expected output:
# "ramen place" -> food_preference (confidence: 0.82)
# "sf" -> current_location (confidence: 0.75)
```

## 4. Evaluate on Your Data

```bash
# IMPORTANT: use compatibility runner for reliable GLiNER behavior
scripts/run_gliner_eval_compat.sh \
  --gold training_data/gliner_goldset/candidate_gold.json \
  --no-label-min \
  --mode pipeline \
  --context-window 0

# Metrics are written to:
# training_data/gliner_goldset/gliner_metrics.json
```

## 5. Train Classifier (if needed)

If GLiNER precision < 80%, train a filter:

```bash
# 1. Build candidate-level train/dev data from candidate_gold.json
scripts/run_gliner_compat.sh scripts/build_fact_filter_dataset.py \
  --gold training_data/gliner_goldset/candidate_gold.json \
  --output-all training_data/fact_candidates.jsonl \
  --output-train training_data/fact_candidates_train.jsonl \
  --output-dev training_data/fact_candidates_dev.jsonl \
  --manifest training_data/fact_candidates_manifest.json \
  --threshold 0.35 \
  --no-label-min \
  --context-window 0

# 2. Train classifier
python scripts/train_fact_filter.py \
  --input training_data/fact_candidates_train.jsonl \
  --test training_data/fact_candidates_dev.jsonl \
  --evaluate \
  --output models/fact_filter.pkl

# 3. Model artifact
# models/fact_filter.pkl
```

## 6. Expand Goldset (Manual)

Build a new annotation pack focused on weak labels (org/location/health):

```bash
scripts/run_gliner_compat.sh scripts/build_gliner_candidate_goldset.py \
  --total 300 \
  --org-count 100 \
  --location-count 100 \
  --health-count 70 \
  --negative-count 30 \
  --label-profile balanced \
  --threshold 0.35 \
  --no-label-min \
  --output-dir training_data/gliner_goldset_round3 \
  --overwrite
```

Outputs:
- `training_data/gliner_goldset_round3/sampled_messages.csv`
- `training_data/gliner_goldset_round3/sampled_messages.json`
- `training_data/gliner_goldset_round3/batch_*.json`

Fill `expected_candidates` manually, then merge into your gold corpus.

## Expected Timeline

| Task | Time | Result |
|------|------|--------|
| Install & run comparison | 30 min | See GLiNER vs regex on your data |
| Review 100 extractions | 1 hour | Know if GLiNER alone is enough |
| Label 200 examples (if needed) | 2-3 hours | Training data for classifier |
| Train & integrate | 2-4 hours | Production-ready pipeline |

**Total**: 1-2 days if GLiNER alone works, 2-3 days if classifier needed.

## Quick Decision Matrix

| GLiNER Precision | Recommendation |
|------------------|----------------|
| 70-80% | Use GLiNER alone, tune thresholds |
| 60-70% | GLiNER + simple rule filter |
| <60% | GLiNER + trained classifier |

## Memory Check

```python
# Before loading GLiNER
import psutil
print(f"Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB")

# Load GLiNER
from gliner import GLiNER
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# After loading
print(f"Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB")

# Should show ~350MB increase - well within 8GB budget
```

## Next Steps

1. ✅ Run comparison script
2. ✅ Review output
3. ⬜ Decide: GLiNER only or + classifier?
4. ⬜ Integrate into JARVIS
5. ⬜ Deploy to production
