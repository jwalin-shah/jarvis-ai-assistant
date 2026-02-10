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
# Generate training data from your messages (creates JSONL)
python scripts/prepare_gliner_training.py \
    --source imessage \
    --limit 100 \
    --output training_data/my_facts.jsonl \
    --apply-heuristics

# Review the output
cat training_data/my_facts.jsonl | head -20
```

## 5. Train Classifier (if needed)

If GLiNER precision < 80%, train a filter:

```bash
# 1. Label ~100-200 examples manually
# Edit training_data/my_facts.jsonl and set "label": 1 or 0

# 2. Train classifier
python scripts/train_fact_filter.py \
    --input training_data/my_facts.jsonl \
    --output models/fact_filter.pkl

# 3. Test the classifier
python scripts/train_fact_filter.py \
    --input training_data/my_facts.jsonl \
    --evaluate
```

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
