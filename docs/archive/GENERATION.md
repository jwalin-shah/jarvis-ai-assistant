# Response Generation Improvements

## 2A. Response Style Fine-Tuning (LoRA)

**Goal**: Fine-tune LFM2.5 with LoRA to match user's texting style.

### Implementation

```python
from mlx_lm import lora

train_data = prepare_user_responses(jarvis_db)  # 10k+ examples
lora.train(
    model="LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit",
    data=train_data,
    rank=8,
    output_dir="~/.jarvis/models/lfm-lora/"
)
```

### Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| Generation latency | 600-3000ms | +5% |
| Memory | ~1.5GB | +100MB |
| Style match | 0.60 | 0.80+ |

**Pros**: Personalizes to exact style, small adapter (~10MB), can have per-contact adapters
**Cons**: Requires 10k+ messages, risk of overfitting
**Effort**: Medium (2-3 weeks)

---

## 2B. Multi-Option Generation Improvements

**Goal**: Improve quality and diversity of generated options.

### Current State

`jarvis/multi_option.py` generates AGREE/DECLINE/DEFER but lacks:
- Temperature diversity
- Beam search alternatives
- Quality ranking

### Implementation

```python
def generate_diverse_options(trigger, temps=[0.3, 0.6, 0.9]):
    options = []
    for temp in temps:
        option = generator.generate(trigger, temperature=temp)
        options.append(option)
    return rank_by_quality(options)
```

**Nucleus sampling by type:**
- AGREE: top_p=0.7 (focused)
- DECLINE: top_p=0.9 (need polite alternatives)

**Expected Impact**: Option diversity 0.40 → 0.70+
**Effort**: Low (1 week)

---

## 2C. Pre-Generation Quality Scoring

**Goal**: Predict response quality before expensive generation.

### Implementation

```python
def should_generate(trigger, context):
    quality_score = quality_predictor.predict(trigger, context)
    if quality_score < 0.5:
        return False, "Need more context"
    return True, None
```

**Benefits:**
- Reduces wasted generation on low-quality outputs
- Improves overall response quality
- Faster average response time

**Expected Impact**: 30% early exit, quality 0.65 → 0.80+
**Effort**: Medium (2 weeks)
**Dependency**: Quality metric ground truth
