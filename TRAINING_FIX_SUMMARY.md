# Training Fix Summary

## Problems Found and Fixed

### 1. EMPTY TRAINING DATA DIRECTORY (CRITICAL)
**Problem:** All configs with `cataware` variant pointed to `data/personal/category_aware/` which was **EMPTY**.

**Root Cause:** `scripts/training/generate_ft_configs.py` included a non-existent data variant.

**Fix:** Changed `DATA_VARIANTS` in `generate_ft_configs.py`:
```python
# Before (BROKEN):
DATA_VARIANTS = [
    {"id": "cataware", "path": "data/personal/category_aware"},  # EMPTY!
    {"id": "rawstyle", "path": "data/personal/raw_style"},
]

# After (FIXED):
DATA_VARIANTS = [
    {"id": "variable", "path": "data/personal/raw_style_variable"},  # Has 10K examples
    {"id": "rawstyle", "path": "data/personal/raw_style"},
]
```

**Action:** Regenerated all configs and removed broken `cataware` configs.

---

### 2. TRAINING/INFERENCE FORMAT MISMATCH
**Problem:** Training data format didn't match inference prompt format.

**Training Format (what model learned):**
```
<|im_start|>system
You are Jwalin. Reply to text messages...
Rules:
- Match your typical reply length (9 words avg)
...
<|im_end|>
<|im_start|>user
+1402...: Will be back jan
+1402...: We gotto pay rent
Jwalin: yea u right
<|im_end|>
<|im_start|>assistant
what's the split on the 1905<|im_end|>
```

**Inference Format (what model saw at runtime):**
```
<|im_start|>system
You are Jwalin Shah, a tech founder. Text like a real person...
<|im_end|>
<|im_start|>user
Context:
+1402...: Will be back jan

Last Message: Them: +1402...: Will be back jan<|im_end|>
<|im_start|>assistant
```

**Key Differences:**
1. Different system prompts
2. Different user content format (no "Context:" / "Last Message:" wrappers in training)

**Fixes Applied:**

#### File: `jarvis/prompts/constants.py`
- Updated `SYSTEM_PREFIX` to match training system prompt
- Updated `SIMPLE_REPLY_PROMPT.template` to remove "Context:" and "Last Message:" wrappers

#### File: `jarvis/prompts/rag.py`
- Updated `build_simple_reply_prompt()` to match training format

#### File: `tests/unit/test_prompt_assembly.py`
- Updated tests to expect new system prompt format

---

## Data Statistics (After Fix)

```
data/personal/raw_style_variable/
- train.jsonl: 5,807,622 bytes (~10K examples)
- valid.jsonl: 730,799 bytes  (~1.5K examples)
- test.jsonl: 755,852 bytes   (~1.5K examples)

data/personal/category_aware/
- EMPTY (0 bytes) - CONFIGS REMOVED
```

---

## Next Steps for Training

1. **Verify data exists:**
   ```bash
   ls -la data/personal/raw_style_variable/
   wc -l data/personal/raw_style_variable/train.jsonl
   ```

2. **Run small test** (100 iterations) to verify format alignment:
   ```bash
   uv run mlx_lm.lora --config ft_configs/personal_0.3b_lora_variable.yaml \
       --iters 100 --steps-per-eval 10
   ```

3. **Compare outputs** between:
   - Base model (no adapter)
   - Fine-tuned model with properly aligned format

4. **If successful**, scale to larger models:
   ```bash
   uv run mlx_lm.lora --config ft_configs/personal_0.7b_lora_variable.yaml
   uv run mlx_lm.lora --config ft_configs/personal_1.2b_lora_variable.yaml
   ```

5. **Only then** try advanced techniques (GaLore, PiSSA) if needed.

---

## Files Changed

1. `scripts/training/generate_ft_configs.py` - Fixed data paths
2. `jarvis/prompts/constants.py` - Aligned system prompt and template
3. `jarvis/prompts/rag.py` - Aligned inference format
4. `tests/unit/test_prompt_assembly.py` - Updated tests
5. `ft_configs/personal_*.yaml` - Regenerated with correct paths
6. `ft_configs/personal_*_cataware.yaml` - Removed (pointed to empty data)

---

## Root Cause Summary

The model wasn't learning because:
1. Training configs pointed to empty directory → no training data
2. Format mismatch → model learned one pattern, saw different pattern at inference
3. Combined effect → adapter weights were noise, degrading performance from baseline

Both issues are now fixed. The next training run should actually learn from the data.
