# Training Debug Report: Why Fine-Tuning Isn't Working

**Date:** 2026-02-20

## Executive Summary

Two major issues explain why fine-tuning produced worse-than-baseline results:

1. **EMPTY TRAINING DATA**: The `cataware` configs pointed to an empty directory
2. **TRAINING/INFERENCE FORMAT MISMATCH**: Training data format doesn't match inference prompt format

---

## Issue 1: Empty Training Data Directory (CRITICAL)

### Problem
- All configs with `cataware` variant pointed to `data/personal/category_aware/`
- This directory was **EMPTY** (0 bytes)
- Training ran with NO DATA

### Evidence
```bash
$ ls -la data/personal/category_aware/
total 0
drwxr-xr-x 2 jwalinshah staff 64 Feb 8 21:33 .
```

### Root Cause
The `generate_ft_configs.py` script included a non-existent data variant:
```python
DATA_VARIANTS = [
    {"id": "cataware", "path": "data/personal/category_aware"},  # EMPTY!
    {"id": "rawstyle", "path": "data/personal/raw_style"},
]
```

### Fix Applied
Changed to use the actual training data location:
```python
DATA_VARIANTS = [
    {"id": "variable", "path": "data/personal/raw_style_variable"},  # Has data
    {"id": "rawstyle", "path": "data/personal/raw_style"},           # Has data
]
```

---

## Issue 2: Training/Inference Format Mismatch

### Training Format (from extract_finetuning_data.py)
```json
{
  "messages": [
    {"role": "system", "content": "You are Jwalin. Reply to text messages..."},
    {"role": "user", "content": "Contact: Hello\nJwalin: Hey"},
    {"role": "assistant", "content": "what's up"}
  ]
}
```

Converted by `mlx_lm.lora` using tokenizer's `apply_chat_template()` to:
```
<|im_start|>system
You are Jwalin. Reply to text messages...
<|im_end|>
<|im_start|>user
Contact: Hello
Jwalin: Hey
<|im_end|>
<|im_start|>assistant
what's up<|im_end|>
```

### Inference Format (from jarvis/prompts/constants.py)
```
<|im_start|>system
You are Jwalin Shah, a tech founder. Text like a real person... {instruction}<|im_end|>
<|im_start|>user
Context:
{context}

Last Message: {last_message}<|im_end|>
<|im_start|>assistant
```

### Key Differences

| Aspect | Training | Inference | Impact |
|--------|----------|-----------|--------|
| System prompt | "You are Jwalin..." | "You are Jwalin Shah, a tech founder..." | Different persona |
| User content format | "Name: message" | "Context:\n...\nLast Message: ..." | Different structure |
| Assistant format | Ends with `<|im_end|>` | No end token | Different termination |

---

## Why This Causes Worse-Than-Baseline Results

1. **No actual training** (empty data) = model never learned anything new
2. **Format mismatch** = model sees different patterns than trained on
3. **Combined effect** = adapter weights are noise, degrading from base model

---

## Recommended Fixes

### Immediate (Before Next Training)

1. **✅ FIXED**: Regenerate configs to point to correct data directory

2. **Align system prompts**: Make training and inference use identical system prompts

3. **Align user content format**: Either:
   - Option A: Change training to use "Context/Last Message" format
   - Option B: Change inference to use simple "Name: message" format

### Code Changes Needed

#### Option B (Recommended): Simplify inference to match training

In `jarvis/prompts/rag.py`, modify `build_simple_reply_prompt`:
```python
def build_simple_reply_prompt(context, last_message, ...):
    # Current: Uses "Context:\n...\nLast Message:" format
    # Change to: Simple "Name: message" format like training
    
    formatted_context = context  # Already in "Name: message" format
    user_content = formatted_context  # Just the conversation
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    # Use tokenizer.apply_chat_template instead of hardcoded format
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

#### Alternative: Update training data format

In `extract_finetuning_data.py`, change `to_messages_format`:
```python
def to_messages_format(self, system_prompt: str) -> dict:
    # Build "Context:\n...\nLast Message:" format to match inference
    # instead of simple "Name: message" format
```

---

## Next Steps

1. **Verify data exists** before training:
   ```bash
   ls -la data/personal/raw_style_variable/
   wc -l data/personal/raw_style_variable/train.jsonl
   ```

2. **Choose format alignment strategy** (Option A or B above)

3. **Run small test** (100 iterations) to verify format alignment:
   ```bash
   uv run mlx_lm.lora --config ft_configs/personal_0.3b_lora_variable.yaml \
       --iters 100 --steps-per-eval 10
   ```

4. **Compare outputs** between:
   - Base model (no adapter)
   - Fine-tuned model (with properly aligned format)

5. Only then scale to larger models (0.7B, 1.2B) or try advanced techniques (GaLore, PiSSA)

---

## Data Statistics

After fix:
```
data/personal/raw_style_variable/
- train.jsonl: 5,807,622 bytes (~10K examples)
- valid.jsonl: 730,799 bytes  (~1.5K examples)
- test.jsonl: 755,852 bytes   (~1.5K examples)
```

Before fix:
```
data/personal/category_aware/
- EMPTY (0 bytes)
```
