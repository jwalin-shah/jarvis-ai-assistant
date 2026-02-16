# Reply Pipeline Architecture Guide

> **Last Updated:** 2026-02-10

## Overview

This guide covers the complete architecture for generating text message replies that match your personal style.

## Architecture Components

### 1. Classification Layer (Fast Path)

**Purpose**: Decide if/what to reply

```python
from jarvis.classifiers.category_classifier import classify_category
from jarvis.classifiers.response_mobilization import classify_mobilization

# Classify incoming message
category = classify_category(incoming_text, context=recent_messages)
mobilization = classify_mobilization(incoming_text)

# Decision tree
if category.category == "acknowledge":
    # Brief acknowledgment - reply if mobilization is HIGH
    return generate_brief_reply()
elif mobilization.pressure == "none":
    # No reply needed
    return None
else:
    # Generate full reply
    return generate_reply()
```

**Categories** (6):
- `acknowledge` - "ok", "sure", "thanks"
- `question` - "What time?", "Where are you?"
- `request` - "Can you pick me up?"
- `emotion` - "omg!", "so excited!"
- `closing` - "bye", "talk later"
- `statement` - "Running late", "Just got home"

**Mobilization** (4 levels):
- `HIGH` - Question/request that needs answer
- `MEDIUM` - Emotional content, casual reply expected
- `LOW` - Statement, optional acknowledgment
- `NONE` - No reply needed

### 2. Context Assembly

**What to include in the prompt**:

```python
context_parts = {
    # Recent conversation (essential)
    "conversation_history": get_last_n_messages(10),
    
    # Contact knowledge (from fact DB)
    "facts": get_relevant_facts(contact_id, query=incoming_text, k=5),
    
    # Similar past replies (RAG)
    "similar_examples": search_similar_replies(incoming_text, k=3),
    
    # Relationship type
    "relationship": classify_relationship(contact_id),
    
    # Time context
    "time_since_last": time_since_last_message(),
}
```

### 3. Prompt Engineering

**System Prompt Template**:

```
<system>
You are NOT an AI assistant. You are replying to a text message from your phone.
Just text back. No helpfulness, no formality, no assistant behavior.

Rules:
- Keep it brief (1-2 sentences max)
- Use casual language, abbreviations ok
- Match the energy of the incoming message
- Don't ask follow-up questions unless necessary

Category: {category}
Relationship: {relationship}
{style_section}
</system>
```

**Style Section** (auto-generated from your history):

```
Style:
- Very short messages (1-5 words)
- Rarely uses emoji
- Uses abbreviations: lol, idk, nvm, tbh
- Mostly lowercase
```

### 4. LLM Selection

| Model | Recommendation |
|-------|---------------|
| **Qwen 1.5B** | Default - best speed/quality balance |
| **Gemma 3 4B** | Use if Qwen quality insufficient |
| **LFM 0.3B** | Testing only - very fast, lower quality |

**Configuration** (in `config.yaml`):

```yaml
model:
  name: "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
  max_tokens: 50  # Keep replies short
  temperature: 0.7
  top_p: 0.9
```

### 5. RAG Implementation

**Setup**:

```python
from jarvis.search.vec_search import get_vec_searcher

# Index your message history
searcher = get_vec_searcher(db)

# Search for similar conversations
results = searcher.search_with_chunks(
    query=incoming_text,
    k=3,
    contact_id=current_chat_id  # Optional: same contact only
)

# Use as examples in prompt
examples = []
for r in results:
    examples.append({
        "they_said": r.trigger_text,
        "you_replied": r.response_text
    })
```

**Quality Filtering for RAG**:

Only index high-quality pairs:
- Response length: 3-100 characters
- Not acknowledgments only (avoid "ok" → "ok" training)
- Quality score > 0.7 (from `filter_quality_pairs.py`)

### 6. Training Data Preparation

**From iMessage**:

```bash
# 1. Extract pairs (already done)
uv run python scripts/extract_personal_data.py

# 2. Filter for quality
uv run python scripts/filter_quality_pairs.py \
    --input data/personal/raw_pairs.jsonl \
    --output data/personal/quality_pairs.jsonl

# 3. Prepare training format
uv run python scripts/prepare_personal_data.py --both
# Outputs:
#   - data/personal/category_aware/{train,valid,test}.jsonl
#   - data/personal/raw_style/{train,valid,test}.jsonl
```

**Training Data Format**:

```json
{
  "messages": [
    {"role": "system", "content": "You are NOT an AI assistant... Category: question"},
    {"role": "user", "content": "<conversation>...\n<last_message>Friend: Want to grab lunch?</last_message>"},
    {"role": "assistant", "content": "sure what time?"}
  ]
}
```

### 7. Optional: LoRA Fine-tuning

**When to fine-tune**:
- You have 5,000+ high-quality pairs
- RAG alone doesn't capture your style
- You want to reduce latency (no retrieval step)

**Fine-tuning command**:

```bash
mlx_lm.lora \
    --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --train data/personal/category_aware/train.jsonl \
    --valid data/personal/category_aware/valid.jsonl \
    --batch-size 4 \
    --lora-layers 8 \
    --iters 1000 \
    --learning-rate 1e-5 \
    --output-dir models/lora/personal
```

**Using fine-tuned model**:

```python
from mlx_lm import load

model, tokenizer = load(
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    adapter_path="models/lora/personal"
)
```

## Fact Extraction Improvements

### Current Limitations

1. **No temporal reasoning** - "moving to Austin" vs "moved to Austin 2 years ago"
2. **No fact expiration** - old jobs, old relationships still treated as current
3. **Limited relationship types** - only extracts "my X Y" patterns
4. **No implicit facts** - misses "finally got that promotion" → works at X

### Recommended Enhancements

1. **Add temporal markers**:
   - Pattern: "moving to", "just moved", "lived in", "grew up in"
   - Store: `valid_from`, `valid_until` timestamps

2. **Extract implicit relationships**:
   - "Mom says hi" → relationship: family, subject: "my mom"
   - "Working with Sarah" → relationship: coworker

3. **Fact confidence decay**:
   - Facts older than 2 years: reduce confidence
   - Conflicting new fact: replace old one

## End-to-End Example

```python
from jarvis.reply_service import get_reply_service

reply_service = get_reply_service()

result = reply_service.route_legacy(
    incoming="Want to grab lunch?",
    thread=["You: hey", "Friend: want to grab lunch?"],
    chat_id="chat123"
)

print(result["response"])  # "sure what time?"
print(result["confidence"])  # "high"
print(result["category"])  # "request"
```

## Performance Targets

- **Classification**: < 10ms (category + mobilization)
- **Context assembly**: < 50ms (facts + RAG)
- **Generation**: < 200ms (Qwen 1.5B, 20 tokens)
- **Total**: < 300ms end-to-end

## Next Steps

1. **Validate your data quality**:
   ```bash
   uv run python scripts/filter_quality_pairs.py
   # Check quality_metrics.txt output
   ```

2. **Test RAG-only first**:
   ```bash
   # Don't fine-tune yet, just use RAG
   # Test with 100 messages, see quality
   ```

3. **Measure before optimizing**:
   ```bash
   uv run python evals/run_comparison.py
   # Check if replies match your style
   ```

4. **Consider fine-tuning only if**:
   - RAG quality < 70% satisfaction
   - You have 5k+ quality pairs
   - Latency is critical (< 100ms target)
