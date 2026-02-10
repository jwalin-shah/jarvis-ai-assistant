# JARVIS Code Cleanup Guide

## The Problem You're Facing

Your codebase has grown organically and now has:
- **Overlapping concerns**: Fact extraction, classification, and generation are tangled
- **Multiple classification layers**: Mobilization → Category → Intent → ???
- **Blocking operations**: Fact extraction slows down reply generation
- **Too many files**: Hard to understand the data flow

## The Solution: Three Clean Pipelines

```
┌─────────────────────────────────────────────────────────────┐
│                    THREE PIPELINES                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. UNDERSTAND (Background)                                 │
│     Input:  Message text                                    │
│     Output: Updated knowledge graph                         │
│     Speed:  Can be slow (async)                             │
│     Files:  pipelines/understand.py                         │
│                                                             │
│  2. CLASSIFY (Fast)                                         │
│     Input:  Message text + context                          │
│     Output: Decision (reply? template? generate?)           │
│     Speed:  <50ms                                           │
│     Files:  pipelines/classify.py                           │
│                                                             │
│  3. GENERATE (When needed)                                  │
│     Input:  Message + classification                        │
│     Output: Reply text                                      │
│     Speed:  <500ms                                          │
│     Files:  pipelines/generate.py                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Principles

### 1. Classification Should Be Simple

**Before**: 3 classifiers, 700+ lines, confusing interactions
**After**: 1 classifier, ~150 lines, clear flow

```python
# The entire public API:
from jarvis.pipelines import classify_message

result = classify_message("Want to grab lunch?")
# result.category = "question"
# result.urgency = "high"
# result.should_reply = True
# result.use_template = False
```

### 2. Understanding Should Be Async

**Before**: Extract facts → Block → Generate reply
**After**: Generate reply (with existing KG) → Extract facts (background)

```python
# Don't block on this:
asyncio.create_task(process_for_knowledge(message, contact_id))

# Continue immediately with reply
reply = generate_reply(message, classification)
```

### 3. Generation Should Be Layered

**Before**: Reply service mixes RAG, prompt building, and SLM calling
**After**: Clear stages

```python
context = assemble_context(message, classification)  # RAG + profile
prompt = build_prompt(context)                       # Text construction
reply = slm.generate(prompt)                         # Model call
```

## Files Created for You

| File | Purpose |
|------|---------|
| `docs/ARCHITECTURE_CLEANUP_PLAN.md` | Full architectural vision |
| `docs/CLEANUP_ACTION_PLAN.md` | Step-by-step migration plan |
| `docs/EXAMPLE_CLEAN_CLASSIFIER.py` | Ready-to-use classifier |
| `docs/EXAMPLE_CLEAN_GENERATION.py` | Ready-to-use generation pipeline |
| `docs/EXAMPLE_CLEAN_UNDERSTANDING.py` | Ready-to-use understanding pipeline |

## Quick Start (Do This Now)

### Step 1: Create the pipelines directory
```bash
mkdir -p jarvis/pipelines
touch jarvis/pipelines/__init__.py
```

### Step 2: Copy the clean classifier
```bash
cp docs/EXAMPLE_CLEAN_CLASSIFIER.py jarvis/pipelines/classify.py
```

Edit it to remove the `if __name__ == "__main__"` block and add to `__init__.py`:
```python
from jarvis.pipelines.classify import classify_message, Classification, Category, Urgency
```

### Step 3: Replace classification in your code
Find where you currently call classification and replace:
```python
# OLD
mobilization = classify_with_cascade(text)
category = classify_category(text)

# NEW
classification = classify_message(text)
if not classification.should_reply:
    return skip_reply()
```

### Step 4: Test
```bash
python -c "from jarvis.pipelines import classify_message; print(classify_message('ok'))"
```

## Migration Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Consolidate classification | `pipelines/classify.py` working |
| 2 | Clean up generation | `pipelines/generate.py` working |
| 3 | Make understanding async | Background KG updates |
| 4 | Delete old code | Remove cascade, router, etc. |

## What Gets Deleted

After migration, these files can be removed:
- `jarvis/classifiers/cascade.py` → merged into `pipelines/classify.py`
- `jarvis/classifiers/response_mobilization.py` → simplified
- `jarvis/classifiers/intent_classifier.py` → not needed
- `jarvis/router.py` → merged into `pipelines/generate.py`
- `jarvis/generation.py` → merged into `pipelines/generate.py`

## What Stays

- `jarvis/classifiers/category_classifier.py` → Move model loading to new pipeline
- `jarvis/contacts/fact_extractor.py` → Move to `pipelines/understand.py`
- `jarvis/graph/knowledge_graph.py` → Keep as `knowledge/graph.py`
- `jarvis/reply_service.py` → Simplify to use new pipelines

## Success Metrics

After cleanup:
- [ ] New developer understands system in < 10 minutes
- [ ] Classification runs in < 50ms
- [ ] No blocking operations in reply path
- [ ] < 200 lines per pipeline file
- [ ] Clear data flow: Message → Classify → (Generate) → Reply

## Questions?

The example files are complete, runnable Python. You can:
1. Copy them directly
2. Modify to fit your needs
3. Use them as reference while refactoring

Start with the classifier - it's the simplest and will give you immediate clarity.
