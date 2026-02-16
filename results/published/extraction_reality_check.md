# Extraction Reality Check: What's Working vs What's Hard

## Current State

### ✅ What's Working: Entity Extraction (GLiNER)
The production pipeline extracts **surface entities** successfully:

| Entity Type | Count | Examples |
|------------|-------|----------|
| food_item | ~300 | cake, chocolate milk, guacamole |
| person_name | ~134 | Mom, sister, Ashutosh |
| place | ~40 | Houston, Austin, Dallas |
| job_role | ~10 | CEO, senior engineer, CTO |
| org | ~6 | Pixar, Crossover SOMA |

**Performance**: 90%+ confidence on 1,000+ facts
**Use case**: "Who mentioned chocolate milk?" → Search works

### ❌ What's Hard: Semantic Relation Extraction
The gold labels want **understanding**, not just entities:

| Text | Gold Fact | Problem |
|------|-----------|---------|
| "take my kids to the dentist" | has children | Requires inference |
| "Tall boy" | tall physical stature | Requires trait recognition |
| "played mini ultimate" | plays ultimate frisbee | Requires activity inference |
| "me and Deevy" | associate of Deevy | Requires relationship parsing |

**Best Attempts**:
- 350M base model: ~5% recall (mostly hallucinations)
- 350M Extract model: Outputs template variables instead of values
- 1.2B base model: ~15% recall (better but still poor)
- 1.2B Instruct: Not accessible (gated repo)
- 1.2B Extract: Not working

### Why It's Hard

1. **Models are too small** (350M-1.2B parameters)
   - Semantic understanding requires 3B+ parameters typically
   - Fine-tuning helps but base capacity matters

2. **Task complexity**
   - Entity extraction: Pattern matching (easy)
   - Semantic extraction: Natural language understanding (hard)

3. **Chat text challenges**
   - Informal language
   - Abbreviations ("yuhhh", "omw")
   - Context-dependent meanings

## Hybrid Approach Results

Tried combining:
1. **GLiNER**: Extract entities (working)
2. **LLM**: Extract semantic facts with few-shot prompting (struggling)

**Result**: Entities ✓, Semantic facts ✗

The LLM still outputs:
- Template variables: `[person] | [category]: [specific value]`
- Wrong inferences: "dentist appointment" as location
- Hallucinations: Made-up facts

## Recommended Next Steps

### Option 1: Accept Entity-Only (Recommended)
Use what works:
- Entity search: "Who mentioned [food/place/person]?"
- Contact profiles: Aggregate mentions over time
- Knowledge graph: Link contacts through shared entities

**Pros**: Working now, scalable, 90%+ accuracy
**Cons**: No semantic understanding

### Option 2: Use Larger Models
Try 3B+ parameter models:
- `Qwen2.5-3B-Instruct`
- `Llama-3.2-3B-Instruct`
- Fine-tune on your gold labels

**Pros**: Better semantic understanding
**Cons**: Slower (3-5x), more memory, still may struggle

### Option 3: Rule-Based Semantic Parsing
Build pattern matchers for common relations:
```python
r'(my|our)\s+(mom|dad|mother|father)' → family: has_parent
r'(?:live|lived)\s+in\s+(\w+)' → location: lives_in
```

**Pros**: Fast, deterministic, no model needed
**Cons**: Brittle, misses edge cases, maintenance burden

### Option 4: Two-Stage Pipeline
1. Extract entities (GLiNER - working)
2. Classify entity pairs into relations (small classifier)

Example:
- Input: "my sister Sarah" → entities: [sister, Sarah]
- Classifier: (speaker, sister) → family: has_sister

**Pros**: Structured, trainable, scales well
**Cons**: Requires training data, limited to known patterns

## Immediate Recommendation

**Go with Option 1 (Entity-Only) + Option 4 (Two-Stage)**:

1. **Ship entity extraction** - It's working well
2. **Build simple classifiers** for high-value relations:
   - Family: "my [sister/brother/mom/dad]"
   - Location: "live in [place]"
   - Work: "work at [org]"
3. **Use LLM for edge cases** - Not primary extraction

This gives you a working knowledge graph now, with semantic enrichment later.
