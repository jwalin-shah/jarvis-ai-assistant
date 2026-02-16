# 350M Model Fact Extraction & Knowledge Graph Summary

## Overview
Ran the LFM-350M base model on conversation segment chunks with rule-based extraction 
to populate the knowledge graph with personal facts about contacts.

## Results Summary

### Extraction Metrics (on segments with gold labels)
- **Segments Processed**: 119
- **Total Facts Extracted**: 23
- **Gold Facts Available**: 223
- **Precision**: 52.2% (12/23 correct)
- **Recall**: 5.4% (12/223 found)
- **F1 Score**: 9.8%

### Per-Category Performance
| Category    | Precision | Recall | F1   | Support |
|-------------|-----------|--------|------|---------|
| family      | 62%       | 21%    | 31%  | 24      |
| location    | 50%       | 9%     | 16%  | 32      |
| hobby       | 67%       | 5%     | 10%  | 39      |
| education   | 100%      | 3%     | 6%   | 33      |
| preference  | 33%       | 4%     | 7%   | 25      |

### Knowledge Graph Statistics
- **Total Nodes**: 1,229
- **Total Edges**: 1,312
- **Contacts**: 207
- **Entity Nodes**: 1,022

### Top Fact Categories
1. other (preferences, misc): 954
2. relationship: 199
3. preference: 88
4. location: 40
5. personal: 7

## Sample Correct Extractions ✓
- Mariela Costello | family | has children
- Asian Tim | hobby | plays mini ultimate
- Asian Tim | location | Austin
- Tejas Polkham | family | has mom/sister
- Kimiya Ganjooi | location | slo / education for MC

## Sample Missed Facts (False Negatives)
- Asian Tim | personality | Tall physical stature
- Asian Tim | education | Taking a physics class
- Sheethal | location | Lives in Dallas
- Jwalin | hobby | plays ultimate frisbee
- Ashutosh Kulkarni | hobby | knows about cars

## Key Observations

### What Works
- **Family relationships**: Best performance (62% precision, 21% recall)
- **Location mentions**: Moderate success with clear location patterns
- **Hobby extraction**: Good when patterns match ("play X", "like Y")

### Limitations
- **350M model output is noisy**: Doesn't reliably follow structured formats
- **Pattern-based extraction misses nuance**: 
  - "Taking a physics class" → missed because pattern looks for "taking class"
  - "Tall physical stature" → missed (personality trait not caught by patterns)
- **Low recall overall**: Missing ~95% of facts

### Knowledge Graph Capabilities Demonstrated
✅ Contact-entity relationships stored and queryable
✅ Search across facts (e.g., "Austin", "mom", "sister")
✅ Connection discovery (shared entities between contacts)
✅ Integration with existing GLiNER-extracted facts

## Files Generated
- `results/chunk_facts_350m.jsonl` - Raw extraction results with model responses
- Facts stored in `~/.jarvis/jarvis.db` (contact_facts table)

## Next Steps for Improvement
1. **Fine-tune 350M model** on fact extraction task for better structured output
2. **Expand pattern library** for more fact types (personality, nuanced hobbies)
3. **Use NLI verification** like GLiNER pipeline does
4. **Combine with spaCy NER** for better entity recognition
5. **Process all chunks** (~298 segments with facts) for more complete coverage

## Usage

### Run extraction on more chunks:
```bash
uv run python scripts/extract_facts_350m_chunks.py --only-with-facts --all --save
```

### Query knowledge graph:
```python
from jarvis.graph.knowledge_graph import KnowledgeGraph
kg = KnowledgeGraph()
kg.build_from_db()

# Query contact
result = kg.query_contact('contact_id')

# Search facts
results = kg.search_facts('Austin', limit=10)
```
