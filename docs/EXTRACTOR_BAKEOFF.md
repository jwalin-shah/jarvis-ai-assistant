# Phase 1: Candidate Extractor Bakeoff

This document describes the Phase 1 implementation of the Fact KG Roadmap - a bakeoff evaluation system for comparing multiple candidate extraction backends.

## Overview

The bakeoff system provides a common adapter interface for evaluating and comparing different entity extraction tools:

- **GLiNER** (baseline): The current production-ready extractor
- **GLiNER2**: Newer architecture with improved multi-task support
- **NuExtract**: Schema-driven LLM-based extraction

## Architecture

### Adapter Interface

All extractors implement the `ExtractorAdapter` base class:

```python
from jarvis.contacts.extractors import ExtractorAdapter, ExtractedCandidate

class MyExtractor(ExtractorAdapter):
    def extract_from_text(self, text: str, message_id: int, ...) -> list[ExtractedCandidate]:
        # Implementation
        ...
    
    def extract_batch(self, messages: list[dict], ...) -> list[ExtractionResult]:
        # Batch implementation
        ...
```

### Common Output Schema

All extractors produce normalized `ExtractedCandidate` objects:

```python
@dataclass
class ExtractedCandidate:
    span_text: str        # The extracted entity text
    span_label: str       # Normalized label (place, org, etc.)
    score: float          # Confidence score (0.0-1.0)
    start_char: int       # Start offset in source text
    end_char: int         # End offset in source text
    fact_type: str        # Mapped fact type (work.employer, etc.)
    extractor_metadata: dict  # Tool-specific extras
```

## Usage

### Running the Bakeoff

Evaluate all extractors against the frozen goldset:

```bash
# Full evaluation
uv run python scripts/run_extractor_bakeoff.py

# Evaluate specific extractors
uv run python scripts/run_extractor_bakeoff.py --extractors gliner,gliner2

# Limit for quick testing
uv run python scripts/run_extractor_bakeoff.py --limit 100

# Custom goldset
uv run python scripts/run_extractor_bakeoff.py --gold training_data/gliner_goldset/candidate_gold_merged_r4.json
```

### Extracting Candidates

Use the updated extraction script with any adapter:

```bash
# GLiNER baseline (default)
uv run python scripts/extract_candidates.py --limit 500

# GLiNER2
uv run python scripts/extract_candidates.py --extractor gliner2 --limit 500

# NuExtract (slower, uses LLM)
uv run python scripts/extract_candidates.py --extractor nuextract --limit 100 --batch-size 8
```

### Using Adapters Directly

```python
from jarvis.contacts.extractors import create_extractor

# Create extractor
extractor = create_extractor("gliner2", config={"threshold": 0.30})

# Extract from single message
candidates = extractor.extract_from_text(
    text="I live in Austin and work at Google",
    message_id=123,
)

# Batch extraction
messages = [
    {"text": "I love sushi", "message_id": 1},
    {"text": "My brother lives in NYC", "message_id": 2},
]
results = extractor.extract_batch(messages, batch_size=32)
```

## Evaluation Metrics

The bakeoff uses precision-weighted F0.5 score as the primary metric, with a recall floor:

- **F0.5**: Weights precision twice as much as recall (beta=0.5)
- **Recall floor**: Minimum 0.40 recall required
- **Secondary metrics**: F1, precision, recall per label/type/slice

## Exit Gate Criteria

Phase 1 exit gate (from roadmap):

> Select primary extractor and fallback extractor with written benchmark evidence.

The bakeoff script produces a recommendation based on:
1. F0.5 score ranking
2. Recall >= 0.40 floor
3. Per-slice performance analysis

## Output Files

Results are saved to `results/extractor_bakeoff/`:

```
results/extractor_bakeoff/
├── bakeoff_results.json      # Combined results from all extractors
├── gliner_metrics.json       # Per-extractor detailed metrics
├── gliner2_metrics.json
└── nuextract_metrics.json
```

## Adding New Extractors

1. Create a new adapter class in `jarvis/contacts/extractors/my_extractor.py`:

```python
from jarvis.contacts.extractors.base import (
    ExtractorAdapter, ExtractedCandidate, register_extractor
)

class MyExtractor(ExtractorAdapter):
    def __init__(self, config: dict | None = None):
        super().__init__("my_extractor", config)
        # Initialize
    
    def extract_from_text(self, text, message_id, **kwargs) -> list[ExtractedCandidate]:
        # Implementation
        ...
    
    # ... other required methods

register_extractor("my_extractor", MyExtractor)
```

2. Import in `jarvis/contacts/extractors/__init__.py`

3. Run bakeoff: `uv run python scripts/run_extractor_bakeoff.py --extractors my_extractor`

## Dependencies

- **GLiNER**: `uv pip install gliner`
- **GLiNER2**: Same as GLiNER (different model name)
- **NuExtract**: `uv pip install nuextract` (optional, falls back to mock mode)

## Notes

- NuExtract requires significant compute (LLM inference)
- GLiNER2 may need different thresholds than GLiNER baseline
- The mock mode for NuExtract provides a testing fallback without installing the full package
