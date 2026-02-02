# Text Normalization for Embeddings

## Overview

All text is normalized before embedding to ensure consistency between training and inference.

## Why Normalize?

**Problem:** Training data and inference data can have subtle differences:

| Issue | Training Data | Inference Data |
|-------|--------------|----------------|
| Multi-message turns | Single messages (mostly) | Grouped with `\n` separators |
| Unicode variants | Mixed (smart quotes, etc.) | iOS smart quotes `"` `'` |
| Whitespace | Clean | May have extra spaces/newlines |

**Result:** Same semantic text embeds differently, hurting classifier accuracy.

## Normalization Function

```python
import unicodedata

def normalize_text(text: str) -> str:
    """Normalize text for consistent embedding.

    Applied in embedding_adapter.encode() before computing embeddings.
    """
    # Unicode normalize (NFKC converts smart quotes, em-dashes, etc.)
    text = unicodedata.normalize("NFKC", text)
    # Collapse all whitespace (including newlines) to single spaces
    text = " ".join(text.split())
    return text
```

## What It Does

| Before | After | Why |
|--------|-------|-----|
| `"hey\nwant lunch?"` | `"hey want lunch?"` | Multi-message turns → single line |
| `"don't"` (smart apostrophe) | `"don't"` (ASCII) | Unicode consistency |
| `"hello"` (smart quotes) | `"hello"` (ASCII) | Unicode consistency |
| `"hello   world"` | `"hello world"` | Collapse extra spaces |
| `"  trimmed  "` | `"trimmed"` | Strip leading/trailing |

## What It Preserves

| Feature | Preserved? | Why |
|---------|------------|-----|
| Case | Yes | "YES" vs "yes" carries meaning |
| Punctuation | Yes | "lunch?" vs "lunch" is different |
| Emoji | Yes | Emoji carry sentiment |
| URLs | Yes | Context about link sharing |

## Where Applied

Normalization happens in `embedding_adapter.encode()`, so it applies everywhere:

- Trigger classifier (train + inference)
- Response classifier (train + inference)
- Similarity search (indexed + queries)
- Topic detection
- Any future embedding use

## Turn Bundling

The extraction system (`jarvis/extract.py`) groups rapid-fire messages into turns:

```python
# Config: turn_bundle_minutes = 10.0
# Messages within 10 min from same sender are bundled

# Raw messages:
Them: "hey"          (10:00)
Them: "want lunch?"  (10:02)
Them: "thinking sushi" (10:03)

# After bundling:
trigger_text = "hey\nwant lunch?\nthinking sushi"

# After normalization:
normalized = "hey want lunch? thinking sushi"
```

## Training Data

Current training data is mostly single messages:
- `data/trigger_labeling.jsonl`: ~4,865 examples, only 8 have `\n`
- `data/response_labeling.jsonl`: ~4,865 examples

Normalization makes the training data compatible with inference (where multi-message turns are common for rapid-fire texters).

## Implementation Location

```
jarvis/embedding_adapter.py
├── normalize_text()           # The normalization function
└── UnifiedEmbedder.encode()   # Calls normalize_text() before embedding
```
