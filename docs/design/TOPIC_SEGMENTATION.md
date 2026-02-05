# Topic Segmentation

Semantic topic boundary detection for conversation chunking. Replaces arbitrary time-based turn bundling with intelligent boundary detection.

**Module:** `jarvis/topic_segmenter.py`

## Overview

Topic segmentation divides long conversations into coherent topic chunks, enabling:
- More accurate context retrieval (fetch entire topic, not arbitrary windows)
- Better extraction pipeline processing (group related messages)
- Improved summarization (summarize by topic, not by time)

## Architecture

```
Raw Messages
    |
normalize_for_task_with_entities()  -> text + entities
    |
CorefResolver.resolve() (optional)  -> resolved pronouns
    |
embedder.encode()                   -> (N, 384) embeddings
    |
TopicSegmenter._compute_boundary_scores()
    |
Split at boundaries + merge small segments
    |
list[TopicSegment]
```

## Boundary Detection Algorithm

The segmenter computes a boundary score between consecutive messages:

```python
boundary_score = 0.4 * embedding_drift
               + 0.3 * (1 - entity_jaccard)
               + 0.2 * time_penalty
               + 0.4 * topic_shift_marker  # only if marker present
```

A boundary is created when `score >= threshold` (default 0.5).

### Signal Components

| Signal | Weight | Description |
|--------|--------|-------------|
| Embedding Drift | 0.4 | Cosine similarity drop between sliding window centroids |
| Entity Discontinuity | 0.3 | Jaccard overlap of named entities (PERSON, ORG, etc.) |
| Time Gap | 0.2 | Soft penalty for gaps > 10 min, hard boundary at 30 min |
| Topic Shift Markers | 0.4 | Text markers like "btw", "anyway", "oh also" |

### Sliding Window Centroids

For each position `i`, the algorithm computes a centroid (mean embedding) of the previous `window_size` messages (default 3). Comparing adjacent centroids smooths out single-message noise and detects gradual topic drift.

## Configuration

```json
{
  "segmentation": {
    "window_size": 3,
    "similarity_threshold": 0.55,
    "entity_weight": 0.3,
    "entity_jaccard_threshold": 0.2,
    "time_gap_minutes": 30.0,
    "soft_gap_minutes": 10.0,
    "coreference_enabled": false,
    "use_topic_shift_markers": true,
    "topic_shift_weight": 0.4,
    "min_segment_messages": 1,
    "max_segment_messages": 50,
    "boundary_threshold": 0.5
  }
}
```

## Data Types

### TopicSegment

```python
@dataclass
class TopicSegment:
    segment_id: str                           # UUID
    messages: list[SegmentMessage]            # Messages in segment
    start_time: datetime                      # First message timestamp
    end_time: datetime                        # Last message timestamp
    centroid: NDArray[np.float32] | None      # Mean embedding
    entities: dict[str, list[str]]            # Aggregated entities
    topic_label: str | None                   # Optional topic label
    confidence: float                         # Segmentation confidence
```

### SegmentBoundaryReason

```python
class SegmentBoundaryReason(Enum):
    EMBEDDING_DRIFT = "embedding_drift"
    ENTITY_DISCONTINUITY = "entity_discontinuity"
    TIME_GAP = "time_gap"
    TOPIC_SHIFT_MARKER = "topic_shift_marker"
```

## Usage

### Basic Segmentation

```python
from jarvis.topic_segmenter import segment_conversation

segments = segment_conversation(messages, contact_id="...")

for seg in segments:
    print(f"Topic: {seg.message_count} messages, {seg.duration_seconds:.0f}s")
    print(f"Entities: {seg.entities}")
```

### Extraction Pipeline Integration

```python
from jarvis.topic_segmenter import segment_for_extraction

# Returns list[list[Message]] for direct use in extraction
message_groups = segment_for_extraction(messages)

for group in message_groups:
    # Process each topic group
    extract_from_messages(group)
```

## Optional: Coreference Resolution

When `coreference_enabled=True`, the segmenter resolves pronouns before embedding:

- "He said he'd be late" -> "Jake said Jake'd be late"
- Requires `fastcoref` package (optional dependency)
- Improves entity continuity detection

## Performance

- Embedding computation: ~1ms per message (batched)
- Boundary scoring: O(n) linear scan
- Total: ~3-5ms for 100 messages
- Embeddings cached via `CachedEmbedder`

## Topic Shift Markers

Detected from `text_normalizer.TOPIC_SHIFT_MARKERS`:

```python
TOPIC_SHIFT_MARKERS = {
    "btw", "anyway", "oh also", "random but",
    "unrelated", "side note", "speaking of",
    "quick question", "totally different topic"
}
```

## Related Modules

- `jarvis/text_normalizer.py` - Entity extraction, topic shift detection
- `jarvis/embedding_adapter.py` - Embedding computation
- `jarvis/coref_resolver.py` - Coreference resolution (optional)
- `jarvis/ner_client.py` - Named entity recognition
