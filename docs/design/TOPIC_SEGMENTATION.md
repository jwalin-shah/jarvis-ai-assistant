# Topic Segmentation

> **Last Updated:** 2026-02-13

Semantic topic boundary detection for conversation chunking. Replaces arbitrary time-based turn bundling with intelligent boundary detection.

**Module:** `jarvis/topics/topic_segmenter.py`

## Overview

Topic segmentation divides long conversations into coherent topic chunks, enabling:
- More accurate context retrieval (fetch entire topic, not arbitrary windows)
- Better extraction pipeline processing (group related messages)
- Improved summarization (summarize by topic, not by time)

## Architecture

```
Raw Messages
    |
normalize_text()  -> expanded slang, cleaned text
    |
embedder.encode()                   -> (N, 384) embeddings via get_embedder()
    |
EntityAnchorTracker.get_anchors()  -> entity continuity check
    |
boundary_score = 0.4 * embedding_drift + 0.3 * entity_component + 0.2 * time_penalty + shift_penalty
    |
Split at boundaries (score >= threshold OR hard time gap)
    |
_create_segment() + _compute_segment_metadata()
    |
list[TopicSegment]
```

## Boundary Detection Algorithm

The segmenter computes a boundary score between consecutive messages:

```python
embedding_component = drift  # cosine similarity drop
entity_component = 1.0 - entity_jaccard if entity_jaccard > 0 else 1.0
time_penalty = min(time_diff_hours / dynamic_gap_threshold, 1.0) if is_large_gap else 0.0
shift_penalty = topic_shift_weight if has_topic_shift else 0.0

boundary_score = (
    0.4 * embedding_component + 0.3 * entity_component + 0.2 * time_penalty + shift_penalty
)
```

A boundary is created when `score >= boundary_threshold` (default 0.5) OR when there's a hard time gap (30+ minutes by default).

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
    "boundary_threshold": 0.5,
    "drift_threshold": 0.35
  }
}
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `boundary_threshold` | 0.5 | Score above this creates a boundary |
| `drift_threshold` | 0.35 | Default threshold in `segment_conversation()` function |
| `similarity_threshold` | 0.55 | Cosine similarity below this indicates drift |
| `time_gap_minutes` | 30.0 | Hard gap threshold (always splits) |
| `soft_gap_minutes` | 10.0 | Soft gap threshold (contributes to score) |

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
from jarvis.topics.topic_segmenter import segment_conversation

# With pre-fetched embeddings (for efficiency)
segments = segment_conversation(
    messages, 
    contact_id="...",
    drift_threshold=0.35,
    pre_fetched_embeddings={msg.id: embedding for msg, embedding in zip(messages, embeddings)}
)

for seg in segments:
    print(f"Segment: {seg.message_count} messages")
    print(f"Topic label: {seg.topic_label}")
    print(f"Summary: {seg.summary}")
    print(f"Keywords: {seg.keywords}")
```

### Extraction Pipeline Integration

```python
from jarvis.topics.segment_pipeline import process_segments

# Full pipeline: persist segments, optionally extract facts
results = process_segments(messages, contact_id="...", extract_facts=True)

for seg in results.segments:
    print(f"Segment {seg.segment_id}: {seg.message_count} messages")
```

## Optional: Coreference Resolution

When `coreference_enabled=True`, the segmenter resolves pronouns before embedding:

- "He said he'd be late" -> "Jake said Jake'd be late"
- Requires `fastcoref` package (optional dependency)
- Improves entity continuity detection

## Performance

- Embedding computation: ~1ms per message (batched via `embed_batch`)
- Boundary scoring: O(n) linear scan
- Total: ~3-5ms for 100 messages
- Embeddings cached via `CachedEmbedder` (1000 entry LRU)
- Entity anchors cached via `EntityAnchorTracker` singleton

## Topic Shift Markers

Detected from `jarvis.text_normalizer.TOPIC_SHIFT_MARKERS`:

```python
TOPIC_SHIFT_MARKERS = frozenset({
    "btw", "anyway", "oh also", "random but",
    "unrelated", "side note", "speaking of",
    "quick question", "totally different topic",
    "by the way", "side note", "changing subject"
})
```

## Related Modules

- `jarvis/topics/entity_anchor.py` - EntityAnchorTracker for continuity detection
- `jarvis/topics/segment_labeler.py` - Topic label generation
- `jarvis/topics/segment_storage.py` - Database persistence
- `jarvis/topics/segment_pipeline.py` - Full extraction pipeline orchestration
- `jarvis/topics/segment_extractor.py` - Bridge to fact extraction
- `jarvis/text_normalizer.py` - Text normalization, topic shift detection
- `jarvis/embedding_adapter.py` - Embedding computation via get_embedder()
