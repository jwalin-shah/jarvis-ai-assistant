"""Topic Segmenter - Semantic topic boundary detection for conversation chunking.

Replaces arbitrary time-based turn bundling with intelligent boundary detection using:
1. Sliding window embedding similarity - detect topic drift
2. Entity continuity - same people/places = same topic (Jaccard overlap)
3. Coreference resolution - resolve "he/she/it" to referents (optional, via FastCoref)
4. Text features - topic shift markers from text_normalizer

Architecture:
    Raw Messages
        |
    normalize_for_task_with_entities()  -> text + entities
        |
    CorefResolver.resolve() (optional)  -> resolved text
        |
    embedder.encode()                   -> (N, 384) embeddings
        |
    TopicSegmenter._compute_boundary_scores()
        |
    Split at boundaries + merge small segments
        |
    list[TopicSegment]

Usage:
    from jarvis.topic_segmenter import segment_conversation, get_segmenter

    # Segment a conversation
    segments = segment_conversation(messages, contact_id="...")

    # For extraction pipeline
    from jarvis.topic_segmenter import segment_for_extraction
    message_groups = segment_for_extraction(messages)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from jarvis.ner_client import Entity
from jarvis.text_normalizer import (
    TOPIC_SHIFT_MARKERS,
    extract_entities,
    normalize_for_task_with_entities,
    starts_new_topic,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from contracts.imessage import Message
    from jarvis.coref_resolver import CorefResolver
    from jarvis.embedding_adapter import CachedEmbedder

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


class SegmentBoundaryReason(Enum):
    """Reason why a segment boundary was created."""

    EMBEDDING_DRIFT = "embedding_drift"
    ENTITY_DISCONTINUITY = "entity_discontinuity"
    TIME_GAP = "time_gap"
    TOPIC_SHIFT_MARKER = "topic_shift_marker"


@dataclass
class SegmentBoundary:
    """A detected boundary between topic segments."""

    position: int  # Index in message list where boundary occurs (before this message)
    score: float  # Boundary score (higher = stronger boundary)
    reason: SegmentBoundaryReason  # Primary reason for boundary


@dataclass
class SegmentMessage:
    """A message within a topic segment with computed features."""

    text: str
    timestamp: datetime
    is_from_me: bool
    embedding: NDArray[np.float32] | None = None
    entities: list[Entity] = field(default_factory=list)
    coreference_resolved_text: str | None = None
    raw_text: str | None = None  # Original text before normalization


@dataclass
class TopicSegment:
    """A segment of conversation on a single topic."""

    segment_id: str
    messages: list[SegmentMessage]
    start_time: datetime
    end_time: datetime
    centroid: NDArray[np.float32] | None = None  # Mean embedding
    entities: dict[str, list[str]] = field(default_factory=dict)  # {"PERSON": ["Jake"]}
    topic_label: str | None = None
    confidence: float = 1.0

    @property
    def text(self) -> str:
        """Get combined text of all messages in segment."""
        return "\n".join(m.text for m in self.messages if m.text)

    @property
    def message_count(self) -> int:
        """Number of messages in this segment."""
        return len(self.messages)

    @property
    def duration_seconds(self) -> float:
        """Duration of segment in seconds."""
        return (self.end_time - self.start_time).total_seconds()


# =============================================================================
# Helper Functions (reused from topic_discovery.py)
# =============================================================================


def _entities_to_label_set(entities: list[Entity]) -> set[str]:
    """Convert entities to normalized label set for overlap similarity.

    Includes both full entity text and individual tokens (>2 chars) for
    fuzzy matching. This helps match "Jake Smith" with "Jake".

    Args:
        entities: List of Entity objects.

    Returns:
        Set of strings like {"PERSON:jake", "PERSON:jake smith", "ORG:google"}.
    """
    result: set[str] = set()
    for e in entities:
        result.add(f"{e.label}:{e.text.lower()}")
        for token in e.text.lower().split():
            if len(token) > 2:
                result.add(f"{e.label}:{token}")
    return result


def _compute_jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets.

    Args:
        set_a: First set.
        set_b: Second set.

    Returns:
        Jaccard similarity (0.0 to 1.0).
    """
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _aggregate_entities(messages: list[SegmentMessage]) -> dict[str, list[str]]:
    """Aggregate entities from a list of messages.

    Args:
        messages: List of SegmentMessage objects.

    Returns:
        Dict mapping entity labels to lists of entity texts.
    """
    from collections import Counter

    label_counts: dict[str, Counter[str]] = {}

    for msg in messages:
        for e in msg.entities:
            if e.label not in label_counts:
                label_counts[e.label] = Counter()
            label_counts[e.label][e.text.lower()] += 1

    # Keep top entities per label
    result: dict[str, list[str]] = {}
    for label, counts in label_counts.items():
        result[label] = [text for text, _ in counts.most_common(5)]

    return result


# =============================================================================
# TopicSegmenter
# =============================================================================


class TopicSegmenter:
    """Semantic topic segmenter using embedding drift and entity continuity.

    Boundary Detection Algorithm:
        boundary_score = 0.4 * embedding_drift
                       + 0.3 * (1 - entity_jaccard)
                       + 0.2 * time_penalty
                       + 0.4 * topic_shift_marker  # only if marker present

    Boundary created when score >= threshold (default 0.5).
    """

    def __init__(
        self,
        embedder: CachedEmbedder | None = None,
        window_size: int = 3,
        similarity_threshold: float = 0.55,
        entity_weight: float = 0.3,
        entity_jaccard_threshold: float = 0.2,
        time_gap_minutes: float = 30.0,
        soft_gap_minutes: float = 10.0,
        coreference_enabled: bool = False,
        use_topic_shift_markers: bool = True,
        topic_shift_weight: float = 0.4,
        min_segment_messages: int = 1,
        max_segment_messages: int = 50,
        boundary_threshold: float = 0.5,
    ) -> None:
        """Initialize topic segmenter.

        Args:
            embedder: Embedder for computing message embeddings. If None, uses default.
            window_size: Size of sliding window for computing centroids.
            similarity_threshold: Below this cosine similarity, consider topic drift.
            entity_weight: Weight for entity continuity in boundary score (0-1).
            entity_jaccard_threshold: Below this Jaccard, entity discontinuity.
            time_gap_minutes: Hard boundary if time gap exceeds this.
            soft_gap_minutes: Contributes to boundary score if gap exceeds this.
            coreference_enabled: Whether to resolve pronouns before embedding.
            use_topic_shift_markers: Whether to use text markers like "btw", "anyway".
            topic_shift_weight: Weight for topic shift markers in boundary score.
            min_segment_messages: Minimum messages per segment (merge smaller).
            max_segment_messages: Maximum messages per segment (force split larger).
            boundary_threshold: Score threshold for creating a boundary.
        """
        self._embedder = embedder
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.entity_weight = entity_weight
        self.entity_jaccard_threshold = entity_jaccard_threshold
        self.time_gap_minutes = time_gap_minutes
        self.soft_gap_minutes = soft_gap_minutes
        self.coreference_enabled = coreference_enabled
        self.use_topic_shift_markers = use_topic_shift_markers
        self.topic_shift_weight = topic_shift_weight
        self.min_segment_messages = min_segment_messages
        self.max_segment_messages = max_segment_messages
        self.boundary_threshold = boundary_threshold

        # Lazy-load coref resolver
        self._coref_resolver: CorefResolver | None = None

    def _get_embedder(self) -> CachedEmbedder:
        """Get embedder, initializing if needed."""
        if self._embedder is None:
            from jarvis.embedding_adapter import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    def _get_coref_resolver(self) -> CorefResolver | None:
        """Get coreference resolver, initializing if needed."""
        if self._coref_resolver is None and self.coreference_enabled:
            try:
                from jarvis.coref_resolver import get_coref_resolver

                self._coref_resolver = get_coref_resolver()
            except ImportError:
                logger.warning("Coreference resolution not available (fastcoref not installed)")
                self._coref_resolver = None
        return self._coref_resolver

    def segment(
        self,
        messages: list[Message],
        contact_id: str | None = None,
    ) -> list[TopicSegment]:
        """Segment a conversation into topic-coherent chunks.

        Args:
            messages: List of messages, sorted by date ascending.
            contact_id: Optional contact ID for context.

        Returns:
            List of TopicSegment objects.
        """
        if not messages:
            return []

        # Sort by timestamp
        sorted_messages = sorted(messages, key=lambda m: m.date)

        # 1. Normalize and extract entities for each message
        segment_messages = self._prepare_messages(sorted_messages)

        if not segment_messages:
            return []

        # 2. Compute embeddings
        self._compute_embeddings(segment_messages)

        # 3. Compute boundary scores
        boundaries = self._compute_boundary_scores(segment_messages)

        # 4. Split at boundaries
        segments = self._split_at_boundaries(segment_messages, boundaries)

        # 5. Merge small segments
        segments = self._merge_small_segments(segments)

        # 6. Compute segment metadata (centroids, entity aggregation)
        for segment in segments:
            self._compute_segment_metadata(segment)

        return segments

    def _prepare_messages(self, messages: list[Message]) -> list[SegmentMessage]:
        """Prepare messages for segmentation by normalizing and extracting entities.

        Args:
            messages: Raw messages.

        Returns:
            List of SegmentMessage with normalized text and entities.
        """
        segment_messages: list[SegmentMessage] = []

        # Optionally resolve coreferences
        coref_resolver = self._get_coref_resolver()
        texts_for_coref: list[str] = []
        raw_texts: list[str] = []

        for msg in messages:
            if not msg.text:
                continue

            raw_texts.append(msg.text)
            texts_for_coref.append(msg.text)

        # Batch coreference resolution if available
        resolved_texts: list[str | None] = [None] * len(texts_for_coref)
        if coref_resolver is not None and coref_resolver.is_available():
            try:
                batch_result = coref_resolver.resolve_batch(texts_for_coref)
                # Convert str to str | None (batch_result returns list[str])
                resolved_texts = [r for r in batch_result]
            except Exception as e:
                logger.warning("Coreference resolution failed: %s", e)

        # Now process each message
        msg_idx = 0
        for msg in messages:
            if not msg.text:
                continue

            # Use resolved text if available, otherwise original
            text_to_normalize = resolved_texts[msg_idx] or msg.text

            # Normalize and extract entities
            result = normalize_for_task_with_entities(text_to_normalize, "classification")

            if result.text:
                segment_messages.append(
                    SegmentMessage(
                        text=result.text,
                        timestamp=msg.date,
                        is_from_me=msg.is_from_me,
                        entities=result.entities,
                        coreference_resolved_text=(
                            resolved_texts[msg_idx] if resolved_texts[msg_idx] else None
                        ),
                        raw_text=raw_texts[msg_idx],
                    )
                )

            msg_idx += 1

        return segment_messages

    def _compute_embeddings(self, messages: list[SegmentMessage]) -> None:
        """Compute embeddings for all messages.

        Args:
            messages: List of SegmentMessage (modified in place).
        """
        if not messages:
            return

        embedder = self._get_embedder()
        texts = [m.text for m in messages]

        try:
            embeddings = embedder.encode(texts, normalize=True)
            for i, msg in enumerate(messages):
                msg.embedding = embeddings[i]
        except Exception as e:
            logger.warning("Failed to compute embeddings: %s", e)

    def _compute_boundary_scores(
        self,
        messages: list[SegmentMessage],
    ) -> list[SegmentBoundary]:
        """Compute boundary scores between consecutive messages.

        Boundary score formula:
            score = 0.4 * embedding_drift
                  + 0.3 * (1 - entity_jaccard)
                  + 0.2 * time_penalty
                  + 0.4 * topic_shift_marker (only if present)

        Args:
            messages: List of SegmentMessage with embeddings and entities.

        Returns:
            List of SegmentBoundary for positions that exceed threshold.
        """
        if len(messages) < 2:
            return []

        boundaries: list[SegmentBoundary] = []

        # Pre-compute entity sets for each message
        entity_sets = [_entities_to_label_set(m.entities) for m in messages]

        # Compute sliding window centroids for embedding drift detection
        window_centroids = self._compute_window_centroids(messages)

        for i in range(1, len(messages)):
            score = 0.0
            primary_reason = SegmentBoundaryReason.EMBEDDING_DRIFT

            # 1. Embedding drift (compare window centroids)
            embedding_drift = 0.0
            if (
                window_centroids is not None
                and i < len(window_centroids)
                and i > 0
            ):
                prev_centroid = window_centroids[i - 1]
                curr_centroid = window_centroids[i]
                if prev_centroid is not None and curr_centroid is not None:
                    similarity = float(np.dot(prev_centroid, curr_centroid))
                    if similarity < self.similarity_threshold:
                        embedding_drift = 1.0 - similarity
            score += 0.4 * embedding_drift

            # 2. Entity discontinuity (Jaccard between adjacent messages)
            entity_jaccard = _compute_jaccard(entity_sets[i - 1], entity_sets[i])
            if entity_jaccard < self.entity_jaccard_threshold:
                entity_discontinuity = 1.0 - entity_jaccard
                score += self.entity_weight * entity_discontinuity
                if entity_discontinuity > embedding_drift:
                    primary_reason = SegmentBoundaryReason.ENTITY_DISCONTINUITY

            # 3. Time gap penalty
            time_gap_mins = (
                messages[i].timestamp - messages[i - 1].timestamp
            ).total_seconds() / 60.0

            # Hard boundary for large time gaps
            if time_gap_mins >= self.time_gap_minutes:
                boundaries.append(
                    SegmentBoundary(
                        position=i,
                        score=1.0,
                        reason=SegmentBoundaryReason.TIME_GAP,
                    )
                )
                continue

            # Soft penalty for moderate time gaps
            if time_gap_mins >= self.soft_gap_minutes:
                time_penalty = min(
                    1.0,
                    (time_gap_mins - self.soft_gap_minutes)
                    / (self.time_gap_minutes - self.soft_gap_minutes),
                )
                score += 0.2 * time_penalty
                if time_penalty > embedding_drift and time_penalty > (1.0 - entity_jaccard):
                    primary_reason = SegmentBoundaryReason.TIME_GAP

            # 4. Topic shift markers (btw, anyway, etc.)
            if self.use_topic_shift_markers:
                raw_text = messages[i].raw_text or messages[i].text
                if starts_new_topic(raw_text):
                    score += self.topic_shift_weight
                    primary_reason = SegmentBoundaryReason.TOPIC_SHIFT_MARKER

            # Create boundary if score exceeds threshold
            if score >= self.boundary_threshold:
                boundaries.append(
                    SegmentBoundary(
                        position=i,
                        score=score,
                        reason=primary_reason,
                    )
                )

        return boundaries

    def _compute_window_centroids(
        self,
        messages: list[SegmentMessage],
    ) -> NDArray[np.float32] | None:
        """Compute sliding window centroids for embedding drift detection.

        For each position i, computes the mean embedding of messages
        [max(0, i - window_size + 1), i + 1].

        Args:
            messages: List of SegmentMessage with embeddings.

        Returns:
            Array of shape (n_messages, embedding_dim) with centroids,
            or None if no embeddings available.
        """
        if not messages or messages[0].embedding is None:
            return None

        n = len(messages)
        dim = len(messages[0].embedding)
        centroids = np.zeros((n, dim), dtype=np.float32)

        for i in range(n):
            # Window: [max(0, i - window_size + 1), i + 1]
            start = max(0, i - self.window_size + 1)
            window_embeddings = [
                m.embedding for m in messages[start : i + 1] if m.embedding is not None
            ]

            if window_embeddings:
                centroid = np.mean(window_embeddings, axis=0)
                # L2 normalize
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                centroids[i] = centroid

        return centroids

    def _split_at_boundaries(
        self,
        messages: list[SegmentMessage],
        boundaries: list[SegmentBoundary],
    ) -> list[TopicSegment]:
        """Split messages into segments at boundary positions.

        Args:
            messages: List of SegmentMessage.
            boundaries: List of SegmentBoundary.

        Returns:
            List of TopicSegment.
        """
        if not messages:
            return []

        # Get boundary positions
        boundary_positions = sorted(b.position for b in boundaries)

        segments: list[TopicSegment] = []
        start = 0

        for pos in boundary_positions:
            if pos > start:
                segment_messages = messages[start:pos]
                if segment_messages:
                    segments.append(
                        TopicSegment(
                            segment_id=str(uuid.uuid4()),
                            messages=segment_messages,
                            start_time=segment_messages[0].timestamp,
                            end_time=segment_messages[-1].timestamp,
                        )
                    )
            start = pos

        # Don't forget the last segment
        if start < len(messages):
            segment_messages = messages[start:]
            if segment_messages:
                segments.append(
                    TopicSegment(
                        segment_id=str(uuid.uuid4()),
                        messages=segment_messages,
                        start_time=segment_messages[0].timestamp,
                        end_time=segment_messages[-1].timestamp,
                    )
                )

        return segments

    def _merge_small_segments(
        self,
        segments: list[TopicSegment],
    ) -> list[TopicSegment]:
        """Merge segments that are smaller than min_segment_messages.

        Merges small segments into the previous segment when possible.

        Args:
            segments: List of TopicSegment.

        Returns:
            List of TopicSegment with small segments merged.
        """
        if not segments or self.min_segment_messages <= 1:
            return segments

        merged: list[TopicSegment] = []

        for segment in segments:
            if merged and segment.message_count < self.min_segment_messages:
                # Merge into previous segment
                prev = merged[-1]
                prev.messages.extend(segment.messages)
                prev.end_time = segment.end_time
            else:
                merged.append(segment)

        # Handle case where last segment is still too small
        if len(merged) > 1 and merged[-1].message_count < self.min_segment_messages:
            last = merged.pop()
            merged[-1].messages.extend(last.messages)
            merged[-1].end_time = last.end_time

        return merged

    def _compute_segment_metadata(self, segment: TopicSegment) -> None:
        """Compute centroid and aggregate entities for a segment.

        Args:
            segment: TopicSegment (modified in place).
        """
        # Compute centroid
        embeddings = [m.embedding for m in segment.messages if m.embedding is not None]
        if embeddings:
            centroid = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            segment.centroid = centroid.astype(np.float32)

        # Aggregate entities
        segment.entities = _aggregate_entities(segment.messages)


# =============================================================================
# Singleton and Public API
# =============================================================================

_segmenter: TopicSegmenter | None = None


def get_segmenter() -> TopicSegmenter:
    """Get the singleton TopicSegmenter instance.

    Reads configuration from config.segmentation.

    Returns:
        TopicSegmenter instance.
    """
    global _segmenter
    if _segmenter is None:
        from jarvis.config import get_config

        config = get_config()

        # Check if segmentation config exists (may not if not added yet)
        seg_config = getattr(config, "segmentation", None)

        if seg_config is not None:
            _segmenter = TopicSegmenter(
                window_size=seg_config.window_size,
                similarity_threshold=seg_config.similarity_threshold,
                entity_weight=seg_config.entity_weight,
                entity_jaccard_threshold=seg_config.entity_jaccard_threshold,
                time_gap_minutes=seg_config.time_gap_minutes,
                soft_gap_minutes=seg_config.soft_gap_minutes,
                coreference_enabled=seg_config.coreference_enabled,
                use_topic_shift_markers=seg_config.use_topic_shift_markers,
                topic_shift_weight=seg_config.topic_shift_weight,
                min_segment_messages=seg_config.min_segment_messages,
                max_segment_messages=seg_config.max_segment_messages,
                boundary_threshold=seg_config.boundary_threshold,
            )
        else:
            _segmenter = TopicSegmenter()

    return _segmenter


def reset_segmenter() -> None:
    """Reset the singleton segmenter for testing."""
    global _segmenter
    _segmenter = None


def segment_conversation(
    messages: list[Message],
    contact_id: str | None = None,
) -> list[TopicSegment]:
    """Segment a conversation into topic-coherent chunks.

    Convenience function using the singleton segmenter.

    Args:
        messages: List of messages, sorted by date ascending.
        contact_id: Optional contact ID for context.

    Returns:
        List of TopicSegment objects.
    """
    segmenter = get_segmenter()
    return segmenter.segment(messages, contact_id)


def segment_for_extraction(
    messages: list[Message],
) -> list[list[Message]]:
    """Segment messages for the extraction pipeline.

    Returns groups of messages (as original Message objects) that belong
    to the same topic segment. This is designed to replace time-based
    turn bundling in extract.py.

    Args:
        messages: List of messages, sorted by date ascending.

    Returns:
        List of message groups, where each group is a topic segment.
    """
    if not messages:
        return []

    # Get segments
    segments = segment_conversation(messages)

    # Map segment messages back to original messages by timestamp
    # Build a lookup from timestamp to original message
    msg_by_time: dict[datetime, Message] = {m.date: m for m in messages}

    result: list[list[Message]] = []
    for segment in segments:
        group: list[Message] = []
        for seg_msg in segment.messages:
            orig_msg = msg_by_time.get(seg_msg.timestamp)
            if orig_msg:
                group.append(orig_msg)
        if group:
            result.append(group)

    return result


__all__ = [
    # Data classes
    "SegmentBoundaryReason",
    "SegmentBoundary",
    "SegmentMessage",
    "TopicSegment",
    # Main class
    "TopicSegmenter",
    # Public API
    "get_segmenter",
    "reset_segmenter",
    "segment_conversation",
    "segment_for_extraction",
]
