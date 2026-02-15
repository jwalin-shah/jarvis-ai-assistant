"""Topic Segmenter - Splits conversations into semantic segments using BGE + Anchors.

Uses BGE-Small for embedding drift detection and EntityAnchorTracker
for multi-signal continuity (contacts, noun chunks, semantic subjects).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from jarvis.contracts.imessage import Message

from enum import Enum

logger = logging.getLogger(__name__)


class SegmentBoundaryReason(Enum):
    """Reason why a boundary was detected."""

    EMBEDDING_DRIFT = "embedding_drift"
    ENTITY_SHIFT = "entity_shift"
    GAP_DETECTION = "gap_detection"
    MANUAL_SPLIT = "manual_split"


@dataclass
class SegmentBoundary:
    """A point in the conversation where a topic likely changed."""

    index: int
    drift: float
    confidence: float
    reason: SegmentBoundaryReason


@dataclass
class SegmentMessage:
    """Minimal message representation for segment storage."""

    id: int
    text: str
    is_from_me: bool
    date: datetime


@dataclass
class TopicSegment:
    """A contiguous set of messages belonging to one topic."""

    chat_id: str
    contact_id: str | None
    messages: list[Message]
    start_time: datetime
    end_time: datetime
    message_count: int
    segment_id: str | None = None
    text: str = ""  # Full concatenated text
    keywords: list[str] = field(default_factory=list)
    centroid: NDArray[np.float32] | None = None
    entities: dict[str, list[str]] = field(default_factory=dict)
    topic_label: str | None = None
    summary: str | None = None
    confidence: float = 1.0


def segment_conversation(
    messages: list[Message],
    contact_id: str | None = None,
    drift_threshold: float = 0.35,
    pre_fetched_embeddings: dict[int, Any] | None = None,
) -> list[TopicSegment]:
    """Segment a list of messages into topics."""
    if not messages:
        return []

    # Ensure oldest-first ordering (messages may come newest-first from DB)
    messages = sorted(messages, key=lambda m: m.date)

    # Pre-filter junk/system/spam messages before embedding and boundary scoring.
    from jarvis.contacts.junk_filters import is_junk_message
    from jarvis.text_normalizer import normalize_text

    filtered_messages: list[Message] = []
    filtered_norm_texts: list[str] = []
    for m in messages:
        raw_text = m.text or ""
        norm_text = normalize_text(
            raw_text,
            expand_slang=True,
            filter_garbage=True,
            filter_attributed_artifacts=True,
            strip_signatures=True,
        )
        if not norm_text:
            continue
        if is_junk_message(norm_text, m.chat_id):
            continue
        filtered_messages.append(m)
        filtered_norm_texts.append(norm_text)

    # If everything is filtered, preserve behavior by returning one segment on original messages.
    if not filtered_messages:
        if messages:
            return [_create_segment(messages, contact_id)]
        return []

    messages = filtered_messages

    n = len(messages)
    if n < 2:
        # Create single segment for short conversations
        seg = _create_segment(messages, contact_id)
        return [seg]

    # Signal 1: Get BGE Embeddings (on normalized text)
    from jarvis.embedding_adapter import get_embedder

    embedder = get_embedder()

    # Normalize text before embedding (expands slang, cleans text)
    msg_texts = filtered_norm_texts

    # Use pre-fetched if available, otherwise batch encode
    embeddings: list[NDArray[np.float32]] = []
    to_encode_indices = []
    to_encode_texts = []

    # Initialize list
    embeddings = [None] * n

    if pre_fetched_embeddings:
        for i, m in enumerate(messages):
            if m.id in pre_fetched_embeddings:
                embeddings[i] = pre_fetched_embeddings[m.id]

    for i, emb in enumerate(embeddings):
        if emb is None:
            to_encode_indices.append(i)
            to_encode_texts.append(msg_texts[i])

    if to_encode_texts:
        # Filter out empty texts after normalization
        valid_pairs = [(idx, text) for idx, text in zip(to_encode_indices, to_encode_texts) if text]
        if valid_pairs:
            valid_indices = [p[0] for p in valid_pairs]
            valid_texts = [p[1] for p in valid_pairs]
            new_embs = embedder.embed_batch(valid_texts)
            for idx, emb in zip(valid_indices, new_embs):
                embeddings[idx] = emb

    # Signal 2: Entity Anchor Continuity (Contact-Aware spaCy)
    from jarvis.text_normalizer import TOPIC_SHIFT_MARKERS
    from jarvis.topics.entity_anchor import get_tracker

    anchor_tracker = get_tracker()

    # Load config with defaults
    cfg = _get_segmentation_config()
    topic_shift_weight = cfg.topic_shift_weight
    boundary_threshold = cfg.boundary_threshold
    use_topic_shift = cfg.use_topic_shift_markers

    # Compute dynamic time gap threshold based on conversation patterns
    # Use median gap - if messages typically arrive 1hr apart, a 4hr gap is significant
    time_gaps: list[float] = []
    for i in range(1, n):
        gap_hours = (messages[i].date - messages[i - 1].date).total_seconds() / 3600.0
        if gap_hours > 0 and gap_hours < 168:  # Ignore gaps > 1 week (likely new conversation)
            time_gaps.append(gap_hours)

    if time_gaps:
        import statistics

        median_gap = statistics.median(time_gaps)
        # Gap is "large" if it's 4x the typical gap, minimum 15 min
        dynamic_gap_threshold = max(median_gap * 4, 0.25)  # 0.25 hours = 15 min
    else:
        dynamic_gap_threshold = 24.0  # Default: 24 hours if only 1 message

    segments: list[TopicSegment] = []
    current_chunk: list[Message] = [messages[0]]
    current_segment_anchors: set[str] = anchor_tracker.get_anchors(messages[0].text or "")

    for i in range(1, n):
        prev_msg = messages[i - 1]
        curr_msg = messages[i]

        # Signal 0: Dynamic Time Gap - based on conversation's typical pace
        time_diff_hours = (curr_msg.date - prev_msg.date).total_seconds() / 3600.0
        is_large_gap = time_diff_hours > dynamic_gap_threshold

        # Signal 1: Embedding Drift (semantic meaning change)
        v1 = embeddings[i - 1]
        v2 = embeddings[i]

        # Skip if either embedding is missing (empty text after normalization)
        if v1 is None or v2 is None:
            # Treat as no drift (keep in same segment)
            continue

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        sim = np.dot(v1, v2) / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 1.0
        drift = 1.0 - sim

        # Signal 2: Entity Continuity (Jaccard) - are we still talking about same things?
        msg_text = curr_msg.text or ""
        msg_anchors = anchor_tracker.get_anchors(msg_text)

        if current_segment_anchors and msg_anchors:
            intersection = len(msg_anchors & current_segment_anchors)
            union = len(msg_anchors | current_segment_anchors)
            entity_jaccard = intersection / union if union > 0 else 0.0
        else:
            entity_jaccard = 0.0

        # Signal 3: Topic Shift Markers - explicit "btw", "anyway", etc.
        has_topic_shift = False
        if use_topic_shift:
            lower_text = msg_text.lower()
            has_topic_shift = any(marker in lower_text for marker in TOPIC_SHIFT_MARKERS)

        # Update segment anchors
        current_segment_anchors.update(msg_anchors)

        # Weighted boundary scoring: meaning-based signals
        # Time is a soft signal, not hard boundary - conversation pace varies
        embedding_component = drift
        entity_component = 1.0 - entity_jaccard if entity_jaccard > 0 else 1.0
        time_penalty = min(time_diff_hours / dynamic_gap_threshold, 1.0) if is_large_gap else 0.0
        shift_penalty = topic_shift_weight if has_topic_shift else 0.0

        boundary_score = (
            0.4 * embedding_component + 0.3 * entity_component + 0.2 * time_penalty + shift_penalty
        )

        # Split if boundary score exceeds threshold OR hard time gap
        if boundary_score >= boundary_threshold or is_large_gap:
            seg_embeddings = embeddings[i - len(current_chunk) : i]
            segments.append(_create_segment(current_chunk, contact_id, seg_embeddings))
            current_chunk = [messages[i]]
            current_segment_anchors = msg_anchors
        else:
            current_chunk.append(messages[i])

    # Add final segment
    if current_chunk:
        seg_embeddings = embeddings[n - len(current_chunk) : n]
        segments.append(_create_segment(current_chunk, contact_id, seg_embeddings))

    # Compute metadata (labels, summaries) for all segments
    for seg in segments:
        _compute_segment_metadata(seg)

    return segments


def _create_segment(
    messages: list[Message],
    contact_id: str | None,
    embeddings: list[NDArray[np.float32]] | None = None,
) -> TopicSegment:
    """Helper to initialize a TopicSegment from a list of messages."""
    import uuid

    centroid = None
    if embeddings:
        # Filter out None embeddings
        valid_embs = [e for e in embeddings if e is not None]
        if valid_embs:
            centroid = np.mean(valid_embs, axis=0)

    return TopicSegment(
        chat_id=messages[0].chat_id,
        contact_id=contact_id,
        messages=messages,
        start_time=messages[0].date,
        end_time=messages[-1].date,
        message_count=len(messages),
        segment_id=str(uuid.uuid4()),
        text="\n".join([m.text for m in messages if m.text]),
        centroid=centroid,
    )


def _get_segmentation_config():
    """Load segmentation config with defaults."""
    from jarvis.config import get_config

    try:
        return get_config().segmentation
    except Exception:
        # Return defaults if config unavailable
        from dataclasses import dataclass

        @dataclass
        class Defaults:
            time_gap_minutes = 30.0
            similarity_threshold = 0.55
            entity_weight = 0.3
            entity_jaccard_threshold = 0.2
            use_topic_shift_markers = True
            topic_shift_weight = 0.4
            boundary_threshold = 0.5

        return Defaults()


def _compute_segment_metadata(segment: TopicSegment):
    """Generate labels, keywords and summary for a segment."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    from jarvis.text_normalizer import normalize_text

    if not segment.text:
        return

    # Normalize text before TF-IDF (expands slang like "lmao" → "laughing my ass off")
    normalized_text = normalize_text(segment.text, expand_slang=True)

    # 1. Keywords via simple TF-IDF on normalized text
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5)
        # We need at least a few words to make TF-IDF useful
        if normalized_text and len(normalized_text.split()) > 3:
            vectorizer.fit([normalized_text])
            segment.keywords = list(vectorizer.get_feature_names_out())
    except Exception:
        pass

    # 2. Label and Summary via SegmentLabeler
    try:
        from jarvis.topics.segment_labeler import get_labeler

        labeler = get_labeler()
        segment.topic_label = labeler.label_segment(segment)
        segment.summary = labeler.summarize_segment(segment)
    except Exception as e:
        logger.debug("Labeler failed: %s", e)
        segment.topic_label = segment.keywords[0].title() if segment.keywords else "General Chat"
        segment.summary = segment.topic_label


class TopicSegmenter:
    """Class-based wrapper for segmentation (legacy support)."""

    def segment(self, messages: list[Message], **kwargs) -> list[TopicSegment]:
        return segment_conversation(messages, **kwargs)


def segment_for_extraction(messages: list[Message], **kwargs) -> list[TopicSegment]:
    """Helper for extraction pipeline."""
    return segment_conversation(messages, **kwargs)


_segmenter: TopicSegmenter | None = None


def get_segmenter() -> TopicSegmenter:
    global _segmenter
    if _segmenter is None:
        _segmenter = TopicSegmenter()
    return _segmenter


def reset_segmenter():
    global _segmenter
    _segmenter = None
