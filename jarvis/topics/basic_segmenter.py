"""Basic Segmenter - Simple conversation segmentation without topic labels.

This is a simplified version of topic segmentation that:
1. Detects segment boundaries (embedding drift, entity shifts, time gaps)
2. Groups messages into segments
3. Stores segments WITHOUT topic labels/keywords (they were low quality)

Use this instead of topic_segmenter when you want clean segmentation
without the overhead of TF-IDF labeling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from jarvis.contracts.imessage import Message

logger = logging.getLogger(__name__)


@dataclass
class BasicSegment:
    """A simple conversation segment without topic metadata.

    Unlike TopicSegment, this has NO:
    - topic_label (often inaccurate)
    - keywords (TF-IDF was poor quality)
    - entities (redundant with fact extraction)
    - summary (not useful)

    It keeps:
    - messages (the actual content)
    - timestamps (for ordering)
    - text (concatenated for embedding)
    """

    chat_id: str
    contact_id: str | None
    messages: list[Message]
    start_time: datetime
    end_time: datetime
    message_count: int
    text: str = ""  # Concatenated for embedding/search
    segment_id: str | None = None

    # Optional: simple label from first few words (not TF-IDF)
    preview: str = ""


def segment_conversation_basic(
    messages: list[Message],
    contact_id: str | None = None,
    drift_threshold: float = 0.35,
    pre_fetched_embeddings: dict[int, Any] | None = None,
) -> list[BasicSegment]:
    """Segment messages into basic chunks (no topic labeling).

    Uses the same boundary detection as topic_segmenter but skips
    the low-quality TF-IDF labeling step.

    Args:
        messages: List of messages to segment
        contact_id: Optional contact ID
        drift_threshold: Threshold for embedding drift (0.0-1.0)
        pre_fetched_embeddings: Optional cached embeddings

    Returns:
        List of BasicSegment objects
    """
    if not messages:
        return []

    # Use shared preparation logic (sorting, junk filtering)
    from jarvis.topics.utils import get_embeddings_for_segmentation, prepare_messages_for_segmentation

    messages, norm_texts = prepare_messages_for_segmentation(messages)

    # If all messages are filtered, preserve fallback behavior.
    if not messages:
        return []

    if len(messages) < 2:
        return [_create_basic_segment(messages, contact_id)]

    # Use shared embedding logic
    embeddings_list = get_embeddings_for_segmentation(
        messages, norm_texts, pre_fetched_embeddings
    )

    # Detect boundaries
    boundaries = _detect_boundaries_basic(messages, embeddings_list, drift_threshold)

    # Split into segments
    segments = []
    start_idx = 0

    for boundary_idx in boundaries:
        seg_messages = messages[start_idx:boundary_idx]
        if seg_messages:
            segments.append(_create_basic_segment(seg_messages, contact_id))
        start_idx = boundary_idx

    # Add final segment
    if start_idx < len(messages):
        seg_messages = messages[start_idx:]
        segments.append(_create_basic_segment(seg_messages, contact_id))

    return segments


def _detect_boundaries_basic(
    messages: list[Message],
    embeddings: list[Any],
    drift_threshold: float,
) -> list[int]:
    """Detect segment boundaries using embedding drift + time gaps."""
    boundaries = []

    for i in range(1, len(messages)):
        is_boundary = False

        # Check 1: Embedding drift
        prev_emb = embeddings[i - 1]
        curr_emb = embeddings[i]

        if prev_emb is not None and curr_emb is not None:
            similarity = float(np.dot(prev_emb, curr_emb))
            if similarity < drift_threshold:
                is_boundary = True
                logger.debug(f"Boundary at {i}: drift={1 - similarity:.3f}")

        # Check 2: Time gap (>30 minutes)
        time_gap = (messages[i].date - messages[i - 1].date).total_seconds() / 60
        if time_gap > 30:
            is_boundary = True
            logger.debug(f"Boundary at {i}: time_gap={time_gap:.1f}min")

        if is_boundary:
            boundaries.append(i)

    return boundaries


def _create_basic_segment(
    messages: list[Message],
    contact_id: str | None,
) -> BasicSegment:
    """Create a BasicSegment from messages."""
    chat_id = messages[0].chat_id if messages else ""

    # Concatenate text
    texts = []
    for m in messages:
        prefix = "Me: " if m.is_from_me else "Them: "
        texts.append(f"{prefix}{m.text or ''}")

    full_text = "\n".join(texts)

    # Simple preview: first 50 chars of first message
    preview = ""
    if messages and messages[0].text:
        preview = messages[0].text[:50] + "..." if len(messages[0].text) > 50 else messages[0].text

    return BasicSegment(
        chat_id=chat_id,
        contact_id=contact_id,
        messages=messages,
        start_time=messages[0].date,
        end_time=messages[-1].date,
        message_count=len(messages),
        text=full_text,
        preview=preview,
    )


# Simple cache for embedder
_basic_segmenter_embedder = None


def get_basic_segmenter_embedder():
    global _basic_segmenter_embedder
    if _basic_segmenter_embedder is None:
        from jarvis.embedding_adapter import get_embedder

        _basic_segmenter_embedder = get_embedder()
    return _basic_segmenter_embedder
