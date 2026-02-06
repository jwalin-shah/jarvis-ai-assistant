"""Segment-Based Ingestion - Extract topic segments from iMessage conversations.

Replaces the legacy ExchangeBuilder pipeline with NLP-based topic segmentation.
Segments become the primary retrieval unit in vec_chunks.

Usage:
    jarvis db extract   # CLI command (updated to use this module)

Pipeline:
    1. For each conversation: segment_conversation(messages) -> list[TopicSegment]
    2. For each segment: index into vec_chunks (centroid embedding + metadata)
    3. Within each segment: extract trigger/response pairs for few-shot retrieval
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.db import JarvisDB
    from jarvis.search.vec_search import VecSearcher

logger = logging.getLogger(__name__)


@dataclass
class SegmentExtractionStats:
    """Statistics from segment-based extraction."""

    conversations_processed: int = 0
    total_messages_scanned: int = 0
    segments_created: int = 0
    segments_indexed: int = 0
    segments_skipped_no_response: int = 0
    segments_skipped_too_short: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversations_processed": self.conversations_processed,
            "total_messages_scanned": self.total_messages_scanned,
            "segments_created": self.segments_created,
            "segments_indexed": self.segments_indexed,
            "segments_skipped_no_response": self.segments_skipped_no_response,
            "segments_skipped_too_short": self.segments_skipped_too_short,
            "errors": self.errors,
        }


def extract_segments(
    chat_db_reader: Any,
    jarvis_db: JarvisDB,
    vec_searcher: VecSearcher,
    progress_callback: Any | None = None,
    limit: int = 1000,
) -> dict[str, Any]:
    """Extract topic segments from all conversations and index into vec_chunks.

    Args:
        chat_db_reader: ChatDBReader instance for reading iMessage.
        jarvis_db: JarvisDB instance.
        vec_searcher: VecSearcher instance for indexing.
        progress_callback: Optional callback(current, total, chat_id).
        limit: Max conversations to process.

    Returns:
        Stats dictionary.
    """
    from jarvis.topics.topic_segmenter import segment_conversation

    stats = SegmentExtractionStats()

    conversations = chat_db_reader.get_conversations(limit=limit)
    total = len(conversations)

    for idx, conv in enumerate(conversations):
        if progress_callback:
            progress_callback(idx, total, conv.chat_id)

        try:
            # Resolve contact
            contact = jarvis_db.get_contact_by_chat_id(conv.chat_id)
            contact_id = contact.id if contact else None

            # Get messages
            messages = chat_db_reader.get_messages(conv.chat_id, limit=10000)
            stats.total_messages_scanned += len(messages)

            if len(messages) < 2:
                continue

            # Segment the conversation
            segments = segment_conversation(
                messages, contact_id=str(contact_id) if contact_id else None
            )
            stats.segments_created += len(segments)

            # Index each segment
            for segment in segments:
                # Skip segments without any response (me) messages
                has_response = any(m.is_from_me for m in segment.messages)
                if not has_response:
                    stats.segments_skipped_no_response += 1
                    continue

                if segment.message_count < 2:
                    stats.segments_skipped_too_short += 1
                    continue

                if vec_searcher.index_segment(segment, contact_id, conv.chat_id):
                    stats.segments_indexed += 1

            stats.conversations_processed += 1

        except Exception as e:
            logger.warning("Error segmenting %s: %s", conv.chat_id, e)
            stats.errors.append({"chat_id": conv.chat_id, "error": str(e)})

    return stats.to_dict()
