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

from jarvis.db.contacts import _CONTACT_COLUMNS

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

    # Batch-load all contacts to avoid N+1 queries in the loop
    chat_ids = [conv.chat_id for conv in conversations]
    contact_by_chat_id: dict[str, Any] = {}
    if chat_ids:
        with jarvis_db.connection() as conn:
            # Build placeholders for IN clause
            placeholders = ",".join("?" * len(chat_ids))
            cursor = conn.execute(
                f"SELECT {_CONTACT_COLUMNS} FROM contacts WHERE chat_id IN ({placeholders})",
                chat_ids,
            )
            for row in cursor.fetchall():
                contact = jarvis_db._row_to_contact(row)
                if contact:
                    contact_by_chat_id[contact.chat_id] = contact

            # Also try extracting identifiers for iMessage-format chat_ids
            # (e.g., "iMessage;-;+15551234567" -> "+15551234567")
            missing_ids = [cid for cid in chat_ids if cid not in contact_by_chat_id]
            identifier_map: dict[str, str] = {}  # identifier -> original chat_id
            for cid in missing_ids:
                if ";" in cid:
                    identifier = cid.rsplit(";", 1)[-1]
                    if identifier:
                        identifier_map[identifier] = cid
            if identifier_map:
                id_placeholders = ",".join("?" * len(identifier_map))
                cursor = conn.execute(
                    f"SELECT {_CONTACT_COLUMNS} FROM contacts WHERE chat_id IN ({id_placeholders})",
                    list(identifier_map.keys()),
                )
                for row in cursor.fetchall():
                    contact = jarvis_db._row_to_contact(row)
                    if contact and contact.chat_id in identifier_map:
                        orig_chat_id = identifier_map[contact.chat_id]
                        contact_by_chat_id[orig_chat_id] = contact

    # Batch-load messages for all conversations (single query per chunk of 900)
    conv_messages: dict[str, list] = {}
    if chat_ids:
        conv_messages = chat_db_reader.get_messages_batch(chat_ids, limit_per_chat=10000)
        for messages in conv_messages.values():
            stats.total_messages_scanned += len(messages)

    # Batch-fetch all cached embeddings once across all contacts (PERF-08)
    all_msg_ids: list[int] = []
    for messages in conv_messages.values():
        for m in messages:
            msg_id = getattr(m, "id", None)
            if msg_id is not None:
                all_msg_ids.append(msg_id)

    pre_fetched_embeddings: dict[int, Any] = {}
    if all_msg_ids:
        try:
            pre_fetched_embeddings = vec_searcher.get_embeddings_by_ids(all_msg_ids)
            logger.info(
                "Pre-fetched %d/%d embeddings from vec_messages cache",
                len(pre_fetched_embeddings),
                len(all_msg_ids),
            )
        except Exception:
            pass  # Fall through to per-contact encoding

    for idx, conv in enumerate(conversations):
        if progress_callback:
            progress_callback(idx, total, conv.chat_id)

        try:
            # Resolve contact from batch-loaded dict
            contact = contact_by_chat_id.get(conv.chat_id)
            contact_id = contact.id if contact else None

            # Get messages (already batch-loaded above)
            messages = conv_messages.get(conv.chat_id, [])

            if len(messages) < 2:
                continue

            # Segment the conversation with pre-fetched embeddings
            segments = segment_conversation(
                messages,
                contact_id=str(contact_id) if contact_id else None,
                pre_fetched_embeddings=pre_fetched_embeddings,
            )
            stats.segments_created += len(segments)

            # Filter segments before batch indexing
            eligible_segments = []
            for segment in segments:
                # Skip segments without any response (me) messages
                has_response = any(m.is_from_me for m in segment.messages)
                if not has_response:
                    stats.segments_skipped_no_response += 1
                    continue

                if segment.message_count < 2:
                    stats.segments_skipped_too_short += 1
                    continue

                eligible_segments.append(segment)

            # Batch index all eligible segments (single connection + executemany)
            if eligible_segments:
                stats.segments_indexed += len(
                    vec_searcher.index_segments(
                        eligible_segments, contact_id, conv.chat_id
                    )
                )

            stats.conversations_processed += 1

        except Exception as e:
            logger.warning("Error segmenting %s: %s", conv.chat_id, e)
            stats.errors.append({"chat_id": conv.chat_id, "error": str(e)})

    return stats.to_dict()
