"""Segment-Based Ingestion - Extract conversation segments from iMessage.

Two-phase sequential extraction optimized for 8GB RAM:
Phase 1: Segment conversations using basic segmentation (no topic labels).
Phase 2: Extract facts from segments using fine-tuned LLM.

Uses basic_segmenter for clean boundaries without low-quality topic metadata.
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
    tier: str = "0.7b",  # CHANGED: Use 0.7B model for optimal extraction quality/speed balance
) -> dict[str, Any]:
    """Extract conversation segments and index into vec_chunks."""
    from jarvis.topics.basic_segmenter import segment_conversation_basic

    stats = SegmentExtractionStats()
    conversations = chat_db_reader.get_conversations(limit=limit)
    total = len(conversations)

    # Batch-load all contacts
    chat_ids = [conv.chat_id for conv in conversations]
    contact_by_chat_id: dict[str, Any] = {}
    if chat_ids:
        with jarvis_db.connection() as conn:
            placeholders = ",".join("?" * len(chat_ids))
            cursor = conn.execute(
                f"SELECT {_CONTACT_COLUMNS} FROM contacts WHERE chat_id IN ({placeholders})",
                chat_ids,
            )
            for row in cursor.fetchall():
                contact = jarvis_db._row_to_contact(row)
                if contact:
                    contact_by_chat_id[contact.chat_id] = contact

    # Batch-load messages
    conv_messages: dict[str, list] = {}
    if chat_ids:
        conv_messages = chat_db_reader.get_messages_batch(chat_ids, limit_per_chat=10000)
        for messages in conv_messages.values():
            stats.total_messages_scanned += len(messages)

    # Pre-fetch embeddings
    all_msg_ids: list[int] = []
    for messages in conv_messages.values():
        for m in messages:
            if hasattr(m, "id") and m.id is not None:
                all_msg_ids.append(m.id)

    pre_fetched_embeddings: dict[int, Any] = {}
    if all_msg_ids:
        try:
            pre_fetched_embeddings = vec_searcher.get_embeddings_by_ids(all_msg_ids)
        except Exception:
            pass

    # --- Phase 1: Segmentation ---
    all_created_segments: list[tuple[int, Any, str]] = []  # (db_id, segment_obj, chat_id)

    for idx, conv in enumerate(conversations):
        if progress_callback:
            progress_callback(idx, total, conv.chat_id)

        try:
            # Ensure contact exists
            contact = contact_by_chat_id.get(conv.chat_id)
            if not contact:
                contact = jarvis_db.add_contact(
                    chat_id=conv.chat_id, display_name=conv.display_name or conv.chat_id
                )
                contact_by_chat_id[conv.chat_id] = contact

            contact_id = contact.id
            messages = conv_messages.get(conv.chat_id, [])
            if len(messages) < 2:
                continue

            segments = segment_conversation_basic(
                messages, contact_id=str(contact_id), pre_fetched_embeddings=pre_fetched_embeddings
            )
            stats.segments_created += len(segments)

            # Filter for indexing (more permissive to catch facts from small exchanges)
            eligible = [s for s in segments if s.message_count >= 1]

            if eligible:
                from jarvis.topics.segment_storage import link_vec_chunk_rowids, persist_segments

                with jarvis_db.connection() as conn:
                    db_ids = persist_segments(conn, eligible, conv.chat_id, str(contact_id))

                vec_ids = vec_searcher.index_segments(eligible, contact_id, conv.chat_id)
                stats.segments_indexed += len(vec_ids)

                if db_ids and vec_ids:
                    with jarvis_db.connection() as conn:
                        link_vec_chunk_rowids(conn, db_ids, vec_ids)

                    for db_id, seg_obj in zip(db_ids, eligible):
                        all_created_segments.append((db_id, seg_obj, conv.chat_id))

            stats.conversations_processed += 1
        except Exception as e:
            logger.warning("Error segmenting %s: %s", conv.chat_id, e)
            stats.errors.append({"chat_id": conv.chat_id, "error": str(e)})

    # --- Phase 2: Batched Fact Extraction (Model kept warm) ---
    if all_created_segments:
        logger.info("Phase 1 complete. Freeing memory for Phase 2...")

        # Explicitly unload Phase 1 resources
        from jarvis.embedding_adapter import reset_embedder

        reset_embedder()

        from jarvis.contacts.batched_extractor import get_batched_instruction_extractor
        from jarvis.contacts.fact_storage import save_facts

        # Use batched extractor with model kept warm
        extractor = get_batched_instruction_extractor(tier=tier, batch_size=5)
        logger.info(
            "Phase 2: Extracting facts from %d segments using %s model (batch_size=5)...",
            len(all_created_segments),
            tier,
        )

        # Load model ONCE and keep warm for all extractions
        if extractor.load():
            try:
                # Collect all segments for batch processing
                all_segments = [seg for _, seg, _ in all_created_segments]
                segment_db_ids = [db_id for db_id, _, _ in all_created_segments]
                chat_ids = [chat_id for _, _, chat_id in all_created_segments]

                # Extract facts in batches (5 segments per LLM call)
                batch_results = extractor.extract_facts_from_segments_batch(
                    all_segments,
                    contact_id="",  # Per-segment attribution handled in results
                    contact_name="Contact",
                    user_name="Me",
                )

                # Save facts with proper segment linking
                total_facts = 0
                for batch_idx, (local_idx, facts) in enumerate(batch_results):
                    global_idx = batch_idx * extractor._batch_size + local_idx
                    if global_idx < len(segment_db_ids) and facts:
                        seg_db_id = segment_db_ids[global_idx]
                        chat_id = chat_ids[global_idx]

                        # Update contact_id for each fact
                        for fact in facts:
                            fact.contact_id = chat_id

                        save_facts(facts, chat_id, segment_id=seg_db_id)
                        total_facts += len(facts)

                logger.info(
                    "Extracted %d facts total from %d segments",
                    total_facts,
                    len(all_created_segments),
                )

            finally:
                # Unload model ONCE at the end
                extractor.unload()
        else:
            logger.error("Failed to load extractor model")

        logger.info("Extraction complete.")

    return stats.to_dict()
