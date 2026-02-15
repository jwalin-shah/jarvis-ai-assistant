"""Segment-Based Ingestion - Extract conversation segments from iMessage.

Two-phase sequential extraction optimized for 8GB RAM:
Phase 1: Segment conversations using basic segmentation (no topic labels).
Phase 2: Extract facts from segments using fine-tuned LLM.

Uses basic_segmenter for clean boundaries without low-quality topic metadata.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jarvis.db.contacts import _CONTACT_COLUMNS

if TYPE_CHECKING:
    from jarvis.db import JarvisDB
    from jarvis.search.vec_search import VecSearcher

logger = logging.getLogger(__name__)

_LOW_INFO_VALUES = {
    "me",
    "you",
    "that",
    "this",
    "it",
    "they",
    "them",
    "him",
    "her",
}


def _should_verify_fact_value(value: str) -> bool:
    """Fast gate before NLI verification to avoid low-signal checks."""
    normalized = " ".join((value or "").strip().lower().split())
    if not normalized or normalized in _LOW_INFO_VALUES:
        return False
    if len(normalized) < 3:
        return False
    if not any(ch.isalpha() for ch in normalized):
        return False
    return True


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


def ingest_and_extract_segments(
    chat_db_reader: Any,
    jarvis_db: JarvisDB,
    vec_searcher: VecSearcher,
    progress_callback: Any | None = None,
    limit: int = 1000,
    tier: str = "0.7b",  # tier for extraction quality/speed balance
    force: bool = False,
) -> dict[str, Any]:
    """Ingest segments, index embeddings, and extract facts using a streaming delta approach."""
    from jarvis.model_manager import get_model_manager
    from jarvis.observability.pipeline_monitor import PipelineMonitor
    from jarvis.topics.basic_segmenter import segment_conversation_basic
    from jarvis.topics.segment_storage import delete_segments_for_chat, mark_facts_extracted

    monitor = PipelineMonitor()
    manager = get_model_manager()
    
    # Ensure memory is ready for embeddings
    manager.prepare_for("embedder")

    stats = SegmentExtractionStats()
    conversations = chat_db_reader.get_conversations(limit=limit)
    total = len(conversations)

    # Batch-load all contacts into cache
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

    # --- Phase 1: Segmentation (Streaming) ---
    monitor.start_stage("segmentation")
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
            
            # --- DELTA CHECK: Find last processed timestamp for this chat ---
            last_processed_at = None
            if not force:
                with jarvis_db.connection() as conn:
                    row = conn.execute(
                        "SELECT MAX(end_time) FROM conversation_segments WHERE chat_id = ?",
                        (conv.chat_id,),
                    ).fetchone()
                    if row and row[0]:
                        from datetime import datetime
                        last_processed_at = datetime.fromisoformat(row[0])

            # Fetch messages for THIS chat only (Streaming)
            messages = chat_db_reader.get_messages(
                conv.chat_id, 
                limit=None, 
                after_date=last_processed_at if not force else None
            )
            
            if not messages or len(messages) < 2:
                continue

            stats.total_messages_scanned += len(messages)

            # If force is requested, clear existing segments for this chat
            if force:
                with jarvis_db.connection() as conn:
                    delete_segments_for_chat(conn, conv.chat_id)

            # --- Automatically update contact profiles and preferences ---
            from jarvis.contacts.contact_profile import (
                ContactProfileBuilder,
                save_profile,
                update_preference_tables,
            )

            # Build/update profile with the available history
            builder = ContactProfileBuilder()
            if len(messages) >= builder.min_messages:
                try:
                    profile = builder.build_profile(
                        contact_id=conv.chat_id,
                        messages=messages,
                        contact_name=conv.display_name,
                    )
                    save_profile(profile)
                    update_preference_tables(profile, messages)
                except Exception as e:
                    logger.debug("Failed to update preferences for %s: %s", conv.chat_id, e)

            # Pre-fetch embeddings for THIS chat's messages only
            msg_ids = [m.id for m in messages if hasattr(m, "id") and m.id is not None]
            pre_fetched_embeddings = {}
            if msg_ids:
                try:
                    pre_fetched_embeddings = vec_searcher.get_embeddings_by_ids(msg_ids)
                except Exception:
                    pass

            segments = segment_conversation_basic(
                messages, contact_id=str(contact_id), pre_fetched_embeddings=pre_fetched_embeddings
            )
            
            # Filter for indexing
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
                
                stats.segments_created += len(eligible)
                monitor.stages["segmentation"].items_processed += len(eligible)
                monitor.stages["segmentation"].success_count += 1

            stats.conversations_processed += 1
            
            # Clear memory for next chat
            del messages
            del pre_fetched_embeddings
            
        except Exception as e:
            logger.warning("Error segmenting %s: %s", conv.chat_id, e)
            stats.errors.append({"chat_id": conv.chat_id, "error": str(e)})
            if "segmentation" in monitor.stages:
                monitor.stages["segmentation"].failure_count += 1

    monitor.end_stage("segmentation")

    # --- Phase 2: Batched Fact Extraction ---
    if all_created_segments:
        monitor.start_stage("extraction")
        logger.info("Phase 1 complete. Preparing memory for Phase 2...")

        from jarvis.contacts.batched_extractor import get_batched_instruction_extractor
        from jarvis.contacts.fact_verifier import FactVerifier
        from jarvis.contacts.fact_storage import save_facts

        # Use batched extractor (ModelManager handled inside extractor.load)
        batch_size = max(1, int(os.getenv("FACT_EXTRACT_BATCH_SIZE", "2")))
        extractor = get_batched_instruction_extractor(tier=tier, batch_size=batch_size)
        logger.info(
            "Phase 2: Extracting facts from %d segments using %s model (batch_size=%d)...",
            len(all_created_segments),
            tier,
            batch_size,
        )

        if extractor.load():
            try:
                verifier = FactVerifier(threshold=0.05)
                # Collect all segments for batch processing
                all_segments = [seg for _, seg, _ in all_created_segments]
                segment_db_ids = [db_id for db_id, _, _ in all_created_segments]
                chat_ids = [chat_id for _, _, chat_id in all_created_segments]

                # Extract facts in batches
                batch_results, rejection_count = extractor.extract_facts_from_segments_batch(
                    all_segments,
                    contact_id="", 
                    contact_name="Contact",
                    user_name="Me",
                )

                monitor.stages["extraction"].rejection_count += rejection_count

                # Save facts with proper segment linking
                facts_by_chat: dict[str, list[Any]] = {}
                total_rejected_by_nli = 0
                for abs_idx, facts in batch_results:
                    if abs_idx < len(segment_db_ids) and facts:
                        seg_db_id = segment_db_ids[abs_idx]
                        chat_id = chat_ids[abs_idx]
                        segment_obj = all_segments[abs_idx]

                        # Update contact_id for each fact
                        for fact in facts:
                            fact.contact_id = chat_id

                        segment_text = getattr(segment_obj, "text", "") or ""
                        if not segment_text:
                            segment_text = "\n".join(
                                (m.text or "") for m in getattr(segment_obj, "messages", [])
                            )
                        fast_gated_facts = [
                            f for f in facts if _should_verify_fact_value(getattr(f, "value", ""))
                        ]
                        if not fast_gated_facts:
                            continue
                        verified_facts, rejected = verifier.verify_facts(fast_gated_facts, segment_text)
                        total_rejected_by_nli += rejected
                        if not verified_facts:
                            continue
                        for fact in verified_facts:
                            setattr(fact, "_segment_db_id", seg_db_id)
                        facts_by_chat.setdefault(chat_id, []).extend(verified_facts)
                        
                    monitor.stages["extraction"].items_processed += 1
                    monitor.stages["extraction"].success_count += 1

                total_facts = 0
                for chat_id, chat_facts in facts_by_chat.items():
                    new_count = save_facts(
                        chat_facts,
                        chat_id,
                        segment_id=None,
                        log_raw_facts=True,
                        log_chat_id=chat_id,
                        log_stage="segment_ingest",
                    )
                    total_facts += new_count

                logger.info(
                    "Extracted %d new unique facts from %d segments",
                    total_facts,
                    len(all_created_segments),
                )
                with jarvis_db.connection() as conn:
                    mark_facts_extracted(conn, segment_db_ids)
                    conn.commit()
                if total_rejected_by_nli:
                    logger.info(
                        "NLI verification rejected %d facts during segment ingest",
                        total_rejected_by_nli,
                    )

            finally:
                extractor.unload()
                monitor.end_stage("extraction")
        else:
            logger.error("Failed to load extractor model")
            monitor.stages["extraction"].failure_count += 1
            monitor.end_stage("extraction")

    # Clean up and report
    manager.unload_all()
    
    final_stats = stats.to_dict()
    final_stats["pipeline_summary"] = monitor.get_summary()
    return final_stats


def extract_segments(
    chat_db_reader: Any,
    jarvis_db: JarvisDB,
    vec_searcher: VecSearcher,
    progress_callback: Any | None = None,
    limit: int = 1000,
    tier: str = "0.7b",
    force: bool = False,
) -> dict[str, Any]:
    """Backward-compatible alias for ingest_and_extract_segments."""
    return ingest_and_extract_segments(
        chat_db_reader=chat_db_reader,
        jarvis_db=jarvis_db,
        vec_searcher=vec_searcher,
        progress_callback=progress_callback,
        limit=limit,
        tier=tier,
        force=force,
    )
