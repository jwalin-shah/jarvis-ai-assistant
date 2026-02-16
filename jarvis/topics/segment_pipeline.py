"""Unified segment pipeline: persist → index → extract facts.

Orchestrates the full lifecycle of topic segments, replacing the
previously-separate segmentation and extraction flows in the watcher.
"""

from __future__ import annotations

import logging
import os
from hashlib import sha1
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.topics.topic_segmenter import TopicSegment

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
    # Skip values with no letters (often timestamps/noise).
    if not any(ch.isalpha() for ch in normalized):
        return False
    return True


def _segment_fingerprint(segment: TopicSegment) -> str:
    text = getattr(segment, "text", "") or ""
    if not text:
        text = "\n".join((m.text or "") for m in getattr(segment, "messages", []))
    normalized = " ".join(text.lower().split())
    return sha1(normalized.encode("utf-8")).hexdigest()


def process_segments(
    segments: list[TopicSegment],
    chat_id: str,
    contact_id: str | None = None,
    extract_facts: bool = True,
) -> dict[str, Any]:
    """Persist segments, index into vec_chunks, and optionally extract facts.

    This is the single entry point that replaces calling segmentation,
    indexing, and extraction separately.

    Uses SINGLE TRANSACTION for all database operations to ensure atomicity.

    Args:
        segments: TopicSegment objects from the segmenter.
        chat_id: iMessage chat identifier.
        contact_id: Optional contact identifier.
        extract_facts: Whether to run fact extraction on segments.

    Returns:
        Stats dict: {"persisted": N, "indexed": N, "facts_extracted": N}
    """
    from jarvis.db import get_db
    from jarvis.search.vec_search import get_vec_searcher
    from jarvis.topics.segment_storage import (
        link_vec_chunk_rowids,
        mark_facts_extracted,
        persist_segments,
    )

    stats: dict[str, Any] = {"persisted": 0, "indexed": 0, "facts_extracted": 0}

    if not segments:
        return stats

    db = get_db()
    searcher = get_vec_searcher()

    # OPTIMIZED: Single transaction for all DB operations
    with db.connection() as conn:
        try:
            # 1. Persist to conversation_segments + segment_messages
            segment_db_ids = persist_segments(conn, segments, chat_id, contact_id)
            stats["persisted"] = len(segment_db_ids)

            if not segment_db_ids:
                conn.rollback()
                return stats

            # 2. Index into vec_chunks (returns list of rowids)
            # Note: vec_chunks is a virtual table - uses separate connection internally
            # But we keep the transaction for other operations
            chunk_rowids = searcher.index_segments(segments, chat_id=chat_id)
            stats["indexed"] = len(chunk_rowids)

            # 3. Link vec_chunk_rowids back to conversation_segments
            if chunk_rowids and len(chunk_rowids) == len(segment_db_ids):
                link_vec_chunk_rowids(conn, segment_db_ids, chunk_rowids)

            # Commit all DB operations together
            conn.commit()

        except Exception as e:
            conn.rollback()
            logger.error("Failed to process segments for %s: %s", chat_id, e)
            return stats

    # 4. Extract facts per segment (outside transaction - can be slow)
    if extract_facts:
        extract_segments: list[TopicSegment] = []
        extract_segment_db_ids: list[int] = []
        with db.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS segment_fact_fingerprints (
                    chat_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (chat_id, fingerprint)
                )
                """
            )
            # Batch: fetch existing fingerprints once, build insert batch, determine new segments
            # First occurrence of each new fingerprint triggers extraction (matches INSERT+changes).
            existing_rows = conn.execute(
                "SELECT fingerprint FROM segment_fact_fingerprints WHERE chat_id = ?",
                (chat_id,),
            ).fetchall()
            existing = {r[0] for r in existing_rows}
            batch: list[tuple[str, str]] = []
            seen_new: set[str] = set()
            for segment, segment_db_id in zip(segments, segment_db_ids):
                fingerprint = _segment_fingerprint(segment)
                batch.append((chat_id, fingerprint))
                if fingerprint not in existing and fingerprint not in seen_new:
                    seen_new.add(fingerprint)
                    extract_segments.append(segment)
                    extract_segment_db_ids.append(segment_db_id)
            if batch:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO segment_fact_fingerprints (chat_id, fingerprint)
                    VALUES (?, ?)
                    """,
                    batch,
                )
            conn.commit()

        facts_count = _extract_facts_from_segments(
            extract_segments,
            extract_segment_db_ids,
            contact_id or chat_id,
        )
        stats["facts_extracted"] = facts_count

        # Mark extraction attempt complete for these segments even when dedup inserts 0.
        with db.connection() as conn:
            mark_facts_extracted(conn, segment_db_ids)
            conn.commit()

    logger.info(
        "Segment pipeline for %s: persisted=%d, indexed=%d, facts=%d",
        chat_id[:20],
        stats["persisted"],
        stats["indexed"],
        stats["facts_extracted"],
    )
    return stats


def _extract_facts_from_segments(
    segments: list[TopicSegment],
    segment_db_ids: list[int],
    contact_id: str,
) -> int:
    """Run fact extraction on segments using the V4 InstructionFactExtractor.

    Args:
        segments: TopicSegment objects.
        segment_db_ids: Database IDs for linking facts to segments.
        contact_id: Contact to associate facts with.

    Returns:
        Total number of facts extracted.
    """
    try:
        from integrations.imessage import ChatDBReader
        from jarvis.contacts.fact_storage import log_pass1_claims, save_facts
        from jarvis.contacts.fact_verifier import FactVerifier
        from jarvis.contacts.instruction_extractor import get_instruction_extractor
        from jarvis.db import get_db

        db = get_db()
        reader = ChatDBReader()
        user_name = reader.get_user_name()

        # Resolve contact name
        contact_name = "Contact"
        with db.connection() as conn:
            row = conn.execute(
                "SELECT display_name FROM contacts WHERE chat_id = ?", (contact_id,)
            ).fetchone()
            if row and row[0]:
                contact_name = row[0]

        tier = os.getenv("FACT_EXTRACT_TIER", "0.7b")
        extractor = get_instruction_extractor(tier=tier)
        if not extractor.is_loaded():
            extractor.load()

        total_rejected = 0
        total_raw = 0
        total_prefilter_rejected = 0
        batch_size = max(1, int(os.getenv("FACT_EXTRACT_BATCH_SIZE", "2")))
        
        # Deferred NLI collection
        candidates_to_verify: list[tuple[Any, str, int]] = []

        for i in range(0, len(segments), batch_size):
            batch_segments = segments[i : i + batch_size]
            batch_db_ids = segment_db_ids[i : i + batch_size]

            end = min(i + batch_size, len(segments))
            print(f"    - Extracting facts segments {i + 1}-{end}/{len(segments)}...", flush=True)

            # Extract facts for the whole batch in one LLM call
            batch_results = extractor.extract_facts_from_batch(
                batch_segments,
                contact_id=contact_id,
                contact_name=contact_name,
                user_name=user_name,
            )
            batch_extract_stats = extractor.get_last_batch_stats()
            total_raw += batch_extract_stats.get("raw_triples", 0)
            total_prefilter_rejected += batch_extract_stats.get("prefilter_rejected", 0)
            claims_by_segment = extractor.get_last_batch_pass1_claims()
            log_pass1_claims(
                contact_id=contact_id,
                chat_id=contact_id,
                segment_db_ids=batch_db_ids,
                claims_by_segment=claims_by_segment,
                stage="segment_pipeline",
            )

            # Collect candidates for batch verification later
            for segment_obj, segment_facts, db_id in zip(
                batch_segments, batch_results, batch_db_ids
            ):
                if segment_facts:
                    fast_gated_facts = [
                        f
                        for f in segment_facts
                        if _should_verify_fact_value(getattr(f, "value", ""))
                    ]
                    if not fast_gated_facts:
                        continue
                    segment_text = getattr(segment_obj, "text", "") or ""
                    if not segment_text:
                        segment_text = "\n".join(
                            (m.text or "") for m in getattr(segment_obj, "messages", [])
                        )
                    
                    # Store for batch NLI
                    for f in fast_gated_facts:
                        candidates_to_verify.append((f, segment_text, db_id))

        # --- BATCH VERIFICATION (Deferred Stage) ---
        all_verified_facts = []
        if candidates_to_verify:
            print(f"    - Verifying {len(candidates_to_verify)} candidate facts via NLI...", flush=True)
            verifier = FactVerifier(threshold=0.05)
            
            # Prepare data for verifier
            verification_input = [(f, text) for f, text, _ in candidates_to_verify]
            verified_facts, rejected_count = verifier.verify_facts_batch(verification_input)
            
            total_rejected = rejected_count
            
            # Re-link segment IDs to verified facts
            # Note: verified_facts is a subset of candidates_to_verify (ordered)
            # We match them back by Fact object identity or index
            # Fact objects in verified_facts are the SAME instances from candidates_to_verify
            # But since verify_facts_batch returns new list, we need to ensure segment_id is set
            
            # Map fact instance to its db_id from our collection
            fact_to_db_id = {id(f): db_id for f, _, db_id in candidates_to_verify}
            
            for f in verified_facts:
                db_id = fact_to_db_id.get(id(f))
                if db_id is not None:
                    setattr(f, "_segment_db_id", db_id)
            
            all_verified_facts = verified_facts

        total_inserted = save_facts(
            all_verified_facts,
            contact_id,
            segment_id=None,
            log_raw_facts=True,
            log_chat_id=contact_id,
            log_stage="segment_pipeline",
            raw_count=total_raw,
            prefilter_rejected=total_prefilter_rejected,
            verifier_rejected=total_rejected,
        )

        if total_rejected:
            logger.info(
                "NLI verification rejected %d facts in segment pipeline for %s",
                total_rejected,
                contact_id[:20],
            )

        return total_inserted

    except Exception as e:
        logger.warning("Fact extraction in segment pipeline failed: %s", e)
        return 0


def _link_candidates_to_segments(
    candidates: list[Any],
    segments: list[TopicSegment],
    segment_db_ids: list[int],
) -> None:
    """Set segment_db_id on candidates by matching message_id to segment membership.

    This allows fact traceability back to the segment that produced it.
    Modifies candidates in-place (adds _segment_db_id attribute if possible).
    """
    # Build message_id → segment_db_id lookup
    msg_to_seg: dict[int, int] = {}
    for segment, db_id in zip(segments, segment_db_ids):
        for msg in segment.messages:
            if msg.message_id is not None:
                msg_to_seg[msg.message_id] = db_id

    for candidate in candidates:
        msg_id = getattr(candidate, "message_id", None)
        if msg_id is not None and msg_id in msg_to_seg:
            candidate._segment_db_id = msg_to_seg[msg_id]
