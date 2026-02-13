"""Unified segment pipeline: persist → index → extract facts.

Orchestrates the full lifecycle of topic segments, replacing the
previously-separate segmentation and extraction flows in the watcher.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.topics.topic_segmenter import TopicSegment

logger = logging.getLogger(__name__)


def process_segments(
    segments: list[TopicSegment],
    chat_id: str,
    contact_id: str | None = None,
    extract_facts: bool = True,
) -> dict[str, Any]:
    """Persist segments, index into vec_chunks, and optionally extract facts.

    This is the single entry point that replaces calling segmentation,
    indexing, and extraction separately.

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

    # 1. Persist to conversation_segments + segment_messages
    with db.connection() as conn:
        segment_db_ids = persist_segments(conn, segments, chat_id, contact_id)
    stats["persisted"] = len(segment_db_ids)

    if not segment_db_ids:
        return stats

    # 2. Index into vec_chunks (returns list of rowids)
    chunk_rowids = searcher.index_segments(segments, chat_id=chat_id)
    stats["indexed"] = len(chunk_rowids)

    # 3. Link vec_chunk_rowids back to conversation_segments
    if chunk_rowids and len(chunk_rowids) == len(segment_db_ids):
        with db.connection() as conn:
            link_vec_chunk_rowids(conn, segment_db_ids, chunk_rowids)

    # 4. Extract facts per segment
    if extract_facts:
        facts_count = _extract_facts_from_segments(
            segments, segment_db_ids, contact_id or chat_id
        )
        stats["facts_extracted"] = facts_count

        if facts_count > 0:
            with db.connection() as conn:
                mark_facts_extracted(conn, segment_db_ids)

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
    """Run fact extraction on segments using the existing SegmentExtractor.

    Args:
        segments: TopicSegment objects.
        segment_db_ids: Database IDs for linking facts to segments.
        contact_id: Contact to associate facts with.

    Returns:
        Total number of facts extracted.
    """
    try:
        from jarvis.contacts.candidate_extractor import CandidateExtractor
        from jarvis.contacts.fact_storage import save_candidate_facts
        from jarvis.contacts.segment_extractor import extract_facts_from_segments

        extractor = CandidateExtractor(label_profile="balanced", use_entailment=True)
        candidates = extract_facts_from_segments(segments, extractor)

        if not candidates:
            return 0

        # Associate segment_db_ids with candidates based on message_id membership
        _link_candidates_to_segments(candidates, segments, segment_db_ids)

        inserted = save_candidate_facts(candidates, contact_id)
        return inserted

    except Exception as e:
        logger.warning("Fact extraction in segment pipeline failed: %s", e)
        return 0


def _link_candidates_to_segments(
    candidates: list,
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
