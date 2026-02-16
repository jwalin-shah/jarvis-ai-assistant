"""Unified segment pipeline: persist -> index -> optional fact extraction."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from jarvis.topics.segment_pipeline_collect import FactCandidate, collect_fact_candidates
from jarvis.topics.segment_pipeline_persist import (
    choose_segments_for_fact_extraction,
    mark_fact_extraction_complete,
    persist_and_index_segments,
)
from jarvis.topics.segment_pipeline_verify import verify_fact_candidates

if TYPE_CHECKING:
    from jarvis.topics.topic_segmenter import TopicSegment

logger = logging.getLogger(__name__)


def process_segments(
    segments: list[TopicSegment],
    chat_id: str,
    contact_id: str | None = None,
    extract_facts: bool = True,
) -> dict[str, Any]:
    """Persist segments, index vec chunks, and optionally extract facts."""
    stats: dict[str, Any] = {"persisted": 0, "indexed": 0, "facts_extracted": 0}
    if not segments:
        return stats

    segment_db_ids, indexed_count = persist_and_index_segments(segments, chat_id, contact_id)
    stats["persisted"] = len(segment_db_ids)
    stats["indexed"] = indexed_count
    if not segment_db_ids:
        return stats

    if extract_facts:
        extract_segments, extract_segment_db_ids = choose_segments_for_fact_extraction(
            segments,
            segment_db_ids,
            chat_id,
        )
        stats["facts_extracted"] = _extract_facts_from_segments(
            extract_segments,
            extract_segment_db_ids,
            contact_id or chat_id,
        )
        mark_fact_extraction_complete(segment_db_ids)

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
    """Extract facts with batch LLM extraction and deferred NLI verification."""
    if not segments:
        return 0

    try:
        from integrations.imessage import ChatDBReader
        from jarvis.contacts.fact_storage import log_pass1_claims, save_facts
        from jarvis.contacts.instruction_extractor import get_instruction_extractor
        from jarvis.db import get_db

        db = get_db()
        reader = ChatDBReader()
        user_name = reader.get_user_name()

        contact_name = "Contact"
        with db.connection() as conn:
            row = conn.execute(
                "SELECT display_name FROM contacts WHERE chat_id = ?",
                (contact_id,),
            ).fetchone()
            if row and row[0]:
                contact_name = row[0]

        tier = os.getenv("FACT_EXTRACT_TIER", "0.7b")
        extractor = get_instruction_extractor(tier=tier)
        if not extractor.is_loaded():
            extractor.load()

        total_raw = 0
        total_prefilter_rejected = 0
        total_rejected = 0
        batch_size = max(1, int(os.getenv("FACT_EXTRACT_BATCH_SIZE", "2")))
        candidates_to_verify: list[FactCandidate] = []

        for start in range(0, len(segments), batch_size):
            end = min(start + batch_size, len(segments))
            batch_segments = segments[start:end]
            batch_db_ids = segment_db_ids[start:end]

            print(
                f"    - Extracting facts segments {start + 1}-{end}/{len(segments)}...",
                flush=True,
            )

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
            candidates_to_verify.extend(
                collect_fact_candidates(batch_segments, batch_results, batch_db_ids)
            )

        verified_facts: list[Any] = []
        if candidates_to_verify:
            print(
                f"    - Verifying {len(candidates_to_verify)} candidate facts via NLI...",
                flush=True,
            )
            verified_facts, total_rejected = verify_fact_candidates(candidates_to_verify)

        save_result = save_facts(
            verified_facts,
            contact_id,
            segment_id=None,
            log_raw_facts=True,
            log_chat_id=contact_id,
            log_stage="segment_pipeline",
            raw_count=total_raw,
            prefilter_rejected=total_prefilter_rejected,
            verifier_rejected=total_rejected,
        )

        total_inserted = save_result[0] if isinstance(save_result, tuple) else save_result

        if total_rejected:
            logger.info(
                "NLI verification rejected %d facts in segment pipeline for %s",
                total_rejected,
                contact_id[:20],
            )

        return int(total_inserted)

    except Exception as exc:
        logger.warning("Fact extraction in segment pipeline failed: %s", exc)
        return 0
