"""Persistence stage helpers for the segment pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jarvis.db import get_db
from jarvis.search.vec_search import get_vec_searcher
from jarvis.topics.segment_pipeline_collect import segment_fingerprint
from jarvis.topics.segment_storage import (
    link_vec_chunk_rowids,
    mark_facts_extracted,
    persist_segments,
)

if TYPE_CHECKING:
    from jarvis.topics.topic_segmenter import TopicSegment

logger = logging.getLogger(__name__)


def persist_and_index_segments(
    segments: list[TopicSegment],
    chat_id: str,
    contact_id: str | None,
) -> tuple[list[int], int]:
    """Persist segments and index them in vec_chunks."""
    db = get_db()
    searcher = get_vec_searcher()

    with db.connection() as conn:
        try:
            segment_db_ids = persist_segments(conn, segments, chat_id, contact_id)
            if not segment_db_ids:
                conn.rollback()
                return [], 0

            chunk_rowids = searcher.index_segments(segments, chat_id=chat_id)
            indexed = len(chunk_rowids)

            if chunk_rowids and len(chunk_rowids) == len(segment_db_ids):
                link_vec_chunk_rowids(conn, segment_db_ids, chunk_rowids)

            conn.commit()
            return segment_db_ids, indexed

        except Exception as exc:
            conn.rollback()
            logger.error("Failed to process segments for %s: %s", chat_id, exc)
            return [], 0


def choose_segments_for_fact_extraction(
    segments: list[TopicSegment],
    segment_db_ids: list[int],
    chat_id: str,
) -> tuple[list[TopicSegment], list[int]]:
    """Select first-seen segment fingerprints that should run extraction."""
    db = get_db()
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

        existing_rows = conn.execute(
            "SELECT fingerprint FROM segment_fact_fingerprints WHERE chat_id = ?",
            (chat_id,),
        ).fetchall()
        existing = {row[0] for row in existing_rows}

        inserts: list[tuple[str, str]] = []
        seen_new: set[str] = set()
        for segment, segment_db_id in zip(segments, segment_db_ids):
            fingerprint = segment_fingerprint(segment)
            inserts.append((chat_id, fingerprint))
            if fingerprint not in existing and fingerprint not in seen_new:
                seen_new.add(fingerprint)
                extract_segments.append(segment)
                extract_segment_db_ids.append(segment_db_id)

        if inserts:
            conn.executemany(
                """
                INSERT OR IGNORE INTO segment_fact_fingerprints (chat_id, fingerprint)
                VALUES (?, ?)
                """,
                inserts,
            )
        conn.commit()

    return extract_segments, extract_segment_db_ids


def mark_fact_extraction_complete(segment_db_ids: list[int]) -> None:
    """Mark segments as having completed an extraction attempt."""
    if not segment_db_ids:
        return

    db = get_db()
    with db.connection() as conn:
        mark_facts_extracted(conn, segment_db_ids)
        conn.commit()
