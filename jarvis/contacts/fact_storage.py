"""Persist extracted facts to contact_facts table.

Provides CRUD operations for the contact_facts table in jarvis.db.
Facts are deduplicated by (contact_id, category, subject, predicate) UNIQUE constraint.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from jarvis.contacts.contact_profile import Fact
from jarvis.utils.latency_tracker import track_latency

logger = logging.getLogger(__name__)
_SEMANTIC_DEDUPER: Any | None = None


def _get_semantic_deduper() -> Any:
    """Reuse one deduper instance to avoid repeated model/bootstrap overhead."""
    global _SEMANTIC_DEDUPER
    if _SEMANTIC_DEDUPER is None:
        from jarvis.contacts.fact_deduplicator import FactDeduplicator

        _SEMANTIC_DEDUPER = FactDeduplicator()
    return _SEMANTIC_DEDUPER


@dataclass
class FactCandidate:
    """A candidate fact extracted by NER/extraction models, pending downstream filtering."""

    message_id: int  # iMessage ROWID
    span_text: str  # extracted entity text ("Austin", "Google", "Sarah")
    span_label: str  # entity label (place, org, person_name, allergy, etc.)
    gliner_score: float  # extraction confidence (legacy name for backward compatibility)
    fact_type: str  # mapped type (location.future, work.employer, etc.)
    start_char: int  # character offset start in source_text
    end_char: int  # character offset end in source_text
    source_text: str = ""  # full message text for context
    chat_id: int | None = None
    is_from_me: bool | None = None  # True if sent by user
    sender_handle_id: int | None = None  # handle ROWID of sender
    message_date: int | None = None  # iMessage date (Core Data timestamp)
    status: str = "pending"  # pending | accepted | rejected

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSONL output."""
        return {
            "message_id": self.message_id,
            "span_text": self.span_text,
            "span_label": self.span_label,
            "gliner_score": self.gliner_score,
            "fact_type": self.fact_type,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "source_text": self.source_text,
            "chat_id": self.chat_id,
            "is_from_me": self.is_from_me,
            "sender_handle_id": self.sender_handle_id,
            "message_date": self.message_date,
            "status": self.status,
        }


def _log_raw_fact_candidates(
    conn: sqlite3.Connection,
    facts: list[Fact],
    contact_id: str,
    log_chat_id: str | None,
    segment_id: int | None,
    stage: str,
) -> None:
    """Insert every incoming fact into fact_candidates_log for auditing."""
    if not facts:
        return

    entries: list[
        tuple[str, str, int | None, str, str, str, str, float, str, str, int | None, str]
    ] = []
    chat_hint = log_chat_id or contact_id
    for fact in facts:
        resolved_segment_id = getattr(fact, "_segment_db_id", segment_id)
        entries.append(
            (
                contact_id,
                chat_hint,
                fact.source_message_id,
                fact.subject,
                fact.predicate,
                fact.value,
                fact.category,
                fact.confidence,
                (fact.source_text or "")[:500],
                fact.attribution or "contact",
                resolved_segment_id,
                stage,
            )
        )

    conn.executemany(
        """
        INSERT INTO fact_candidates_log
        (contact_id, chat_id, message_id, subject, predicate, value,
         category, confidence, source_text, attribution, segment_id, log_stage)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        entries,
    )


def _record_fact_pipeline_metrics(
    conn: sqlite3.Connection,
    *,
    contact_id: str,
    chat_id: str,
    stage: str,
    raw_count: int,
    prefilter_rejected: int,
    verifier_rejected: int,
    semantic_dedup_rejected: int,
    unique_conflict_rejected: int,
    saved_count: int,
) -> None:
    """Persist aggregate pipeline counters for dashboard/query reporting."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_pipeline_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contact_id TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            raw_count INTEGER DEFAULT 0,
            prefilter_rejected INTEGER DEFAULT 0,
            verifier_rejected INTEGER DEFAULT 0,
            semantic_dedup_rejected INTEGER DEFAULT 0,
            unique_conflict_rejected INTEGER DEFAULT 0,
            saved_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO fact_pipeline_metrics
        (contact_id, chat_id, stage, raw_count, prefilter_rejected, verifier_rejected,
         semantic_dedup_rejected, unique_conflict_rejected, saved_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            contact_id,
            chat_id,
            stage,
            raw_count,
            prefilter_rejected,
            verifier_rejected,
            semantic_dedup_rejected,
            unique_conflict_rejected,
            saved_count,
        ),
    )


def log_pass1_claims(
    contact_id: str,
    chat_id: str,
    segment_db_ids: list[int],
    claims_by_segment: list[list[str]],
    *,
    stage: str = "segment_pipeline",
) -> int:
    """Persist pass-1 natural-language claims for audit/debug."""
    from jarvis.db import get_db

    if not segment_db_ids:
        return 0

    normalized_claims = claims_by_segment or []
    if len(normalized_claims) < len(segment_db_ids):
        normalized_claims = normalized_claims + [
            [] for _ in range(len(segment_db_ids) - len(normalized_claims))
        ]

    rows: list[tuple[str, str, int, str, str]] = []
    for seg_id, seg_claims in zip(segment_db_ids, normalized_claims):
        for claim in seg_claims:
            clean = " ".join((claim or "").split()).strip()
            if clean:
                rows.append((contact_id, chat_id, seg_id, clean[:500], stage))

    db = get_db()
    with db.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fact_pass1_claims_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contact_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                segment_id INTEGER,
                claim_text TEXT NOT NULL,
                stage TEXT DEFAULT 'segment_pipeline',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        if rows:
            conn.executemany(
                """
                INSERT INTO fact_pass1_claims_log
                (contact_id, chat_id, segment_id, claim_text, stage)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
    return len(rows)


def save_facts(
    facts: list[Fact],
    contact_id: str,
    segment_id: int | None = None,
    *,
    log_raw_facts: bool = False,
    log_chat_id: str | None = None,
    log_stage: str = "extraction",
    raw_count: int | None = None,
    prefilter_rejected: int = 0,
    verifier_rejected: int = 0,
    return_embeddings: bool = False,
) -> int | tuple[int, np.ndarray]:
    """Save facts to contact_facts table, skip duplicates.

    This is a pure DB operation. For saving + semantic indexing,
    use ``save_and_index_facts()`` instead.

    Args:
        facts: Extracted facts to persist.
        contact_id: Contact these facts belong to.
        segment_id: Optional segment DB ID for traceability.
        log_raw_facts: Whether to persist every incoming fact to the audit log.
        log_chat_id: Optional chat_id to record in the audit log (defaults to contact_id).
        log_stage: Identifier describing where the facts came from (defaults to "extraction").
        return_embeddings: If True, return embeddings alongside count (for indexing).

    Returns:
        Number of new facts inserted.
        If return_embeddings=True, returns (count, embeddings) tuple.
    """
    import time

    from jarvis.db import get_db

    if not facts:
        if raw_count is not None:
            from jarvis.db import get_db

            db = get_db()
            with db.connection() as conn:
                _record_fact_pipeline_metrics(
                    conn,
                    contact_id=contact_id,
                    chat_id=log_chat_id or contact_id,
                    stage=log_stage,
                    raw_count=raw_count,
                    prefilter_rejected=prefilter_rejected,
                    verifier_rejected=verifier_rejected,
                    semantic_dedup_rejected=0,
                    unique_conflict_rejected=0,
                    saved_count=0,
                )
        return 0

    with track_latency("fact_save", contact_id=contact_id[:16], count=len(facts)):
        db = get_db()
        start_time = time.perf_counter()

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build lookup of facts to insert, keyed by unique constraint
        # (contact_id, category, subject, predicate, attribution)
        fact_keys: dict[tuple[str, str, str, str, str], Fact] = {}
        for fact in facts:
            key = (
                contact_id,
                fact.category,
                fact.subject,
                fact.predicate,
                fact.attribution or "contact",
            )
            # Keep the last occurrence if duplicates in input list
            fact_keys[key] = fact

        incoming_list = list(fact_keys.values())
        semantic_min_batch = max(1, int(os.getenv("FACT_SEMANTIC_DEDUP_MIN_BATCH", "6")))

        # Skip expensive semantic search for very small batches; SQL unique keys still apply.
        fact_embeddings = np.array([]) if return_embeddings else None
        if len(incoming_list) < semantic_min_batch:
            final_new_facts = incoming_list
            semantic_dedup_rejected = 0
            # For small batches where we need embeddings, compute them directly
            if return_embeddings and final_new_facts:
                from jarvis.embedding_adapter import get_embedder

                texts = [f.value or "" for f in final_new_facts]
                fact_embeddings = get_embedder().encode(texts, normalize=True)
        else:
            deduper = _get_semantic_deduper()
            # Pass empty list for existing_facts - deduper uses vec_facts index.
            # Request embeddings back for indexing to avoid re-computation.
            dedup_result = deduper.deduplicate(
                incoming_list, [], return_embeddings=return_embeddings
            )
            if return_embeddings:
                final_new_facts, fact_embeddings = dedup_result
            else:
                final_new_facts = dedup_result
                fact_embeddings = np.array([])
            semantic_dedup_rejected = max(0, len(incoming_list) - len(final_new_facts))
        inserted_count = 0
        unique_conflict_rejected = 0

        with db.connection() as conn:
            if log_raw_facts:
                _log_raw_fact_candidates(
                    conn,
                    facts,
                    contact_id,
                    log_chat_id,
                    segment_id,
                    log_stage,
                )

            if final_new_facts:
                # Prepare data for only new facts
                fact_data = [
                    (
                        contact_id,
                        fact.category,
                        fact.subject,
                        fact.predicate,
                        fact.value or "",
                        fact.confidence,
                        fact.source_message_id,
                        fact.source_text[:500] if fact.source_text else "",
                        current_time,
                        fact.linked_contact_id,
                        fact.valid_from,
                        fact.valid_until,
                        fact.attribution,
                        getattr(fact, "_segment_db_id", segment_id),
                    )
                    for fact in final_new_facts
                ]

                # Batch insert only new facts
                total_changes_before = conn.execute("SELECT total_changes()").fetchone()[0]
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO contact_facts
                    (contact_id, category, subject, predicate, value, confidence,
                     source_message_id, source_text, extracted_at, linked_contact_id,
                     valid_from, valid_until, attribution, segment_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    fact_data,
                )
                total_changes_after = conn.execute("SELECT total_changes()").fetchone()[0]
                inserted_count = total_changes_after - total_changes_before
                unique_conflict_rejected = max(0, len(final_new_facts) - inserted_count)

            if raw_count is not None:
                _record_fact_pipeline_metrics(
                    conn,
                    contact_id=contact_id,
                    chat_id=log_chat_id or contact_id,
                    stage=log_stage,
                    raw_count=raw_count,
                    prefilter_rejected=prefilter_rejected,
                    verifier_rejected=verifier_rejected,
                    semantic_dedup_rejected=semantic_dedup_rejected,
                    unique_conflict_rejected=unique_conflict_rejected,
                    saved_count=inserted_count,
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if inserted_count:
            logger.info(
                "Saved %d new facts for %s in %.1fms (batch insert)",
                inserted_count,
                contact_id[:16],
                elapsed_ms,
            )

        if return_embeddings:
            # fact_embeddings is always set when return_embeddings=True
            return inserted_count, fact_embeddings  # type: ignore[return-value]
        return inserted_count


def save_and_index_facts(
    facts: list[Fact],
    contact_id: str,
    segment_id: int | None = None,
    *,
    log_raw_facts: bool = False,
    log_chat_id: str | None = None,
    log_stage: str = "extraction",
) -> int:
    """Save facts to DB and index them for semantic search.

    Combines ``save_facts()`` (pure DB insert) with ``index_facts()``
    (embedding + vec_facts). Indexing failures are logged but don't
    affect the save.

    Uses embeddings from deduplication phase to avoid double computation.

    Args:
        facts: Extracted facts to persist and index.
        contact_id: Contact these facts belong to.
        segment_id: Optional segment DB ID for traceability.

    Returns:
        Number of new facts inserted.
    """
    # Request embeddings back from save_facts to avoid re-encoding in index_facts
    result = save_facts(
        facts,
        contact_id,
        segment_id=segment_id,
        log_raw_facts=log_raw_facts,
        log_chat_id=log_chat_id,
        log_stage=log_stage,
        return_embeddings=True,
    )

    if isinstance(result, tuple):
        inserted, embeddings = result
    else:
        inserted = result
        embeddings = None

    if inserted:
        try:
            from jarvis.contacts.fact_index import index_facts

            # Pass embeddings to avoid double computation
            index_facts(facts, contact_id, embeddings=embeddings)
        except Exception as e:
            logger.debug("Fact indexing skipped: %s", e)

    return inserted


def get_facts_for_contact(contact_id: str) -> list[Fact]:
    """Load all facts for a contact from DB."""
    from jarvis.db import get_db

    db = get_db()

    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT category, subject, predicate, value, confidence,
                   source_text, source_message_id, extracted_at,
                   valid_from, valid_until, attribution
            FROM contact_facts
            WHERE contact_id = ?
            ORDER BY confidence DESC
            """,
            (contact_id,),
        ).fetchall()

    return [
        Fact(
            category=row["category"],
            subject=row["subject"],
            predicate=row["predicate"],
            value=row["value"],
            confidence=row["confidence"],
            source_text=row["source_text"] or "",
            source_message_id=row["source_message_id"],
            contact_id=contact_id,
            extracted_at=row["extracted_at"] or "",
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            attribution=row["attribution"] or "contact",
        )
        for row in rows
    ]


def get_facts_for_contacts(contact_ids: list[str]) -> dict[str, list[Fact]]:
    """Batch load facts for multiple contacts.

    More efficient than calling get_facts_for_contact() N times.
    Uses a single query with IN clause.

    Args:
        contact_ids: List of contact IDs to fetch facts for.

    Returns:
        Dict mapping contact_id -> list of Facts.
    """
    from jarvis.db import get_db
    from jarvis.db.query_builder import QueryBuilder

    if not contact_ids:
        return {}

    db = get_db()

    # Build chunked IN clause for safety
    results: dict[str, list[Fact]] = {cid: [] for cid in contact_ids}

    with db.connection() as conn:
        # Process in chunks to stay within SQLite parameter limits
        for chunk_start in range(0, len(contact_ids), 900):
            chunk = contact_ids[chunk_start : chunk_start + 900]
            placeholders, params = QueryBuilder.in_clause(chunk)

            rows = conn.execute(
                f"""
                SELECT contact_id, category, subject, predicate, value, confidence,
                       source_text, source_message_id, extracted_at,
                       valid_from, valid_until, attribution
                FROM contact_facts
                WHERE contact_id IN ({placeholders})
                ORDER BY confidence DESC
                """,
                params,
            ).fetchall()

            for row in rows:
                cid = row["contact_id"]
                results[cid].append(
                    Fact(
                        category=row["category"],
                        subject=row["subject"],
                        predicate=row["predicate"],
                        value=row["value"],
                        confidence=row["confidence"],
                        source_text=row["source_text"] or "",
                        source_message_id=row["source_message_id"],
                        contact_id=cid,
                        extracted_at=row["extracted_at"] or "",
                        valid_from=row["valid_from"],
                        valid_until=row["valid_until"],
                        attribution=row["attribution"] or "contact",
                    )
                )

    return results


def count_facts_for_contact(contact_id: str) -> int:
    """Quick check for whether a contact has any facts. Returns count."""
    from jarvis.db import get_db

    db = get_db()

    with db.connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM contact_facts WHERE contact_id = ?",
            (contact_id,),
        ).fetchone()
        return row[0] if row else 0


def get_all_facts() -> list[Fact]:
    """Load all facts across all contacts."""
    from jarvis.db import get_db

    db = get_db()

    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT contact_id, category, subject, predicate, value,
                   confidence, source_text, source_message_id, extracted_at,
                   valid_from, valid_until, attribution
            FROM contact_facts
            ORDER BY confidence DESC
            """,
        ).fetchall()

    return [
        Fact(
            category=row["category"],
            subject=row["subject"],
            predicate=row["predicate"],
            value=row["value"],
            confidence=row["confidence"],
            source_text=row["source_text"] or "",
            source_message_id=row["source_message_id"],
            contact_id=row["contact_id"],
            extracted_at=row["extracted_at"] or "",
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            attribution=row["attribution"] or "contact",
        )
        for row in rows
    ]


def delete_facts_for_contact(contact_id: str) -> int:
    """Delete all facts for a contact. Returns count deleted."""
    from jarvis.db import get_db

    db = get_db()

    with db.connection() as conn:
        cursor = conn.execute(
            "DELETE FROM contact_facts WHERE contact_id = ?",
            (contact_id,),
        )
        deleted = cursor.rowcount

    if deleted:
        logger.info("Deleted %d facts for %s", deleted, contact_id[:16])
    return deleted


def save_candidate_facts(
    candidates: list[FactCandidate],
    contact_id: str,
    segment_id: int | None = None,
    *,
    log_raw_facts: bool = False,
    log_chat_id: str | None = None,
    log_stage: str = "extraction",
) -> int:
    """Convert FactCandidates to Facts and save them.

    Maps fact_type (e.g. 'location.current') to category+predicate used by
    the Fact storage schema.

    Args:
        candidates: List of FactCandidate objects from CandidateExtractor.
        contact_id: Contact ID to associate facts with.
        segment_id: Optional segment DB ID for traceability.
        log_raw_facts: Whether to persist every candidate to the audit log.
        log_chat_id: Optional chat_id to record (defaults to contact_id).
        log_stage: Identifier describing where the candidates came from.

    Returns:
        Number of new facts inserted.
    """
    # fact_type → (category, predicate)
    type_to_schema: dict[str, tuple[str, str]] = {
        "location.current": ("location", "lives_in"),
        "location.past": ("location", "lived_in"),
        "location.future": ("location", "moving_to"),
        "location.hometown": ("location", "from"),
        "work.employer": ("work", "works_at"),
        "work.former_employer": ("work", "worked_at"),
        "work.job_title": ("work", "job_title"),
        "relationship.family": ("relationship", "is_family_of"),
        "relationship.friend": ("relationship", "is_friend_of"),
        "relationship.partner": ("relationship", "is_partner_of"),
        "preference.food_like": ("preference", "likes_food"),
        "preference.food_dislike": ("preference", "dislikes_food"),
        "preference.activity": ("preference", "enjoys"),
        "health.allergy": ("health", "allergic_to"),
        "health.dietary": ("health", "dietary"),
        "health.condition": ("health", "has_condition"),
        "personal.birthday": ("personal", "birthday_is"),
        "personal.school": ("personal", "attends"),
        "personal.pet": ("personal", "has_pet"),
    }

    facts: list[Fact] = []
    for c in candidates:
        if not isinstance(c, FactCandidate):
            continue
        mapping = type_to_schema.get(c.fact_type)
        if mapping is None:
            continue
        category, predicate = mapping

        # Resolve attribution: if contact sent message, fact is about contact
        # if user sent message, fact is about user
        is_from_me = getattr(c, "is_from_me", None) or False
        attribution = "user" if is_from_me else "contact"

        facts.append(
            Fact(
                category=category,
                subject=c.span_text,
                predicate=predicate,
                value=c.span_label,
                source_text=c.source_text[:500] if c.source_text else "",
                confidence=c.gliner_score if c.gliner_score > 0 else 0.5,
                contact_id=contact_id,
                source_message_id=c.message_id,
                attribution=attribution,
            )
        )

    if not facts:
        return 0

    return save_and_index_facts(
        facts,
        contact_id,
        segment_id=segment_id,
        log_raw_facts=log_raw_facts,
        log_chat_id=log_chat_id,
        log_stage=log_stage,
    )


def delete_facts_by_predicate_prefix(prefix: str) -> int:
    """Delete all facts whose predicate starts with the given prefix.

    Args:
        prefix: Predicate prefix to match (e.g. 'legacy_').

    Returns:
        Number of facts deleted.
    """
    from jarvis.db import get_db

    db = get_db()

    with db.connection() as conn:
        cursor = conn.execute(
            "DELETE FROM contact_facts WHERE predicate LIKE ?",
            (f"{prefix}%",),
        )
        deleted = cursor.rowcount

    if deleted:
        logger.info("Deleted %d facts with predicate prefix '%s'", deleted, prefix)
    return deleted


def get_fact_count() -> int:
    """Get total number of facts in the database."""
    from jarvis.db import get_db

    db = get_db()

    with db.connection() as conn:
        row = conn.execute("SELECT COUNT(*) FROM contact_facts").fetchone()
        return row[0] if row else 0
