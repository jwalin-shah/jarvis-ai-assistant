"""Persist extracted facts to contact_facts table.

Provides CRUD operations for the contact_facts table in jarvis.db.
Facts are deduplicated by (contact_id, category, subject, predicate) UNIQUE constraint.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from jarvis.contacts.contact_profile import Fact

if TYPE_CHECKING:
    from jarvis.contacts.candidate_extractor import FactCandidate
from jarvis.utils.latency_tracker import track_latency

logger = logging.getLogger(__name__)


def save_facts(
    facts: list[Fact],
    contact_id: str,
    segment_id: int | None = None,
) -> int:
    """Save facts to contact_facts table, skip duplicates.

    This is a pure DB operation. For saving + semantic indexing,
    use ``save_and_index_facts()`` instead.

    Args:
        facts: Extracted facts to persist.
        contact_id: Contact these facts belong to.
        segment_id: Optional segment DB ID for traceability.

    Returns:
        Number of new facts inserted.
    """
    import time

    from jarvis.db import get_db

    if not facts:
        return 0

    with track_latency("fact_save", contact_id=contact_id[:16], count=len(facts)):
        db = get_db()
        start_time = time.perf_counter()

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build lookup of facts to insert, keyed by unique constraint
        fact_keys: dict[tuple[str, str, str, str], Fact] = {}
        for fact in facts:
            key = (contact_id, fact.category, fact.subject, fact.predicate)
            # Keep the last occurrence if duplicates in input list
            fact_keys[key] = fact

        with db.connection() as conn:
            # Check which facts already exist (single query for all)
            existing_keys: set[tuple[str, str, str, str]] = set()
            if fact_keys:
                # Build OR conditions for efficient lookup
                conditions = []
                params = []
                for key in fact_keys:
                    conditions.append(
                        "(contact_id = ? AND category = ? AND subject = ? AND predicate = ?)"
                    )
                    params.extend(key)

                # Query in chunks to avoid SQLite parameter limit
                chunk_size = 250  # SQLite has 999 param limit, 250*4=1000
                keys_list = list(fact_keys.keys())
                for i in range(0, len(keys_list), chunk_size):
                    chunk = keys_list[i : i + chunk_size]
                    chunk_conditions = []
                    chunk_params = []
                    for key in chunk:
                        chunk_conditions.append(
                            "(contact_id = ? AND category = ? AND subject = ? AND predicate = ?)"
                        )
                        chunk_params.extend(key)

                    query = f"SELECT contact_id, category, subject, predicate FROM contact_facts WHERE {' OR '.join(chunk_conditions)}"
                    cursor = conn.execute(query, chunk_params)
                    for row in cursor:
                        existing_keys.add((row[0], row[1], row[2], row[3]))

            # Filter to only new facts
            new_facts = [fact for key, fact in fact_keys.items() if key not in existing_keys]
            inserted_count = len(new_facts)

            if new_facts:
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
                    for fact in new_facts
                ]

                # Batch insert only new facts
                conn.executemany(
                    """
                    INSERT INTO contact_facts
                    (contact_id, category, subject, predicate, value, confidence,
                     source_message_id, source_text, extracted_at, linked_contact_id,
                     valid_from, valid_until, attribution, segment_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    fact_data,
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if inserted_count:
            logger.info(
                "Saved %d new facts for %s in %.1fms (batch insert)",
                inserted_count,
                contact_id[:16],
                elapsed_ms,
            )

        return inserted_count


def save_and_index_facts(
    facts: list[Fact],
    contact_id: str,
    segment_id: int | None = None,
) -> int:
    """Save facts to DB and index them for semantic search.

    Combines ``save_facts()`` (pure DB insert) with ``index_facts()``
    (embedding + vec_facts). Indexing failures are logged but don't
    affect the save.

    Args:
        facts: Extracted facts to persist and index.
        contact_id: Contact these facts belong to.
        segment_id: Optional segment DB ID for traceability.

    Returns:
        Number of new facts inserted.
    """
    inserted = save_facts(facts, contact_id, segment_id=segment_id)

    if inserted:
        try:
            from jarvis.contacts.fact_index import index_facts

            index_facts(facts, contact_id)
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
) -> int:
    """Convert FactCandidates to Facts and save them.

    Maps fact_type (e.g. 'location.current') to category+predicate used by
    the Fact storage schema.

    Args:
        candidates: List of FactCandidate objects from CandidateExtractor.
        contact_id: Contact ID to associate facts with.
        segment_id: Optional segment DB ID for traceability.

    Returns:
        Number of new facts inserted.
    """
    from jarvis.contacts.candidate_extractor import FactCandidate

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

    return save_and_index_facts(facts, contact_id, segment_id=segment_id)


def delete_facts_by_predicate_prefix(prefix: str) -> int:
    """Delete all facts whose predicate starts with the given prefix.

    Args:
        prefix: Predicate prefix to match (e.g. 'gliner_').

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
