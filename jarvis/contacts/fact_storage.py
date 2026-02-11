"""Persist extracted facts to contact_facts table.

Provides CRUD operations for the contact_facts table in jarvis.db.
Facts are deduplicated by (contact_id, category, subject, predicate) UNIQUE constraint.
"""

from __future__ import annotations

import logging
from datetime import datetime

from jarvis.contacts.contact_profile import Fact
from jarvis.utils.latency_tracker import track_latency

logger = logging.getLogger(__name__)


def save_facts(facts: list[Fact], contact_id: str) -> int:
    """Save facts to contact_facts table, skip duplicates.

    Args:
        facts: Extracted facts to persist.
        contact_id: Contact these facts belong to.

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

        # PERF FIX: Use batch INSERT with executemany() instead of loop
        # Before: 50 individual INSERT statements = ~150ms
        # After: 1 batch INSERT = ~3ms
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare all fact data as tuples for batch insert
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
            )
            for fact in facts
        ]

        with db.connection() as conn:
            # Get initial count to calculate how many were actually inserted
            cursor = conn.execute(
                "SELECT COUNT(*) FROM contact_facts WHERE contact_id = ?",
                (contact_id,),
            )
            count_before = cursor.fetchone()[0]

            # Batch insert all facts at once
            conn.executemany(
                """
                INSERT OR IGNORE INTO contact_facts
                (contact_id, category, subject, predicate, value, confidence,
                 source_message_id, source_text, extracted_at, linked_contact_id,
                 valid_from, valid_until)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                fact_data,
            )

            # Get final count to see how many were inserted (executemany rowcount is unreliable)
            cursor = conn.execute(
                "SELECT COUNT(*) FROM contact_facts WHERE contact_id = ?",
                (contact_id,),
            )
            count_after = cursor.fetchone()[0]
            inserted = count_after - count_before

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if inserted:
            logger.info(
                "Saved %d new facts for %s in %.1fms (batch insert)",
                inserted,
                contact_id[:16],
                elapsed_ms,
            )

            # Index newly inserted facts into vec_facts for semantic retrieval
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
                   valid_from, valid_until
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
                   valid_from, valid_until
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
) -> int:
    """Convert FactCandidates to Facts and save them.

    Maps fact_type (e.g. 'location.current') to category+predicate used by
    the Fact storage schema.

    Args:
        candidates: List of FactCandidate objects from CandidateExtractor.
        contact_id: Contact ID to associate facts with.

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
            )
        )

    if not facts:
        return 0

    return save_facts(facts, contact_id)


def get_fact_count() -> int:
    """Get total number of facts in the database."""
    from jarvis.db import get_db

    db = get_db()

    with db.connection() as conn:
        row = conn.execute("SELECT COUNT(*) FROM contact_facts").fetchone()
        return row[0] if row else 0
