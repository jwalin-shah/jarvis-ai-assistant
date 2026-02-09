"""Persist extracted facts to contact_facts table.

Provides CRUD operations for the contact_facts table in jarvis.db.
Facts are deduplicated by (contact_id, category, subject, predicate) UNIQUE constraint.
"""

from __future__ import annotations

import logging
from datetime import datetime

from jarvis.contacts.contact_profile import Fact

logger = logging.getLogger(__name__)


def save_facts(facts: list[Fact], contact_id: str) -> int:
    """Save facts to contact_facts table, skip duplicates.

    Args:
        facts: Extracted facts to persist.
        contact_id: Contact these facts belong to.

    Returns:
        Number of new facts inserted.
    """
    from jarvis.db import get_db

    db = get_db()
    inserted = 0

    with db.connection() as conn:
        for fact in facts:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO contact_facts
                (contact_id, category, subject, predicate, value, confidence,
                 source_message_id, source_text, extracted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    contact_id,
                    fact.category,
                    fact.subject,
                    fact.predicate,
                    fact.value or "",
                    fact.confidence,
                    fact.source_message_id,
                    fact.source_text[:500] if fact.source_text else "",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            inserted += cursor.rowcount

    if inserted:
        logger.info("Saved %d new facts for %s", inserted, contact_id[:16])
    return inserted


def get_facts_for_contact(contact_id: str) -> list[Fact]:
    """Load all facts for a contact from DB."""
    from jarvis.db import get_db

    db = get_db()

    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT category, subject, predicate, value, confidence,
                   source_text, source_message_id, extracted_at
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
        )
        for row in rows
    ]


def get_all_facts() -> list[Fact]:
    """Load all facts across all contacts."""
    from jarvis.db import get_db

    db = get_db()

    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT contact_id, category, subject, predicate, value,
                   confidence, source_text, source_message_id, extracted_at
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


def get_fact_count() -> int:
    """Get total number of facts in the database."""
    from jarvis.db import get_db

    db = get_db()

    with db.connection() as conn:
        row = conn.execute("SELECT COUNT(*) FROM contact_facts").fetchone()
        return row[0] if row else 0
