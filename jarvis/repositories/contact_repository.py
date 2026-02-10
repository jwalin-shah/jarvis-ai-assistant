"""Repository for contact-related data access.

Wraps ``jarvis.contacts.fact_storage`` (DB-backed facts) and
``jarvis.contacts.contact_profile`` (file-backed profiles) behind a
single interface.  Callers are NOT migrated yet -- this is the extraction
step only.
"""

from __future__ import annotations

import logging
from pathlib import Path

from jarvis.contacts.contact_profile import (
    ContactProfile,
    Fact,
    load_profile,
    save_profile,
)
from jarvis.repositories.base import BaseRepository
from jarvis.utils.latency_tracker import track_latency

logger = logging.getLogger(__name__)


class ContactRepository(BaseRepository):
    """Data access for contacts, facts, and profiles.

    Facts are stored in ``contact_facts`` table via JarvisDB.
    Profiles are stored as JSON files under ``~/.jarvis/profiles/``.
    """

    # ------------------------------------------------------------------
    # Fact CRUD (wraps fact_storage.py queries)
    # ------------------------------------------------------------------

    def save_facts(self, facts: list[Fact], contact_id: str) -> int:
        """Batch-insert facts, skipping duplicates.

        Returns:
            Number of newly inserted facts.
        """
        import time
        from datetime import datetime

        if not facts:
            return 0

        with track_latency("fact_save", contact_id=contact_id[:16], count=len(facts)):
            start_time = time.perf_counter()
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

            with self.db.connection() as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM contact_facts WHERE contact_id = ?",
                    (contact_id,),
                )
                count_before = cursor.fetchone()[0]

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
            return inserted

    def get_facts_for_contact(self, contact_id: str) -> list[Fact]:
        """Load all facts for a contact, ordered by confidence desc."""
        with self.db.connection() as conn:
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

    def get_all_facts(self) -> list[Fact]:
        """Load all facts across all contacts."""
        with self.db.connection() as conn:
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

    def delete_facts_for_contact(self, contact_id: str) -> int:
        """Delete all facts for a contact. Returns count deleted."""
        with self.db.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM contact_facts WHERE contact_id = ?",
                (contact_id,),
            )
            deleted = cursor.rowcount

        if deleted:
            logger.info("Deleted %d facts for %s", deleted, contact_id[:16])
        return deleted

    def get_fact_count(self) -> int:
        """Get total number of facts in the database."""
        with self.db.connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM contact_facts").fetchone()
            return row[0] if row else 0

    # ------------------------------------------------------------------
    # Profile persistence (wraps contact_profile.py file I/O)
    # ------------------------------------------------------------------

    def save_profile(self, profile: ContactProfile) -> bool:
        """Save a contact profile to disk."""
        return save_profile(profile)

    def load_profile(self, contact_id: str) -> ContactProfile | None:
        """Load a contact profile from disk."""
        return load_profile(contact_id)
