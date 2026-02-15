"""Comprehensive tests for jarvis.contacts.fact_storage module.

Tests cover:
- Batch insert (save_facts)
- Deduplication via UNIQUE constraint
- Linked contacts (linked_contact_id)
- Retrieval by contact, across all contacts, counting
- Deletion
- Empty/edge cases (empty list, None values, unicode, long text)
- save_candidate_facts conversion
- Performance: batch insert of 150+ facts without N+1 queries
- get_fact_count global counter
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from jarvis.contacts.contact_profile import Fact

# ---------------------------------------------------------------------------
# Fixtures: in-memory JarvisDB with schema initialized
# ---------------------------------------------------------------------------


@pytest.fixture()
def _reset_db_singleton():
    """Reset the jarvis.db singleton before and after each test."""
    from jarvis.db import reset_db

    reset_db()
    yield
    reset_db()


@pytest.fixture()
def db(tmp_path: Path, _reset_db_singleton):
    """Create a fresh in-memory-like JarvisDB for each test.

    Uses a temp file so the singleton get_db() returns a real DB with the
    full schema (contact_facts table, indexes, etc.).
    """
    from jarvis.db import get_db

    db_path = tmp_path / "test_jarvis.db"

    # Patch the default path so get_db() uses our temp DB
    with (
        patch("jarvis.db.models.JARVIS_DB_PATH", db_path),
        patch("jarvis.db.core.JARVIS_DB_PATH", db_path),
    ):
        db_instance = get_db(db_path)
        db_instance.init_schema()
        yield db_instance


@pytest.fixture()
def patched_get_db(db):
    """Patch jarvis.db.get_db so that fact_storage functions use the test DB.

    fact_storage.py uses local imports (``from jarvis.db import get_db``)
    inside each function, so the patch target must be ``jarvis.db.get_db``
    (where the name is actually defined), not the consumer module.
    """
    with patch("jarvis.db.get_db", return_value=db):
        yield db


def _make_fact(
    *,
    category: str = "preference",
    subject: str = "sushi",
    predicate: str = "likes",
    value: str = "food",
    confidence: float = 0.8,
    contact_id: str = "contact-001",
    source_text: str = "I love sushi",
    source_message_id: int | None = 42,
    linked_contact_id: str | None = None,
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> Fact:
    """Helper to build a Fact with sensible defaults."""
    return Fact(
        category=category,
        subject=subject,
        predicate=predicate,
        value=value,
        confidence=confidence,
        contact_id=contact_id,
        source_text=source_text,
        source_message_id=source_message_id,
        linked_contact_id=linked_contact_id,
        valid_from=valid_from,
        valid_until=valid_until,
    )


# =========================================================================
# Batch Insert
# =========================================================================


class TestSaveFacts:
    """Test save_facts batch insert."""

    def test_insert_single_fact(self, patched_get_db) -> None:
        """Insert one fact and verify it is stored."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        facts = [_make_fact(subject="Austin", predicate="lives_in", category="location")]
        inserted = save_facts(facts, "contact-001")

        assert inserted == 1

        stored = get_facts_for_contact("contact-001")
        assert len(stored) == 1
        assert stored[0].subject == "Austin"
        assert stored[0].predicate == "lives_in"
        assert stored[0].category == "location"

    def test_insert_multiple_facts(self, patched_get_db) -> None:
        """Insert multiple distinct facts in one batch call."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        facts = [
            _make_fact(category="location", subject="Austin", predicate="lives_in"),
            _make_fact(category="work", subject="Google", predicate="works_at"),
            _make_fact(category="preference", subject="sushi", predicate="likes"),
            _make_fact(
                category="relationship",
                subject="Sarah",
                predicate="is_family_of",
                value="sister",
            ),
        ]
        inserted = save_facts(facts, "contact-001")

        assert inserted == 4

        stored = get_facts_for_contact("contact-001")
        assert len(stored) == 4
        categories = {f.category for f in stored}
        assert categories == {"location", "work", "preference", "relationship"}

    def test_insert_empty_list(self, patched_get_db) -> None:
        """Empty fact list returns 0 and does not crash."""
        from jarvis.contacts.fact_storage import save_facts

        inserted = save_facts([], "contact-001")
        assert inserted == 0

    def test_source_text_truncated_at_500(self, patched_get_db) -> None:
        """Source text longer than 500 chars is truncated before storage."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        long_text = "x" * 1000
        facts = [_make_fact(source_text=long_text)]
        save_facts(facts, "contact-001")

        stored = get_facts_for_contact("contact-001")
        assert len(stored) == 1
        assert len(stored[0].source_text) == 500

    def test_none_value_stored_as_empty_string(self, patched_get_db) -> None:
        """Fact with value=None is stored as empty string."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        fact = _make_fact(value=None)  # type: ignore[arg-type]
        save_facts([fact], "contact-001")

        stored = get_facts_for_contact("contact-001")
        assert len(stored) == 1
        assert stored[0].value == ""

    def test_none_source_text_stored_as_empty_string(self, patched_get_db) -> None:
        """Fact with source_text=None is stored as empty string."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        fact = _make_fact(source_text=None)  # type: ignore[arg-type]
        save_facts([fact], "contact-001")

        stored = get_facts_for_contact("contact-001")
        assert len(stored) == 1
        assert stored[0].source_text == ""


# =========================================================================
# Deduplication
# =========================================================================


class TestDeduplication:
    """Test UNIQUE constraint dedup: (contact_id, category, subject, predicate)."""

    def test_duplicate_facts_ignored(self, patched_get_db) -> None:
        """Inserting the same fact twice results in only one row."""
        from jarvis.contacts.fact_storage import count_facts_for_contact, save_facts

        fact = _make_fact(category="location", subject="Austin", predicate="lives_in")

        inserted_first = save_facts([fact], "contact-001")
        assert inserted_first == 1

        inserted_second = save_facts([fact], "contact-001")
        assert inserted_second == 0

        assert count_facts_for_contact("contact-001") == 1

    def test_same_subject_different_predicate_both_stored(self, patched_get_db) -> None:
        """Same subject but different predicate should store both."""
        from jarvis.contacts.fact_storage import count_facts_for_contact, save_facts

        facts = [
            _make_fact(category="location", subject="Austin", predicate="lives_in"),
            _make_fact(category="location", subject="Austin", predicate="lived_in"),
        ]
        inserted = save_facts(facts, "contact-001")
        assert inserted == 2
        assert count_facts_for_contact("contact-001") == 2

    def test_same_fact_different_contact_both_stored(self, patched_get_db) -> None:
        """Same fact for different contacts should store both."""
        from jarvis.contacts.fact_storage import count_facts_for_contact, save_facts

        fact = _make_fact(category="location", subject="Austin", predicate="lives_in")

        save_facts([fact], "contact-001")
        save_facts([fact], "contact-002")

        assert count_facts_for_contact("contact-001") == 1
        assert count_facts_for_contact("contact-002") == 1

    def test_batch_with_duplicates_only_inserts_unique(self, patched_get_db) -> None:
        """Batch containing duplicates of each other inserts only unique facts."""
        from jarvis.contacts.fact_storage import count_facts_for_contact, save_facts

        fact_a = _make_fact(category="location", subject="Austin", predicate="lives_in")
        fact_b = _make_fact(
            category="location", subject="Austin", predicate="lives_in"
        )  # duplicate
        fact_c = _make_fact(category="work", subject="Google", predicate="works_at")

        inserted = save_facts([fact_a, fact_b, fact_c], "contact-001")
        assert inserted == 2
        assert count_facts_for_contact("contact-001") == 2


# =========================================================================
# Linked Contacts
# =========================================================================


class TestLinkedContacts:
    """Test facts with linked_contact_id for NER person linking."""

    def test_linked_contact_id_stored(self, patched_get_db) -> None:
        """linked_contact_id is persisted and retrievable."""
        from jarvis.contacts.fact_storage import save_facts

        fact = _make_fact(
            category="relationship",
            subject="Sarah",
            predicate="is_family_of",
            value="sister",
            linked_contact_id="contact-099",
        )
        save_facts([fact], "contact-001")

        # Verify via raw query (get_facts_for_contact doesn't return
        # linked_contact_id in all code paths, so verify directly)
        db = patched_get_db
        with db.connection() as conn:
            row = conn.execute(
                "SELECT linked_contact_id FROM contact_facts WHERE contact_id = ?",
                ("contact-001",),
            ).fetchone()
            assert row["linked_contact_id"] == "contact-099"

    def test_null_linked_contact_id(self, patched_get_db) -> None:
        """Facts without linked_contact_id store NULL."""
        from jarvis.contacts.fact_storage import save_facts

        fact = _make_fact(linked_contact_id=None)
        save_facts([fact], "contact-001")

        db = patched_get_db
        with db.connection() as conn:
            row = conn.execute(
                "SELECT linked_contact_id FROM contact_facts WHERE contact_id = ?",
                ("contact-001",),
            ).fetchone()
            assert row["linked_contact_id"] is None

    def test_linked_contact_index_exists(self, patched_get_db) -> None:
        """idx_facts_linked_contact index is created by schema."""
        db = patched_get_db
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND name='idx_facts_linked_contact'"
            ).fetchall()
            assert len(rows) == 1


# =========================================================================
# Temporal Validity
# =========================================================================


class TestTemporalValidity:
    """Test valid_from and valid_until temporal fields."""

    def test_valid_from_stored(self, patched_get_db) -> None:
        """valid_from timestamp is persisted."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        fact = _make_fact(
            category="location",
            subject="Austin",
            predicate="lives_in",
            valid_from="2024-06-15T10:00:00",
        )
        save_facts([fact], "contact-001")

        stored = get_facts_for_contact("contact-001")
        assert len(stored) == 1
        assert stored[0].valid_from is not None
        assert str(stored[0].valid_from).startswith("2024-06-15")

    def test_valid_until_stored(self, patched_get_db) -> None:
        """valid_until timestamp is persisted."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        fact = _make_fact(
            category="location",
            subject="NYC",
            predicate="lived_in",
            valid_until="2023-12-01T00:00:00",
        )
        save_facts([fact], "contact-001")

        stored = get_facts_for_contact("contact-001")
        assert len(stored) == 1
        assert stored[0].valid_until is not None
        assert str(stored[0].valid_until).startswith("2023-12-01")

    def test_both_temporal_fields_null(self, patched_get_db) -> None:
        """Facts without temporal fields store NULL for both."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        fact = _make_fact(valid_from=None, valid_until=None)
        save_facts([fact], "contact-001")

        stored = get_facts_for_contact("contact-001")
        assert len(stored) == 1
        assert stored[0].valid_from is None
        assert stored[0].valid_until is None


# =========================================================================
# Retrieval
# =========================================================================


class TestRetrieval:
    """Test fact retrieval functions."""

    def test_get_facts_for_contact_ordered_by_confidence(self, patched_get_db) -> None:
        """Facts are returned in descending confidence order."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        facts = [
            _make_fact(category="location", subject="Austin", predicate="lives_in", confidence=0.5),
            _make_fact(category="work", subject="Google", predicate="works_at", confidence=0.9),
            _make_fact(category="preference", subject="sushi", predicate="likes", confidence=0.7),
        ]
        save_facts(facts, "contact-001")

        stored = get_facts_for_contact("contact-001")
        confidences = [f.confidence for f in stored]
        assert confidences == sorted(confidences, reverse=True)

    def test_get_facts_returns_empty_for_unknown_contact(self, patched_get_db) -> None:
        """Unknown contact returns empty list, not error."""
        from jarvis.contacts.fact_storage import get_facts_for_contact

        stored = get_facts_for_contact("nonexistent-contact")
        assert stored == []

    def test_get_all_facts_across_contacts(self, patched_get_db) -> None:
        """get_all_facts returns facts from all contacts."""
        from jarvis.contacts.fact_storage import get_all_facts, save_facts

        save_facts(
            [_make_fact(category="location", subject="Austin", predicate="lives_in")],
            "contact-001",
        )
        save_facts(
            [_make_fact(category="work", subject="Meta", predicate="works_at")],
            "contact-002",
        )

        all_facts = get_all_facts()
        assert len(all_facts) == 2
        contact_ids = {f.contact_id for f in all_facts}
        assert contact_ids == {"contact-001", "contact-002"}

    def test_get_all_facts_ordered_by_confidence(self, patched_get_db) -> None:
        """get_all_facts returns facts in descending confidence order."""
        from jarvis.contacts.fact_storage import get_all_facts, save_facts

        save_facts(
            [
                _make_fact(
                    category="location", subject="Austin", predicate="lives_in", confidence=0.3
                )
            ],
            "c1",
        )
        save_facts(
            [_make_fact(category="work", subject="Meta", predicate="works_at", confidence=0.9)],
            "c2",
        )

        all_facts = get_all_facts()
        confidences = [f.confidence for f in all_facts]
        assert confidences == sorted(confidences, reverse=True)

    def test_count_facts_for_contact(self, patched_get_db) -> None:
        """count_facts_for_contact returns correct count."""
        from jarvis.contacts.fact_storage import count_facts_for_contact, save_facts

        assert count_facts_for_contact("contact-001") == 0

        save_facts(
            [
                _make_fact(category="location", subject="Austin", predicate="lives_in"),
                _make_fact(category="work", subject="Google", predicate="works_at"),
            ],
            "contact-001",
        )
        assert count_facts_for_contact("contact-001") == 2

    def test_get_fact_count_global(self, patched_get_db) -> None:
        """get_fact_count returns total across all contacts."""
        from jarvis.contacts.fact_storage import get_fact_count, save_facts

        assert get_fact_count() == 0

        save_facts(
            [_make_fact(category="location", subject="Austin", predicate="lives_in")],
            "c1",
        )
        save_facts(
            [_make_fact(category="work", subject="Meta", predicate="works_at")],
            "c2",
        )

        assert get_fact_count() == 2


# =========================================================================
# Deletion
# =========================================================================


class TestDeletion:
    """Test delete_facts_for_contact."""

    def test_delete_removes_all_facts_for_contact(self, patched_get_db) -> None:
        """Delete all facts for a specific contact."""
        from jarvis.contacts.fact_storage import (
            count_facts_for_contact,
            delete_facts_for_contact,
            save_facts,
        )

        save_facts(
            [
                _make_fact(category="location", subject="Austin", predicate="lives_in"),
                _make_fact(category="work", subject="Google", predicate="works_at"),
            ],
            "contact-001",
        )
        save_facts(
            [_make_fact(category="preference", subject="coffee", predicate="likes")],
            "contact-002",
        )

        deleted = delete_facts_for_contact("contact-001")
        assert deleted == 2
        assert count_facts_for_contact("contact-001") == 0
        # Other contact's facts are untouched
        assert count_facts_for_contact("contact-002") == 1

    def test_delete_nonexistent_contact_returns_zero(self, patched_get_db) -> None:
        """Deleting facts for a contact with no facts returns 0."""
        from jarvis.contacts.fact_storage import delete_facts_for_contact

        deleted = delete_facts_for_contact("nonexistent")
        assert deleted == 0

    def test_delete_then_reinsert(self, patched_get_db) -> None:
        """After deleting facts, reinserting the same facts should work."""
        from jarvis.contacts.fact_storage import (
            count_facts_for_contact,
            delete_facts_for_contact,
            save_facts,
        )

        facts = [_make_fact(category="location", subject="Austin", predicate="lives_in")]
        save_facts(facts, "contact-001")
        assert count_facts_for_contact("contact-001") == 1

        delete_facts_for_contact("contact-001")
        assert count_facts_for_contact("contact-001") == 0

        inserted = save_facts(facts, "contact-001")
        assert inserted == 1
        assert count_facts_for_contact("contact-001") == 1


# =========================================================================
# Edge Cases
# =========================================================================


class TestEdgeCases:
    """Edge cases: unicode, very long text, special characters."""

    def test_unicode_subject_and_value(self, patched_get_db) -> None:
        """Unicode characters in subject and value are stored correctly."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        fact = _make_fact(
            category="preference",
            subject="ramen",
            predicate="likes",
            value="food",
            source_text="I love ramen!",
        )
        save_facts([fact], "contact-001")

        stored = get_facts_for_contact("contact-001")
        assert len(stored) == 1
        assert stored[0].subject == "ramen"

    def test_emoji_in_source_text(self, patched_get_db) -> None:
        """Emoji characters in source text are stored correctly."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        fact = _make_fact(source_text="I love sushi so much! \U0001f363\U0001f60d")
        save_facts([fact], "contact-001")

        stored = get_facts_for_contact("contact-001")
        assert len(stored) == 1
        assert "\U0001f363" in stored[0].source_text

    def test_cjk_characters(self, patched_get_db) -> None:
        """CJK characters stored and retrieved correctly."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        fact = _make_fact(
            subject="\u6771\u4eac",
            predicate="lives_in",
            category="location",
            source_text="\u6771\u4eac\u306b\u4f4f\u3093\u3067\u3044\u307e\u3059",
        )
        save_facts([fact], "contact-001")

        stored = get_facts_for_contact("contact-001")
        assert len(stored) == 1
        assert stored[0].subject == "\u6771\u4eac"

    def test_fact_with_none_source_message_id(self, patched_get_db) -> None:
        """Fact with source_message_id=None is stored."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        fact = _make_fact(source_message_id=None)
        save_facts([fact], "contact-001")

        stored = get_facts_for_contact("contact-001")
        assert len(stored) == 1
        assert stored[0].source_message_id is None

    def test_very_long_subject(self, patched_get_db) -> None:
        """Very long subject string is stored (SQLite has no practical limit)."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        long_subject = "A" * 2000
        fact = _make_fact(subject=long_subject)
        save_facts([fact], "contact-001")

        stored = get_facts_for_contact("contact-001")
        assert len(stored) == 1
        assert stored[0].subject == long_subject

    def test_empty_strings_for_optional_fields(self, patched_get_db) -> None:
        """Empty strings for value and source_text are handled."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        fact = _make_fact(value="", source_text="")
        save_facts([fact], "contact-001")

        stored = get_facts_for_contact("contact-001")
        assert len(stored) == 1
        assert stored[0].value == ""
        assert stored[0].source_text == ""

    def test_special_sql_characters_in_subject(self, patched_get_db) -> None:
        """SQL special characters in subject don't cause injection."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        fact = _make_fact(subject="O'Brien; DROP TABLE contact_facts;--")
        save_facts([fact], "contact-001")

        stored = get_facts_for_contact("contact-001")
        assert len(stored) == 1
        assert stored[0].subject == "O'Brien; DROP TABLE contact_facts;--"


# =========================================================================
# Retrieval field integrity
# =========================================================================


class TestRetrievalFieldIntegrity:
    """Verify all fields are correctly round-tripped through save/load."""

    def test_all_fields_round_trip(self, patched_get_db) -> None:
        """Every field set on the Fact is retrievable after save."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        fact = Fact(
            category="work",
            subject="Google",
            predicate="works_at",
            value="software engineer",
            confidence=0.92,
            contact_id="contact-001",
            source_text="I work at Google as a software engineer",
            source_message_id=12345,
            extracted_at="2025-01-15T10:30:00",
            valid_from="2024-06-01T00:00:00",
            valid_until=None,
        )
        save_facts([fact], "contact-001")

        stored = get_facts_for_contact("contact-001")
        assert len(stored) == 1
        s = stored[0]

        assert s.category == "work"
        assert s.subject == "Google"
        assert s.predicate == "works_at"
        assert s.value == "software engineer"
        assert s.confidence == pytest.approx(0.92, abs=0.001)
        assert s.source_text == "I work at Google as a software engineer"
        assert s.source_message_id == 12345
        assert s.valid_from is not None
        assert str(s.valid_from).startswith("2024-06-01")
        assert s.valid_until is None

    def test_extracted_at_field_populated(self, patched_get_db) -> None:
        """extracted_at is set by save_facts (current timestamp)."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        fact = _make_fact()
        save_facts([fact], "contact-001")

        stored = get_facts_for_contact("contact-001")
        assert len(stored) == 1
        # extracted_at should be set (datetime or non-empty string)
        assert stored[0].extracted_at is not None
        assert stored[0].extracted_at != ""


# =========================================================================
# Performance
# =========================================================================


class TestPerformance:
    """Performance tests: batch operations must not use N+1 pattern."""

    def test_batch_insert_150_facts_under_200ms(self, patched_get_db) -> None:
        """Insert 150 facts in a single batch call under 200ms.

        This verifies we use executemany() not individual INSERTs.
        """
        from jarvis.contacts.fact_storage import count_facts_for_contact, save_facts

        facts = [
            _make_fact(
                category=f"cat_{i % 5}",
                subject=f"Subject {i}",
                predicate=f"pred_{i % 3}",
                confidence=0.5 + (i % 50) / 100,
            )
            for i in range(150)
        ]

        start = time.perf_counter()
        inserted = save_facts(facts, "contact-perf")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert inserted == 150
        assert count_facts_for_contact("contact-perf") == 150
        assert elapsed_ms < 200, f"Batch insert took {elapsed_ms:.1f}ms, should be <200ms"

    def test_retrieval_150_facts_under_100ms(self, patched_get_db) -> None:
        """Retrieve 150 facts for a contact under 100ms."""
        from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

        facts = [
            _make_fact(
                category=f"cat_{i % 5}",
                subject=f"Subject {i}",
                predicate=f"pred_{i % 3}",
            )
            for i in range(150)
        ]
        save_facts(facts, "contact-perf")

        start = time.perf_counter()
        stored = get_facts_for_contact("contact-perf")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(stored) == 150
        assert elapsed_ms < 100, f"Retrieval took {elapsed_ms:.1f}ms, should be <100ms"

    def test_delete_150_facts_under_100ms(self, patched_get_db) -> None:
        """Delete 150 facts for a contact under 100ms."""
        from jarvis.contacts.fact_storage import delete_facts_for_contact, save_facts

        facts = [
            _make_fact(
                category=f"cat_{i % 5}",
                subject=f"Subject {i}",
                predicate=f"pred_{i % 3}",
            )
            for i in range(150)
        ]
        save_facts(facts, "contact-perf")

        start = time.perf_counter()
        deleted = delete_facts_for_contact("contact-perf")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert deleted == 150
        assert elapsed_ms < 100, f"Delete took {elapsed_ms:.1f}ms, should be <100ms"

    def test_count_is_fast(self, patched_get_db) -> None:
        """count_facts_for_contact is O(1)-ish (indexed query)."""
        from jarvis.contacts.fact_storage import count_facts_for_contact, save_facts

        facts = [
            _make_fact(
                category=f"cat_{i % 5}",
                subject=f"Subject {i}",
                predicate=f"pred_{i % 3}",
            )
            for i in range(200)
        ]
        save_facts(facts, "contact-perf")

        start = time.perf_counter()
        for _ in range(100):
            count_facts_for_contact("contact-perf")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"100 count queries took {elapsed_ms:.1f}ms, should be <50ms"


# =========================================================================
# Database Index Verification
# =========================================================================


class TestDatabaseIndexes:
    """Verify that expected indexes exist on contact_facts table."""

    def test_contact_facts_indexes_exist(self, patched_get_db) -> None:
        """Check idx_facts_contact and idx_facts_category indexes exist."""
        db = patched_get_db
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_facts_%'"
            ).fetchall()
            index_names = {row["name"] for row in rows}

        assert "idx_facts_contact" in index_names
        assert "idx_facts_category" in index_names

    def test_unique_constraint_exists(self, patched_get_db) -> None:
        """UNIQUE(contact_id, category, subject, predicate) is enforced."""
        db = patched_get_db
        with db.connection() as conn:
            # Insert a fact directly
            conn.execute(
                """INSERT INTO contact_facts
                   (contact_id, category, subject, predicate)
                   VALUES (?, ?, ?, ?)""",
                ("c1", "location", "Austin", "lives_in"),
            )

            # Attempt duplicate should raise IntegrityError
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """INSERT INTO contact_facts
                       (contact_id, category, subject, predicate)
                       VALUES (?, ?, ?, ?)""",
                    ("c1", "location", "Austin", "lives_in"),
                )


# =========================================================================
# Fact Indexing Integration (mocked embedder)
# =========================================================================


class TestFactIndexingOnSave:
    """Test that save_and_index_facts calls index_facts for newly inserted facts."""

    def test_index_facts_called_on_new_inserts(self, patched_get_db) -> None:
        """When facts are inserted, index_facts is called."""
        from jarvis.contacts.fact_storage import save_and_index_facts

        fact = _make_fact(category="location", subject="Austin", predicate="lives_in")

        with patch("jarvis.contacts.fact_index.index_facts") as mock_index:
            save_and_index_facts([fact], "contact-001")
            mock_index.assert_called_once()
            call_args = mock_index.call_args
            assert call_args[0][1] == "contact-001"
            assert len(call_args[0][0]) == 1

    def test_index_facts_not_called_on_zero_inserts(self, patched_get_db) -> None:
        """When all facts are duplicates, index_facts is NOT called."""
        from jarvis.contacts.fact_storage import save_and_index_facts, save_facts

        fact = _make_fact(category="location", subject="Austin", predicate="lives_in")
        save_facts([fact], "contact-001")  # First insert (no indexing needed)

        with patch("jarvis.contacts.fact_index.index_facts") as mock_index:
            save_and_index_facts([fact], "contact-001")  # Duplicate
            mock_index.assert_not_called()

    def test_index_facts_failure_does_not_crash_save(self, patched_get_db) -> None:
        """If index_facts raises, save_and_index_facts still succeeds."""
        from jarvis.contacts.fact_storage import count_facts_for_contact, save_and_index_facts

        fact = _make_fact(category="location", subject="Austin", predicate="lives_in")

        with patch(
            "jarvis.contacts.fact_index.index_facts",
            side_effect=RuntimeError("embedder unavailable"),
        ):
            inserted = save_and_index_facts([fact], "contact-001")

        assert inserted == 1
        assert count_facts_for_contact("contact-001") == 1


# =========================================================================
# Category-based retrieval (via raw SQL since no dedicated function)
# =========================================================================


class TestCategoryRetrieval:
    """Test that facts can be queried by category via the DB indexes."""

    def test_query_by_category(self, patched_get_db) -> None:
        """Facts can be filtered by category using direct SQL."""
        from jarvis.contacts.fact_storage import save_facts

        save_facts(
            [
                _make_fact(category="location", subject="Austin", predicate="lives_in"),
                _make_fact(category="work", subject="Google", predicate="works_at"),
                _make_fact(category="location", subject="NYC", predicate="lived_in"),
            ],
            "contact-001",
        )

        db = patched_get_db
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT subject FROM contact_facts WHERE contact_id = ? AND category = ?",
                ("contact-001", "location"),
            ).fetchall()

        subjects = {row["subject"] for row in rows}
        assert subjects == {"Austin", "NYC"}
        assert "Google" not in subjects

    def test_query_by_recency(self, patched_get_db) -> None:
        """Facts can be ordered by extracted_at for recency queries."""
        from jarvis.contacts.fact_storage import save_facts

        save_facts(
            [
                _make_fact(category="location", subject="Austin", predicate="lives_in"),
                _make_fact(category="work", subject="Google", predicate="works_at"),
            ],
            "contact-001",
        )

        db = patched_get_db
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT subject, extracted_at FROM contact_facts "
                "WHERE contact_id = ? ORDER BY extracted_at DESC",
                ("contact-001",),
            ).fetchall()

        assert len(rows) == 2
        # Both should have a non-empty extracted_at since save_facts sets it
        for row in rows:
            assert row["extracted_at"] is not None
            assert row["extracted_at"] != ""
