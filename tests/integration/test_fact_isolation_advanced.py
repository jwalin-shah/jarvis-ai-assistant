"""Advanced isolation and edge-case integration tests for fact extraction.

Tests cross-contact fact isolation, NER linking boundaries, temporal validity,
confidence thresholds, bulk extraction, and empty/new contact behavior.
Uses in-memory DB with fake embedder (no MLX required).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from jarvis.contacts.contact_profile import Fact
from jarvis.contacts.fact_extractor import FactExtractor
from jarvis.contacts.fact_storage import get_facts_for_contact, save_facts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONTACT_A = "iMessage;-;+15551111111"
CONTACT_B = "iMessage;-;+15552222222"
CONTACT_C = "iMessage;-;+15553333333"


def _make_test_db(tmp_path: Path):
    from jarvis.db import JarvisDB

    db_path = tmp_path / "test_isolation.db"
    db = JarvisDB(db_path=db_path)
    db.init_schema()
    return db


def _make_fact(
    subject: str,
    category: str = "preference",
    predicate: str = "likes",
    confidence: float = 0.8,
    contact_id: str = "",
    **kwargs,
) -> Fact:
    return Fact(
        category=category,
        subject=subject,
        predicate=predicate,
        confidence=confidence,
        contact_id=contact_id,
        extracted_at=datetime.now().isoformat(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCrossContactIsolation:
    """Facts stored for one contact don't leak into another."""

    def test_same_subject_different_contacts_no_collision(self, tmp_path):
        """Two contacts can both like pizza without collision."""
        db = _make_test_db(tmp_path)

        facts_a = [_make_fact("pizza", contact_id=CONTACT_A)]
        facts_b = [_make_fact("pizza", contact_id=CONTACT_B)]

        with patch("jarvis.db.get_db", return_value=db):
            save_facts(facts_a, CONTACT_A)
            save_facts(facts_b, CONTACT_B)

            retrieved_a = get_facts_for_contact(CONTACT_A)
            retrieved_b = get_facts_for_contact(CONTACT_B)

        assert len(retrieved_a) == 1
        assert len(retrieved_b) == 1
        assert retrieved_a[0].subject == "pizza"
        assert retrieved_b[0].subject == "pizza"
        assert retrieved_a[0].contact_id == CONTACT_A
        assert retrieved_b[0].contact_id == CONTACT_B

    def test_different_facts_stay_separated(self, tmp_path):
        """Contact A has sushi, Contact B has tacos. No cross-contamination."""
        db = _make_test_db(tmp_path)

        with patch("jarvis.db.get_db", return_value=db):
            save_facts([_make_fact("sushi", contact_id=CONTACT_A)], CONTACT_A)
            save_facts([_make_fact("tacos", contact_id=CONTACT_B)], CONTACT_B)

            a_facts = get_facts_for_contact(CONTACT_A)
            b_facts = get_facts_for_contact(CONTACT_B)

        a_subjects = {f.subject for f in a_facts}
        b_subjects = {f.subject for f in b_facts}
        assert "sushi" in a_subjects
        assert "tacos" not in a_subjects
        assert "tacos" in b_subjects
        assert "sushi" not in b_subjects


class TestNERLinkingBoundaries:
    """NER person linking doesn't cross contact boundaries."""

    def test_resolve_only_matches_cached_contacts(self):
        """_resolve_person_to_contact uses only its contacts cache."""
        ext = FactExtractor()
        cache = [(CONTACT_A, "Sarah Johnson", {"sarah", "johnson"})]
        ext._get_contacts_for_resolution = lambda: cache
        assert ext._resolve_person_to_contact("Sarah Johnson") == CONTACT_A
        assert ext._resolve_person_to_contact("Mike Thompson") is None

    def test_different_contact_caches_independent(self):
        """Two extractors with different caches don't interfere."""
        ext1 = FactExtractor()
        ext1._get_contacts_for_resolution = lambda: [(CONTACT_A, "Alice", {"alice"})]

        ext2 = FactExtractor()
        ext2._get_contacts_for_resolution = lambda: [(CONTACT_B, "Bob", {"bob"})]

        assert ext1._resolve_person_to_contact("Alice") == CONTACT_A
        assert ext1._resolve_person_to_contact("Bob") is None

        assert ext2._resolve_person_to_contact("Bob") == CONTACT_B
        assert ext2._resolve_person_to_contact("Alice") is None


class TestTemporalValidityStorage:
    """Temporal fields are stored and retrieved correctly."""

    def test_valid_from_persists(self, tmp_path):
        db = _make_test_db(tmp_path)
        fact = _make_fact(
            "Austin",
            category="location",
            predicate="lives_in",
            contact_id=CONTACT_A,
            valid_from="2025-01-15T10:00:00",
        )

        with patch("jarvis.db.get_db", return_value=db):
            save_facts([fact], CONTACT_A)
            retrieved = get_facts_for_contact(CONTACT_A)

        assert len(retrieved) == 1
        assert retrieved[0].subject == "Austin"
        assert retrieved[0].valid_from is not None
        # SQLite may return datetime or string; just verify the value round-trips
        assert "2025" in str(retrieved[0].valid_from)

    def test_valid_until_persists(self, tmp_path):
        db = _make_test_db(tmp_path)
        fact = _make_fact(
            "Texas",
            category="location",
            predicate="lived_in",
            contact_id=CONTACT_A,
            valid_until="2025-01-15T10:00:00",
        )

        with patch("jarvis.db.get_db", return_value=db):
            save_facts([fact], CONTACT_A)
            retrieved = get_facts_for_contact(CONTACT_A)

        assert len(retrieved) == 1
        assert retrieved[0].valid_until is not None
        assert "2025" in str(retrieved[0].valid_until)


class TestConfidenceThresholdFiltering:
    """Extraction respects confidence thresholds."""

    def test_high_threshold_filters_more(self):
        ext_strict = FactExtractor(confidence_threshold=0.9)
        ext_lenient = FactExtractor(confidence_threshold=0.3)

        messages = [{"text": "I live in Austin and love spicy ramen every day"}]
        strict_facts = ext_strict.extract_facts(messages, CONTACT_A)
        lenient_facts = ext_lenient.extract_facts(messages, CONTACT_A)

        assert len(lenient_facts) >= len(strict_facts)


class TestBulkExtraction:
    """Bulk extraction across many messages and contacts."""

    def test_100_messages_3_contacts_no_leakage(self, tmp_path):
        """Extract from messages per contact, verify no cross-contact leakage."""
        db = _make_test_db(tmp_path)
        ext = FactExtractor()

        contacts = {
            CONTACT_A: "I live in Austin",
            CONTACT_B: "I work at Google",
            CONTACT_C: "My sister Sarah is visiting",
        }

        with patch("jarvis.db.get_db", return_value=db):
            for contact_id, base_text in contacts.items():
                messages = [{"text": base_text}] * 10
                facts = ext.extract_facts(messages, contact_id)
                if facts:
                    save_facts(facts, contact_id)

            a_facts = get_facts_for_contact(CONTACT_A)
            b_facts = get_facts_for_contact(CONTACT_B)
            c_facts = get_facts_for_contact(CONTACT_C)

        for f in a_facts:
            assert f.contact_id == CONTACT_A
        for f in b_facts:
            assert f.contact_id == CONTACT_B
        for f in c_facts:
            assert f.contact_id == CONTACT_C

        a_categories = {f.category for f in a_facts}
        b_categories = {f.category for f in b_facts}
        c_categories = {f.category for f in c_facts}

        assert "location" in a_categories
        assert "work" in b_categories
        assert "relationship" in c_categories


class TestEmptyNewContact:
    """Empty or new contacts don't inherit facts from others."""

    def test_new_contact_returns_empty(self, tmp_path):
        db = _make_test_db(tmp_path)

        with patch("jarvis.db.get_db", return_value=db):
            save_facts([_make_fact("sushi", contact_id=CONTACT_A)], CONTACT_A)
            new_facts = get_facts_for_contact("new_unknown_contact")

        assert len(new_facts) == 0

    def test_empty_messages_produce_no_facts(self):
        ext = FactExtractor()
        facts = ext.extract_facts([], CONTACT_A)
        assert len(facts) == 0

    def test_whitespace_only_messages_produce_no_facts(self):
        ext = FactExtractor()
        facts = ext.extract_facts([{"text": "   "}, {"text": ""}], CONTACT_A)
        assert len(facts) == 0


class TestDeduplicationAcrossMessages:
    """Duplicate facts from different messages are properly deduplicated."""

    def test_same_fact_different_messages_stored_once(self, tmp_path):
        db = _make_test_db(tmp_path)
        ext = FactExtractor()

        messages = [
            {"text": "I live in Austin"},
            {"text": "Yeah I live in Austin too"},
        ]
        facts = ext.extract_facts(messages, CONTACT_A)

        with patch("jarvis.db.get_db", return_value=db):
            save_facts(facts, CONTACT_A)
            retrieved = get_facts_for_contact(CONTACT_A)

        austin_facts = [f for f in retrieved if f.subject.lower() == "austin"]
        assert len(austin_facts) == 1
