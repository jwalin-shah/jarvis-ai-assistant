"""Tests for graph context extraction (jarvis/graph/context.py).

Uses a real sqlite DB with contact_facts and contacts tables to verify
that relationship descriptions, shared connections, and the combined
context string are built correctly from actual data.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from jarvis.graph.context import (
    _get_relationship_description,
    _get_shared_connections,
    get_graph_context,
)

CONTACT_ID = "iMessage;-;+15551234567"
LINKED_CONTACT_CHAT_ID = "iMessage;-;+15559876543"


def _make_test_db(tmp_path):
    """Create a real JarvisDB with contacts and facts."""
    from jarvis.db import JarvisDB

    db_path = tmp_path / "test_graph.db"
    db = JarvisDB(db_path=db_path)
    db.init_schema()
    return db


def _seed_data(db) -> None:
    """Seed contacts and facts for testing."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with db.connection() as conn:
        # Insert contacts
        conn.execute(
            """
            INSERT INTO contacts (chat_id, display_name, phone_or_email, relationship)
            VALUES (?, ?, ?, ?)
            """,
            (CONTACT_ID, "Sarah", "+15551234567", "family"),
        )
        conn.execute(
            """
            INSERT INTO contacts (chat_id, display_name, phone_or_email, relationship)
            VALUES (?, ?, ?, ?)
            """,
            (LINKED_CONTACT_CHAT_ID, "John Smith", "+15559876543", "friend"),
        )

        # Insert relationship facts
        conn.execute(
            """
            INSERT INTO contact_facts
            (contact_id, category, subject, predicate, value, confidence, extracted_at)
            VALUES (?, 'relationship', 'Sarah', 'is_family_of', 'sister', 0.95, ?)
            """,
            (CONTACT_ID, now),
        )
        conn.execute(
            """
            INSERT INTO contact_facts
            (contact_id, category, subject, predicate, value, confidence, extracted_at)
            VALUES (?, 'relationship', 'Mom', 'is_family_of', 'mother', 0.9, ?)
            """,
            (CONTACT_ID, now),
        )

        # Insert a fact that links to another contact
        conn.execute(
            """
            INSERT INTO contact_facts
            (contact_id, category, subject, predicate, value, confidence,
             linked_contact_id, extracted_at)
            VALUES (?, 'relationship', 'John', 'is_friend_of', '', 0.8, ?, ?)
            """,
            (CONTACT_ID, LINKED_CONTACT_CHAT_ID, now),
        )

        # Insert non-relationship facts (should be ignored by relationship query)
        conn.execute(
            """
            INSERT INTO contact_facts
            (contact_id, category, subject, predicate, value, confidence, extracted_at)
            VALUES (?, 'work', 'Google', 'works_at', 'engineer', 0.9, ?)
            """,
            (CONTACT_ID, now),
        )


class TestRelationshipDescription:
    def test_returns_relationship_facts_as_natural_language(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_data(db)

        with patch("jarvis.db.get_db", return_value=db):
            result = _get_relationship_description(CONTACT_ID)

        # Should mention the relationships
        assert "Sarah" in result
        assert "sister" in result
        assert "Mom" in result
        assert "mother" in result
        # Should NOT include work facts
        assert "Google" not in result
        db.close()

    def test_empty_when_no_relationship_facts(self, tmp_path):
        db = _make_test_db(tmp_path)
        # Don't seed any data
        with patch("jarvis.db.get_db", return_value=db):
            result = _get_relationship_description(CONTACT_ID)
        assert result == ""
        db.close()

    def test_empty_for_unknown_contact(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_data(db)
        with patch("jarvis.db.get_db", return_value=db):
            result = _get_relationship_description("nonexistent_chat_id")
        assert result == ""
        db.close()


class TestSharedConnections:
    def test_finds_linked_contacts(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_data(db)

        with patch("jarvis.db.get_db", return_value=db):
            result = _get_shared_connections(CONTACT_ID)

        # Should find John Smith via linked_contact_id
        assert "John Smith" in result
        db.close()

    def test_empty_when_no_linked_contacts(self, tmp_path):
        db = _make_test_db(tmp_path)
        # Seed only basic facts without linked_contact_id
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO contact_facts
                (contact_id, category, subject, predicate, value, confidence, extracted_at)
                VALUES (?, 'relationship', 'Sarah', 'is_family_of', 'sister', 0.95, ?)
                """,
                (CONTACT_ID, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )

        with patch("jarvis.db.get_db", return_value=db):
            result = _get_shared_connections(CONTACT_ID)

        assert result == ""
        db.close()


class TestGetGraphContext:
    """Test the full pipeline: relationship + recency + connections combined."""

    def test_combines_relationship_and_connections(self, tmp_path):
        db = _make_test_db(tmp_path)
        _seed_data(db)

        with (
            patch("jarvis.db.get_db", return_value=db),
            # Mock recency since we can't query real iMessage DB in tests
            patch(
                "jarvis.graph.context._get_interaction_recency",
                return_value="Last messaged 2 days ago.",
            ),
        ):
            result = get_graph_context(CONTACT_ID, CONTACT_ID)

        # All three components should be present
        assert "Sarah" in result
        assert "sister" in result
        assert "Last messaged 2 days ago." in result
        assert "John Smith" in result
        db.close()

    def test_graceful_when_all_empty(self, tmp_path):
        db = _make_test_db(tmp_path)
        # No data seeded

        with (
            patch("jarvis.db.get_db", return_value=db),
            patch("jarvis.graph.context._get_interaction_recency", return_value=""),
        ):
            result = get_graph_context("unknown_chat", "unknown_chat")

        assert result == ""
        db.close()

    def test_partial_data_still_works(self, tmp_path):
        """If only relationship facts exist (no connections, no recency), still useful."""
        db = _make_test_db(tmp_path)
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO contact_facts
                (contact_id, category, subject, predicate, value, confidence, extracted_at)
                VALUES (?, 'relationship', 'Sarah', 'is_family_of', 'sister', 0.95, ?)
                """,
                (CONTACT_ID, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )

        with (
            patch("jarvis.db.get_db", return_value=db),
            patch("jarvis.graph.context._get_interaction_recency", return_value=""),
        ):
            result = get_graph_context(CONTACT_ID, CONTACT_ID)

        assert "Sarah" in result
        assert "sister" in result
        # No crash, just relationship info
        db.close()
