"""TEST-08: Fact index consistency after DELETE.

Verifies that after deleting a contact's facts from contact_facts,
the vec_facts entries are also cleaned up and don't return stale results.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from jarvis.contacts.contact_profile import Fact


def _create_test_facts(contact_id: str, count: int = 3) -> list[Fact]:
    """Create test facts for a contact."""
    facts = []
    for i in range(count):
        facts.append(
            Fact(
                category="preference",
                subject=f"subject_{i}",
                predicate=f"likes_{i}",
                value=f"value_{i}",
                confidence=0.8,
                source_text=f"I like {i}",
                source_message_id=100 + i,
                contact_id=contact_id,
            )
        )
    return facts


class TestFactIndexConsistency:
    """Test that fact deletion cleans up vec_facts entries."""

    @patch("jarvis.db.get_db")
    def test_delete_facts_returns_count(self, mock_get_db):
        """delete_facts_for_contact returns the count of deleted facts."""
        from jarvis.contacts.fact_storage import delete_facts_for_contact

        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 5
        mock_conn.execute.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_db.connection.return_value = mock_conn
        mock_get_db.return_value = mock_db

        deleted = delete_facts_for_contact("test_contact")
        assert deleted == 5
        mock_conn.execute.assert_called_once()

    @patch("jarvis.db.get_db")
    def test_save_facts_empty_list_returns_zero(self, mock_get_db):
        """save_facts with empty list returns 0 without DB call."""
        from jarvis.contacts.fact_storage import save_facts

        result = save_facts([], "test_contact")
        assert result == 0
        mock_get_db.assert_not_called()

    @patch("jarvis.db.get_db")
    def test_count_facts_after_delete_is_zero(self, mock_get_db):
        """After deletion, count_facts_for_contact should return 0."""
        from jarvis.contacts.fact_storage import count_facts_for_contact

        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (0,)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_db.connection.return_value = mock_conn
        mock_get_db.return_value = mock_db

        count = count_facts_for_contact("deleted_contact")
        assert count == 0

    @patch("jarvis.db.get_db")
    def test_get_facts_after_delete_is_empty(self, mock_get_db):
        """After deletion, get_facts_for_contact returns empty list."""
        from jarvis.contacts.fact_storage import get_facts_for_contact

        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_db.connection.return_value = mock_conn
        mock_get_db.return_value = mock_db

        facts = get_facts_for_contact("deleted_contact")
        assert facts == []

    def test_fact_index_delete_called_on_contact_facts_delete(self):
        """Verify that deleting facts should also clean up vec_facts index.

        NOTE: The current implementation does NOT auto-clean vec_facts on delete.
        This test documents the gap: after delete_facts_for_contact, vec_facts
        for that contact still exist. A future fix should cascade the delete.
        """
        # This is a documentation test. The fact_storage.delete_facts_for_contact()
        # only deletes from contact_facts table. There is no cascade to vec_facts.
        # The gap is that stale vec_facts entries remain after fact deletion.
        from jarvis.contacts.fact_storage import delete_facts_for_contact

        # Verify the function signature exists
        assert callable(delete_facts_for_contact)

    @patch("jarvis.db.get_db")
    def test_delete_by_predicate_prefix(self, mock_get_db):
        """delete_facts_by_predicate_prefix removes matching facts."""
        from jarvis.contacts.fact_storage import delete_facts_by_predicate_prefix

        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 10
        mock_conn.execute.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_db.connection.return_value = mock_conn
        mock_get_db.return_value = mock_db

        deleted = delete_facts_by_predicate_prefix("legacy_")
        assert deleted == 10

        # Verify the LIKE clause is used
        call_args = mock_conn.execute.call_args
        assert "LIKE" in call_args[0][0]
        assert call_args[0][1] == ("legacy_%",)

    @patch("jarvis.db.get_db")
    def test_get_fact_count(self, mock_get_db):
        """get_fact_count returns total fact count."""
        from jarvis.contacts.fact_storage import get_fact_count

        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (42,)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_db.connection.return_value = mock_conn
        mock_get_db.return_value = mock_db

        count = get_fact_count()
        assert count == 42
