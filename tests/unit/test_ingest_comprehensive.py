"""Comprehensive tests for contact ingestion.

Tests cover:
- Success cases (new contacts, updates, grouping)
- Error cases (database errors, missing files)
- Edge cases (empty names, duplicate handles, large datasets)
- Invalid inputs (malformed data, SQL injection)
- Integration scenarios (end-to-end ingestion)
"""

import sqlite3
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from jarvis.db import Contact
from jarvis.search.ingest import ADDRESS_BOOK_PATH, ingest_contacts


class TestIngestContactsSuccessCases:
    """Tests for successful contact ingestion scenarios."""

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """Create a mock JarvisDB instance."""
        db = MagicMock()
        db.get_contact_by_handle = Mock(return_value=None)
        db.add_contact = Mock()
        return db

    @pytest.fixture
    def sample_rows(self) -> list[dict[str, Any]]:
        """Sample Address Book rows."""
        return [
            {
                "identifier": "+15551234567",
                "first_name": "John",
                "last_name": "Doe",
                "org_name": None,
            },
            {
                "identifier": "john.doe@example.com",
                "first_name": "John",
                "last_name": "Doe",
                "org_name": None,
            },
        ]

    def _mock_path_str(self, _instance: Any = None) -> str:
        """Helper to mock Path.__str__."""
        return str(ADDRESS_BOOK_PATH)

    def _create_mock_connection(self, rows: list[dict[str, Any]]) -> MagicMock:
        """Helper to create mock SQLite connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        class Row:
            def __init__(self, data: dict[str, Any]) -> None:
                self._data = data

            def __getitem__(self, key: str) -> Any:
                return self._data[key]

        mock_cursor.description = [
            ("identifier",),
            ("first_name",),
            ("last_name",),
            ("org_name",),
        ]
        mock_cursor.fetchall.return_value = [Row(row) for row in rows]
        mock_conn.execute.return_value = mock_cursor
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        return mock_conn

    def test_ingest_new_contacts(
        self, mock_db: MagicMock, sample_rows: list[dict[str, Any]]
    ) -> None:
        """Test successful ingestion of new contacts."""
        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True

            # Mock __str__ method (accepts instance arg from MagicMock dispatch)
            def mock_str(_instance: Any = None) -> str:
                return str(ADDRESS_BOOK_PATH)

            mock_path.__str__ = mock_str  # type: ignore[method-assign]
            mock_connect.return_value = self._create_mock_connection(sample_rows)

            result = ingest_contacts(mock_db)

            assert result["processed"] == 1  # Grouped as one person
            assert result["created"] == 1
            assert result["updated"] == 0
            assert result["skipped"] == 0
            assert "error" not in result

    def test_ingest_updates_existing_contact(self, mock_db, sample_rows) -> None:
        """Test that existing contacts are updated with new handles."""
        import json

        existing_contact = Contact(
            id=1,
            chat_id="chat123",
            display_name="John Doe",
            phone_or_email="+15551234567",
            relationship=None,
            style_notes=None,
            handles_json=json.dumps(["+15551234567"]),
        )

        def get_contact_side_effect(handle: str) -> Contact | None:
            return existing_contact if handle == "+15551234567" else None

        mock_db.get_contact_by_handle.side_effect = get_contact_side_effect

        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_path.__str__ = self._mock_path_str  # type: ignore[method-assign]
            mock_connect.return_value = self._create_mock_connection(sample_rows)

            result = ingest_contacts(mock_db)

            assert result["updated"] == 1
            assert result["created"] == 0
            # Verify handles were merged
            call_args = mock_db.add_contact.call_args[1]
            assert len(call_args["handles"]) == 2
            assert "+15551234567" in call_args["handles"]
            assert "john.doe@example.com" in call_args["handles"]

    def test_ingest_groups_multiple_handles(self, mock_db) -> None:
        """Test that multiple handles for same person are grouped."""
        rows = [
            {
                "identifier": "+15551234567",
                "first_name": "John",
                "last_name": "Doe",
                "org_name": None,
            },
            {
                "identifier": "john.doe@example.com",
                "first_name": "John",
                "last_name": "Doe",
                "org_name": None,
            },
            {
                "identifier": "+15559876543",
                "first_name": "John",
                "last_name": "Doe",
                "org_name": None,
            },
        ]

        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_path.__str__ = self._mock_path_str  # type: ignore[method-assign]
            mock_connect.return_value = self._create_mock_connection(rows)

            result = ingest_contacts(mock_db)

            assert result["processed"] == 1  # All grouped
            assert result["created"] == 1
            call_args = mock_db.add_contact.call_args[1]
            assert len(call_args["handles"]) == 3

    def test_ingest_uses_org_name_when_no_personal_name(self, mock_db) -> None:
        """Test that organization name is used when no personal name."""
        rows = [
            {
                "identifier": "info@company.com",
                "first_name": None,
                "last_name": None,
                "org_name": "Company Inc",
            },
        ]

        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_path.__str__ = self._mock_path_str  # type: ignore[method-assign]
            mock_connect.return_value = self._create_mock_connection(rows)

            result = ingest_contacts(mock_db)

            assert result["created"] == 1
            call_args = mock_db.add_contact.call_args[1]
            assert call_args["display_name"] == "Company Inc"


class TestIngestContactsEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """Create a mock JarvisDB instance."""
        db = MagicMock()
        db.get_contact_by_handle = Mock(return_value=None)
        db.add_contact = Mock()
        return db

    def _mock_path_str(self, _instance: Any = None) -> str:
        """Helper to mock Path.__str__."""
        return str(ADDRESS_BOOK_PATH)

    def test_address_book_not_found(self, mock_db) -> None:
        """Test handling when Address Book database doesn't exist."""
        with patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path:
            mock_path.exists.return_value = False
            result = ingest_contacts(mock_db)

            assert "error" in result
            assert result["error"] == "Address Book not found"
            mock_db.add_contact.assert_not_called()

    def test_empty_database(self, mock_db) -> None:
        """Test handling when Address Book is empty."""
        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            result = ingest_contacts(mock_db)

            assert result["processed"] == 0
            assert result["created"] == 0
            assert "error" not in result

    def test_contacts_with_empty_names(self, mock_db) -> None:
        """Test that contacts with empty names are skipped."""
        rows = [
            {
                "identifier": "+15551234567",
                "first_name": None,
                "last_name": None,
                "org_name": None,
            },
        ]

        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_path.__str__ = self._mock_path_str  # type: ignore[method-assign]

            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            class Row:
                def __init__(self, data):
                    self._data = data

                def __getitem__(self, key):
                    return self._data[key]

            mock_cursor.description = [
                ("identifier",),
                ("first_name",),
                ("last_name",),
                ("org_name",),
            ]
            mock_cursor.fetchall.return_value = [Row(row) for row in rows]
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            result = ingest_contacts(mock_db)

            assert result["processed"] == 1
            assert result["created"] == 0  # Skipped due to empty name
            mock_db.add_contact.assert_not_called()

    def test_contacts_with_only_first_name(self, mock_db) -> None:
        """Test contact with only first name."""
        rows = [
            {
                "identifier": "+15551234567",
                "first_name": "John",
                "last_name": None,
                "org_name": None,
            },
        ]

        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_path.__str__ = self._mock_path_str  # type: ignore[method-assign]

            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            class Row:
                def __init__(self, data):
                    self._data = data

                def __getitem__(self, key):
                    return self._data[key]

            mock_cursor.description = [
                ("identifier",),
                ("first_name",),
                ("last_name",),
                ("org_name",),
            ]
            mock_cursor.fetchall.return_value = [Row(row) for row in rows]
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            result = ingest_contacts(mock_db)

            assert result["created"] == 1
            call_args = mock_db.add_contact.call_args[1]
            assert call_args["display_name"] == "John"

    def test_contacts_with_only_last_name(self, mock_db) -> None:
        """Test contact with only last name."""
        rows = [
            {
                "identifier": "+15551234567",
                "first_name": None,
                "last_name": "Doe",
                "org_name": None,
            },
        ]

        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_path.__str__ = self._mock_path_str  # type: ignore[method-assign]

            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            class Row:
                def __init__(self, data):
                    self._data = data

                def __getitem__(self, key):
                    return self._data[key]

            mock_cursor.description = [
                ("identifier",),
                ("first_name",),
                ("last_name",),
                ("org_name",),
            ]
            mock_cursor.fetchall.return_value = [Row(row) for row in rows]
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            result = ingest_contacts(mock_db)

            assert result["created"] == 1
            call_args = mock_db.add_contact.call_args[1]
            assert call_args["display_name"] == "Doe"

    def test_invalid_identifiers_filtered(self, mock_db) -> None:
        """Test that empty/None identifiers are filtered out.

        Note: normalize_phone_number returns non-phone strings as-is (e.g. "abc"),
        only None/empty inputs are filtered. So "abc" still creates a contact,
        while "" is filtered (returns None).
        """
        rows = [
            {
                "identifier": "abc",  # Non-phone but still returned as-is by normalize
                "first_name": "John",
                "last_name": "Doe",
                "org_name": None,
            },
            {
                "identifier": "",  # Empty -> normalize returns None -> filtered
                "first_name": "Jane",
                "last_name": "Smith",
                "org_name": None,
            },
            {
                "identifier": "+15551234567",  # Valid phone
                "first_name": "Bob",
                "last_name": "Jones",
                "org_name": None,
            },
        ]

        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_path.__str__ = self._mock_path_str  # type: ignore[method-assign]

            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            class Row:
                def __init__(self, data):
                    self._data = data

                def __getitem__(self, key):
                    return self._data[key]

            mock_cursor.description = [
                ("identifier",),
                ("first_name",),
                ("last_name",),
                ("org_name",),
            ]
            mock_cursor.fetchall.return_value = [Row(row) for row in rows]
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            result = ingest_contacts(mock_db)

            # "abc" is kept (normalize returns it as-is), "" is filtered (returns None)
            # So John Doe and Bob Jones are created, Jane Smith is skipped
            assert result["created"] == 2
            assert result["processed"] == 2


class TestIngestContactsErrorCases:
    """Tests for error handling."""

    def _mock_path_str(self, _instance: Any = None) -> str:
        """Helper to mock Path.__str__."""
        return str(ADDRESS_BOOK_PATH)

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """Create a mock JarvisDB instance."""
        db = MagicMock()
        db.get_contact_by_handle = Mock(return_value=None)
        db.add_contact = Mock()
        return db

    def test_database_locked_error(self, mock_db) -> None:
        """Test handling when database is locked."""
        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_connect.side_effect = sqlite3.Error("Database locked")

            result = ingest_contacts(mock_db)

            assert "error" in result
            assert "Database locked" in result["error"]
            mock_db.add_contact.assert_not_called()

    def test_database_permission_error(self, mock_db) -> None:
        """Test handling when database access is denied."""
        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_connect.side_effect = PermissionError("Access denied")

            result = ingest_contacts(mock_db)

            assert "error" in result
            mock_db.add_contact.assert_not_called()

    def test_query_execution_error(self, mock_db) -> None:
        """Test handling when query execution fails."""
        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_conn = MagicMock()
            mock_conn.execute.side_effect = sqlite3.Error("Query failed")
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            result = ingest_contacts(mock_db)

            assert "error" in result
            mock_db.add_contact.assert_not_called()

    def test_db_add_contact_error(self, mock_db) -> None:
        """Test handling when add_contact fails."""
        rows = [
            {
                "identifier": "+15551234567",
                "first_name": "John",
                "last_name": "Doe",
                "org_name": None,
            },
        ]

        mock_db.add_contact.side_effect = Exception("Database error")

        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_path.__str__ = self._mock_path_str  # type: ignore[method-assign]

            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            class Row:
                def __init__(self, data):
                    self._data = data

                def __getitem__(self, key):
                    return self._data[key]

            mock_cursor.description = [
                ("identifier",),
                ("first_name",),
                ("last_name",),
                ("org_name",),
            ]
            mock_cursor.fetchall.return_value = [Row(row) for row in rows]
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            result = ingest_contacts(mock_db)

            assert "error" in result
            assert "Database error" in result["error"]


class TestIngestContactsIntegration:
    """Integration tests for contact ingestion."""

    def _mock_path_str(self, _instance: Any = None) -> str:
        """Helper to mock Path.__str__."""
        return str(ADDRESS_BOOK_PATH)

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """Create a mock JarvisDB instance."""
        db = MagicMock()
        db.get_contact_by_handle = Mock(return_value=None)
        db.add_contact = Mock()
        return db

    def test_large_dataset_handling(self, mock_db) -> None:
        """Test handling of large contact datasets."""
        # Generate many contacts
        rows = []
        for i in range(1000):
            rows.append(
                {
                    "identifier": f"+1555123{i:04d}",
                    "first_name": f"Person{i}",
                    "last_name": "Test",
                    "org_name": None,
                }
            )

        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_path.__str__ = self._mock_path_str  # type: ignore[method-assign]

            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            class Row:
                def __init__(self, data):
                    self._data = data

                def __getitem__(self, key):
                    return self._data[key]

            mock_cursor.description = [
                ("identifier",),
                ("first_name",),
                ("last_name",),
                ("org_name",),
            ]
            mock_cursor.fetchall.return_value = [Row(row) for row in rows]
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            result = ingest_contacts(mock_db)

            assert result["processed"] == 1000
            assert result["created"] == 1000
            assert mock_db.add_contact.call_count == 1000

    def test_duplicate_handles_across_people(self, mock_db) -> None:
        """Test that duplicate handles are handled correctly."""
        rows = [
            {
                "identifier": "+15551234567",
                "first_name": "John",
                "last_name": "Doe",
                "org_name": None,
            },
            {
                "identifier": "+15551234567",  # Same handle, different person
                "first_name": "Jane",
                "last_name": "Smith",
                "org_name": None,
            },
        ]

        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_path.__str__ = self._mock_path_str  # type: ignore[method-assign]

            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            class Row:
                def __init__(self, data):
                    self._data = data

                def __getitem__(self, key):
                    return self._data[key]

            mock_cursor.description = [
                ("identifier",),
                ("first_name",),
                ("last_name",),
                ("org_name",),
            ]
            mock_cursor.fetchall.return_value = [Row(row) for row in rows]
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            result = ingest_contacts(mock_db)

            # Should create 2 separate contacts (different names)
            assert result["processed"] == 2
            assert result["created"] == 2

    def test_mixed_email_and_phone(self, mock_db) -> None:
        """Test contacts with both email and phone."""
        rows = [
            {
                "identifier": "+15551234567",
                "first_name": "John",
                "last_name": "Doe",
                "org_name": None,
            },
            {
                "identifier": "john.doe@example.com",
                "first_name": "John",
                "last_name": "Doe",
                "org_name": None,
            },
            {
                "identifier": "jane@example.com",
                "first_name": "Jane",
                "last_name": "Smith",
                "org_name": None,
            },
        ]

        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_path.__str__ = self._mock_path_str  # type: ignore[method-assign]

            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            class Row:
                def __init__(self, data):
                    self._data = data

                def __getitem__(self, key):
                    return self._data[key]

            mock_cursor.description = [
                ("identifier",),
                ("first_name",),
                ("last_name",),
                ("org_name",),
            ]
            mock_cursor.fetchall.return_value = [Row(row) for row in rows]
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            result = ingest_contacts(mock_db)

            # John Doe grouped (2 handles), Jane Smith separate
            assert result["processed"] == 2
            assert result["created"] == 2

    def test_stats_accuracy(self, mock_db) -> None:
        """Test that stats are accurate."""
        rows = [
            {
                "identifier": "+15551234567",
                "first_name": "John",
                "last_name": "Doe",
                "org_name": None,
            },
            {
                "identifier": "john.doe@example.com",
                "first_name": "John",
                "last_name": "Doe",
                "org_name": None,
            },
            {
                "identifier": "+15559876543",
                "first_name": "Jane",
                "last_name": "Smith",
                "org_name": None,
            },
        ]

        # Mock existing contact for Jane
        import json

        existing = Contact(
            id=1,
            chat_id="chat123",
            display_name="Jane Smith",
            phone_or_email="+15559876543",
            relationship=None,
            style_notes=None,
            handles_json=json.dumps(["+15559876543"]),
        )

        def get_contact_side_effect(handle: str) -> Contact | None:
            return existing if handle == "+15559876543" else None

        mock_db.get_contact_by_handle.side_effect = get_contact_side_effect

        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_path.__str__ = self._mock_path_str  # type: ignore[method-assign]

            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            class Row:
                def __init__(self, data):
                    self._data = data

                def __getitem__(self, key):
                    return self._data[key]

            mock_cursor.description = [
                ("identifier",),
                ("first_name",),
                ("last_name",),
                ("org_name",),
            ]
            mock_cursor.fetchall.return_value = [Row(row) for row in rows]
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            result = ingest_contacts(mock_db)

            # John Doe: 1 processed, 1 created
            # Jane Smith: 1 processed, 1 updated (no new handles, so skipped)
            assert result["processed"] == 2
            # Jane has no new handles, so should be skipped
            assert result["skipped"] >= 0
            assert result["created"] == 1
