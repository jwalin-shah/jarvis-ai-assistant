"""Tests for jarvis/ingest.py - Contact ingestion from macOS Address Book."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from jarvis.ingest import (
    ADDRESS_BOOK_PATH,
    ALL_CONTACTS_QUERY,
    ingest_contacts,
    normalize_phone_number,
)


class TestNormalizePhoneNumber:
    """Tests for normalize_phone_number function."""

    def test_phone_with_plus(self) -> None:
        """Test phone number with leading +."""
        assert normalize_phone_number("+1 (555) 123-4567") == "+15551234567"
        assert normalize_phone_number("+44 20 7946 0958") == "+442079460958"

    def test_phone_without_plus(self) -> None:
        """Test phone number without leading + gets +1 prefix for 10 digits."""
        assert normalize_phone_number("(555) 123-4567") == "+15551234567"
        assert normalize_phone_number("555-123-4567") == "+15551234567"
        assert normalize_phone_number("555.123.4567") == "+15551234567"

    def test_email_address(self) -> None:
        """Test email address returned as-is (no lowercase normalization)."""
        assert normalize_phone_number("John.Doe@Example.COM") == "John.Doe@Example.COM"
        assert normalize_phone_number("test@test.com") == "test@test.com"
        assert normalize_phone_number("  Test@Test.com  ") == "Test@Test.com"

    def test_empty_input(self) -> None:
        """Test empty/None input."""
        assert normalize_phone_number("") is None
        assert normalize_phone_number(None) is None  # type: ignore
        assert normalize_phone_number("   ") is None

    def test_no_digits(self) -> None:
        """Test input with no digits - letters kept, formatting chars stripped."""
        assert normalize_phone_number("abc") == "abc"
        assert normalize_phone_number("---") == ""

    def test_phone_with_spaces(self) -> None:
        """Test phone number with spaces - 10-digit gets +1 prefix."""
        assert normalize_phone_number("555 123 4567") == "+15551234567"
        assert normalize_phone_number("+1 555 123 4567") == "+15551234567"

    def test_phone_with_extensions(self) -> None:
        """Test phone number with extension - 'ext' and 'x' text are NOT stripped."""
        assert normalize_phone_number("555-123-4567 ext 123") == "5551234567ext123"
        assert normalize_phone_number("555-123-4567 x123") == "5551234567x123"


class TestIngestContacts:
    """Tests for ingest_contacts function."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock JarvisDB instance."""
        db = MagicMock()
        db.get_contact_by_handle = Mock(return_value=None)
        db.add_contact = Mock()
        return db

    @pytest.fixture
    def sample_address_book_rows(self):
        """Sample rows from Address Book query."""
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
            {
                "identifier": "+15559876543",
                "first_name": "Jane",
                "last_name": "Smith",
                "org_name": None,
            },
            {
                "identifier": "info@company.com",
                "first_name": None,
                "last_name": None,
                "org_name": "Company Inc",
            },
        ]

    def test_address_book_not_found(self, mock_db) -> None:
        """Test handling when Address Book database doesn't exist."""
        with patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path:
            mock_path.exists.return_value = False
            result = ingest_contacts(mock_db)

            assert "error" in result
            assert result["error"] == "Address Book not found"
            mock_db.add_contact.assert_not_called()

    def test_successful_ingestion_new_contacts(self, mock_db, sample_address_book_rows) -> None:
        """Test successful ingestion of new contacts."""
        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_path.__str__ = lambda x: "/path/to/AddressBook-v22.abcddb"

            # Mock SQLite connection
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = sample_address_book_rows
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            # Mock row factory
            def row_factory(cursor, row):
                return dict(zip([col[0] for col in cursor.description], row))

            mock_conn.row_factory = row_factory

            # Convert sample rows to Row objects
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
            mock_cursor.fetchall.return_value = [
                Row(row) for row in sample_address_book_rows
            ]

            result = ingest_contacts(mock_db)

            # John Doe (grouped), Jane Smith, and Company Inc = 3 contacts
            assert result["processed"] == 3
            assert result["created"] == 3
            assert result["updated"] == 0
            assert result["skipped"] == 0

            # Verify add_contact was called for new contacts
            assert mock_db.add_contact.call_count == 3

    def test_ingestion_updates_existing_contact(self, mock_db, sample_address_book_rows) -> None:
        """Test that existing contacts are updated."""
        # Mock existing contact
        existing_contact = MagicMock()
        existing_contact.display_name = "John Doe"
        existing_contact.chat_id = "existing_chat_id"
        existing_contact.phone_or_email = "+15551234567"
        existing_contact.relationship = None
        existing_contact.style_notes = None
        existing_contact.handles = ["+15551234567"]

        mock_db.get_contact_by_handle.side_effect = lambda handle: (
            existing_contact if handle == "+15551234567" else None
        )

        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_path.__str__ = lambda x: "/path/to/AddressBook-v22.abcddb"

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
            # Only return John Doe rows
            john_rows = [r for r in sample_address_book_rows if r["first_name"] == "John"]
            mock_cursor.fetchall.return_value = [Row(row) for row in john_rows]
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            result = ingest_contacts(mock_db)

            assert result["processed"] == 1
            # Should update because new handle (email) was added
            assert result["updated"] == 1
            assert result["created"] == 0

            # Verify update was called with merged handles
            mock_db.add_contact.assert_called_once()
            call_args = mock_db.add_contact.call_args[1]
            assert set(call_args["handles"]) == {"+15551234567", "john.doe@example.com"}

    def test_ingestion_skips_unchanged_contact(self, mock_db, sample_address_book_rows) -> None:
        """Test that unchanged contacts are skipped."""
        # Mock existing contact with same data
        existing_contact = MagicMock()
        existing_contact.display_name = "John Doe"
        existing_contact.chat_id = "existing_chat_id"
        existing_contact.phone_or_email = "+15551234567"
        existing_contact.relationship = None
        existing_contact.style_notes = None
        existing_contact.handles = {"+15551234567", "john.doe@example.com"}

        mock_db.get_contact_by_handle.side_effect = lambda handle: (
            existing_contact if handle in ("+15551234567", "john.doe@example.com") else None
        )

        with (
            patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
            patch("sqlite3.connect") as mock_connect,
        ):
            mock_path.exists.return_value = True
            mock_path.__str__ = lambda x: "/path/to/AddressBook-v22.abcddb"

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
            john_rows = [r for r in sample_address_book_rows if r["first_name"] == "John"]
            mock_cursor.fetchall.return_value = [Row(row) for row in john_rows]
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            result = ingest_contacts(mock_db)

            assert result["processed"] == 1
            assert result["skipped"] == 1
            assert result["updated"] == 0
            assert result["created"] == 0

    def test_ingestion_handles_empty_name(self, mock_db) -> None:
        """Test that contacts with empty names are skipped."""
        empty_name_rows = [
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
            mock_path.__str__ = lambda x: "/path/to/AddressBook-v22.abcddb"

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
            mock_cursor.fetchall.return_value = [Row(row) for row in empty_name_rows]
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            result = ingest_contacts(mock_db)

            # Should process but skip empty names
            assert result["processed"] == 1
            assert result["created"] == 0
            mock_db.add_contact.assert_not_called()

    def test_ingestion_uses_org_name_when_no_personal_name(self, mock_db) -> None:
        """Test that organization name is used when no personal name."""
        org_rows = [
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
            mock_path.__str__ = lambda x: "/path/to/AddressBook-v22.abcddb"

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
            mock_cursor.fetchall.return_value = [Row(row) for row in org_rows]
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            result = ingest_contacts(mock_db)

            assert result["created"] == 1
            call_args = mock_db.add_contact.call_args[1]
            assert call_args["display_name"] == "Company Inc"

    def test_ingestion_groups_multiple_handles(self, mock_db) -> None:
        """Test that multiple handles for same person are grouped."""
        multi_handle_rows = [
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
            mock_path.__str__ = lambda x: "/path/to/AddressBook-v22.abcddb"

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
            mock_cursor.fetchall.return_value = [Row(row) for row in multi_handle_rows]
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            result = ingest_contacts(mock_db)

            assert result["processed"] == 1  # All grouped as one person
            assert result["created"] == 1
            call_args = mock_db.add_contact.call_args[1]
            # All handles should be included
            assert len(call_args["handles"]) == 3
            assert "+15551234567" in call_args["handles"]
            assert "john.doe@example.com" in call_args["handles"]
            assert "+15559876543" in call_args["handles"]

    def test_ingestion_handles_database_error(self, mock_db) -> None:
        """Test error handling when database query fails."""
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

    def test_ingestion_filters_invalid_identifiers(self, mock_db) -> None:
        """Test that empty identifiers are filtered, but non-digit strings pass through."""
        invalid_rows = [
            {
                "identifier": "abc",  # No digits but normalize returns "abc" (truthy)
                "first_name": "John",
                "last_name": "Doe",
                "org_name": None,
            },
            {
                "identifier": "",  # Empty -> normalize returns None (filtered)
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
            mock_path.__str__ = lambda x: "/path/to/AddressBook-v22.abcddb"

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
            mock_cursor.fetchall.return_value = [Row(row) for row in invalid_rows]
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_connect.return_value = mock_conn

            result = ingest_contacts(mock_db)

            # John Doe ("abc" is truthy) and Bob Jones are created; Jane Smith filtered (empty)
            assert result["created"] == 2
            assert mock_db.add_contact.call_count == 2

    def test_display_name_construction(self, mock_db) -> None:
        """Test various display name construction scenarios."""
        test_cases = [
            {
                "rows": [
                    {
                        "identifier": "+15551234567",
                        "first_name": "John",
                        "last_name": "Doe",
                        "org_name": None,
                    }
                ],
                "expected": "John Doe",
            },
            {
                "rows": [
                    {
                        "identifier": "+15551234567",
                        "first_name": "John",
                        "last_name": None,
                        "org_name": None,
                    }
                ],
                "expected": "John",
            },
            {
                "rows": [
                    {
                        "identifier": "+15551234567",
                        "first_name": None,
                        "last_name": "Doe",
                        "org_name": None,
                    }
                ],
                "expected": "Doe",
            },
            {
                "rows": [
                    {
                        "identifier": "info@company.com",
                        "first_name": None,
                        "last_name": None,
                        "org_name": "Company Inc",
                    }
                ],
                "expected": "Company Inc",
            },
        ]

        for case in test_cases:
            with (
                patch("jarvis.ingest.ADDRESS_BOOK_PATH") as mock_path,
                patch("sqlite3.connect") as mock_connect,
            ):
                mock_path.exists.return_value = True
                mock_path.__str__ = lambda x: "/path/to/AddressBook-v22.abcddb"

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
                mock_cursor.fetchall.return_value = [Row(row) for row in case["rows"]]
                mock_conn.execute.return_value = mock_cursor
                mock_conn.__enter__ = Mock(return_value=mock_conn)
                mock_conn.__exit__ = Mock(return_value=False)
                mock_connect.return_value = mock_conn

                ingest_contacts(mock_db)

                call_args = mock_db.add_contact.call_args[1]
                assert call_args["display_name"] == case["expected"]

                # Reset mock for next iteration
                mock_db.reset_mock()
