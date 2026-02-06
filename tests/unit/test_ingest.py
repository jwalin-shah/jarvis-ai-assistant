"""Tests for jarvis/search/ingest.py - Contact ingestion from macOS Address Book."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from jarvis.search.ingest import (
    ingest_contacts,
    normalize_phone_number,
)


class TestNormalizePhoneNumber:
    """Tests for normalize_phone_number function."""

    def test_phone_with_plus(self) -> None:
        assert normalize_phone_number("+1 (555) 123-4567") == "+15551234567"
        assert normalize_phone_number("+44 20 7946 0958") == "+442079460958"

    def test_phone_without_plus(self) -> None:
        assert normalize_phone_number("(555) 123-4567") == "+15551234567"
        assert normalize_phone_number("555-123-4567") == "+15551234567"
        assert normalize_phone_number("555.123.4567") == "+15551234567"

    def test_email_address(self) -> None:
        assert normalize_phone_number("John.Doe@Example.COM") == "John.Doe@Example.COM"
        assert normalize_phone_number("test@test.com") == "test@test.com"
        assert normalize_phone_number("  Test@Test.com  ") == "Test@Test.com"

    def test_empty_input(self) -> None:
        assert normalize_phone_number("") is None
        assert normalize_phone_number(None) is None  # type: ignore
        assert normalize_phone_number("   ") is None

    def test_no_digits(self) -> None:
        assert normalize_phone_number("abc") == "abc"
        assert normalize_phone_number("---") == ""

    def test_phone_with_spaces(self) -> None:
        assert normalize_phone_number("555 123 4567") == "+15551234567"
        assert normalize_phone_number("+1 555 123 4567") == "+15551234567"

    def test_phone_with_extensions(self) -> None:
        assert normalize_phone_number("555-123-4567 ext 123") == "5551234567ext123"
        assert normalize_phone_number("555-123-4567 x123") == "5551234567x123"


def _row(identifier: str, first: str | None, last: str | None, org: str | None = None) -> dict:
    """Helper to create a mock Address Book row."""
    return {
        "identifier": identifier,
        "first_name": first,
        "last_name": last,
        "org_name": org,
    }


class TestIngestContacts:
    """Tests for ingest_contacts function."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.get_contact_by_handle = Mock(return_value=None)
        db.add_contact = Mock()
        return db

    def test_no_contacts_found(self, mock_db) -> None:
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=[]):
            result = ingest_contacts(mock_db)
            assert "error" in result
            mock_db.add_contact.assert_not_called()

    def test_successful_ingestion_new_contacts(self, mock_db) -> None:
        rows = [
            _row("+15551234567", "John", "Doe"),
            _row("john@example.com", "John", "Doe"),
            _row("+15559876543", "Jane", "Smith"),
            _row("info@company.com", None, None, "Company Inc"),
        ]
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=rows):
            result = ingest_contacts(mock_db)

            assert result["processed"] == 3  # John (grouped), Jane, Company
            assert result["created"] == 3
            assert mock_db.add_contact.call_count == 3

    def test_ingestion_updates_existing_phone_name_contact(self, mock_db) -> None:
        existing = MagicMock()
        existing.display_name = "+15551234567"
        existing.chat_id = "+15551234567"
        existing.phone_or_email = "+15551234567"
        existing.relationship = None
        existing.style_notes = None

        mock_db.get_contact_by_handle.side_effect = lambda h: (
            existing if h == "+15551234567" else None
        )

        rows = [_row("+1 (555) 123-4567", "John", "Doe")]
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=rows):
            result = ingest_contacts(mock_db)

            assert result["updated"] == 1
            assert result["created"] == 0
            assert mock_db.add_contact.call_args[1]["display_name"] == "John Doe"

    def test_ingestion_skips_unchanged_contact(self, mock_db) -> None:
        existing = MagicMock()
        existing.display_name = "John Doe"
        existing.chat_id = "chat123"
        existing.phone_or_email = "+15551234567"

        mock_db.get_contact_by_handle.side_effect = lambda h: (
            existing if h == "+15551234567" else None
        )

        rows = [_row("+15551234567", "John", "Doe")]
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=rows):
            result = ingest_contacts(mock_db)

            assert result["skipped"] == 1
            assert result["updated"] == 0

    def test_ingestion_skips_empty_name(self, mock_db) -> None:
        rows = [_row("+15551234567", None, None)]
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=rows):
            ingest_contacts(mock_db)
            mock_db.add_contact.assert_not_called()

    def test_ingestion_uses_org_name(self, mock_db) -> None:
        rows = [_row("info@company.com", None, None, "Company Inc")]
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=rows):
            result = ingest_contacts(mock_db)
            assert result["created"] == 1
            assert mock_db.add_contact.call_args[1]["display_name"] == "Company Inc"

    def test_display_name_construction(self, mock_db) -> None:
        cases = [
            (_row("+15551234567", "John", "Doe"), "John Doe"),
            (_row("+15551234567", "John", None), "John"),
            (_row("+15551234567", None, "Doe"), "Doe"),
            (_row("info@co.com", None, None, "Company"), "Company"),
        ]

        for row_data, expected_name in cases:
            mock_db.reset_mock()
            mock_db.get_contact_by_handle = Mock(return_value=None)
            with patch("jarvis.search.ingest._read_all_source_dbs", return_value=[row_data]):
                ingest_contacts(mock_db)
                assert mock_db.add_contact.call_args[1]["display_name"] == expected_name
