"""Comprehensive tests for contact ingestion from AddressBook source DBs."""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from jarvis.db import Contact
from jarvis.search.ingest import ingest_contacts


def _row(identifier: str, first: str | None, last: str | None, org: str | None = None) -> dict:
    """Helper to create a mock Address Book row."""
    return {
        "identifier": identifier,
        "first_name": first,
        "last_name": last,
        "org_name": org,
    }


class TestIngestContactsSuccessCases:
    @pytest.fixture
    def mock_db(self) -> MagicMock:
        db = MagicMock()
        db.get_contact_by_handles = Mock(return_value=None)
        db.add_contact = Mock()
        return db

    def test_ingest_new_contacts(self, mock_db) -> None:
        rows = [
            _row("+15551234567", "John", "Doe"),
            _row("john@example.com", "John", "Doe"),
        ]
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=rows):
            result = ingest_contacts(mock_db)

            assert result["processed"] == 1  # Grouped as one person
            assert result["created"] == 1
            assert "error" not in result

    def test_ingest_updates_phone_name_contact(self, mock_db) -> None:
        existing = Contact(
            id=1,
            chat_id="+15551234567",
            display_name="+15551234567",
            phone_or_email="+15551234567",
            relationship=None,
            style_notes=None,
            handles_json=json.dumps(["+15551234567"]),
        )

        mock_db.get_contact_by_handles.side_effect = lambda handles: (
            existing if "+15551234567" in handles else None
        )

        rows = [_row("+1 (555) 123-4567", "John", "Doe")]
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=rows):
            result = ingest_contacts(mock_db)

            assert result["updated"] == 1
            assert mock_db.add_contact.call_args[1]["display_name"] == "John Doe"

    def test_ingest_uses_org_name(self, mock_db) -> None:
        rows = [_row("info@company.com", None, None, "Company Inc")]
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=rows):
            result = ingest_contacts(mock_db)
            assert result["created"] == 1
            assert mock_db.add_contact.call_args[1]["display_name"] == "Company Inc"


class TestIngestContactsEdgeCases:
    @pytest.fixture
    def mock_db(self) -> MagicMock:
        db = MagicMock()
        db.get_contact_by_handles = Mock(return_value=None)
        db.add_contact = Mock()
        return db

    def test_no_contacts_returns_error(self, mock_db) -> None:
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=[]):
            result = ingest_contacts(mock_db)
            assert "error" in result

    def test_empty_names_skipped(self, mock_db) -> None:
        rows = [_row("+15551234567", None, None)]
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=rows):
            ingest_contacts(mock_db)
            mock_db.add_contact.assert_not_called()

    def test_empty_identifiers_filtered(self, mock_db) -> None:
        rows = [_row("", "John", "Doe")]
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=rows):
            ingest_contacts(mock_db)
            mock_db.add_contact.assert_not_called()

    def test_only_first_name(self, mock_db) -> None:
        rows = [_row("+15551234567", "John", None)]
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=rows):
            result = ingest_contacts(mock_db)
            assert result["created"] == 1
            assert mock_db.add_contact.call_args[1]["display_name"] == "John"

    def test_only_last_name(self, mock_db) -> None:
        rows = [_row("+15551234567", None, "Doe")]
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=rows):
            result = ingest_contacts(mock_db)
            assert result["created"] == 1
            assert mock_db.add_contact.call_args[1]["display_name"] == "Doe"

    def test_skips_unchanged_real_name(self, mock_db) -> None:
        existing = MagicMock()
        existing.display_name = "John Doe"
        mock_db.get_contact_by_handles.side_effect = lambda handles: (
            existing if "+15551234567" in handles else None
        )
        rows = [_row("+15551234567", "John", "Doe")]
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=rows):
            result = ingest_contacts(mock_db)
            assert result["skipped"] == 1

    def test_handles_grouped_by_name(self, mock_db) -> None:
        rows = [
            _row("+15551234567", "John", "Doe"),
            _row("john@example.com", "John", "Doe"),
            _row("+15559876543", "John", "Doe"),
        ]
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=rows):
            result = ingest_contacts(mock_db)
            assert result["processed"] == 1
            assert result["created"] == 1


class TestIngestContactsIntegration:
    @pytest.fixture
    def mock_db(self) -> MagicMock:
        db = MagicMock()
        db.get_contact_by_handles = Mock(return_value=None)
        db.add_contact = Mock()
        return db

    def test_large_dataset(self, mock_db) -> None:
        rows = [_row(f"+1555123{i:04d}", f"Person{i}", "Test") for i in range(1000)]
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=rows):
            result = ingest_contacts(mock_db)
            assert result["processed"] == 1000
            assert result["created"] == 1000

    def test_mixed_email_and_phone(self, mock_db) -> None:
        rows = [
            _row("+15551234567", "John", "Doe"),
            _row("john@example.com", "John", "Doe"),
            _row("jane@example.com", "Jane", "Smith"),
        ]
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=rows):
            result = ingest_contacts(mock_db)
            assert result["processed"] == 2
            assert result["created"] == 2

    def test_stats_accuracy(self, mock_db) -> None:
        existing = Contact(
            id=1,
            chat_id="+15559876543",
            display_name="Jane Smith",
            phone_or_email="+15559876543",
            relationship=None,
            style_notes=None,
            handles_json=json.dumps(["+15559876543"]),
        )

        mock_db.get_contact_by_handles.side_effect = lambda handles: (
            existing if "+15559876543" in handles else None
        )

        rows = [
            _row("+15551234567", "John", "Doe"),
            _row("+15559876543", "Jane", "Smith"),
        ]
        with patch("jarvis.search.ingest._read_all_source_dbs", return_value=rows):
            result = ingest_contacts(mock_db)

            assert result["processed"] == 2
            assert result["created"] == 1
            assert result["skipped"] == 1
