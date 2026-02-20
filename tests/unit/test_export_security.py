"""Security tests for the export module.

Tests that export functionality handles malicious input safely.
"""

import csv
import io
from datetime import datetime

import pytest

from contracts.imessage import Message
from jarvis.export import export_messages_csv


class TestExportSecurity:
    """Tests for export security vulnerabilities."""

    def test_csv_formula_injection_prevention(self):
        """CSV export prevents formula injection in text field."""
        malicious_message = Message(
            id=1,
            chat_id="chat1",
            sender="user",
            sender_name="User",
            text="=cmd|' /C calc'!A0",
            date=datetime.now(),
            is_from_me=False,
        )

        result = export_messages_csv([malicious_message])
        reader = csv.DictReader(io.StringIO(result))
        row = next(reader)

        # The text should be escaped with a single quote
        assert row["text"].startswith("'")
        assert row["text"] == "'=cmd|' /C calc'!A0"

    def test_csv_injection_sender_fields(self):
        """CSV export prevents injection in sender fields."""
        malicious_message = Message(
            id=1,
            chat_id="@malicious",
            sender="+1234567890",
            sender_name="-BadGuy",
            text="Normal text",
            date=datetime.now(),
            is_from_me=False,
        )

        result = export_messages_csv([malicious_message])
        reader = csv.DictReader(io.StringIO(result))
        row = next(reader)

        # Fields starting with dangerous chars should be escaped
        assert row["chat_id"].startswith("'@")
        assert row["sender_name"].startswith("'-")

        # Safe fields should not be modified (other than normal CSV escaping if needed)
        # Note: Phone numbers starting with + are also escaped to preserve them as text
        # and prevent formula interpretation, which is desired behavior.
        assert row["sender"] == "'+1234567890"

    def test_csv_injection_plus_minus(self):
        """CSV export escapes + and - at the start of fields."""
        malicious_message = Message(
            id=1,
            chat_id="chat1",
            sender="+1234567890",
            sender_name="User",
            text="-100",
            date=datetime.now(),
            is_from_me=False,
        )

        result = export_messages_csv([malicious_message])
        reader = csv.DictReader(io.StringIO(result))
        row = next(reader)

        assert row["sender"].startswith("'+")
        assert row["text"].startswith("'-")
