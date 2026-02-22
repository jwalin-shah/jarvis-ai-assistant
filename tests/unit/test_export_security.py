"""Security tests for the export module.

Tests that export functionality handles malicious input safely.
"""

import csv
import io
from datetime import datetime

from jarvis.contracts.imessage import Message
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

        assert row["chat_id"].startswith("'@")
        assert row["sender_name"].startswith("'-")

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

    def test_csv_normal_text_unchanged(self):
        """CSV export does not modify normal text."""
        normal_message = Message(
            id=1,
            chat_id="chat1",
            sender="user@example.com",
            sender_name="Normal User",
            text="Hello, this is a normal message!",
            date=datetime.now(),
            is_from_me=False,
        )

        result = export_messages_csv([normal_message])
        reader = csv.DictReader(io.StringIO(result))
        row = next(reader)

        assert row["chat_id"] == "chat1"
        assert row["sender"] == "user@example.com"
        assert row["sender_name"] == "Normal User"
        assert row["text"] == "Hello, this is a normal message!"

    def test_csv_attachment_filenames_sanitized(self):
        """CSV export sanitizes attachment filenames."""
        malicious_message = Message(
            id=1,
            chat_id="chat1",
            sender="user",
            sender_name="User",
            text="Normal text",
            date=datetime.now(),
            is_from_me=False,
            attachments=[type("Attachment", (), {"filename": '=HYPERLINK("http://evil.com")'})()],
        )

        result = export_messages_csv([malicious_message], include_attachments=True)
        reader = csv.DictReader(io.StringIO(result))
        row = next(reader)

        assert row["attachment_filenames"].startswith("'=")
