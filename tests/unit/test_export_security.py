"""Security tests for the export module.

Tests prevention of CSV injection vulnerabilities.
"""

import csv
import io
from datetime import datetime

import pytest

from contracts.imessage import Message
from jarvis.export import export_messages_csv

@pytest.fixture
def malicious_messages() -> list[Message]:
    """Create messages with malicious payloads."""
    return [
        Message(
            id=1,
            chat_id="safe_id",
            sender="attacker",
            sender_name="=cmd|' /C calc'!A0",
            text="=cmd|' /C calc'!A0",
            date=datetime(2024, 1, 15, 10, 30),
            is_from_me=False,
            attachments=[],
            reactions=[],
            reply_to_id=None,
            is_system_message=False,
        ),
        Message(
            id=2,
            chat_id="+malicious_chat_id",
            sender="-malicious_sender",
            sender_name="@malicious_name",
            text="+1+1",
            date=datetime(2024, 1, 15, 10, 31),
            is_from_me=False,
            attachments=[],
            reactions=[],
            reply_to_id=None,
            is_system_message=False,
        ),
    ]

class TestCsvInjectionPrevention:
    """Tests for CSV injection prevention."""

    def test_csv_injection_escaped(self, malicious_messages):
        """Malicious payloads are escaped with a leading single quote."""
        result = export_messages_csv(malicious_messages)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)

        # First message
        row1 = rows[0]
        assert row1["text"].startswith("'=")
        assert row1["sender_name"].startswith("'=")
        assert "'=cmd" in row1["text"]

        # Second message
        row2 = rows[1]
        assert row2["chat_id"].startswith("'+")
        assert row2["sender"].startswith("'-")
        assert row2["sender_name"].startswith("'@")
        assert row2["text"].startswith("'+")

    def test_safe_values_not_escaped(self):
        """Safe values are not escaped."""
        message = Message(
            id=1,
            chat_id="safe",
            sender="safe",
            sender_name="safe",
            text="safe",
            date=datetime.now(),
            is_from_me=False,
            attachments=[],
            reactions=[],
            reply_to_id=None,
            is_system_message=False,
        )
        result = export_messages_csv([message])
        reader = csv.DictReader(io.StringIO(result))
        row = next(reader)

        assert not row["text"].startswith("'")
        assert row["text"] == "safe"
