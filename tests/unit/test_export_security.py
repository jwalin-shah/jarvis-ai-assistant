"""Security tests for export functionality."""

import csv
import io
from datetime import datetime
from contracts.imessage import Message
from jarvis.export import export_messages_csv

def test_csv_injection_prevention():
    """Test that CSV injection characters are escaped."""
    malicious_text = "=cmd|' /C calc'!A0"
    message = Message(
        id=1,
        chat_id="test_chat",
        sender="attacker",
        sender_name="Attacker",
        text=malicious_text,
        date=datetime.now(),
        is_from_me=False,
    )

    csv_output = export_messages_csv([message])

    # Parse the output
    reader = csv.DictReader(io.StringIO(csv_output))
    row = next(reader)

    # The text should be escaped with a leading single quote
    assert row["text"] == f"'{malicious_text}"

    # Verify other dangerous characters are also escaped
    dangerous_chars = ["+", "-", "@"]
    for char in dangerous_chars:
        message.text = f"{char}payload"
        csv_output = export_messages_csv([message])
        reader = csv.DictReader(io.StringIO(csv_output))
        row = next(reader)
        assert row["text"] == f"'{char}payload"

def test_csv_injection_prevention_other_fields():
    """Test that other fields are also sanitized."""
    message = Message(
        id=1,
        chat_id="=malicious_chat",
        sender="+1234567890",  # Phone numbers often start with +
        sender_name="@attacker",
        text="Normal text",
        date=datetime.now(),
        is_from_me=False,
    )

    csv_output = export_messages_csv([message])
    reader = csv.DictReader(io.StringIO(csv_output))
    row = next(reader)

    assert row["chat_id"] == "'=malicious_chat"
    assert row["sender"] == "'+1234567890"
    assert row["sender_name"] == "'@attacker"

def test_normal_text_not_escaped():
    """Test that normal text is not escaped."""
    normal_text = "Just a normal message"
    message = Message(
        id=1,
        chat_id="normal_chat",
        sender="1234567890",
        sender_name="Normal User",
        text=normal_text,
        date=datetime.now(),
        is_from_me=False,
    )

    csv_output = export_messages_csv([message])
    reader = csv.DictReader(io.StringIO(csv_output))
    row = next(reader)

    assert row["text"] == normal_text
    assert row["chat_id"] == "normal_chat"
    assert row["sender"] == "1234567890"
    assert row["sender_name"] == "Normal User"
