"""Data models and constants for JARVIS database."""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# Register custom timestamp converter that handles timezone-aware timestamps
def _convert_timestamp(val: bytes) -> datetime:
    """Convert timestamp bytes to datetime, handling timezone info."""
    # Handle both space separator ("2025-02-11 14:30:00") and ISO T separator
    # ("2025-02-11T14:30:00") — the latter is produced by datetime.isoformat()
    # and stored by FactExtractor for valid_from/valid_until fields.
    if b"T" in val:
        datepart, timepart = val.split(b"T", 1)
    else:
        datepart, timepart = val.split(b" ", 1)
    year, month, day = (int(x) for x in datepart.split(b"-"))

    # Handle timezone offset (e.g., "14:30:00.123456+00:00")
    if b"+" in timepart:
        timepart, tz_offset = timepart.rsplit(b"+", 1)
    elif timepart.count(b"-") == 1:
        timepart, tz_offset = timepart.rsplit(b"-", 1)
    # Note: timezone offset is intentionally discarded; JARVIS uses naive
    # datetimes in local time throughout, consistent with iMessage's storage.

    # Handle microseconds
    if b"." in timepart:
        timepart, microseconds = timepart.split(b".")
        microseconds = int(microseconds)
    else:
        microseconds = 0

    hours, minutes, seconds = (int(x) for x in timepart.split(b":"))

    return datetime(year, month, day, hours, minutes, seconds, microseconds)


# Register the custom converter
sqlite3.register_converter("TIMESTAMP", _convert_timestamp)

# Default database path
JARVIS_DB_PATH = Path.home() / ".jarvis" / "jarvis.db"
INDEXES_DIR = Path.home() / ".jarvis" / "indexes"


@dataclass
class Contact:
    """Contact with relationship metadata."""

    id: int | None
    chat_id: str | None
    display_name: str
    phone_or_email: str | None
    relationship: str | None
    style_notes: str | None
    handles_json: str | None = None  # JSON array of handles ["phone", "email"]
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def handles(self) -> list[str]:
        """Get list of handles from JSON."""
        if self.handles_json:
            try:
                return json.loads(self.handles_json)
            except json.JSONDecodeError:
                return []
        return []


@dataclass
class ContactStyleTargets:
    """Style targets for a contact (computed from conversation history)."""

    contact_id: int
    median_reply_length: int = 10  # Median word count
    punctuation_rate: float = 0.5  # Fraction with ending punctuation
    emoji_rate: float = 0.1  # Fraction containing emojis
    greeting_rate: float = 0.2  # Fraction starting with greeting
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contact_id": self.contact_id,
            "median_reply_length": self.median_reply_length,
            "punctuation_rate": self.punctuation_rate,
            "emoji_rate": self.emoji_rate,
            "greeting_rate": self.greeting_rate,
        }


@dataclass
class IndexVersion:
    """Metadata for a vector index version."""

    id: int | None
    version_id: str  # e.g., "20240115-143022"
    model_name: str
    embedding_dim: int
    num_vectors: int
    index_path: str
    is_active: bool
    created_at: datetime | None = None

