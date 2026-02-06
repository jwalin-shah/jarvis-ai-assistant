"""Data models, cache, and constants for JARVIS database."""

import json
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# TTL-enabled LRU Cache for query results
# ---------------------------------------------------------------------------


class TTLCache:
    """Thread-safe LRU cache with TTL expiration.

    Provides caching for frequently called database queries with automatic
    expiration to prevent stale data.
    """

    def __init__(self, maxsize: int = 128, ttl_seconds: float = 30.0) -> None:
        """Initialize cache.

        Args:
            maxsize: Maximum number of entries to cache.
            ttl_seconds: Time-to-live for cache entries in seconds.
        """
        self._cache: OrderedDict[Any, tuple[Any, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._lock = threading.RLock()

    def get(self, key: Any) -> tuple[bool, Any]:
        """Get value from cache.

        Args:
            key: Cache key.

        Returns:
            Tuple of (hit, value). hit is True if found and not expired.
        """
        with self._lock:
            if key not in self._cache:
                return (False, None)

            value, timestamp = self._cache[key]
            if time.time() - timestamp > self._ttl:
                # Expired - remove and return miss
                del self._cache[key]
                return (False, None)

            # Update access order for LRU (move to end)
            self._cache.move_to_end(key)
            return (True, value)

    def set(self, key: Any, value: Any) -> None:
        """Set value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)  # Remove oldest (FIFO)

            self._cache[key] = (value, time.time())
            self._cache.move_to_end(key)  # Move to end (newest)

    def invalidate(self, key: Any) -> None:
        """Remove a specific key from cache."""
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
            }


# Register custom timestamp converter that handles timezone-aware timestamps
def _convert_timestamp(val: bytes) -> datetime:
    """Convert timestamp bytes to datetime, handling timezone info."""
    datepart, timepart = val.split(b" ")
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
class Pair:
    """A (trigger, response) pair extracted from message history."""

    id: int | None
    contact_id: int | None
    trigger_text: str
    response_text: str
    trigger_timestamp: datetime
    response_timestamp: datetime
    chat_id: str
    # Message IDs for debugging/deduplication
    trigger_msg_id: int | None = None
    response_msg_id: int | None = None  # First response msg ID if multi-message
    trigger_msg_ids_json: str | None = None  # JSON array for multi-message triggers
    response_msg_ids_json: str | None = None  # JSON array for multi-message responses
    # Conversation context (messages leading up to the trigger)
    context_text: str | None = None  # Previous messages before trigger for LLM context
    # Quality and filtering
    quality_score: float = 1.0
    flags_json: str | None = None  # JSON: {"attachment_only":true, "short":true}
    is_group: bool = False  # True if from group chat
    is_holdout: bool = False  # True if reserved for evaluation (not in training index)
    # Validity gate results (v6+)
    gate_a_passed: bool | None = None  # Rule gate result
    gate_b_score: float | None = None  # Embedding similarity score
    gate_c_verdict: str | None = None  # NLI verdict (accept/reject/uncertain)
    validity_status: str | None = None  # Final: valid/invalid/uncertain
    # Dialogue act classification (v7+)
    trigger_da_type: str | None = None  # e.g., WH_QUESTION, INFO_STATEMENT
    trigger_da_conf: float | None = None  # Classifier confidence 0-1
    response_da_type: str | None = None  # e.g., STATEMENT, AGREE, ACKNOWLEDGE
    response_da_conf: float | None = None  # Classifier confidence 0-1
    # HDBSCAN cluster assignment
    cluster_id: int | None = None  # -1 for noise, else cluster number
    # Freshness and usage tracking
    usage_count: int = 0
    last_used_at: datetime | None = None
    last_verified_at: datetime | None = None
    source_timestamp: datetime | None = None  # For decay/freshness

    @property
    def flags(self) -> dict[str, Any]:
        """Get flags dict from JSON."""
        if self.flags_json:
            try:
                return json.loads(self.flags_json)
            except json.JSONDecodeError:
                return {}
        return {}

    @property
    def trigger_msg_ids(self) -> list[int]:
        """Get list of trigger message IDs."""
        if self.trigger_msg_ids_json:
            try:
                return json.loads(self.trigger_msg_ids_json)
            except json.JSONDecodeError:
                return []
        return [self.trigger_msg_id] if self.trigger_msg_id else []

    @property
    def response_msg_ids(self) -> list[int]:
        """Get list of response message IDs."""
        if self.response_msg_ids_json:
            try:
                return json.loads(self.response_msg_ids_json)
            except json.JSONDecodeError:
                return []
        return [self.response_msg_id] if self.response_msg_id else []


@dataclass
class PairArtifact:
    """Heavy artifacts for a pair (stored separately to keep pairs table lean).

    Stored in pair_artifacts table to avoid bloating the main pairs table.
    """

    pair_id: int
    context_json: str | None = None  # Structured context window (JSON list of messages)
    gate_a_reason: str | None = None  # Why Gate A rejected (if rejected)
    gate_c_scores_json: str | None = None  # Raw NLI scores (JSON dict)
    raw_trigger_text: str | None = None  # Original text before normalization
    raw_response_text: str | None = None  # Original text before normalization

    @property
    def context_messages(self) -> list[dict]:
        """Get context messages from JSON."""
        if self.context_json:
            try:
                return json.loads(self.context_json)
            except json.JSONDecodeError:
                return []
        return []

    @property
    def gate_c_scores(self) -> dict[str, float]:
        """Get Gate C scores from JSON."""
        if self.gate_c_scores_json:
            try:
                return json.loads(self.gate_c_scores_json)
            except json.JSONDecodeError:
                return {}
        return {}


@dataclass
class ContactStyleTargets:
    """Style targets for a contact (computed from their pairs)."""

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
class Cluster:
    """An intent cluster grouping similar responses."""

    id: int | None
    name: str
    description: str | None
    example_triggers: list[str] = field(default_factory=list)
    example_responses: list[str] = field(default_factory=list)
    created_at: datetime | None = None


@dataclass
class PairEmbedding:
    """Links a pair to its FAISS vector position."""

    pair_id: int  # PRIMARY KEY - stable reference
    faiss_id: int  # Position in FAISS index (can change on rebuild)
    cluster_id: int | None
    index_version: str | None = None  # Which index version this belongs to


@dataclass
class IndexVersion:
    """Metadata for a FAISS index version."""

    id: int | None
    version_id: str  # e.g., "20240115-143022"
    model_name: str
    embedding_dim: int
    num_vectors: int
    index_path: str
    is_active: bool
    created_at: datetime | None = None
