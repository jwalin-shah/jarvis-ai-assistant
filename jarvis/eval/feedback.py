"""Feedback Loop System - Track user acceptance/rejection of AI suggestions.

This module provides tracking for user feedback on AI-generated suggestions,
enabling analysis and improvement of suggestion quality over time.

Schema:
- feedback_id: Primary key (auto-increment)
- message_id: ID of the message the suggestion was for
- suggestion_id: Unique identifier for the suggestion
- action: User action (accepted/rejected/edited)
- timestamp: When the feedback was recorded
- contact_id: Optional link to contact for per-contact analytics
- original_suggestion: The AI-generated suggestion text
- edited_text: The user's edited text (when action is 'edited')

Usage:
    from jarvis.eval.feedback import get_feedback_store, FeedbackAction

    store = get_feedback_store()
    store.init_schema()

    # Record feedback with full context
    feedback = store.record_feedback(
        message_id="msg_123",
        suggestion_id="sug_456",
        action=FeedbackAction.EDITED,
        contact_id=1,
        original_suggestion="How about tomorrow?",
        edited_text="How about tomorrow at 3pm?",
    )

    # Query feedback
    feedback = store.get_feedback(feedback_id=1)
    all_feedback = store.list_feedback(limit=100)

    # Analytics
    daily_stats = store.get_stats_by_day(days=30)
    contact_stats = store.get_stats_by_contact()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, cast

from jarvis.errors import (
    ErrorCode,
    FeedbackError,
    FeedbackInvalidActionError,
)

logger = logging.getLogger(__name__)

# Default database path (same directory as main jarvis.db)
FEEDBACK_DB_PATH = Path.home() / ".jarvis" / "jarvis.db"

# Current schema version for migrations
FEEDBACK_SCHEMA_VERSION = 2


class FeedbackAction(str, Enum):
    """User actions on AI suggestions."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EDITED = "edited"


@dataclass
class Feedback:
    """A feedback record for a suggestion."""

    id: int | None
    message_id: str
    suggestion_id: str
    action: FeedbackAction
    timestamp: datetime
    metadata_json: str | None = None  # Optional JSON for extra context
    contact_id: int | None = None  # Link to contact for per-contact analytics
    original_suggestion: str | None = None  # The AI-generated suggestion
    edited_text: str | None = None  # User's edited text (when action is 'edited')

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "message_id": self.message_id,
            "suggestion_id": self.suggestion_id,
            "action": self.action.value,
            "timestamp": self.timestamp.isoformat(),
            "contact_id": self.contact_id,
            "original_suggestion": self.original_suggestion,
            "edited_text": self.edited_text,
        }


@dataclass
class DailyStats:
    """Daily feedback statistics."""

    date: str  # YYYY-MM-DD format
    total: int
    accepted: int
    rejected: int
    edited: int
    acceptance_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "date": self.date,
            "total": self.total,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "edited": self.edited,
            "acceptance_rate": self.acceptance_rate,
        }


@dataclass
class ContactStats:
    """Per-contact feedback statistics."""

    contact_id: int
    total: int
    accepted: int
    rejected: int
    edited: int
    acceptance_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "contact_id": self.contact_id,
            "total": self.total,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "edited": self.edited,
            "acceptance_rate": self.acceptance_rate,
        }


# Schema SQL for feedback table (v2 with enhanced fields)
FEEDBACK_SCHEMA_SQL = """
-- Feedback tracking for AI suggestions
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    message_id TEXT NOT NULL,
    suggestion_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('accepted', 'rejected', 'edited')),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT,
    contact_id INTEGER,
    original_suggestion TEXT,
    edited_text TEXT,
    UNIQUE(message_id, suggestion_id)
);

-- Schema version tracking for feedback
CREATE TABLE IF NOT EXISTS feedback_schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for efficient lookups
CREATE INDEX IF NOT EXISTS idx_feedback_message ON feedback(message_id);
CREATE INDEX IF NOT EXISTS idx_feedback_suggestion ON feedback(suggestion_id);
CREATE INDEX IF NOT EXISTS idx_feedback_action ON feedback(action);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_contact ON feedback(contact_id);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp_date ON feedback(DATE(timestamp));
"""

# Migration SQL for v1 -> v2
MIGRATION_V1_TO_V2 = """
-- Add new columns for v2
ALTER TABLE feedback ADD COLUMN contact_id INTEGER;
ALTER TABLE feedback ADD COLUMN original_suggestion TEXT;
ALTER TABLE feedback ADD COLUMN edited_text TEXT;

-- Add new indexes
CREATE INDEX IF NOT EXISTS idx_feedback_contact ON feedback(contact_id);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp_date ON feedback(DATE(timestamp));
"""


class FeedbackStore:
    """Manager for the feedback SQLite table.

    Thread-safe with per-thread connection reuse.

    TODO: This implementation should be unified with the JSONL-based
    FeedbackStore in jarvis/eval/evaluation.py.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the feedback store.

        Args:
            db_path: Path to the SQLite database file.
                    Defaults to ~/.jarvis/jarvis.db
        """
        self.db_path = db_path or FEEDBACK_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                timeout=30.0,
                check_same_thread=False,
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys and optimize
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            self._local.connection.execute("PRAGMA journal_mode = WAL")
            self._local.connection.execute("PRAGMA synchronous = NORMAL")
        return cast(sqlite3.Connection, self._local.connection)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection with automatic commit/rollback."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "connection") and self._local.connection is not None:
            self._local.connection.close()
            self._local.connection = None

    def exists(self) -> bool:
        """Check if the database file exists."""
        return self.db_path.exists()

    def _get_schema_version(self, conn: sqlite3.Connection) -> int:
        """Get current schema version."""
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='feedback_schema_version'"
            )
            if not cursor.fetchone():
                # Check if feedback table exists (v1)
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
                )
                if cursor.fetchone():
                    return 1  # v1 schema exists
                return 0  # No schema
            cursor = conn.execute("SELECT MAX(version) FROM feedback_schema_version")
            row = cursor.fetchone()
            return row[0] if row and row[0] else 0
        except sqlite3.Error:
            return 0

    def _run_migrations(self, conn: sqlite3.Connection, current_version: int) -> None:
        """Run schema migrations."""
        if current_version < 2:
            if current_version == 1:
                # Migrate from v1 to v2
                logger.info("Migrating feedback schema from v1 to v2")
                for statement in MIGRATION_V1_TO_V2.strip().split(";"):
                    statement = statement.strip()
                    if statement:
                        try:
                            conn.execute(statement)
                        except sqlite3.OperationalError as e:
                            # Only ignore duplicate column errors from re-runs
                            # Re-raise all other SQLite errors
                            error_msg = str(e).lower()
                            if "duplicate column" not in error_msg:
                                raise

            # Create version tracking table if needed
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.execute(
                "INSERT OR REPLACE INTO feedback_schema_version (version) VALUES (?)",
                (FEEDBACK_SCHEMA_VERSION,),
            )

    def init_schema(self) -> bool:
        """Initialize the feedback schema.

        Returns:
            True if schema was created or migrated, False if already current.
        """
        with self.connection() as conn:
            current_version = self._get_schema_version(conn)

            if current_version == FEEDBACK_SCHEMA_VERSION:
                logger.debug("Feedback schema already at version %d", current_version)
                return False

            if current_version == 0:
                # Fresh install - create full schema
                conn.executescript(FEEDBACK_SCHEMA_SQL)
                conn.execute(
                    "INSERT INTO feedback_schema_version (version) VALUES (?)",
                    (FEEDBACK_SCHEMA_VERSION,),
                )
                logger.info("Created feedback schema v%d", FEEDBACK_SCHEMA_VERSION)
                return True

            # Run migrations
            self._run_migrations(conn, current_version)
            logger.info(
                "Migrated feedback schema from v%d to v%d",
                current_version,
                FEEDBACK_SCHEMA_VERSION,
            )
            return True

    def record_feedback(
        self,
        message_id: str,
        suggestion_id: str,
        action: FeedbackAction | str,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        contact_id: int | None = None,
        original_suggestion: str | None = None,
        edited_text: str | None = None,
    ) -> Feedback:
        """Record user feedback on a suggestion.

        Args:
            message_id: ID of the message the suggestion was for.
            suggestion_id: Unique identifier for the suggestion.
            action: User action (accepted/rejected/edited).
            timestamp: When the feedback was recorded. Defaults to now.
            metadata: Optional metadata to store with the feedback.
            contact_id: Optional link to contact for per-contact analytics.
            original_suggestion: The AI-generated suggestion text.
            edited_text: The user's edited text (when action is 'edited').

        Returns:
            The created Feedback record.

        Raises:
            FeedbackInvalidActionError: If action is not valid.
            FeedbackError: If recording fails.
        """
        # Validate and convert action
        if isinstance(action, str):
            try:
                action = FeedbackAction(action)
            except ValueError:
                raise FeedbackInvalidActionError(
                    f"Invalid action: {action}. Must be one of: accepted, rejected, edited",
                    code=ErrorCode.FBK_INVALID_ACTION,
                    details={"action": action},
                )

        timestamp = timestamp or datetime.now(UTC)
        metadata_json = None
        if metadata:
            metadata_json = json.dumps(metadata)

        try:
            with self.connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO feedback
                        (message_id, suggestion_id, action, timestamp, metadata_json,
                         contact_id, original_suggestion, edited_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(message_id, suggestion_id) DO UPDATE SET
                        action = excluded.action,
                        timestamp = excluded.timestamp,
                        metadata_json = excluded.metadata_json,
                        contact_id = excluded.contact_id,
                        original_suggestion = excluded.original_suggestion,
                        edited_text = excluded.edited_text
                    """,
                    (
                        message_id,
                        suggestion_id,
                        action.value,
                        timestamp,
                        metadata_json,
                        contact_id,
                        original_suggestion,
                        edited_text,
                    ),
                )
                feedback_id = cursor.lastrowid

                return Feedback(
                    id=feedback_id,
                    message_id=message_id,
                    suggestion_id=suggestion_id,
                    action=action,
                    timestamp=timestamp,
                    metadata_json=metadata_json,
                    contact_id=contact_id,
                    original_suggestion=original_suggestion,
                    edited_text=edited_text,
                )
        except sqlite3.Error as e:
            raise FeedbackError(
                f"Failed to record feedback: {e}",
                suggestion_id=suggestion_id,
                cause=e,
            )

    def record_feedback_bulk(
        self,
        feedback_items: list[dict[str, Any]],
    ) -> int:
        """Record multiple feedback items in bulk.

        Args:
            feedback_items: List of dicts with keys matching record_feedback params.
                Required: message_id, suggestion_id, action
                Optional: timestamp, metadata, contact_id, original_suggestion, edited_text

        Returns:
            Number of feedback items recorded.

        Raises:
            FeedbackError: If bulk recording fails.
        """
        if not feedback_items:
            return 0

        recorded = 0
        try:
            with self.connection() as conn:
                for item in feedback_items:
                    action = item["action"]
                    if isinstance(action, str):
                        action = FeedbackAction(action)

                    timestamp = item.get("timestamp") or datetime.now(UTC)
                    metadata = item.get("metadata")
                    metadata_json = json.dumps(metadata) if metadata else None

                    conn.execute(
                        """
                        INSERT INTO feedback
                            (message_id, suggestion_id, action, timestamp, metadata_json,
                             contact_id, original_suggestion, edited_text)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(message_id, suggestion_id) DO UPDATE SET
                            action = excluded.action,
                            timestamp = excluded.timestamp,
                            metadata_json = excluded.metadata_json,
                            contact_id = excluded.contact_id,
                            original_suggestion = excluded.original_suggestion,
                            edited_text = excluded.edited_text
                        """,
                        (
                            item["message_id"],
                            item["suggestion_id"],
                            action.value,
                            timestamp,
                            metadata_json,
                            item.get("contact_id"),
                            item.get("original_suggestion"),
                            item.get("edited_text"),
                        ),
                    )
                    recorded += 1

            return recorded
        except sqlite3.Error as e:
            raise FeedbackError(
                f"Failed to record bulk feedback: {e}",
                cause=e,
            )

    def _row_to_feedback(self, row: sqlite3.Row) -> Feedback:
        """Convert a database row to a Feedback object."""
        return Feedback(
            id=row["id"],
            message_id=row["message_id"],
            suggestion_id=row["suggestion_id"],
            action=FeedbackAction(row["action"]),
            timestamp=row["timestamp"],
            metadata_json=row["metadata_json"],
            contact_id=row["contact_id"],
            original_suggestion=row["original_suggestion"],
            edited_text=row["edited_text"],
        )

    def get_feedback(self, feedback_id: int) -> Feedback | None:
        """Get a feedback record by ID.

        Args:
            feedback_id: The feedback record ID.

        Returns:
            The Feedback record, or None if not found.
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, message_id, suggestion_id, action, timestamp, metadata_json,
                       contact_id, original_suggestion, edited_text
                FROM feedback
                WHERE id = ?
                """,
                (feedback_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return self._row_to_feedback(row)

    def get_feedback_by_suggestion(self, suggestion_id: str) -> Feedback | None:
        """Get feedback for a specific suggestion.

        Args:
            suggestion_id: The suggestion ID.

        Returns:
            The Feedback record, or None if not found.
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, message_id, suggestion_id, action, timestamp, metadata_json,
                       contact_id, original_suggestion, edited_text
                FROM feedback
                WHERE suggestion_id = ?
                """,
                (suggestion_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return self._row_to_feedback(row)

    def get_feedback_by_message(self, message_id: str) -> list[Feedback]:
        """Get all feedback for a specific message.

        Args:
            message_id: The message ID.

        Returns:
            List of Feedback records.
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, message_id, suggestion_id, action, timestamp, metadata_json,
                       contact_id, original_suggestion, edited_text
                FROM feedback
                WHERE message_id = ?
                ORDER BY timestamp DESC
                """,
                (message_id,),
            )
            return [self._row_to_feedback(row) for row in cursor]

    def get_feedback_by_contact(
        self,
        contact_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Feedback]:
        """Get all feedback for a specific contact.

        Args:
            contact_id: The contact ID.
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            List of Feedback records.
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, message_id, suggestion_id, action, timestamp, metadata_json,
                       contact_id, original_suggestion, edited_text
                FROM feedback
                WHERE contact_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (contact_id, limit, offset),
            )
            return [self._row_to_feedback(row) for row in cursor]

    def list_feedback(
        self,
        action: FeedbackAction | str | None = None,
        contact_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Feedback]:
        """List feedback records with optional filtering.

        Args:
            action: Filter by action type.
            contact_id: Filter by contact ID.
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            List of Feedback records.
        """
        query = """
            SELECT id, message_id, suggestion_id, action, timestamp, metadata_json,
                   contact_id, original_suggestion, edited_text
            FROM feedback
        """
        params: list[Any] = []
        conditions: list[str] = []

        if action is not None:
            if isinstance(action, str):
                action = FeedbackAction(action)
            conditions.append("action = ?")
            params.append(action.value)

        if contact_id is not None:
            conditions.append("contact_id = ?")
            params.append(contact_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self.connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_feedback(row) for row in cursor]

    def list_feedback_in_range(
        self,
        start: datetime,
        end: datetime,
        action: FeedbackAction | str | None = None,
        contact_id: int | None = None,
    ) -> list[Feedback]:
        """List feedback records within a time range.

        Args:
            start: Start of time range (inclusive).
            end: End of time range (inclusive).
            action: Optional filter by action type.
            contact_id: Optional filter by contact ID.

        Returns:
            List of Feedback records.
        """
        query = """
            SELECT id, message_id, suggestion_id, action, timestamp, metadata_json,
                   contact_id, original_suggestion, edited_text
            FROM feedback
            WHERE timestamp >= ? AND timestamp <= ?
        """
        params: list[Any] = [start, end]

        if action is not None:
            if isinstance(action, str):
                action = FeedbackAction(action)
            query += " AND action = ?"
            params.append(action.value)

        if contact_id is not None:
            query += " AND contact_id = ?"
            params.append(contact_id)

        query += " ORDER BY timestamp DESC"

        with self.connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_feedback(row) for row in cursor]

    def count_feedback(
        self,
        action: FeedbackAction | str | None = None,
        contact_id: int | None = None,
    ) -> int:
        """Count feedback records with optional filtering.

        Args:
            action: Filter by action type.
            contact_id: Filter by contact ID.

        Returns:
            Count of feedback records.
        """
        query = "SELECT COUNT(*) FROM feedback"
        params: list[Any] = []
        conditions: list[str] = []

        if action is not None:
            if isinstance(action, str):
                action = FeedbackAction(action)
            conditions.append("action = ?")
            params.append(action.value)

        if contact_id is not None:
            conditions.append("contact_id = ?")
            params.append(contact_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with self.connection() as conn:
            cursor = conn.execute(query, params)
            return cast(int, cursor.fetchone()[0])

    def delete_feedback(self, feedback_id: int) -> bool:
        """Delete a feedback record.

        Args:
            feedback_id: The feedback record ID.

        Returns:
            True if deleted, False if not found.
        """
        with self.connection() as conn:
            cursor = conn.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
            return cursor.rowcount > 0

    def clear_feedback(self) -> int:
        """Delete all feedback records.

        Returns:
            Number of records deleted.
        """
        with self.connection() as conn:
            cursor = conn.execute("DELETE FROM feedback")
            return cursor.rowcount

    def get_stats(self) -> dict[str, Any]:
        """Get feedback statistics.

        Returns:
            Dictionary with feedback statistics.
        """
        with self.connection() as conn:
            # Total count
            total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

            # Count by action
            cursor = conn.execute(
                """
                SELECT action, COUNT(*) as count
                FROM feedback
                GROUP BY action
                """
            )
            by_action = {row["action"]: row["count"] for row in cursor}

            # Acceptance rate
            accepted = by_action.get("accepted", 0)
            acceptance_rate = accepted / total if total > 0 else 0.0

            return {
                "total": total,
                "accepted": by_action.get("accepted", 0),
                "rejected": by_action.get("rejected", 0),
                "edited": by_action.get("edited", 0),
                "acceptance_rate": acceptance_rate,
            }

    def get_stats_by_day(self, days: int = 30) -> list[DailyStats]:
        """Get daily feedback statistics for trending analysis.

        Args:
            days: Number of days to include (default 30).

        Returns:
            List of DailyStats, one per day with data, ordered by date descending.
        """
        start_date = datetime.now(UTC) - timedelta(days=days)

        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    DATE(timestamp) as date,
                    COUNT(*) as total,
                    SUM(CASE WHEN action = 'accepted' THEN 1 ELSE 0 END) as accepted,
                    SUM(CASE WHEN action = 'rejected' THEN 1 ELSE 0 END) as rejected,
                    SUM(CASE WHEN action = 'edited' THEN 1 ELSE 0 END) as edited
                FROM feedback
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY DATE(timestamp) DESC
                """,
                (start_date,),
            )

            results = []
            for row in cursor:
                total = row["total"]
                accepted = row["accepted"]
                acceptance_rate = accepted / total if total > 0 else 0.0
                results.append(
                    DailyStats(
                        date=row["date"],
                        total=total,
                        accepted=accepted,
                        rejected=row["rejected"],
                        edited=row["edited"],
                        acceptance_rate=acceptance_rate,
                    )
                )
            return results

    def get_stats_by_contact(self) -> list[ContactStats]:
        """Get per-contact feedback statistics.

        Returns:
            List of ContactStats for each contact with feedback.
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    contact_id,
                    COUNT(*) as total,
                    SUM(CASE WHEN action = 'accepted' THEN 1 ELSE 0 END) as accepted,
                    SUM(CASE WHEN action = 'rejected' THEN 1 ELSE 0 END) as rejected,
                    SUM(CASE WHEN action = 'edited' THEN 1 ELSE 0 END) as edited
                FROM feedback
                WHERE contact_id IS NOT NULL
                GROUP BY contact_id
                ORDER BY total DESC
                """
            )

            results = []
            for row in cursor:
                total = row["total"]
                accepted = row["accepted"]
                acceptance_rate = accepted / total if total > 0 else 0.0
                results.append(
                    ContactStats(
                        contact_id=row["contact_id"],
                        total=total,
                        accepted=accepted,
                        rejected=row["rejected"],
                        edited=row["edited"],
                        acceptance_rate=acceptance_rate,
                    )
                )
            return results


# ---------------------------------------------------------------------------
# Singleton pattern for FeedbackStore
# ---------------------------------------------------------------------------

_feedback_store: FeedbackStore | None = None
_feedback_store_lock = threading.Lock()


def get_feedback_store(db_path: Path | None = None) -> FeedbackStore:
    """Get the global FeedbackStore instance.

    Uses double-checked locking pattern for thread safety.

    Args:
        db_path: Optional path to database. Only used on first call.

    Returns:
        The global FeedbackStore instance.
    """
    global _feedback_store
    if _feedback_store is None:
        with _feedback_store_lock:
            if _feedback_store is None:
                _feedback_store = FeedbackStore(db_path)
    return _feedback_store


def reset_feedback_store() -> None:
    """Reset the global FeedbackStore instance.

    Useful for testing.
    """
    global _feedback_store
    if _feedback_store is not None:
        _feedback_store.close()
    _feedback_store = None


__all__ = [
    "FeedbackAction",
    "Feedback",
    "DailyStats",
    "ContactStats",
    "FeedbackStore",
    "get_feedback_store",
    "reset_feedback_store",
]
