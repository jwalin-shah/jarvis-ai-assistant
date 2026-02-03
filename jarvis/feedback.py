"""Feedback Loop System - Track user acceptance/rejection of AI suggestions.

This module provides tracking for user feedback on AI-generated suggestions,
enabling analysis and improvement of suggestion quality over time.

Schema:
- feedback_id: Primary key (auto-increment)
- message_id: ID of the message the suggestion was for
- suggestion_id: Unique identifier for the suggestion
- action: User action (accepted/rejected/edited)
- timestamp: When the feedback was recorded

Usage:
    from jarvis.feedback import get_feedback_store, FeedbackAction

    store = get_feedback_store()
    store.init_schema()

    # Record feedback
    feedback = store.record_feedback(
        message_id="msg_123",
        suggestion_id="sug_456",
        action=FeedbackAction.ACCEPTED,
    )

    # Query feedback
    feedback = store.get_feedback(feedback_id=1)
    all_feedback = store.list_feedback(limit=100)
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "message_id": self.message_id,
            "suggestion_id": self.suggestion_id,
            "action": self.action.value,
            "timestamp": self.timestamp.isoformat(),
        }


# Schema SQL for feedback table
FEEDBACK_SCHEMA_SQL = """
-- Feedback tracking for AI suggestions
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    message_id TEXT NOT NULL,
    suggestion_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('accepted', 'rejected', 'edited')),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT,
    UNIQUE(message_id, suggestion_id)
);

-- Index for efficient lookups
CREATE INDEX IF NOT EXISTS idx_feedback_message ON feedback(message_id);
CREATE INDEX IF NOT EXISTS idx_feedback_suggestion ON feedback(suggestion_id);
CREATE INDEX IF NOT EXISTS idx_feedback_action ON feedback(action);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp DESC);
"""


class FeedbackStore:
    """Manager for the feedback SQLite table.

    Thread-safe with per-thread connection reuse.
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

    def init_schema(self) -> bool:
        """Initialize the feedback schema.

        Returns:
            True if schema was created, False if already exists.
        """
        with self.connection() as conn:
            # Check if feedback table already exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
            )
            if cursor.fetchone():
                logger.debug("Feedback table already exists")
                return False

            # Create schema
            conn.executescript(FEEDBACK_SCHEMA_SQL)
            logger.info("Created feedback schema")
            return True

    def record_feedback(
        self,
        message_id: str,
        suggestion_id: str,
        action: FeedbackAction | str,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Feedback:
        """Record user feedback on a suggestion.

        Args:
            message_id: ID of the message the suggestion was for.
            suggestion_id: Unique identifier for the suggestion.
            action: User action (accepted/rejected/edited).
            timestamp: When the feedback was recorded. Defaults to now.
            metadata: Optional metadata to store with the feedback.

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

        timestamp = timestamp or datetime.now()
        metadata_json = None
        if metadata:
            import json

            metadata_json = json.dumps(metadata)

        try:
            with self.connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO feedback
                        (message_id, suggestion_id, action, timestamp, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(message_id, suggestion_id) DO UPDATE SET
                        action = excluded.action,
                        timestamp = excluded.timestamp,
                        metadata_json = excluded.metadata_json
                    """,
                    (message_id, suggestion_id, action.value, timestamp, metadata_json),
                )
                feedback_id = cursor.lastrowid

                return Feedback(
                    id=feedback_id,
                    message_id=message_id,
                    suggestion_id=suggestion_id,
                    action=action,
                    timestamp=timestamp,
                    metadata_json=metadata_json,
                )
        except sqlite3.Error as e:
            raise FeedbackError(
                f"Failed to record feedback: {e}",
                suggestion_id=suggestion_id,
                cause=e,
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
                SELECT id, message_id, suggestion_id, action, timestamp, metadata_json
                FROM feedback
                WHERE id = ?
                """,
                (feedback_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return Feedback(
                id=row["id"],
                message_id=row["message_id"],
                suggestion_id=row["suggestion_id"],
                action=FeedbackAction(row["action"]),
                timestamp=row["timestamp"],
                metadata_json=row["metadata_json"],
            )

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
                SELECT id, message_id, suggestion_id, action, timestamp, metadata_json
                FROM feedback
                WHERE suggestion_id = ?
                """,
                (suggestion_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return Feedback(
                id=row["id"],
                message_id=row["message_id"],
                suggestion_id=row["suggestion_id"],
                action=FeedbackAction(row["action"]),
                timestamp=row["timestamp"],
                metadata_json=row["metadata_json"],
            )

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
                SELECT id, message_id, suggestion_id, action, timestamp, metadata_json
                FROM feedback
                WHERE message_id = ?
                ORDER BY timestamp DESC
                """,
                (message_id,),
            )
            return [
                Feedback(
                    id=row["id"],
                    message_id=row["message_id"],
                    suggestion_id=row["suggestion_id"],
                    action=FeedbackAction(row["action"]),
                    timestamp=row["timestamp"],
                    metadata_json=row["metadata_json"],
                )
                for row in cursor
            ]

    def list_feedback(
        self,
        action: FeedbackAction | str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Feedback]:
        """List feedback records with optional filtering.

        Args:
            action: Filter by action type.
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            List of Feedback records.
        """
        query = """
            SELECT id, message_id, suggestion_id, action, timestamp, metadata_json
            FROM feedback
        """
        params: list[Any] = []

        if action is not None:
            if isinstance(action, str):
                action = FeedbackAction(action)
            query += " WHERE action = ?"
            params.append(action.value)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self.connection() as conn:
            cursor = conn.execute(query, params)
            return [
                Feedback(
                    id=row["id"],
                    message_id=row["message_id"],
                    suggestion_id=row["suggestion_id"],
                    action=FeedbackAction(row["action"]),
                    timestamp=row["timestamp"],
                    metadata_json=row["metadata_json"],
                )
                for row in cursor
            ]

    def count_feedback(self, action: FeedbackAction | str | None = None) -> int:
        """Count feedback records with optional filtering.

        Args:
            action: Filter by action type.

        Returns:
            Count of feedback records.
        """
        query = "SELECT COUNT(*) FROM feedback"
        params: list[Any] = []

        if action is not None:
            if isinstance(action, str):
                action = FeedbackAction(action)
            query += " WHERE action = ?"
            params.append(action.value)

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


# ---------------------------------------------------------------------------
# Singleton pattern for FeedbackStore
# ---------------------------------------------------------------------------

_feedback_store: FeedbackStore | None = None


def get_feedback_store(db_path: Path | None = None) -> FeedbackStore:
    """Get the global FeedbackStore instance.

    Args:
        db_path: Optional path to database. Only used on first call.

    Returns:
        The global FeedbackStore instance.
    """
    global _feedback_store
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
    "FeedbackStore",
    "get_feedback_store",
    "reset_feedback_store",
]
