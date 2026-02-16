"""Window-level progress tracking for fact extraction backfill.

Allows resuming from the last processed window instead of restarting entire contacts.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WindowProgress:
    """Progress record for a single extraction window."""

    contact_id: str
    window_number: int
    processed_at: str
    facts_found: int
    messages_in_window: int


def init_window_progress_table(conn: sqlite3.Connection) -> None:
    """Create the window progress tracking table if not exists."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_extraction_window_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contact_id TEXT NOT NULL,
            window_number INTEGER NOT NULL,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            facts_found INTEGER DEFAULT 0,
            messages_in_window INTEGER DEFAULT 0,
            UNIQUE(contact_id, window_number)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_window_progress_contact
        ON fact_extraction_window_progress(contact_id)
        """
    )


def record_window_progress(
    conn: sqlite3.Connection,
    contact_id: str,
    window_number: int,
    facts_found: int = 0,
    messages_in_window: int = 0,
) -> None:
    """Record that a window has been processed.

    Args:
        conn: Database connection
        contact_id: Contact/chat ID
        window_number: Window number (1-indexed)
        facts_found: Number of facts extracted from this window
        messages_in_window: Number of messages in this window
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO fact_extraction_window_progress
        (contact_id, window_number, processed_at, facts_found, messages_in_window)
        VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?)
        """,
        (contact_id, window_number, facts_found, messages_in_window),
    )


def get_completed_windows(conn: sqlite3.Connection, contact_id: str) -> set[int]:
    """Get the set of window numbers already processed for a contact.

    Args:
        conn: Database connection
        contact_id: Contact/chat ID

    Returns:
        Set of completed window numbers
    """
    cursor = conn.execute(
        """
        SELECT window_number FROM fact_extraction_window_progress
        WHERE contact_id = ?
        """,
        (contact_id,),
    )
    return {row[0] for row in cursor.fetchall()}


def get_last_completed_window(conn: sqlite3.Connection, contact_id: str) -> int:
    """Get the highest window number completed for a contact.

    Args:
        conn: Database connection
        contact_id: Contact/chat ID

    Returns:
        Last completed window number (0 if none)
    """
    cursor = conn.execute(
        """
        SELECT MAX(window_number) FROM fact_extraction_window_progress
        WHERE contact_id = ?
        """,
        (contact_id,),
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else 0


def get_progress_summary(conn: sqlite3.Connection, contact_id: str) -> dict[str, Any]:
    """Get summary statistics for a contact's extraction progress.

    Args:
        conn: Database connection
        contact_id: Contact/chat ID

    Returns:
        Dict with total_windows, completed_windows, total_facts, last_window_at
    """
    cursor = conn.execute(
        """
        SELECT
            COUNT(*) as completed_windows,
            SUM(facts_found) as total_facts,
            MAX(window_number) as last_window,
            MAX(processed_at) as last_processed_at
        FROM fact_extraction_window_progress
        WHERE contact_id = ?
        """,
        (contact_id,),
    )
    row = cursor.fetchone()
    return {
        "completed_windows": row[0] or 0,
        "total_facts": row[1] or 0,
        "last_window": row[2] or 0,
        "last_processed_at": row[3],
    }


def clear_window_progress(conn: sqlite3.Connection, contact_id: str) -> int:
    """Clear progress for a contact (e.g., when re-processing with --force).

    Args:
        conn: Database connection
        contact_id: Contact/chat ID

    Returns:
        Number of progress records deleted
    """
    cursor = conn.execute(
        "DELETE FROM fact_extraction_window_progress WHERE contact_id = ?",
        (contact_id,),
    )
    return cursor.rowcount


def get_contacts_with_partial_progress(
    conn: sqlite3.Connection,
    min_windows: int = 1,
) -> list[tuple[str, int, int]]:
    """Get contacts that have partial window progress but aren't marked complete.

    Args:
        conn: Database connection
        min_windows: Minimum windows to consider

    Returns:
        List of (contact_id, windows_completed, total_facts)
    """
    cursor = conn.execute(
        """
        SELECT
            wp.contact_id,
            COUNT(wp.window_number) as windows_done,
            SUM(wp.facts_found) as facts_found
        FROM fact_extraction_window_progress wp
        LEFT JOIN contacts c ON wp.contact_id = c.chat_id
        WHERE c.last_extracted_rowid IS NULL
        GROUP BY wp.contact_id
        HAVING COUNT(wp.window_number) >= ?
        ORDER BY COUNT(wp.window_number) DESC
        """,
        (min_windows,),
    )
    return [(row[0], row[1], row[2]) for row in cursor.fetchall()]
