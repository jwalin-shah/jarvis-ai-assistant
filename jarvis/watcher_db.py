"""DB helpers for chat watcher polling and schema validation."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from jarvis.utils.datetime_utils import parse_apple_timestamp

logger = logging.getLogger(__name__)


def validate_chat_db_schema(chat_db_path: Path) -> bool:
    """Validate minimal schema requirements for watcher queries."""
    if not chat_db_path.exists():
        logger.warning("chat.db not found, watcher cannot start")
        return False

    try:
        conn = sqlite3.connect(f"file:{chat_db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
            required_tables = {"message", "chat", "handle", "chat_message_join"}
            missing_tables = required_tables - tables
            if missing_tables:
                logger.error("chat.db missing required tables: %s", missing_tables)
                return False

            cursor.execute("PRAGMA table_info(message)")
            column_info = {row[1]: row[2].upper() for row in cursor.fetchall()}
            required_columns = {"text", "date", "is_from_me", "handle_id"}
            missing_columns = required_columns - set(column_info.keys())
            if missing_columns:
                logger.error("chat.db message table missing columns: %s", missing_columns)
                return False

            expected_types = {
                "text": {"TEXT", ""},
                "date": {"INTEGER", "REAL", ""},
                "is_from_me": {"INTEGER", "BOOLEAN", ""},
                "handle_id": {"INTEGER", ""},
            }
            for col_name, valid_types in expected_types.items():
                actual_type = column_info.get(col_name, "")
                if actual_type not in valid_types:
                    logger.error(
                        "chat.db message.%s has unexpected type '%s' (expected one of %s)",
                        col_name,
                        actual_type,
                        valid_types,
                    )
                    return False
            return True
        finally:
            conn.close()
    except Exception as e:
        logger.error("chat.db schema validation error: %s", e)
        return False


def query_last_rowid(conn: sqlite3.Connection) -> int | None:
    """Query latest message rowid using an existing connection."""
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(ROWID) FROM message")
    row = cursor.fetchone()
    return row[0] if row and row[0] else None


def query_new_messages(
    conn: sqlite3.Connection,
    since_rowid: int,
    *,
    limit: int = 500,
) -> list[dict[str, object]]:
    """Query new messages using an existing read-only connection."""
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            message.ROWID as id,
            chat.guid as chat_id,
            COALESCE(handle.id, 'me') as sender,
            message.text,
            message.date,
            message.is_from_me
        FROM message
        JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
        JOIN chat ON chat_message_join.chat_id = chat.ROWID
        LEFT JOIN handle ON message.handle_id = handle.ROWID
        WHERE message.ROWID > ?
        ORDER BY message.date ASC
        LIMIT ?
        """,
        (since_rowid, limit),
    )

    messages: list[dict[str, object]] = []
    for row in cursor.fetchall():
        date_iso = parse_apple_timestamp(row["date"]).isoformat() if row["date"] else None
        messages.append(
            {
                "id": row["id"],
                "chat_id": row["chat_id"],
                "sender": row["sender"],
                "text": row["text"],
                "date": date_iso,
                "is_from_me": bool(row["is_from_me"]),
            }
        )

    return messages
