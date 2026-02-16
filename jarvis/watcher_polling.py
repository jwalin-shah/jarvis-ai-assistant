"""Polling connection/query helpers extracted from chat watcher."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Any, cast

from jarvis.watcher_db import query_last_rowid, query_new_messages

logger = logging.getLogger(__name__)


def get_poll_conn(watcher: Any, chat_db_path: Path) -> sqlite3.Connection | None:
    """Get or create persistent read-only connection for polling."""
    with watcher._poll_conn_lock:
        if watcher._poll_conn is not None:
            return cast(sqlite3.Connection, watcher._poll_conn)

        if not chat_db_path.exists():
            return None
        try:
            watcher._poll_conn = sqlite3.connect(
                f"file:{chat_db_path}?mode=ro",
                uri=True,
                timeout=5.0,
                check_same_thread=False,
            )
            return cast(sqlite3.Connection, watcher._poll_conn)
        except Exception as e:
            logger.debug("Error creating poll connection: %s", e)
            return None


def query_last_rowid_safe(watcher: Any, chat_db_path: Path) -> int | None:
    """Query the last message ROWID with stale-connection recovery."""
    conn = get_poll_conn(watcher, chat_db_path)
    if conn is None:
        return None
    try:
        return query_last_rowid(conn)
    except Exception as e:
        logger.debug("Error getting last ROWID: %s", e)
        with watcher._poll_conn_lock:
            try:
                if watcher._poll_conn:
                    watcher._poll_conn.close()
            except Exception as close_err:
                logger.debug("Error closing stale poll connection: %s", close_err)
            watcher._poll_conn = None
        return None


async def get_new_messages(watcher: Any) -> list[dict[str, Any]]:
    """Get messages newer than last known ROWID in batches."""
    if watcher._last_rowid is None:
        return []

    all_new_messages: list[dict[str, Any]] = []
    current_rowid = watcher._last_rowid

    while True:
        batch = await asyncio.to_thread(watcher._query_new_messages, current_rowid, limit=500)
        if not batch:
            break

        all_new_messages.extend(batch)
        if len(batch) < 500:
            break

        current_rowid = max(msg["id"] for msg in batch)

    return all_new_messages


def query_new_messages_safe(
    watcher: Any,
    chat_db_path: Path,
    since_rowid: int,
    *,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Query new messages with stale-connection recovery."""
    conn = get_poll_conn(watcher, chat_db_path)
    if conn is None:
        return []

    try:
        rows = query_new_messages(conn, since_rowid, limit=limit)
        return [dict(row) for row in rows]
    except Exception as e:
        logger.warning("Error querying new messages: %s", e)
        with watcher._poll_conn_lock:
            watcher._poll_conn = None
        return []
