"""Chat.db file watcher for real-time message detection.

Uses watchfiles for efficient file change detection on macOS.
When chat.db changes, queries for new messages and broadcasts
notifications via the socket server.
"""

import asyncio
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Path to iMessage database
CHAT_DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"

# Minimum interval between checks (debounce)
DEBOUNCE_INTERVAL = 0.5  # seconds

# Apple epoch offset (2001-01-01 in Unix timestamp)
APPLE_EPOCH_OFFSET = 978307200


class BroadcastHandler(Protocol):
    """Protocol for broadcast handlers."""

    async def broadcast(self, method: str, params: dict[str, Any]) -> None:
        """Broadcast a notification."""
        ...


class ChatDBWatcher:
    """Watch chat.db for changes and detect new messages.

    Uses polling with file modification time checks for reliable detection.
    When new messages are detected, broadcasts notifications via the socket server.

    Example:
        watcher = ChatDBWatcher(socket_server)
        await watcher.start()
    """

    def __init__(
        self,
        broadcast_handler: BroadcastHandler,
        poll_interval: float = 2.0,
    ) -> None:
        """Initialize the watcher.

        Args:
            broadcast_handler: Handler with broadcast() method for notifications
            poll_interval: How often to check for changes (seconds)
        """
        self._broadcast_handler = broadcast_handler
        self._poll_interval = poll_interval
        self._running = False
        self._last_rowid: int | None = None
        self._last_mtime: float | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start watching chat.db for changes."""
        if self._running:
            return

        self._running = True

        # Initialize last known ROWID
        self._last_rowid = await self._get_last_rowid()
        if CHAT_DB_PATH.exists():
            self._last_mtime = CHAT_DB_PATH.stat().st_mtime

        logger.info(f"Started watching chat.db (last ROWID: {self._last_rowid})")

        # Start the watch loop
        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        """Stop watching."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Stopped watching chat.db")

    async def _watch_loop(self) -> None:
        """Main watch loop."""
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)

                if not CHAT_DB_PATH.exists():
                    continue

                # Check if file was modified
                current_mtime = CHAT_DB_PATH.stat().st_mtime
                if self._last_mtime is not None and current_mtime <= self._last_mtime:
                    continue

                self._last_mtime = current_mtime

                # Check for new messages
                await self._check_new_messages()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Watcher error: {e}")
                await asyncio.sleep(self._poll_interval)

    async def _check_new_messages(self) -> None:
        """Check for new messages and broadcast notifications."""
        try:
            new_messages = await self._get_new_messages()

            for msg in new_messages:
                # Broadcast new message notification
                await self._broadcast_handler.broadcast(
                    "new_message",
                    {
                        "message_id": msg["id"],
                        "chat_id": msg["chat_id"],
                        "sender": msg["sender"],
                        "text": msg["text"],
                        "date": msg["date"],
                        "is_from_me": msg["is_from_me"],
                    },
                )

                logger.debug(
                    f"New message in {msg['chat_id']}: "
                    f"{msg['text'][:50] if msg['text'] else '[no text]'}..."
                )

            # Update last known ROWID
            if new_messages:
                self._last_rowid = max(m["id"] for m in new_messages)

        except Exception as e:
            logger.debug(f"Error checking new messages: {e}")

    async def _get_last_rowid(self) -> int | None:
        """Get the current maximum message ROWID."""
        return await asyncio.to_thread(self._query_last_rowid)

    def _query_last_rowid(self) -> int | None:
        """Query the last message ROWID (sync)."""
        if not CHAT_DB_PATH.exists():
            return None

        try:
            conn = sqlite3.connect(
                f"file:{CHAT_DB_PATH}?mode=ro",
                uri=True,
                timeout=5.0,
            )
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(ROWID) FROM message")
                row = cursor.fetchone()
                return row[0] if row and row[0] else None
            finally:
                conn.close()
        except Exception as e:
            logger.debug(f"Error getting last ROWID: {e}")
            return None

    async def _get_new_messages(self) -> list[dict[str, Any]]:
        """Get messages newer than last known ROWID."""
        if self._last_rowid is None:
            return []

        return await asyncio.to_thread(self._query_new_messages, self._last_rowid)

    def _query_new_messages(self, since_rowid: int) -> list[dict[str, Any]]:
        """Query for new messages (sync)."""
        if not CHAT_DB_PATH.exists():
            return []

        try:
            conn = sqlite3.connect(
                f"file:{CHAT_DB_PATH}?mode=ro",
                uri=True,
                timeout=5.0,
            )
            conn.row_factory = sqlite3.Row

            try:
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
                    LIMIT 100
                    """,
                    (since_rowid,),
                )

                messages = []
                for row in cursor.fetchall():
                    # Parse Apple timestamp
                    date = None
                    if row["date"]:
                        unix_ts = (row["date"] / 1_000_000_000) + APPLE_EPOCH_OFFSET
                        date = datetime.fromtimestamp(unix_ts).isoformat()

                    messages.append(
                        {
                            "id": row["id"],
                            "chat_id": row["chat_id"],
                            "sender": row["sender"],
                            "text": row["text"],
                            "date": date,
                            "is_from_me": bool(row["is_from_me"]),
                        }
                    )

                return messages

            finally:
                conn.close()

        except Exception as e:
            logger.debug(f"Error querying new messages: {e}")
            return []


async def run_watcher(broadcast_handler: BroadcastHandler) -> None:
    """Run the chat.db watcher.

    Args:
        broadcast_handler: Handler for broadcasting notifications
    """
    watcher = ChatDBWatcher(broadcast_handler)
    await watcher.start()

    try:
        # Keep running until cancelled
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await watcher.stop()
