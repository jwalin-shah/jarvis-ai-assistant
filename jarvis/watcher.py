"""Chat.db file watcher for real-time message detection.

Uses watchfiles (FSEvents on macOS) for efficient file change detection.
Falls back to polling if watchfiles is unavailable.
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

# Debounce interval for rapid file changes
DEBOUNCE_INTERVAL = 0.05  # 50ms - very fast with FSEvents

# Polling interval (fallback when FSEvents unavailable)
POLL_INTERVAL = 2.0  # seconds

# Apple epoch offset (2001-01-01 in Unix timestamp)
APPLE_EPOCH_OFFSET = 978307200

# Check if watchfiles is available
try:
    from watchfiles import Change, awatch

    WATCHFILES_AVAILABLE = True
except ImportError:
    WATCHFILES_AVAILABLE = False
    logger.debug("watchfiles not available, using polling fallback")


class BroadcastHandler(Protocol):
    """Protocol for broadcast handlers."""

    async def broadcast(self, method: str, params: dict[str, Any]) -> None:
        """Broadcast a notification."""
        ...


class ChatDBWatcher:
    """Watch chat.db for changes and detect new messages.

    Uses FSEvents via watchfiles for near-instant detection on macOS (<100ms).
    Falls back to polling if watchfiles is unavailable.
    When new messages are detected, broadcasts notifications via the socket server.

    Example:
        watcher = ChatDBWatcher(socket_server)
        await watcher.start()
    """

    def __init__(
        self,
        broadcast_handler: BroadcastHandler,
        use_fsevents: bool = True,
        poll_interval: float = POLL_INTERVAL,
    ) -> None:
        """Initialize the watcher.

        Args:
            broadcast_handler: Handler with broadcast() method for notifications
            use_fsevents: Whether to use FSEvents (True) or polling (False)
            poll_interval: How often to check for changes when polling (seconds)
        """
        self._broadcast_handler = broadcast_handler
        self._use_fsevents = use_fsevents and WATCHFILES_AVAILABLE
        self._poll_interval = poll_interval
        self._running = False
        self._last_rowid: int | None = None
        self._last_mtime: float | None = None
        self._task: asyncio.Task[None] | None = None
        self._debounce_task: asyncio.Task[None] | None = None
        self._pending_check = False

    async def start(self) -> None:
        """Start watching chat.db for changes."""
        if self._running:
            return

        self._running = True

        # Initialize last known ROWID
        self._last_rowid = await self._get_last_rowid()
        if CHAT_DB_PATH.exists():
            self._last_mtime = CHAT_DB_PATH.stat().st_mtime

        if self._use_fsevents:
            logger.info(f"Started watching chat.db with FSEvents (last ROWID: {self._last_rowid})")
            self._task = asyncio.create_task(self._watch_fsevents())
        else:
            logger.info(f"Started watching chat.db with polling (last ROWID: {self._last_rowid})")
            self._task = asyncio.create_task(self._watch_polling())

    async def stop(self) -> None:
        """Stop watching."""
        self._running = False

        if self._debounce_task:
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass
            self._debounce_task = None

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Stopped watching chat.db")

    async def _watch_fsevents(self) -> None:
        """Watch using FSEvents via watchfiles (macOS native, <100ms latency)."""
        try:
            # Watch the Messages directory (parent of chat.db)
            # watchfiles watches directories, not individual files
            watch_dir = CHAT_DB_PATH.parent

            async for changes in awatch(watch_dir, stop_event=self._make_stop_event()):
                if not self._running:
                    break

                # Check if chat.db was modified
                for change_type, path in changes:
                    if Path(path).name == "chat.db" and change_type == Change.modified:
                        # Debounce rapid changes
                        await self._debounced_check()
                        break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"FSEvents watcher error, falling back to polling: {e}")
            # Fall back to polling on error
            if self._running:
                self._use_fsevents = False
                self._task = asyncio.create_task(self._watch_polling())

    def _make_stop_event(self) -> asyncio.Event:
        """Create a stop event that's set when _running becomes False."""
        event = asyncio.Event()

        async def monitor() -> None:
            while self._running:
                await asyncio.sleep(0.5)
            event.set()

        asyncio.create_task(monitor())
        return event

    async def _debounced_check(self) -> None:
        """Check for new messages with debouncing to coalesce rapid changes."""
        self._pending_check = True

        # Cancel existing debounce timer
        if self._debounce_task and not self._debounce_task.done():
            return  # Already have a pending check

        async def do_check() -> None:
            await asyncio.sleep(DEBOUNCE_INTERVAL)
            if self._pending_check:
                self._pending_check = False
                await self._check_new_messages()

        self._debounce_task = asyncio.create_task(do_check())

    async def _watch_polling(self) -> None:
        """Watch using polling (fallback method)."""
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
                logger.debug(f"Updated last ROWID to {self._last_rowid}")

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
