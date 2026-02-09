"""Chat.db file watcher for real-time message detection.

Uses watchfiles (FSEvents on macOS) for efficient file change detection.
Falls back to polling if watchfiles is unavailable.
When chat.db changes, queries for new messages and broadcasts
notifications via the socket server.
"""

import asyncio
import logging
import sqlite3
from collections import OrderedDict
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
        self._monitor_task: asyncio.Task[None] | None = None
        self._pending_check = False
        self._concurrency = asyncio.Semaphore(4)
        self._chat_msg_counts: dict[str, int] = {}
        self._segment_threshold = 15
        self._segment_window = 50
        self._resegment_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._max_resegment_locks = 100  # LRU eviction limit
        self._resegment_locks_mutex = asyncio.Lock()  # Protect OrderedDict mutations

    async def start(self) -> None:
        """Start watching chat.db for changes."""
        if self._running:
            return

        try:
            # Validate chat.db schema before starting
            if not await asyncio.to_thread(self._validate_schema):
                logger.error("chat.db schema validation failed, watcher not started")
                return

            self._running = True

            # Initialize last known ROWID
            self._last_rowid = await self._get_last_rowid()
            if CHAT_DB_PATH.exists():
                self._last_mtime = CHAT_DB_PATH.stat().st_mtime

            if self._use_fsevents:
                logger.info(
                    "Started watching chat.db with FSEvents (last ROWID: %s)", self._last_rowid
                )
                self._task = asyncio.create_task(self._watch_fsevents())
            else:
                logger.info(
                    "Started watching chat.db with polling (last ROWID: %s)", self._last_rowid
                )
                self._task = asyncio.create_task(self._watch_polling())

        except Exception as e:
            logger.error("Watcher startup failed: %s", e)
            self._running = False
            # Clean up any partially initialized state
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                self._task = None
            raise

    async def stop(self) -> None:
        """Stop watching."""
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

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
        consecutive_errors = 0

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
                        try:
                            # Debounce rapid changes
                            await self._debounced_check()
                            consecutive_errors = 0
                        except Exception as check_error:
                            consecutive_errors += 1
                            logger.warning(
                                "Error processing DB change (consecutive: %d): %s",
                                consecutive_errors,
                                check_error,
                            )
                            # Backoff on consecutive errors
                            if consecutive_errors >= 5:
                                backoff = min(consecutive_errors - 4, 30)
                                await asyncio.sleep(backoff)
                        break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("FSEvents watcher error, falling back to polling: %s", e)
            # Fall back to polling on error
            if self._running:
                self._use_fsevents = False
                self._task = asyncio.create_task(self._watch_polling())

    def _make_stop_event(self) -> asyncio.Event:
        """Create a stop event that's set when _running becomes False."""
        event = asyncio.Event()

        async def monitor() -> None:
            try:
                while self._running:
                    await asyncio.sleep(0.5)
                event.set()
            except asyncio.CancelledError:
                event.set()
            except Exception:
                logger.exception("Stop event monitor failed")
                event.set()

        self._monitor_task = asyncio.create_task(monitor())
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
        consecutive_errors = 0
        backoff_delay = self._poll_interval

        while self._running:
            try:
                await asyncio.sleep(backoff_delay)

                if not CHAT_DB_PATH.exists():
                    continue

                # Check if file was modified
                current_mtime = CHAT_DB_PATH.stat().st_mtime
                if self._last_mtime is not None and current_mtime <= self._last_mtime:
                    continue

                self._last_mtime = current_mtime

                # Check for new messages
                await self._check_new_messages()

                # Reset error counter on success
                consecutive_errors = 0
                backoff_delay = self._poll_interval

            except asyncio.CancelledError:
                break
            except Exception as e:
                consecutive_errors += 1
                logger.warning("Watcher error (consecutive: %d): %s", consecutive_errors, e)

                # Exponential backoff: 2s, 4s, 8s, max 30s
                backoff_delay = min(self._poll_interval * (2**consecutive_errors), 30.0)
                await asyncio.sleep(backoff_delay)

    async def _check_new_messages(self) -> None:
        """Check for new messages and broadcast notifications."""
        try:
            new_messages = await self._get_new_messages()

            if new_messages:
                # Pre-index messages for semantic search in the background
                # This ensures near-instant availability for RAG
                task = asyncio.create_task(self._index_new_messages(new_messages))
                task.add_done_callback(self._log_task_exception)

            for msg in new_messages:
                try:
                    # Broadcast new message notification with concurrency bound
                    async with self._concurrency:
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

                    # Only advance rowid after successful broadcast
                    self._last_rowid = max(self._last_rowid or 0, msg["id"])

                    logger.debug(
                        f"New message in {msg['chat_id']}: "
                        f"{msg['text'][:50] if msg['text'] else '[no text]'}..."
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to broadcast message %d in %s: %s",
                        msg["id"],
                        msg["chat_id"],
                        e,
                    )
                    # Skip advancing rowid on failure - will retry on next check
                    # This means failed messages will be retried, but if the broadcast
                    # handler is persistently broken, we'll retry forever. That's acceptable
                    # since the alternative (skipping messages) loses data.

            # Track per-chat message counts for incremental re-segmentation
            chats_to_resegment: list[str] = []
            for msg in new_messages:
                cid = msg["chat_id"]
                self._chat_msg_counts[cid] = self._chat_msg_counts.get(cid, 0) + 1
                if self._chat_msg_counts[cid] >= self._segment_threshold:
                    chats_to_resegment.append(cid)
                    # Delete key instead of setting to 0 to prevent unbounded dict growth
                    del self._chat_msg_counts[cid]

            # Periodic cleanup: cap dict size to prevent unbounded growth from
            # chats that never reach the segment threshold
            if len(self._chat_msg_counts) > 1000:
                # Keep only the 500 most recent (highest count) entries
                sorted_chats = sorted(
                    self._chat_msg_counts.items(), key=lambda x: x[1], reverse=True
                )
                self._chat_msg_counts = dict(sorted_chats[:500])

            if chats_to_resegment:
                task = asyncio.create_task(self._resegment_chats(chats_to_resegment))
                task.add_done_callback(self._log_task_exception)

        except Exception as e:
            logger.warning("Error checking new messages: %s", e)

    def _log_task_exception(self, task: asyncio.Task[Any]) -> None:
        """Log exceptions from background tasks."""
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("Background task failed: %s", e)

    async def _index_new_messages(self, messages: list[dict[str, Any]]) -> None:
        """Index new messages into vec_messages for semantic search.

        Args:
            messages: List of message dicts from _get_new_messages
        """
        try:
            from contracts.imessage import Message
            from integrations.imessage.parser import parse_iso_datetime
            from jarvis.search.vec_search import get_vec_searcher

            # Filter to messages with text
            text_messages = [m for m in messages if m.get("text")]
            if not text_messages:
                return

            msg_objects = []
            for m in text_messages:
                msg_objects.append(
                    Message(
                        id=m["id"],
                        chat_id=m["chat_id"],
                        sender=m["sender"],
                        text=m["text"],
                        date=parse_iso_datetime(m["date"]),
                        is_from_me=m["is_from_me"],
                    )
                )

            searcher = get_vec_searcher()
            count = await asyncio.to_thread(searcher.index_messages, msg_objects)
            if count > 0:
                logger.debug("Indexed %d new messages into vec_messages", count)

        except Exception as e:
            logger.debug("Error indexing new messages: %s", e)

    async def _get_resegment_lock(self, chat_id: str) -> asyncio.Lock:
        """Get or create a per-chat lock for serializing resegmentation.

        Uses LRU eviction to prevent unbounded memory growth.
        Thread-safe via _resegment_locks_mutex.
        """
        async with self._resegment_locks_mutex:
            # Move to end if exists (LRU update)
            if chat_id in self._resegment_locks:
                self._resegment_locks.move_to_end(chat_id)
                return self._resegment_locks[chat_id]

            # Create new lock
            lock = asyncio.Lock()
            self._resegment_locks[chat_id] = lock

            # Evict oldest if over limit
            if len(self._resegment_locks) > self._max_resegment_locks:
                oldest_key = next(iter(self._resegment_locks))
                del self._resegment_locks[oldest_key]

            return lock

    async def _resegment_chats(self, chat_ids: list[str]) -> None:
        """Re-segment recent messages for chats that hit the threshold.

        Acquires a per-chat lock so concurrent resegmentation tasks for the
        same chat_id are serialized, preventing interleaved delete+index
        operations that corrupt the vector index.
        """
        for chat_id in chat_ids:
            lock = await self._get_resegment_lock(chat_id)
            async with lock:
                try:
                    await asyncio.to_thread(self._do_resegment_one, chat_id)
                except Exception as e:
                    logger.warning("Error re-segmenting %s: %s", chat_id, e)

    def _do_resegment_one(self, chat_id: str) -> None:
        """Sync worker: re-segment recent messages for a single chat."""
        from integrations.imessage import ChatDBReader
        from jarvis.search.vec_search import get_vec_searcher
        from jarvis.topics.topic_segmenter import segment_conversation

        try:
            searcher = get_vec_searcher()
        except Exception as e:
            logger.debug("Cannot get vec_searcher for re-segmentation: %s", e)
            return

        with ChatDBReader() as reader:
            # Get recent messages (newest-first from reader), reverse to chronological
            messages = reader.get_messages(chat_id, limit=self._segment_window)
            messages.reverse()

            if not messages:
                return

            segments = segment_conversation(messages, contact_id=chat_id)

            # Replace old chunks for this chat
            deleted = searcher.delete_chunks_for_chat(chat_id)
            indexed = 0
            for seg in segments:
                if searcher.index_segment(seg, chat_id=chat_id):
                    indexed += 1

            logger.info(
                "Re-segmented %s: deleted=%d, indexed=%d segments",
                chat_id,
                deleted,
                indexed,
            )

    def _validate_schema(self) -> bool:
        """Validate that chat.db has the expected schema.

        Only checks for tables and columns actually used by the watcher.
        This is intentionally permissive - we don't fail if Apple adds new
        tables/columns in macOS updates, only if required ones are missing.
        """
        if not CHAT_DB_PATH.exists():
            logger.warning("chat.db not found, watcher cannot start")
            return False

        try:
            conn = sqlite3.connect(
                f"file:{CHAT_DB_PATH}?mode=ro",
                uri=True,
                timeout=5.0,
            )
            try:
                cursor = conn.cursor()

                # Check required tables (only what we use)
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = {row[0] for row in cursor.fetchall()}
                required_tables = {"message", "chat", "handle", "chat_message_join"}
                missing_tables = required_tables - tables
                if missing_tables:
                    logger.error("chat.db missing required tables: %s", missing_tables)
                    return False

                # Check required columns in message table (only what we query)
                # Note: ROWID is always present in SQLite, no need to check
                cursor.execute("PRAGMA table_info(message)")
                columns = {row[1] for row in cursor.fetchall()}
                required_columns = {"text", "date", "is_from_me", "handle_id"}
                missing_columns = required_columns - columns
                if missing_columns:
                    logger.error("chat.db message table missing columns: %s", missing_columns)
                    return False

                return True
            finally:
                conn.close()
        except Exception as e:
            logger.error("chat.db schema validation error: %s", e)
            return False

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
            logger.debug("Error getting last ROWID: %s", e)
            return None

    async def _get_new_messages(self) -> list[dict[str, Any]]:
        """Get messages newer than last known ROWID.

        Handles bursty writes by fetching in batches until all new messages
        are retrieved (prevents missing messages when >500 arrive at once).
        """
        if self._last_rowid is None:
            return []

        all_new_messages = []
        current_rowid = self._last_rowid

        while True:
            batch = await asyncio.to_thread(self._query_new_messages, current_rowid, limit=500)
            if not batch:
                break

            all_new_messages.extend(batch)

            # If we got fewer than the limit, we've fetched all messages
            if len(batch) < 500:
                break

            # Update current_rowid to the max of this batch to get next batch
            current_rowid = max(msg["id"] for msg in batch)

        return all_new_messages

    def _query_new_messages(self, since_rowid: int, limit: int = 500) -> list[dict[str, Any]]:
        """Query for new messages (sync).

        Args:
            since_rowid: Only fetch messages with ROWID > this value
            limit: Maximum number of messages to fetch in one query
        """
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
                    LIMIT ?
                    """,
                    (since_rowid, limit),
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
            logger.warning("Error querying new messages: %s", e)
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
