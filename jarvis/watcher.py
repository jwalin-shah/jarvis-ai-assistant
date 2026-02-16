"""Chat.db file watcher for real-time message detection.

Uses watchfiles (FSEvents on macOS) for efficient file change detection.
Falls back to polling if watchfiles is unavailable.
When chat.db changes, queries for new messages and broadcasts
notifications via the socket server.
"""

import asyncio
import logging
import sqlite3
import threading
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from jarvis.utils.async_utils import log_task_exception
from jarvis.utils.backoff import AsyncConsecutiveErrorTracker
from jarvis.utils.datetime_utils import (
    APPLE_EPOCH_OFFSET as _APPLE_EPOCH_OFFSET,
)
from jarvis.watcher_db import (
    validate_chat_db_schema,
)
from jarvis.watcher_polling import (
    get_new_messages,
    get_poll_conn,
    query_last_rowid_safe,
    query_new_messages_safe,
)
from jarvis.watcher_resegment import (
    do_resegment_one,
    get_resegment_lock,
    resegment_chats,
)

logger = logging.getLogger(__name__)

# Backward-compatible export for tests and legacy imports.
APPLE_EPOCH_OFFSET = _APPLE_EPOCH_OFFSET


# =============================================================================
# Watcher Exception Hierarchy
# =============================================================================


class WatcherError(Exception):
    """Base exception for watcher errors."""


class TransientWatcherError(WatcherError):
    """Transient error that may resolve on retry (e.g., DB locked, network hiccup)."""


class PermanentWatcherError(WatcherError):
    """Permanent error that will not resolve on retry (e.g., missing schema, bad path)."""


# Path to iMessage database
CHAT_DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"

# Debounce interval for rapid file changes
DEBOUNCE_INTERVAL = 0.01  # 10ms - extremely fast with FSEvents

# Polling interval (fallback when FSEvents unavailable)
POLL_INTERVAL = 2.0  # seconds
FACT_EXTRACTION_MIN_LIMIT = 25
FACT_EXTRACTION_MAX_LIMIT = 100
FACT_EXTRACTION_MESSAGES_TO_WINDOW_MULTIPLIER = 5

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
        self._max_resegment_locks = 100
        self._resegment_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._resegment_locks_guard = asyncio.Lock()
        # Persistent read-only connection for rowid polling (avoids 10-50ms connect overhead)
        self._poll_conn: sqlite3.Connection | None = None
        self._poll_conn_lock = threading.Lock()  # Protects _poll_conn from stop()/query race
        # Health check removed: SQLite read-only connections don't go stale.
        # Error handlers in _query_last_rowid/_query_new_messages reset on failure.

    async def start(self) -> None:
        """Start watching chat.db for changes."""
        if self._running:
            return

        try:
            # Validate chat.db schema before starting
            if not await asyncio.to_thread(self._validate_schema):
                raise PermanentWatcherError("chat.db schema validation failed")

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

        except PermanentWatcherError:
            logger.error("Watcher startup failed: permanent error, not retrying")
            self._running = False
            raise
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

        # Close persistent polling connection (lock protects against query thread race)
        with self._poll_conn_lock:
            if self._poll_conn:
                try:
                    self._poll_conn.close()
                except Exception as e:
                    logger.debug(f"Error closing poll connection: {e}")
                self._poll_conn = None

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
        tracker = AsyncConsecutiveErrorTracker(
            base_delay=1.0, max_delay=30.0, max_consecutive=5, name="watcher-fsevents"
        )

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
                            tracker.reset()
                        except (OSError, sqlite3.OperationalError) as check_error:
                            logger.warning(
                                "Transient error processing DB change: %s",
                                TransientWatcherError(str(check_error)),
                            )
                            await tracker.sleep(tracker.on_error(log_level=logging.DEBUG))
                        except Exception as check_error:
                            logger.warning(
                                "Error processing DB change: %s",
                                check_error,
                            )
                            await tracker.sleep(tracker.on_error(log_level=logging.DEBUG))
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
        tracker = AsyncConsecutiveErrorTracker(
            base_delay=self._poll_interval,
            max_delay=30.0,
            max_consecutive=1,  # Back off immediately on poll error
            name="watcher-polling",
        )

        while self._running:
            try:
                if not CHAT_DB_PATH.exists():
                    await asyncio.sleep(self._poll_interval)
                    continue

                # Check if file was modified
                current_mtime = CHAT_DB_PATH.stat().st_mtime
                if self._last_mtime is not None and current_mtime <= self._last_mtime:
                    await asyncio.sleep(self._poll_interval)
                    continue

                self._last_mtime = current_mtime

                # Check for new messages
                await self._check_new_messages()

                # Reset error counter on success
                tracker.reset()
                await asyncio.sleep(self._poll_interval)

            except asyncio.CancelledError:
                break
            except (OSError, sqlite3.OperationalError) as e:
                logger.warning(
                    "Transient watcher error: %s",
                    TransientWatcherError(str(e)),
                )
                await tracker.sleep(tracker.on_error())
            except Exception as e:
                logger.warning("Watcher error: %s", e)
                await tracker.sleep(tracker.on_error())

    async def _check_new_messages(self) -> None:
        """Check for new messages and broadcast notifications."""
        try:
            new_messages = await self._get_new_messages()

            if new_messages:
                # Pre-index messages for semantic search in the background
                # This ensures near-instant availability for RAG
                task = asyncio.create_task(self._index_new_messages(new_messages))
                task.add_done_callback(log_task_exception)

            # Pre-build all payloads, then broadcast (avoids dict creation inside loop)
            for msg in new_messages:
                try:
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

                except Exception as e:
                    logger.warning(
                        "Failed to broadcast message %d in %s: %s",
                        msg["id"],
                        msg["chat_id"],
                        e,
                    )

            # Extract facts from new messages via background task queue
            if new_messages:
                from jarvis.tasks.models import TaskType
                from jarvis.tasks.queue import get_task_queue

                queue = get_task_queue()
                # Group by chat_id and adapt extraction window to burst size.
                chat_message_counts: dict[str, int] = {}
                for msg in new_messages:
                    chat_id = msg["chat_id"]
                    chat_message_counts[chat_id] = chat_message_counts.get(chat_id, 0) + 1

                for chat_id, message_count in chat_message_counts.items():
                    queue.enqueue(
                        TaskType.FACT_EXTRACTION,
                        {
                            "chat_id": chat_id,
                            "limit": self._get_fact_extraction_limit(message_count),
                        },
                    )

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
                # Keep top 500 by count using heapq (O(n log k) vs O(n log n) sort)
                import heapq

                top_500 = heapq.nlargest(500, self._chat_msg_counts.items(), key=lambda x: x[1])
                self._chat_msg_counts = dict(top_500)

            if chats_to_resegment:
                task = asyncio.create_task(self._resegment_chats(chats_to_resegment))
                task.add_done_callback(log_task_exception)

            # Passive feedback detection (background, non-blocking)
            if new_messages:
                task = asyncio.create_task(self._detect_passive_feedback(new_messages))
                task.add_done_callback(log_task_exception)

        except Exception as e:
            logger.warning("Error checking new messages: %s", e)

    @staticmethod
    def _get_fact_extraction_limit(message_count: int) -> int:
        """Scale extraction window for bursty chats while keeping bounded latency."""
        adaptive_window = message_count * FACT_EXTRACTION_MESSAGES_TO_WINDOW_MULTIPLIER
        return max(
            FACT_EXTRACTION_MIN_LIMIT,
            min(FACT_EXTRACTION_MAX_LIMIT, adaptive_window),
        )

    async def _index_new_messages(self, messages: list[dict[str, Any]]) -> None:
        """Index new messages into vec_messages for semantic search.

        Args:
            messages: List of message dicts from _get_new_messages
        """
        try:
            from contracts.imessage import Message
            from jarvis.search.vec_search import get_vec_searcher

            # Filter to messages with text
            text_messages = [m for m in messages if m.get("text")]
            if not text_messages:
                return

            msg_objects = []
            for m in text_messages:
                # Parse date if it's an ISO string, otherwise assume it's already a datetime
                msg_date = m["date"]
                if isinstance(msg_date, str):
                    msg_date = datetime.fromisoformat(msg_date)

                msg_objects.append(
                    Message(
                        id=m["id"],
                        chat_id=m["chat_id"],
                        sender=m["sender"],
                        sender_name=m.get("sender_name"),
                        text=m["text"],
                        date=msg_date,
                        is_from_me=m["is_from_me"],
                    )
                )

            searcher = get_vec_searcher()
            count = await asyncio.to_thread(searcher.index_messages, msg_objects)
            if count > 0:
                logger.debug("Indexed %d new messages into vec_messages", count)

        except Exception as e:
            logger.debug("Error indexing new messages: %s", e)

    async def _detect_passive_feedback(self, messages: list[dict[str, Any]]) -> None:
        """Detect passive feedback from user messages.

        Identifies when a user sends a message that was previously suggested
        by the AI, or when they send a message that implies feedback.

        This is a simple implementation that logs matches for monitoring.
        A full implementation would:
        1. Store recent suggestions with timestamps and chat_id
        2. Compute semantic similarity between sent message and suggestions
        3. Record to FeedbackStore with action=SENT (>0.92 similarity) or EDITED (>0.55)
        4. Track which suggestions were ignored (wrote_from_scratch)
        """
        try:
            # Filter to outgoing messages with text (is_from_me=True)
            outgoing = [m for m in messages if m.get("is_from_me") and m.get("text")]
            if not outgoing:
                return

            # Get recent feedback entries to check for matches
            from jarvis.eval.evaluation import get_feedback_store

            store = get_feedback_store()
            recent_entries = await asyncio.to_thread(store.get_recent_entries, limit=50)

            # Simple time window: only compare against suggestions from last 5 minutes
            cutoff_time = datetime.now(UTC) - timedelta(minutes=5)

            for msg in outgoing:
                msg_text = msg.get("text", "").strip().lower()
                chat_id = msg.get("chat_id", "")

                # Look for recent suggestions for this chat
                for entry in recent_entries:
                    # Skip if wrong chat or too old
                    if entry.chat_id != chat_id:
                        continue
                    if entry.timestamp < cutoff_time:
                        continue

                    suggestion_text = entry.suggestion_text.strip().lower()

                    # Simple exact match check (a full implementation would use embeddings)
                    # This logs when user sends exactly what was suggested
                    if msg_text == suggestion_text:
                        logger.info(
                            "Passive feedback detected: user sent suggested message "
                            "(chat=%s, suggestion_id=%s)",
                            chat_id[:20],
                            entry.suggestion_id,
                        )
                        # A full implementation would call:
                        # store.record_feedback(
                        #     action=SuggestionAction.SENT,
                        #     suggestion_id=entry.suggestion_id,
                        #     ...
                        # )
                        break

        except Exception as e:
            logger.debug("Passive feedback detection error: %s", e)

    async def _extract_facts(self, messages: list[dict[str, Any]]) -> None:
        """Extract and persist facts from new messages using the Unified V4 Pipeline."""
        try:
            import uuid

            from jarvis.topics.segment_pipeline import process_segments
            from jarvis.topics.topic_segmenter import TopicSegment

            # Group by chat_id
            by_chat: dict[str, list[dict[str, Any]]] = {}
            for msg in messages:
                cid = msg.get("chat_id", "")
                if cid and msg.get("text"):
                    by_chat.setdefault(cid, []).append(msg)

            for chat_id, chat_msgs in by_chat.items():
                # 1. Create a "Live Segment" for these new messages
                # Convert dict messages to Message objects for the segmenter
                from contracts.imessage import Message
                from integrations.imessage.parser import parse_apple_timestamp

                message_objs = []
                for m in chat_msgs:
                    message_objs.append(
                        Message(
                            id=m.get("id", 0),
                            chat_id=chat_id,
                            sender=m.get("sender", ""),
                            sender_name=m.get("sender_name"),
                            text=m.get("text", ""),
                            date=parse_apple_timestamp(m.get("date", 0)),
                            is_from_me=m.get("is_from_me", False),
                            attachments=[],
                            reply_to_id=None,
                            reactions=[],
                            is_system_message=False,
                        )
                    )

                if not message_objs:
                    continue

                live_segment = TopicSegment(
                    chat_id=chat_id,
                    contact_id=chat_id,
                    messages=message_objs,
                    start_time=message_objs[0].date,
                    end_time=message_objs[-1].date,
                    message_count=len(message_objs),
                    segment_id=str(uuid.uuid4()),
                    text="\n".join([m.text or "" for m in message_objs]),
                )

                # 2. Run through Unified Pipeline (Persist -> Index -> Extract)
                # Use a thread pool for the synchronous pipeline calls
                stats = await asyncio.to_thread(
                    process_segments,
                    [live_segment],
                    chat_id,
                    contact_id=chat_id,
                    extract_facts=True,
                )

                if stats.get("facts_extracted", 0) > 0:
                    logger.info(
                        "Live Fact Extraction for %s: %d facts linked to new segment",
                        chat_id[:20],
                        stats["facts_extracted"],
                    )

                    # Invalidate contact profile cache for chats with significant updates
                    try:
                        from jarvis.contacts.contact_profile import invalidate_profile_cache

                        await asyncio.to_thread(invalidate_profile_cache)
                        logger.info("Invalidated contact profile cache for %s", chat_id[:20])
                    except Exception as e:
                        logger.debug("Profile cache invalidation error: %s", e)

        except Exception as e:
            logger.error(f"Live fact extraction pipeline failed: {e}")

    async def _resegment_chats(self, chat_ids: list[str]) -> None:
        """Re-segment recent messages for chats that hit the threshold.

        Acquires a per-chat lock so concurrent resegmentation tasks for the
        same chat_id are serialized, preventing interleaved delete+index
        operations that corrupt the vector index.
        """
        await resegment_chats(self, chat_ids)

    async def _get_resegment_lock(self, chat_id: str) -> asyncio.Lock:
        """Return per-chat async lock with LRU eviction for bounded growth."""
        return await get_resegment_lock(self, chat_id)

    def _do_resegment_one(self, chat_id: str) -> None:
        """Sync worker: incrementally segment new messages for a single chat.

        Instead of deleting all segments and rebuilding, this appends new segments
        for messages that arrived since the last segmentation.
        """
        do_resegment_one(self, chat_id)

    def _validate_schema(self) -> bool:
        """Validate that chat.db has the expected schema.

        Only checks for tables and columns actually used by the watcher.
        This is intentionally permissive - we don't fail if Apple adds new
        tables/columns in macOS updates, only if required ones are missing.
        """
        return validate_chat_db_schema(CHAT_DB_PATH)

    async def _get_last_rowid(self) -> int | None:
        """Get the current maximum message ROWID."""
        return await asyncio.to_thread(self._query_last_rowid)

    def _get_poll_conn(self) -> sqlite3.Connection | None:
        """Get or create persistent read-only connection for polling.

        SQLite read-only connections don't go stale, so no health check needed.
        Error handlers in _query_last_rowid/_query_new_messages reset the
        connection on failure, which forces reconnection on next call.
        """
        return get_poll_conn(self, CHAT_DB_PATH)

    def _query_last_rowid(self) -> int | None:
        """Query the last message ROWID (sync)."""
        return query_last_rowid_safe(self, CHAT_DB_PATH)

    async def _get_new_messages(self) -> list[dict[str, Any]]:
        """Get messages newer than last known ROWID.

        Handles bursty writes by fetching in batches until all new messages
        are retrieved (prevents missing messages when >500 arrive at once).
        """
        return await get_new_messages(self)

    def _query_new_messages(self, since_rowid: int, limit: int = 500) -> list[dict[str, Any]]:
        """Query for new messages (sync).

        Args:
            since_rowid: Only fetch messages with ROWID > this value
            limit: Maximum number of messages to fetch in one query
        """
        return query_new_messages_safe(
            self,
            CHAT_DB_PATH,
            since_rowid,
            limit=limit,
        )


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
