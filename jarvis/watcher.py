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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


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
        # Persistent read-only connection for rowid polling (avoids 10-50ms connect overhead)
        self._poll_conn: sqlite3.Connection | None = None
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

        # Close persistent polling connection
        if self._poll_conn:
            try:
                self._poll_conn.close()
            except Exception:
                pass
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
                        except (OSError, sqlite3.OperationalError) as check_error:
                            consecutive_errors += 1
                            logger.warning(
                                "Transient error processing DB change (consecutive: %d): %s",
                                consecutive_errors,
                                TransientWatcherError(str(check_error)),
                            )
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
            except (OSError, sqlite3.OperationalError) as e:
                consecutive_errors += 1
                logger.warning(
                    "Transient watcher error (consecutive: %d): %s",
                    consecutive_errors,
                    TransientWatcherError(str(e)),
                )

                # Exponential backoff: 2s, 4s, 8s, max 30s
                backoff_delay = min(self._poll_interval * (2**consecutive_errors), 30.0)
                await asyncio.sleep(backoff_delay)
            except Exception as e:
                consecutive_errors += 1
                logger.warning("Watcher error (consecutive: %d): %s", consecutive_errors, e)

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

            # Extract facts from new messages (background, non-blocking)
            if new_messages:
                task = asyncio.create_task(self._extract_facts(new_messages))
                task.add_done_callback(self._log_task_exception)

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
                task.add_done_callback(self._log_task_exception)

            # Passive feedback detection (background, non-blocking)
            if new_messages:
                task = asyncio.create_task(self._detect_passive_feedback(new_messages))
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
        """Extract and persist facts from new messages using GLiNER + NLI."""
        try:
            from jarvis.contacts.candidate_extractor import CandidateExtractor
            from jarvis.contacts.fact_storage import save_candidate_facts

            # Process all messages with text (both incoming and outgoing)
            extractable = [m for m in messages if m.get("text")]
            if not extractable:
                return

            extractor = CandidateExtractor(
                label_profile="balanced", use_entailment=True
            )

            # Group by chat_id (proxy for contact)
            by_chat: dict[str, list[dict[str, Any]]] = {}
            for msg in extractable:
                cid = msg.get("chat_id", "")
                if cid:
                    by_chat.setdefault(cid, []).append(msg)

            # Track chats with significant fact updates for profile cache invalidation
            chats_to_invalidate: list[str] = []

            for chat_id, chat_msgs in by_chat.items():
                try:
                    from jarvis.text_normalizer import normalize_text

                    # Build batch input format for extract_batch()
                    batch_msgs = []
                    for m in chat_msgs:
                        raw = m.get("text")
                        if not raw:
                            continue
                        normalized = normalize_text(raw)
                        if not normalized or not normalized.strip():
                            continue
                        batch_msgs.append(
                            {
                                "text": normalized,
                                "message_id": m.get("id", 0),
                                "is_from_me": m.get("is_from_me", False),
                                "chat_id": chat_id,
                            }
                        )
                    candidates = await asyncio.to_thread(
                        extractor.extract_batch, batch_msgs
                    )
                    if candidates:
                        inserted = await asyncio.to_thread(
                            save_candidate_facts, candidates, chat_id
                        )
                        if inserted:
                            logger.info(
                                "Extracted %d candidates (%d new) for %s",
                                len(candidates),
                                inserted,
                                chat_id[:20],
                            )
                            if inserted >= 5:
                                chats_to_invalidate.append(chat_id)
                except Exception as e:
                    logger.debug("Fact extraction failed for %s: %s", chat_id[:20], e)

            # Invalidate contact profile cache for chats with significant updates
            if chats_to_invalidate:
                try:
                    from jarvis.contacts.contact_profile import invalidate_profile_cache

                    await asyncio.to_thread(invalidate_profile_cache)
                    logger.info(
                        "Invalidated contact profile cache after extracting facts "
                        "for %d chats",
                        len(chats_to_invalidate),
                    )
                except Exception as e:
                    logger.debug("Profile cache invalidation error: %s", e)

        except Exception as e:
            logger.debug("Fact extraction pipeline error: %s", e)

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
            indexed = searcher.index_segments(segments, chat_id=chat_id)

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
                # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
                cursor.execute("PRAGMA table_info(message)")
                column_info = {row[1]: row[2].upper() for row in cursor.fetchall()}
                required_columns = {"text", "date", "is_from_me", "handle_id"}
                missing_columns = required_columns - set(column_info.keys())
                if missing_columns:
                    logger.error("chat.db message table missing columns: %s", missing_columns)
                    return False

                # Validate column types for the columns we depend on
                expected_types = {
                    "text": {"TEXT", ""},  # TEXT or untyped (SQLite is flexible)
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

    async def _get_last_rowid(self) -> int | None:
        """Get the current maximum message ROWID."""
        return await asyncio.to_thread(self._query_last_rowid)

    def _get_poll_conn(self) -> sqlite3.Connection | None:
        """Get or create persistent read-only connection for polling.

        SQLite read-only connections don't go stale, so no health check needed.
        Error handlers in _query_last_rowid/_query_new_messages reset the
        connection on failure, which forces reconnection on next call.
        """
        if self._poll_conn is not None:
            return self._poll_conn

        # Create new connection
        if not CHAT_DB_PATH.exists():
            return None
        try:
            self._poll_conn = sqlite3.connect(
                f"file:{CHAT_DB_PATH}?mode=ro",
                uri=True,
                timeout=5.0,
                check_same_thread=False,
            )
            return self._poll_conn
        except Exception as e:
            logger.debug("Error creating poll connection: %s", e)
            return None

    def _query_last_rowid(self) -> int | None:
        """Query the last message ROWID (sync)."""
        conn = self._get_poll_conn()
        if conn is None:
            return None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(ROWID) FROM message")
            row = cursor.fetchone()
            return row[0] if row and row[0] else None
        except Exception as e:
            logger.debug("Error getting last ROWID: %s", e)
            # Connection may be stale, reset it (forces reconnect on next call)
            try:
                self._poll_conn.close()
            except Exception:
                pass
            self._poll_conn = None
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
        conn = self._get_poll_conn()
        if conn is None:
            return []

        try:
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

            messages = []
            for row in cursor.fetchall():
                # Parse Apple timestamp
                date = None
                if row["date"]:
                    unix_ts = (row["date"] / 1_000_000_000) + APPLE_EPOCH_OFFSET
                    date = datetime.fromtimestamp(unix_ts, tz=UTC).isoformat()

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

        except Exception as e:
            logger.warning("Error querying new messages: %s", e)
            # Connection may be stale, reset it
            self._poll_conn = None
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
