"""Priority queue for scheduled message sends.

Provides a thread-safe priority queue that orders scheduled items by
send time and priority, with persistence to disk.

Usage:
    from jarvis.scheduler.queue import get_scheduler_queue

    queue = get_scheduler_queue()
    item = queue.add(scheduled_item)
    next_item = queue.get_next_due()
"""

from __future__ import annotations

import heapq
import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jarvis.scheduler.models import Priority, ScheduledItem, ScheduledStatus

logger = logging.getLogger(__name__)

# Default persistence path
DEFAULT_QUEUE_PATH = Path.home() / ".jarvis" / "scheduler_queue.json"


class SchedulerQueue:
    """Thread-safe priority queue for scheduled items.

    Items are ordered by (send_at, -priority_weight, created_at) to ensure:
    1. Earlier send times come first
    2. Higher priority items come first when times are equal
    3. FIFO for same time and priority

    Attributes:
        persistence_path: Path to persist queue state.
        max_items: Maximum number of items in queue.
    """

    def __init__(
        self,
        persistence_path: Path | None = None,
        max_items: int = 1000,
        auto_persist: bool = True,
    ) -> None:
        """Initialize the scheduler queue.

        Args:
            persistence_path: Path to persist queue state.
            max_items: Maximum number of items to keep.
            auto_persist: Whether to auto-persist on changes.
        """
        self._items: dict[str, ScheduledItem] = {}
        self._heap: list[tuple[datetime, int, datetime, str]] = []  # Priority heap
        self._lock = threading.RLock()
        self._persistence_path = persistence_path or DEFAULT_QUEUE_PATH
        self._max_items = max_items
        self._auto_persist = auto_persist
        self._callbacks: dict[str, list[Callable[[ScheduledItem], None]]] = {}

        # Load persisted state
        self._load()

    def _heap_key(self, item: ScheduledItem) -> tuple[datetime, int, datetime, str]:
        """Generate heap key for ordering."""
        # Negate priority weight so higher priority comes first (min-heap)
        return (item.send_at, -item.priority.weight, item.created_at, item.id)

    def _rebuild_heap(self) -> None:
        """Rebuild the priority heap from current items."""
        self._heap = [
            self._heap_key(item)
            for item in self._items.values()
            if item.status in (ScheduledStatus.PENDING, ScheduledStatus.QUEUED)
        ]
        heapq.heapify(self._heap)

    def add(self, item: ScheduledItem) -> ScheduledItem:
        """Add a scheduled item to the queue.

        Args:
            item: The item to add.

        Returns:
            The added item.

        Raises:
            ValueError: If queue is at capacity.
        """
        with self._lock:
            if len(self._items) >= self._max_items:
                self._cleanup_old_items()

            if len(self._items) >= self._max_items:
                raise ValueError(f"Queue at maximum capacity ({self._max_items})")

            self._items[item.id] = item
            if item.status in (ScheduledStatus.PENDING, ScheduledStatus.QUEUED):
                heapq.heappush(self._heap, self._heap_key(item))

            if self._auto_persist:
                self._persist()

        logger.info(f"Scheduled item added: {item.id} for {item.send_at}")
        return item

    def get(self, item_id: str) -> ScheduledItem | None:
        """Get an item by ID.

        Args:
            item_id: The item identifier.

        Returns:
            The item if found, None otherwise.
        """
        with self._lock:
            return self._items.get(item_id)

    def get_all(
        self,
        status: ScheduledStatus | None = None,
        contact_id: int | None = None,
        priority: Priority | None = None,
        limit: int | None = None,
    ) -> list[ScheduledItem]:
        """Get all items, optionally filtered.

        Args:
            status: Filter by status.
            contact_id: Filter by contact.
            priority: Filter by priority.
            limit: Maximum items to return.

        Returns:
            List of matching items, sorted by send_at.
        """
        with self._lock:
            items = list(self._items.values())

        # Apply filters
        if status is not None:
            items = [i for i in items if i.status == status]
        if contact_id is not None:
            items = [i for i in items if i.contact_id == contact_id]
        if priority is not None:
            items = [i for i in items if i.priority == priority]

        # Sort by send_at
        items.sort(key=lambda i: (i.send_at, -i.priority.weight))

        # Apply limit
        if limit is not None:
            items = items[:limit]

        return items

    def get_pending(self) -> list[ScheduledItem]:
        """Get all pending items in order.

        Returns:
            List of pending items, ordered by send time and priority.
        """
        return self.get_all(status=ScheduledStatus.PENDING)

    def get_next_due(self) -> ScheduledItem | None:
        """Get the next item due for sending.

        Returns:
            The next due item, or None if no items are due.
        """
        with self._lock:
            now = datetime.now(UTC)

            while self._heap:
                # Peek at the top
                send_at, _, _, item_id = self._heap[0]

                # Check if it's past due
                if send_at > now:
                    return None

                # Pop and verify item still exists and is pending
                heapq.heappop(self._heap)
                item = self._items.get(item_id)

                if item and item.status == ScheduledStatus.PENDING:
                    # Check dependencies
                    if item.depends_on:
                        dep_item = self._items.get(item.depends_on)
                        if dep_item and dep_item.status != ScheduledStatus.SENT:
                            # Re-add to heap for later
                            heapq.heappush(self._heap, self._heap_key(item))
                            continue

                    # Check expiry
                    if item.is_expired:
                        item.mark_expired()
                        if self._auto_persist:
                            self._persist()
                        continue

                    return item

            return None

    def get_due_items(self, limit: int = 10) -> list[ScheduledItem]:
        """Get all items that are due for sending.

        Args:
            limit: Maximum items to return.

        Returns:
            List of due items, ordered by priority.
        """
        due_items: list[ScheduledItem] = []
        now = datetime.now(UTC)

        with self._lock:
            for item in self._items.values():
                if item.status == ScheduledStatus.PENDING and item.send_at <= now:
                    # Check dependencies
                    if item.depends_on:
                        dep_item = self._items.get(item.depends_on)
                        if dep_item and dep_item.status != ScheduledStatus.SENT:
                            continue

                    # Check expiry
                    if item.is_expired:
                        item.mark_expired()
                        continue

                    due_items.append(item)

            if self._auto_persist:
                self._persist()

        # Sort by priority and send_at
        due_items.sort(key=lambda i: (-i.priority.weight, i.send_at))
        return due_items[:limit]

    def update(self, item: ScheduledItem) -> None:
        """Update an item in the queue.

        Args:
            item: The item to update.
        """
        with self._lock:
            if item.id not in self._items:
                logger.warning(f"Attempted to update unknown item: {item.id}")
                return

            self._items[item.id] = item
            self._rebuild_heap()

            if self._auto_persist:
                self._persist()

        self._notify_callbacks(item)

    def cancel(self, item_id: str) -> bool:
        """Cancel a scheduled item.

        Args:
            item_id: The item identifier.

        Returns:
            True if cancelled, False if not found or already terminal.
        """
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return False

            if item.is_terminal:
                logger.warning(f"Cannot cancel terminal item: {item_id}")
                return False

            item.mark_cancelled()
            self._rebuild_heap()

            if self._auto_persist:
                self._persist()

        self._notify_callbacks(item)
        logger.info(f"Scheduled item cancelled: {item_id}")
        return True

    def reschedule(self, item_id: str, new_send_at: datetime) -> ScheduledItem | None:
        """Reschedule an item to a new time.

        Args:
            item_id: The item identifier.
            new_send_at: The new send time.

        Returns:
            The rescheduled item, or None if not found.
        """
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return None

            if item.is_terminal:
                logger.warning(f"Cannot reschedule terminal item: {item_id}")
                return None

            item.reschedule(new_send_at)
            self._rebuild_heap()

            if self._auto_persist:
                self._persist()

        self._notify_callbacks(item)
        logger.info(f"Item rescheduled: {item_id} to {new_send_at}")
        return item

    def mark_sent(self, item_id: str) -> bool:
        """Mark an item as successfully sent.

        Args:
            item_id: The item identifier.

        Returns:
            True if marked, False if not found.
        """
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return False

            item.mark_sent()
            self._rebuild_heap()

            if self._auto_persist:
                self._persist()

        self._notify_callbacks(item)
        logger.info(f"Item marked sent: {item_id}")
        return True

    def mark_failed(
        self, item_id: str, error: str, retry_after: datetime | None = None
    ) -> bool:
        """Mark an item as failed.

        Args:
            item_id: The item identifier.
            error: Error message.
            retry_after: Optional retry time.

        Returns:
            True if marked, False if not found.
        """
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return False

            item.mark_failed(error, retry_after)

            # Auto-retry if possible
            if item.can_retry:
                item.retry()
                heapq.heappush(self._heap, self._heap_key(item))
                logger.info(f"Item will retry: {item_id} (attempt {item.retry_count})")
            else:
                self._rebuild_heap()

            if self._auto_persist:
                self._persist()

        self._notify_callbacks(item)
        return True

    def delete(self, item_id: str) -> bool:
        """Delete an item from the queue.

        Only terminal items can be deleted.

        Args:
            item_id: The item identifier.

        Returns:
            True if deleted, False otherwise.
        """
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return False

            if not item.is_terminal:
                logger.warning(f"Cannot delete non-terminal item: {item_id}")
                return False

            del self._items[item_id]
            self._rebuild_heap()

            if self._auto_persist:
                self._persist()

        logger.info(f"Item deleted: {item_id}")
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary with queue statistics.
        """
        with self._lock:
            items = list(self._items.values())

        by_status: dict[str, int] = {}
        by_priority: dict[str, int] = {}
        by_contact: dict[int, int] = {}

        for status in ScheduledStatus:
            count = sum(1 for i in items if i.status == status)
            if count > 0:
                by_status[status.value] = count

        for priority in Priority:
            count = sum(1 for i in items if i.priority == priority)
            if count > 0:
                by_priority[priority.value] = count

        for item in items:
            by_contact[item.contact_id] = by_contact.get(item.contact_id, 0) + 1

        # Find next due item
        next_due = None
        pending = [i for i in items if i.status == ScheduledStatus.PENDING]
        if pending:
            pending.sort(key=lambda i: i.send_at)
            next_due = pending[0].send_at.isoformat()

        return {
            "total": len(items),
            "by_status": by_status,
            "by_priority": by_priority,
            "contacts_with_scheduled": len(by_contact),
            "next_due": next_due,
        }

    def register_callback(self, item_id: str, callback: Callable[[ScheduledItem], None]) -> None:
        """Register a callback for item updates.

        Args:
            item_id: The item to watch.
            callback: Function to call on updates.
        """
        with self._lock:
            if item_id not in self._callbacks:
                self._callbacks[item_id] = []
            self._callbacks[item_id].append(callback)

    def unregister_callback(
        self, item_id: str, callback: Callable[[ScheduledItem], None]
    ) -> None:
        """Unregister a callback.

        Args:
            item_id: The item being watched.
            callback: The callback to remove.
        """
        with self._lock:
            if item_id in self._callbacks:
                try:
                    self._callbacks[item_id].remove(callback)
                    if not self._callbacks[item_id]:
                        del self._callbacks[item_id]
                except ValueError:
                    pass

    def _notify_callbacks(self, item: ScheduledItem) -> None:
        """Notify registered callbacks of an item update."""
        with self._lock:
            callbacks = list(self._callbacks.get(item.id, []))

        for callback in callbacks:
            try:
                callback(item)
            except Exception as e:
                logger.exception(f"Error in scheduler callback: {e}")

    def _cleanup_old_items(self) -> None:
        """Remove old terminal items to stay within limits."""
        terminal = [i for i in self._items.values() if i.is_terminal]

        if len(terminal) > 0:
            # Sort by updated_at, oldest first
            terminal.sort(key=lambda i: i.updated_at)
            # Remove oldest quarter
            to_remove = max(1, len(terminal) // 4)

            for item in terminal[:to_remove]:
                del self._items[item.id]

            logger.debug(f"Cleaned up {to_remove} old items")

    def _persist(self) -> None:
        """Persist queue state to disk."""
        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": 1,
                "items": [item.to_dict() for item in self._items.values()],
            }

            with self._persistence_path.open("w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Persisted {len(self._items)} items to {self._persistence_path}")

        except Exception as e:
            logger.error(f"Failed to persist scheduler queue: {e}")

    def _load(self) -> None:
        """Load queue state from disk."""
        if not self._persistence_path.exists():
            logger.debug(f"No persisted queue at {self._persistence_path}")
            return

        try:
            with self._persistence_path.open() as f:
                data = json.load(f)

            version = data.get("version", 1)
            if version != 1:
                logger.warning(f"Unknown queue format version: {version}")
                return

            items_data = data.get("items", [])
            for item_data in items_data:
                try:
                    item = ScheduledItem.from_dict(item_data)

                    # Reset sending items to pending (they were interrupted)
                    if item.status in (ScheduledStatus.QUEUED, ScheduledStatus.SENDING):
                        item.status = ScheduledStatus.PENDING
                        item.updated_at = datetime.now(UTC)

                    self._items[item.id] = item
                except Exception as e:
                    logger.warning(f"Failed to load item: {e}")

            self._rebuild_heap()
            logger.info(f"Loaded {len(self._items)} items from {self._persistence_path}")

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in scheduler queue file: {e}")
        except Exception as e:
            logger.error(f"Failed to load scheduler queue: {e}")

    def persist(self) -> None:
        """Manually persist queue state."""
        with self._lock:
            self._persist()

    def clear_terminal(self) -> int:
        """Remove all terminal (sent/cancelled/expired) items.

        Returns:
            Number of items removed.
        """
        with self._lock:
            to_remove = [
                item_id
                for item_id, item in self._items.items()
                if item.is_terminal
            ]
            for item_id in to_remove:
                del self._items[item_id]

            if self._auto_persist and to_remove:
                self._persist()

        logger.info(f"Cleared {len(to_remove)} terminal items")
        return len(to_remove)


# Module-level singleton
_queue: SchedulerQueue | None = None
_queue_lock = threading.Lock()


def get_scheduler_queue() -> SchedulerQueue:
    """Get the singleton scheduler queue instance.

    Returns:
        Shared SchedulerQueue instance.
    """
    global _queue
    if _queue is None:
        with _queue_lock:
            if _queue is None:
                _queue = SchedulerQueue()
    return _queue


def reset_scheduler_queue() -> None:
    """Reset the singleton scheduler queue (for testing)."""
    global _queue
    with _queue_lock:
        _queue = None


# Export all public symbols
__all__ = [
    "SchedulerQueue",
    "get_scheduler_queue",
    "reset_scheduler_queue",
    "DEFAULT_QUEUE_PATH",
]
