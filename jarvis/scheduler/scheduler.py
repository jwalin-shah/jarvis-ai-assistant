"""APScheduler-based draft scheduling system.

Provides a background scheduler that processes scheduled messages,
handles missed schedules gracefully, and integrates with the send executor.

Usage:
    from jarvis.scheduler import get_scheduler, start_scheduler, stop_scheduler

    scheduler = get_scheduler()
    start_scheduler()

    item = scheduler.schedule_draft(
        draft_id="abc123",
        contact_id=1,
        chat_id="chat123",
        message_text="Hello!",
        send_at=datetime.now() + timedelta(hours=2),
    )
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from jarvis.scheduler.executor import SendExecutor, get_executor
from jarvis.scheduler.models import (
    Priority,
    ScheduledItem,
    ScheduledStatus,
    TimingSuggestion,
)
from jarvis.scheduler.queue import SchedulerQueue, get_scheduler_queue
from jarvis.scheduler.timing import TimingAnalyzer, get_timing_analyzer

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Scheduler check interval (seconds)
DEFAULT_CHECK_INTERVAL = 30


class DraftScheduler:
    """Main scheduler for draft scheduling and automated sending.

    Integrates the queue, timing analyzer, and executor to provide
    a complete scheduling system with:
    - Scheduled message sending
    - Smart timing suggestions
    - Priority-based ordering
    - Retry with backoff
    - Graceful handling of missed schedules
    """

    def __init__(
        self,
        queue: SchedulerQueue | None = None,
        executor: SendExecutor | None = None,
        timing_analyzer: TimingAnalyzer | None = None,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
    ) -> None:
        """Initialize the scheduler.

        Args:
            queue: Scheduler queue (uses singleton if None).
            executor: Send executor (uses singleton if None).
            timing_analyzer: Timing analyzer (uses singleton if None).
            check_interval: Seconds between schedule checks.
        """
        self._queue = queue or get_scheduler_queue()
        self._executor = executor or get_executor()
        self._timing = timing_analyzer or get_timing_analyzer()
        self._check_interval = check_interval

        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        # Callbacks for events
        self._on_send_callbacks: list[Callable[[ScheduledItem], None]] = []
        self._on_fail_callbacks: list[Callable[[ScheduledItem, str], None]] = []
        self._on_schedule_callbacks: list[Callable[[ScheduledItem], None]] = []

    @property
    def is_running(self) -> bool:
        """Check if the scheduler is running."""
        return self._running

    def start(self) -> None:
        """Start the background scheduler thread."""
        with self._lock:
            if self._running:
                logger.warning("Scheduler already running")
                return

            self._running = True
            self._stop_event.clear()

            self._thread = threading.Thread(
                target=self._run_loop,
                name="DraftScheduler",
                daemon=True,
            )
            self._thread.start()
            logger.info("Draft scheduler started")

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the background scheduler thread.

        Args:
            timeout: Seconds to wait for thread to stop.
        """
        with self._lock:
            if not self._running:
                return

            self._running = False
            self._stop_event.set()

            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=timeout)
                if self._thread.is_alive():
                    logger.warning("Scheduler thread did not stop cleanly")

            self._thread = None
            logger.info("Draft scheduler stopped")

    def _run_loop(self) -> None:
        """Main scheduler loop."""
        logger.info("Scheduler loop started")

        while not self._stop_event.is_set():
            try:
                self._process_due_items()
                self._process_pending_sends()
            except Exception as e:
                logger.exception(f"Error in scheduler loop: {e}")

            # Wait for next check or stop signal
            self._stop_event.wait(timeout=self._check_interval)

        logger.info("Scheduler loop stopped")

    def _process_due_items(self) -> None:
        """Process all items that are due for sending."""
        due_items = self._queue.get_due_items(limit=10)

        for item in due_items:
            try:
                self._send_item(item)
            except Exception as e:
                logger.exception(f"Error sending item {item.id}: {e}")
                self._queue.mark_failed(item.id, str(e))

    def _process_pending_sends(self) -> None:
        """Process sends pending in undo window."""
        self._executor.process_pending()
        # Results are handled by callbacks in executor

    def _send_item(self, item: ScheduledItem) -> None:
        """Send a single scheduled item.

        Args:
            item: The item to send.
        """
        # Mark as sending
        item.update_status(ScheduledStatus.SENDING)
        self._queue.update(item)

        # Send with retry
        result = self._executor.send_with_retry(
            item,
            max_retries=item.max_retries - item.retry_count,
        )

        if result.success:
            self._queue.mark_sent(item.id)
            # Notify callbacks
            for callback in self._on_send_callbacks:
                try:
                    callback(item)
                except Exception as e:
                    logger.exception(f"Send callback error: {e}")
        else:
            error = result.error or "Unknown error"
            self._queue.mark_failed(item.id, error)
            # Notify callbacks
            for callback in self._on_fail_callbacks:
                try:
                    callback(item, error)
                except Exception as e:
                    logger.exception(f"Fail callback error: {e}")

    def schedule_draft(
        self,
        draft_id: str,
        contact_id: int,
        chat_id: str,
        message_text: str,
        send_at: datetime,
        priority: Priority = Priority.NORMAL,
        timezone: str | None = None,
        depends_on: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ScheduledItem:
        """Schedule a draft for future sending.

        Args:
            draft_id: ID of the draft.
            contact_id: ID of the contact.
            chat_id: Chat ID for sending.
            message_text: The message content.
            send_at: When to send.
            priority: Priority level.
            timezone: Contact's timezone.
            depends_on: ID of item this depends on.
            metadata: Additional metadata.

        Returns:
            The created ScheduledItem.
        """
        item = ScheduledItem(
            draft_id=draft_id,
            contact_id=contact_id,
            chat_id=chat_id,
            message_text=message_text,
            send_at=send_at,
            priority=priority,
            timezone=timezone,
            depends_on=depends_on,
            metadata=metadata or {},
        )

        self._queue.add(item)

        # Notify callbacks
        for callback in self._on_schedule_callbacks:
            try:
                callback(item)
            except Exception as e:
                logger.exception(f"Schedule callback error: {e}")

        logger.info(f"Draft scheduled: {item.id} for {send_at}")
        return item

    def schedule_with_smart_timing(
        self,
        draft_id: str,
        contact_id: int,
        chat_id: str,
        message_text: str,
        earliest: datetime | None = None,
        latest: datetime | None = None,
        priority: Priority = Priority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[ScheduledItem, TimingSuggestion]:
        """Schedule a draft using smart timing analysis.

        Args:
            draft_id: ID of the draft.
            contact_id: ID of the contact.
            chat_id: Chat ID for sending.
            message_text: The message content.
            earliest: Earliest acceptable time.
            latest: Latest acceptable time.
            priority: Priority level.
            metadata: Additional metadata.

        Returns:
            Tuple of (ScheduledItem, TimingSuggestion).
        """
        # Get timing suggestions
        suggestions = self._timing.suggest_time(
            contact_id,
            earliest=earliest,
            latest=latest,
            num_suggestions=1,
        )

        suggestion = (
            suggestions[0]
            if suggestions
            else TimingSuggestion(
                suggested_time=earliest or datetime.now(UTC),
                confidence=0.1,
                reason="no data available",
            )
        )

        # Schedule at suggested time
        item = self.schedule_draft(
            draft_id=draft_id,
            contact_id=contact_id,
            chat_id=chat_id,
            message_text=message_text,
            send_at=suggestion.suggested_time,
            priority=priority,
            metadata=metadata,
        )

        return (item, suggestion)

    def get_scheduled(
        self,
        contact_id: int | None = None,
        status: ScheduledStatus | None = None,
        limit: int = 50,
    ) -> list[ScheduledItem]:
        """Get scheduled items with optional filters.

        Args:
            contact_id: Filter by contact.
            status: Filter by status.
            limit: Maximum items to return.

        Returns:
            List of matching items.
        """
        return self._queue.get_all(
            status=status,
            contact_id=contact_id,
            limit=limit,
        )

    def get_item(self, item_id: str) -> ScheduledItem | None:
        """Get a specific scheduled item.

        Args:
            item_id: The item ID.

        Returns:
            The item if found, None otherwise.
        """
        return self._queue.get(item_id)

    def cancel(self, item_id: str) -> bool:
        """Cancel a scheduled item.

        Args:
            item_id: The item ID.

        Returns:
            True if cancelled, False otherwise.
        """
        # Try to cancel pending send first (undo window)
        if self._executor.cancel_pending(item_id):
            return True

        # Cancel in queue
        return self._queue.cancel(item_id)

    def reschedule(self, item_id: str, new_send_at: datetime) -> ScheduledItem | None:
        """Reschedule an item to a new time.

        Args:
            item_id: The item ID.
            new_send_at: The new send time.

        Returns:
            The rescheduled item, or None if failed.
        """
        return self._queue.reschedule(item_id, new_send_at)

    def update_message(self, item_id: str, new_text: str) -> ScheduledItem | None:
        """Update the message text of a scheduled item.

        Args:
            item_id: The item ID.
            new_text: The new message text.

        Returns:
            The updated item, or None if not found or terminal.
        """
        item = self._queue.get(item_id)
        if item is None or item.is_terminal:
            return None

        item.message_text = new_text
        item.updated_at = datetime.now(UTC)
        self._queue.update(item)

        return item

    def suggest_time(
        self,
        contact_id: int,
        earliest: datetime | None = None,
        latest: datetime | None = None,
        num_suggestions: int = 3,
    ) -> list[TimingSuggestion]:
        """Get timing suggestions for a contact.

        Args:
            contact_id: The contact ID.
            earliest: Earliest acceptable time.
            latest: Latest acceptable time.
            num_suggestions: Number of suggestions.

        Returns:
            List of TimingSuggestion objects.
        """
        return self._timing.suggest_time(
            contact_id,
            earliest=earliest,
            latest=latest,
            num_suggestions=num_suggestions,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary with scheduler stats.
        """
        queue_stats = self._queue.get_stats()
        pending_sends = self._executor.get_pending_count()

        return {
            "running": self._running,
            "check_interval": self._check_interval,
            "pending_in_undo_window": pending_sends,
            **queue_stats,
        }

    def register_on_send(self, callback: Callable[[ScheduledItem], None]) -> None:
        """Register callback for successful sends.

        Args:
            callback: Function to call with sent item.
        """
        self._on_send_callbacks.append(callback)

    def register_on_fail(self, callback: Callable[[ScheduledItem, str], None]) -> None:
        """Register callback for failed sends.

        Args:
            callback: Function to call with item and error.
        """
        self._on_fail_callbacks.append(callback)

    def register_on_schedule(self, callback: Callable[[ScheduledItem], None]) -> None:
        """Register callback for new schedules.

        Args:
            callback: Function to call with scheduled item.
        """
        self._on_schedule_callbacks.append(callback)

    def process_missed_schedules(self) -> int:
        """Process any schedules that were missed (e.g., app was closed).

        Returns:
            Number of missed schedules processed.
        """
        now = datetime.now(UTC)
        processed = 0

        # Get items that should have been sent
        pending = self._queue.get_pending()
        for item in pending:
            if item.send_at < now:
                # Check if still within acceptable window (24 hours)
                hours_overdue = (now - item.send_at).total_seconds() / 3600
                if hours_overdue < 24 and not item.is_expired:
                    logger.info(
                        f"Processing missed schedule: {item.id} ({hours_overdue:.1f}h overdue)"
                    )
                    try:
                        self._send_item(item)
                        processed += 1
                    except Exception as e:
                        logger.error(f"Failed to send missed item {item.id}: {e}")
                        self._queue.mark_failed(item.id, str(e))
                else:
                    logger.warning(f"Marking overdue item as expired: {item.id}")
                    item.mark_expired()
                    self._queue.update(item)

        return processed


# Module-level singleton
_scheduler: DraftScheduler | None = None
_scheduler_lock = threading.Lock()


def get_scheduler() -> DraftScheduler:
    """Get the singleton scheduler instance.

    Returns:
        Shared DraftScheduler instance.
    """
    global _scheduler
    if _scheduler is None:
        with _scheduler_lock:
            if _scheduler is None:
                _scheduler = DraftScheduler()
    return _scheduler


def reset_scheduler() -> None:
    """Reset the singleton scheduler (for testing)."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler and _scheduler.is_running:
            _scheduler.stop()
        _scheduler = None


def start_scheduler() -> None:
    """Start the singleton scheduler."""
    scheduler = get_scheduler()
    if not scheduler.is_running:
        scheduler.start()
        # Process any missed schedules on startup
        missed = scheduler.process_missed_schedules()
        if missed > 0:
            logger.info(f"Processed {missed} missed schedules on startup")


def stop_scheduler() -> None:
    """Stop the singleton scheduler."""
    scheduler = get_scheduler()
    if scheduler.is_running:
        scheduler.stop()


# Export all public symbols
__all__ = [
    "DraftScheduler",
    "get_scheduler",
    "reset_scheduler",
    "start_scheduler",
    "stop_scheduler",
    "DEFAULT_CHECK_INTERVAL",
]
