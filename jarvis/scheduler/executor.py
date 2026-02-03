"""Safe message send executor with retries and rate limiting.

Handles the actual sending of scheduled messages with:
- Exponential backoff on failures
- Rate limiting per contact
- Duplicate detection
- Undo window support

Usage:
    from jarvis.scheduler.executor import SendExecutor, get_executor

    executor = get_executor()
    result = executor.send(scheduled_item)
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from jarvis.scheduler.models import ScheduledItem, SendResult

logger = logging.getLogger(__name__)

# Rate limiting defaults
DEFAULT_RATE_LIMIT_PER_MINUTE = 5  # Max messages per contact per minute
DEFAULT_RATE_LIMIT_PER_HOUR = 30  # Max messages per contact per hour
DEFAULT_UNDO_WINDOW_SECONDS = 10  # Seconds to allow undo


@dataclass
class RateLimitConfig:
    """Rate limiting configuration per contact.

    Attributes:
        per_minute: Max messages per minute.
        per_hour: Max messages per hour.
        per_day: Max messages per day.
        enabled: Whether rate limiting is enabled.
    """

    per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE
    per_hour: int = DEFAULT_RATE_LIMIT_PER_HOUR
    per_day: int = 100
    enabled: bool = True


@dataclass
class PendingSend:
    """A send that is pending (in undo window).

    Attributes:
        item: The scheduled item.
        queued_at: When it was queued.
        expires_at: When the undo window expires.
        cancelled: Whether it was cancelled.
    """

    item: ScheduledItem
    queued_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    cancelled: bool = False

    def __post_init__(self) -> None:
        """Set expiry if not set."""
        if self.expires_at == self.queued_at:
            self.expires_at = self.queued_at + timedelta(seconds=DEFAULT_UNDO_WINDOW_SECONDS)


class SendExecutor:
    """Executes message sends with safety features.

    Provides rate limiting, duplicate detection, undo window,
    and retry logic for sending messages.
    """

    def __init__(
        self,
        rate_limit: RateLimitConfig | None = None,
        undo_window_seconds: int = DEFAULT_UNDO_WINDOW_SECONDS,
        enable_undo: bool = True,
    ) -> None:
        """Initialize the executor.

        Args:
            rate_limit: Rate limiting configuration.
            undo_window_seconds: Seconds to allow undo after queuing.
            enable_undo: Whether to enable the undo window.
        """
        self._rate_limit = rate_limit or RateLimitConfig()
        self._undo_window_seconds = undo_window_seconds
        self._enable_undo = enable_undo

        # Track sends per contact for rate limiting
        self._send_times: dict[int, list[datetime]] = defaultdict(list)

        # Track message hashes for duplicate detection
        self._recent_hashes: dict[str, datetime] = {}

        # Pending sends (in undo window)
        self._pending: dict[str, PendingSend] = {}

        self._lock = threading.RLock()

        # Callbacks for send events
        self._on_send_callbacks: list[Any] = []
        self._on_fail_callbacks: list[Any] = []

    def _message_hash(self, item: ScheduledItem) -> str:
        """Generate a hash for duplicate detection."""
        content = f"{item.contact_id}:{item.chat_id}:{item.message_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _check_rate_limit(self, contact_id: int) -> tuple[bool, str | None]:
        """Check if sending is allowed under rate limits.

        Args:
            contact_id: The contact ID.

        Returns:
            Tuple of (allowed, reason if not allowed).
        """
        if not self._rate_limit.enabled:
            return (True, None)

        now = datetime.now(UTC)

        with self._lock:
            times = self._send_times[contact_id]

            # Clean old entries
            times[:] = [t for t in times if (now - t).total_seconds() < 86400]

            # Check per-minute limit
            minute_ago = now - timedelta(minutes=1)
            recent_minute = sum(1 for t in times if t > minute_ago)
            if recent_minute >= self._rate_limit.per_minute:
                return (False, f"Rate limit: {self._rate_limit.per_minute}/min exceeded")

            # Check per-hour limit
            hour_ago = now - timedelta(hours=1)
            recent_hour = sum(1 for t in times if t > hour_ago)
            if recent_hour >= self._rate_limit.per_hour:
                return (False, f"Rate limit: {self._rate_limit.per_hour}/hour exceeded")

            # Check per-day limit
            day_ago = now - timedelta(days=1)
            recent_day = sum(1 for t in times if t > day_ago)
            if recent_day >= self._rate_limit.per_day:
                return (False, f"Rate limit: {self._rate_limit.per_day}/day exceeded")

        return (True, None)

    def _record_send(self, contact_id: int) -> None:
        """Record a send for rate limiting."""
        with self._lock:
            self._send_times[contact_id].append(datetime.now(UTC))

    def _check_duplicate(self, item: ScheduledItem) -> tuple[bool, str | None]:
        """Check if this message is a duplicate.

        Args:
            item: The scheduled item.

        Returns:
            Tuple of (is_duplicate, reason).
        """
        msg_hash = self._message_hash(item)
        now = datetime.now(UTC)

        with self._lock:
            # Clean old hashes (older than 1 hour)
            old_hashes = [
                h for h, t in self._recent_hashes.items()
                if (now - t).total_seconds() > 3600
            ]
            for h in old_hashes:
                del self._recent_hashes[h]

            # Check for duplicate
            if msg_hash in self._recent_hashes:
                return (True, "Duplicate message detected within last hour")

        return (False, None)

    def _record_message(self, item: ScheduledItem) -> None:
        """Record message hash for duplicate detection."""
        msg_hash = self._message_hash(item)
        with self._lock:
            self._recent_hashes[msg_hash] = datetime.now(UTC)

    def queue_send(self, item: ScheduledItem) -> tuple[bool, str | None]:
        """Queue a message for sending after undo window.

        Args:
            item: The scheduled item.

        Returns:
            Tuple of (success, error_message).
        """
        # Check rate limit
        allowed, reason = self._check_rate_limit(item.contact_id)
        if not allowed:
            return (False, reason)

        # Check for duplicate
        is_dup, reason = self._check_duplicate(item)
        if is_dup:
            return (False, reason)

        # Add to pending
        if self._enable_undo:
            with self._lock:
                pending = PendingSend(
                    item=item,
                    queued_at=datetime.now(UTC),
                    expires_at=datetime.now(UTC) + timedelta(seconds=self._undo_window_seconds),
                )
                self._pending[item.id] = pending

            logger.info(
                f"Message queued for send: {item.id} "
                f"(undo available for {self._undo_window_seconds}s)"
            )
            return (True, None)
        else:
            # Send immediately
            return self._execute_send(item)

    def cancel_pending(self, item_id: str) -> bool:
        """Cancel a pending send (undo).

        Args:
            item_id: The item ID.

        Returns:
            True if cancelled, False if not found or already sent.
        """
        with self._lock:
            pending = self._pending.get(item_id)
            if pending is None:
                return False

            if datetime.now(UTC) > pending.expires_at:
                return False  # Already past undo window

            pending.cancelled = True
            del self._pending[item_id]

        logger.info(f"Pending send cancelled: {item_id}")
        return True

    def process_pending(self) -> list[SendResult]:
        """Process any pending sends that have passed their undo window.

        Returns:
            List of send results for processed items.
        """
        results: list[SendResult] = []
        now = datetime.now(UTC)

        with self._lock:
            to_process = [
                (item_id, pending)
                for item_id, pending in self._pending.items()
                if now > pending.expires_at and not pending.cancelled
            ]

            # Remove from pending
            for item_id, _ in to_process:
                del self._pending[item_id]

        # Process outside lock
        for item_id, pending in to_process:
            success, error = self._execute_send(pending.item)
            if success:
                results.append(SendResult(success=True, sent_at=datetime.now(UTC)))
            else:
                results.append(SendResult(success=False, error=error))

        return results

    def _execute_send(self, item: ScheduledItem) -> tuple[bool, str | None]:
        """Actually send the message via AppleScript.

        Args:
            item: The scheduled item.

        Returns:
            Tuple of (success, error_message).
        """
        try:
            # Build AppleScript command
            # Escape the message text for AppleScript
            escaped_text = (
                item.message_text
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
            )

            # Determine if chat_id is a phone number or email
            chat_id = item.chat_id

            script = f'''
            tell application "Messages"
                set targetService to 1st service whose service type = iMessage
                set targetBuddy to buddy "{chat_id}" of targetService
                send "{escaped_text}" to targetBuddy
            end tell
            '''

            # Execute AppleScript
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                error = result.stderr.strip() or "Unknown error"
                logger.error(f"AppleScript send failed: {error}")
                return (False, f"AppleScript error: {error}")

            # Record for rate limiting and duplicate detection
            self._record_send(item.contact_id)
            self._record_message(item)

            # Notify callbacks
            for callback in self._on_send_callbacks:
                try:
                    callback(item)
                except Exception as e:
                    logger.exception(f"Send callback error: {e}")

            logger.info(f"Message sent successfully: {item.id} to {chat_id}")
            return (True, None)

        except subprocess.TimeoutExpired:
            return (False, "Send timed out after 30 seconds")
        except Exception as e:
            logger.exception(f"Send execution error: {e}")
            return (False, str(e))

    def send(self, item: ScheduledItem) -> SendResult:
        """Send a message immediately (bypassing undo window).

        Args:
            item: The scheduled item.

        Returns:
            SendResult with outcome.
        """
        # Check rate limit
        allowed, reason = self._check_rate_limit(item.contact_id)
        if not allowed:
            return SendResult(success=False, error=reason)

        # Check for duplicate
        is_dup, reason = self._check_duplicate(item)
        if is_dup:
            return SendResult(success=False, error=reason)

        # Execute send
        success, error = self._execute_send(item)

        if success:
            return SendResult(success=True, sent_at=datetime.now(UTC))
        else:
            return SendResult(success=False, error=error)

    def send_with_retry(
        self,
        item: ScheduledItem,
        max_retries: int = 3,
        base_delay_seconds: float = 2.0,
    ) -> SendResult:
        """Send with exponential backoff retry.

        Args:
            item: The scheduled item.
            max_retries: Maximum retry attempts.
            base_delay_seconds: Base delay for exponential backoff.

        Returns:
            SendResult with outcome.
        """
        last_error: str | None = None
        attempts = 0

        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            result = self.send(item)

            if result.success:
                result.attempts = attempts
                return result

            last_error = result.error

            # Check if we should retry
            if attempt < max_retries:
                delay = base_delay_seconds * (2 ** attempt)
                logger.warning(
                    f"Send attempt {attempts} failed, retrying in {delay}s: {last_error}"
                )
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} send attempts failed: {last_error}")

        return SendResult(
            success=False,
            error=last_error,
            attempts=attempts,
        )

    def get_pending_count(self) -> int:
        """Get count of messages pending in undo window."""
        with self._lock:
            return len(self._pending)

    def get_pending_items(self) -> list[ScheduledItem]:
        """Get all items pending in undo window."""
        with self._lock:
            return [p.item for p in self._pending.values() if not p.cancelled]

    def register_on_send(self, callback: Any) -> None:
        """Register callback for successful sends."""
        self._on_send_callbacks.append(callback)

    def register_on_fail(self, callback: Any) -> None:
        """Register callback for failed sends."""
        self._on_fail_callbacks.append(callback)

    def get_rate_limit_status(self, contact_id: int) -> dict[str, Any]:
        """Get current rate limit status for a contact.

        Args:
            contact_id: The contact ID.

        Returns:
            Dictionary with rate limit info.
        """
        now = datetime.now(UTC)

        with self._lock:
            times = self._send_times.get(contact_id, [])

            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)
            day_ago = now - timedelta(days=1)

            recent_minute = sum(1 for t in times if t > minute_ago)
            recent_hour = sum(1 for t in times if t > hour_ago)
            recent_day = sum(1 for t in times if t > day_ago)

        return {
            "contact_id": contact_id,
            "minute": {
                "used": recent_minute,
                "limit": self._rate_limit.per_minute,
                "remaining": max(0, self._rate_limit.per_minute - recent_minute),
            },
            "hour": {
                "used": recent_hour,
                "limit": self._rate_limit.per_hour,
                "remaining": max(0, self._rate_limit.per_hour - recent_hour),
            },
            "day": {
                "used": recent_day,
                "limit": self._rate_limit.per_day,
                "remaining": max(0, self._rate_limit.per_day - recent_day),
            },
        }


# Module-level singleton
_executor: SendExecutor | None = None
_executor_lock = threading.Lock()


def get_executor() -> SendExecutor:
    """Get the singleton executor instance.

    Returns:
        Shared SendExecutor instance.
    """
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = SendExecutor()
    return _executor


def reset_executor() -> None:
    """Reset the singleton executor (for testing)."""
    global _executor
    with _executor_lock:
        _executor = None


# Export all public symbols
__all__ = [
    "SendExecutor",
    "RateLimitConfig",
    "PendingSend",
    "get_executor",
    "reset_executor",
    "DEFAULT_RATE_LIMIT_PER_MINUTE",
    "DEFAULT_RATE_LIMIT_PER_HOUR",
    "DEFAULT_UNDO_WINDOW_SECONDS",
]
