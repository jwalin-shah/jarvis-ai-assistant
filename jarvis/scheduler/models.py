"""Data models for the draft scheduling system.

Defines the data structures for scheduled items, timing suggestions,
and send results.

Usage:
    from jarvis.scheduler.models import ScheduledItem, Priority, ScheduledStatus

    item = ScheduledItem(
        draft_id="abc123",
        contact_id=1,
        send_at=datetime.now() + timedelta(hours=2),
        priority=Priority.NORMAL,
    )
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any


class Priority(str, Enum):
    """Priority levels for scheduled messages."""

    URGENT = "urgent"
    NORMAL = "normal"
    LOW = "low"

    @property
    def weight(self) -> int:
        """Get numeric weight for priority ordering (higher = more urgent)."""
        weights = {Priority.URGENT: 100, Priority.NORMAL: 50, Priority.LOW: 10}
        return weights[self]


class ScheduledStatus(str, Enum):
    """Status of a scheduled item."""

    PENDING = "pending"  # Waiting for send time
    QUEUED = "queued"  # In send queue, about to send
    SENDING = "sending"  # Currently being sent
    SENT = "sent"  # Successfully sent
    FAILED = "failed"  # Send failed (may retry)
    CANCELLED = "cancelled"  # User cancelled
    EXPIRED = "expired"  # Passed expiry without sending


@dataclass
class TimingSuggestion:
    """A suggested send time with reasoning.

    Attributes:
        suggested_time: The recommended send time.
        confidence: Confidence score (0.0 to 1.0).
        reason: Human-readable explanation.
        is_optimal: Whether this is the optimal time.
        factors: Contributing factors to this suggestion.
    """

    suggested_time: datetime
    confidence: float = 0.5
    reason: str = ""
    is_optimal: bool = False
    factors: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suggested_time": self.suggested_time.isoformat(),
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "is_optimal": self.is_optimal,
            "factors": self.factors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimingSuggestion:
        """Create from dictionary."""
        return cls(
            suggested_time=datetime.fromisoformat(data["suggested_time"]),
            confidence=data.get("confidence", 0.5),
            reason=data.get("reason", ""),
            is_optimal=data.get("is_optimal", False),
            factors=data.get("factors", {}),
        )


@dataclass
class SendResult:
    """Result of a send attempt.

    Attributes:
        success: Whether the send succeeded.
        sent_at: When the message was sent (if successful).
        error: Error message if failed.
        retry_after: Suggested retry time if failed.
        attempts: Number of send attempts.
    """

    success: bool = False
    sent_at: datetime | None = None
    error: str | None = None
    retry_after: datetime | None = None
    attempts: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "error": self.error,
            "retry_after": self.retry_after.isoformat() if self.retry_after else None,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SendResult:
        """Create from dictionary."""
        return cls(
            success=data.get("success", False),
            sent_at=datetime.fromisoformat(data["sent_at"]) if data.get("sent_at") else None,
            error=data.get("error"),
            retry_after=(
                datetime.fromisoformat(data["retry_after"]) if data.get("retry_after") else None
            ),
            attempts=data.get("attempts", 1),
        )


@dataclass
class ScheduledItem:
    """A scheduled draft with timing and status information.

    Attributes:
        id: Unique identifier for this scheduled item.
        draft_id: ID of the draft to send.
        contact_id: ID of the contact to send to.
        chat_id: Chat ID for sending.
        message_text: The message content to send.
        send_at: Scheduled send time.
        priority: Priority level.
        status: Current status.
        created_at: When the schedule was created.
        updated_at: Last update time.
        expires_at: When this schedule expires if not sent.
        timezone: Contact's timezone (IANA format, e.g., "America/New_York").
        depends_on: ID of another scheduled item this depends on.
        retry_count: Number of retry attempts made.
        max_retries: Maximum retry attempts allowed.
        result: Result of the last send attempt.
        metadata: Additional metadata (instructions, context, etc.).
    """

    draft_id: str
    contact_id: int
    chat_id: str
    message_text: str
    send_at: datetime
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: Priority = Priority.NORMAL
    status: ScheduledStatus = ScheduledStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    timezone: str | None = None
    depends_on: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    result: SendResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and set defaults after initialization."""
        # Set default expiry to 24 hours after send_at if not set
        if self.expires_at is None:
            self.expires_at = self.send_at + timedelta(hours=24)

    @property
    def is_due(self) -> bool:
        """Check if this item is due for sending."""
        return datetime.now(UTC) >= self.send_at and self.status == ScheduledStatus.PENDING

    @property
    def is_expired(self) -> bool:
        """Check if this item has expired."""
        return self.expires_at is not None and datetime.now(UTC) > self.expires_at

    @property
    def is_terminal(self) -> bool:
        """Check if this item is in a terminal state."""
        return self.status in (
            ScheduledStatus.SENT,
            ScheduledStatus.CANCELLED,
            ScheduledStatus.EXPIRED,
        )

    @property
    def can_retry(self) -> bool:
        """Check if this item can be retried."""
        return self.status == ScheduledStatus.FAILED and self.retry_count < self.max_retries

    @property
    def time_until_send(self) -> timedelta:
        """Get time remaining until scheduled send."""
        return self.send_at - datetime.now(UTC)

    def update_status(self, status: ScheduledStatus) -> None:
        """Update status and timestamp."""
        self.status = status
        self.updated_at = datetime.now(UTC)

    def mark_sent(self) -> None:
        """Mark as successfully sent."""
        self.status = ScheduledStatus.SENT
        self.updated_at = datetime.now(UTC)
        self.result = SendResult(success=True, sent_at=datetime.now(UTC), attempts=self.retry_count)

    def mark_failed(self, error: str, retry_after: datetime | None = None) -> None:
        """Mark as failed with error details."""
        self.status = ScheduledStatus.FAILED
        self.updated_at = datetime.now(UTC)
        self.retry_count += 1
        self.result = SendResult(
            success=False,
            error=error,
            retry_after=retry_after,
            attempts=self.retry_count,
        )

    def mark_cancelled(self) -> None:
        """Mark as cancelled by user."""
        self.status = ScheduledStatus.CANCELLED
        self.updated_at = datetime.now(UTC)

    def mark_expired(self) -> None:
        """Mark as expired."""
        self.status = ScheduledStatus.EXPIRED
        self.updated_at = datetime.now(UTC)

    def reschedule(self, new_send_at: datetime) -> None:
        """Reschedule to a new time."""
        if self.is_terminal:
            raise ValueError("Cannot reschedule terminal item")
        self.send_at = new_send_at
        self.status = ScheduledStatus.PENDING
        self.updated_at = datetime.now(UTC)
        # Reset expiry to 24 hours after new send time
        self.expires_at = new_send_at + timedelta(hours=24)

    def retry(self) -> None:
        """Prepare for retry."""
        if not self.can_retry:
            raise ValueError("Cannot retry: max retries exceeded or not in failed state")
        self.status = ScheduledStatus.PENDING
        self.updated_at = datetime.now(UTC)
        # Set new send time with exponential backoff
        backoff_minutes = 2**self.retry_count
        self.send_at = datetime.now(UTC) + timedelta(minutes=backoff_minutes)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "draft_id": self.draft_id,
            "contact_id": self.contact_id,
            "chat_id": self.chat_id,
            "message_text": self.message_text,
            "send_at": self.send_at.isoformat(),
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "timezone": self.timezone,
            "depends_on": self.depends_on,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "result": self.result.to_dict() if self.result else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduledItem:
        """Create from dictionary."""
        result = None
        if data.get("result"):
            result = SendResult.from_dict(data["result"])

        return cls(
            id=data["id"],
            draft_id=data["draft_id"],
            contact_id=data["contact_id"],
            chat_id=data["chat_id"],
            message_text=data["message_text"],
            send_at=datetime.fromisoformat(data["send_at"]),
            priority=Priority(data.get("priority", "normal")),
            status=ScheduledStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
            timezone=data.get("timezone"),
            depends_on=data.get("depends_on"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            result=result,
            metadata=data.get("metadata", {}),
        )


@dataclass
class QuietHours:
    """Configuration for quiet hours when messages shouldn't be sent.

    Attributes:
        start_hour: Start hour (0-23) in local time.
        end_hour: End hour (0-23) in local time.
        days: Days of week (0=Monday, 6=Sunday). Empty means all days.
        enabled: Whether quiet hours are enabled.
    """

    start_hour: int = 22  # 10 PM
    end_hour: int = 8  # 8 AM
    days: list[int] = field(default_factory=list)  # Empty = all days
    enabled: bool = True

    def is_quiet_time(self, dt: datetime) -> bool:
        """Check if the given time falls within quiet hours.

        Args:
            dt: The datetime to check (should be in local time).

        Returns:
            True if within quiet hours.
        """
        if not self.enabled:
            return False

        # Check day of week if specified
        if self.days and dt.weekday() not in self.days:
            return False

        hour = dt.hour

        # Handle overnight quiet hours (e.g., 22:00 to 08:00)
        if self.start_hour > self.end_hour:
            return hour >= self.start_hour or hour < self.end_hour
        else:
            return self.start_hour <= hour < self.end_hour

    def next_allowed_time(self, dt: datetime) -> datetime:
        """Get the next time outside quiet hours.

        Args:
            dt: The current datetime.

        Returns:
            The next datetime when sending is allowed.
        """
        if not self.is_quiet_time(dt):
            return dt

        # Calculate end of current quiet period
        if self.start_hour > self.end_hour:
            # Overnight quiet hours
            if dt.hour >= self.start_hour:
                # Currently in evening quiet hours, wait until tomorrow morning
                next_day = dt + timedelta(days=1)
                return next_day.replace(hour=self.end_hour, minute=0, second=0, microsecond=0)
            else:
                # Currently in morning quiet hours, wait until end_hour
                return dt.replace(hour=self.end_hour, minute=0, second=0, microsecond=0)
        else:
            # Same-day quiet hours
            return dt.replace(hour=self.end_hour, minute=0, second=0, microsecond=0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "days": self.days,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuietHours:
        """Create from dictionary."""
        return cls(
            start_hour=data.get("start_hour", 22),
            end_hour=data.get("end_hour", 8),
            days=data.get("days", []),
            enabled=data.get("enabled", True),
        )


@dataclass
class ContactTimingPrefs:
    """Timing preferences for a specific contact.

    Attributes:
        contact_id: The contact this applies to.
        timezone: Contact's timezone (IANA format).
        quiet_hours: Contact-specific quiet hours.
        preferred_hours: Preferred send hours (0-23).
        optimal_weekdays: Preferred days (0=Mon, 6=Sun).
        avg_response_time_mins: Average response time in minutes.
        last_interaction: Last interaction timestamp.
    """

    contact_id: int
    timezone: str | None = None
    quiet_hours: QuietHours | None = None
    preferred_hours: list[int] = field(default_factory=list)
    optimal_weekdays: list[int] = field(default_factory=list)
    avg_response_time_mins: float | None = None
    last_interaction: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contact_id": self.contact_id,
            "timezone": self.timezone,
            "quiet_hours": self.quiet_hours.to_dict() if self.quiet_hours else None,
            "preferred_hours": self.preferred_hours,
            "optimal_weekdays": self.optimal_weekdays,
            "avg_response_time_mins": self.avg_response_time_mins,
            "last_interaction": (
                self.last_interaction.isoformat() if self.last_interaction else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContactTimingPrefs:
        """Create from dictionary."""
        quiet_hours = None
        if data.get("quiet_hours"):
            quiet_hours = QuietHours.from_dict(data["quiet_hours"])

        return cls(
            contact_id=data["contact_id"],
            timezone=data.get("timezone"),
            quiet_hours=quiet_hours,
            preferred_hours=data.get("preferred_hours", []),
            optimal_weekdays=data.get("optimal_weekdays", []),
            avg_response_time_mins=data.get("avg_response_time_mins"),
            last_interaction=(
                datetime.fromisoformat(data["last_interaction"])
                if data.get("last_interaction")
                else None
            ),
        )


# Export all public symbols
__all__ = [
    "Priority",
    "ScheduledStatus",
    "TimingSuggestion",
    "SendResult",
    "ScheduledItem",
    "QuietHours",
    "ContactTimingPrefs",
]
