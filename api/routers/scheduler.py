"""Scheduler API endpoints for draft scheduling and smart timing.

Provides endpoints for scheduling drafts, managing scheduled items,
and getting smart timing suggestions.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from jarvis.scheduler import (
    Priority,
    ScheduledItem,
    ScheduledStatus,
    get_scheduler,
    start_scheduler,
    stop_scheduler,
)
from jarvis.scheduler.timing import get_timing_analyzer

router = APIRouter(prefix="/scheduler", tags=["scheduler"])


# =============================================================================
# Pydantic Schemas
# =============================================================================


class PriorityEnum(str, Enum):
    """Priority level enumeration for API."""

    URGENT = "urgent"
    NORMAL = "normal"
    LOW = "low"


class ScheduledStatusEnum(str, Enum):
    """Scheduled item status enumeration for API."""

    PENDING = "pending"
    QUEUED = "queued"
    SENDING = "sending"
    SENT = "sent"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class SendResultResponse(BaseModel):
    """Send result information.

    Example:
        ```json
        {
            "success": true,
            "sent_at": "2024-01-15T10:00:00Z",
            "error": null,
            "attempts": 1
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "sent_at": "2024-01-15T10:00:00Z",
                "error": None,
                "attempts": 1,
            }
        }
    )

    success: bool = Field(..., description="Whether the send succeeded")
    sent_at: datetime | None = Field(default=None, description="When the message was sent")
    error: str | None = Field(default=None, description="Error message if failed")
    attempts: int = Field(default=1, ge=1, description="Number of send attempts")


class TimingSuggestionResponse(BaseModel):
    """Timing suggestion response.

    Example:
        ```json
        {
            "suggested_time": "2024-01-15T14:00:00Z",
            "confidence": 0.85,
            "reason": "high engagement at 2 PM; Tuesdays work well",
            "is_optimal": true
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "suggested_time": "2024-01-15T14:00:00Z",
                "confidence": 0.85,
                "reason": "high engagement at 2 PM; Tuesdays work well",
                "is_optimal": True,
            }
        }
    )

    suggested_time: datetime = Field(..., description="Recommended send time")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    reason: str = Field(default="", description="Human-readable explanation")
    is_optimal: bool = Field(default=False, description="Whether this is the optimal time")


class ScheduledItemResponse(BaseModel):
    """Scheduled item response.

    Example:
        ```json
        {
            "id": "abc123",
            "draft_id": "draft456",
            "contact_id": 1,
            "chat_id": "chat789",
            "message_text": "Hello!",
            "send_at": "2024-01-15T14:00:00Z",
            "priority": "normal",
            "status": "pending"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "abc123",
                "draft_id": "draft456",
                "contact_id": 1,
                "chat_id": "chat789",
                "message_text": "Hello!",
                "send_at": "2024-01-15T14:00:00Z",
                "priority": "normal",
                "status": "pending",
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-15T10:00:00Z",
            }
        }
    )

    id: str = Field(..., description="Unique identifier")
    draft_id: str = Field(..., description="ID of the draft")
    contact_id: int = Field(..., description="Contact ID")
    chat_id: str = Field(..., description="Chat ID for sending")
    message_text: str = Field(..., description="Message content")
    send_at: datetime = Field(..., description="Scheduled send time")
    priority: PriorityEnum = Field(default=PriorityEnum.NORMAL, description="Priority level")
    status: ScheduledStatusEnum = Field(
        default=ScheduledStatusEnum.PENDING, description="Current status"
    )
    created_at: datetime = Field(..., description="Creation time")
    updated_at: datetime = Field(..., description="Last update time")
    expires_at: datetime | None = Field(default=None, description="Expiry time")
    timezone: str | None = Field(default=None, description="Contact's timezone")
    depends_on: str | None = Field(default=None, description="ID of dependent item")
    retry_count: int = Field(default=0, ge=0, description="Retry attempts made")
    max_retries: int = Field(default=3, ge=0, description="Maximum retries")
    result: SendResultResponse | None = Field(default=None, description="Send result")


class ScheduleDraftRequest(BaseModel):
    """Request to schedule a draft.

    Example:
        ```json
        {
            "draft_id": "draft456",
            "contact_id": 1,
            "chat_id": "chat789",
            "message_text": "Hello!",
            "send_at": "2024-01-15T14:00:00Z",
            "priority": "normal"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "draft_id": "draft456",
                "contact_id": 1,
                "chat_id": "chat789",
                "message_text": "Hello!",
                "send_at": "2024-01-15T14:00:00Z",
                "priority": "normal",
            }
        }
    )

    draft_id: str = Field(..., description="ID of the draft to schedule")
    contact_id: int = Field(..., description="Contact ID to send to")
    chat_id: str = Field(..., description="Chat ID for sending")
    message_text: str = Field(..., min_length=1, description="Message content")
    send_at: datetime = Field(..., description="When to send the message")
    priority: PriorityEnum = Field(default=PriorityEnum.NORMAL, description="Priority level")
    timezone: str | None = Field(default=None, description="Contact's timezone")
    depends_on: str | None = Field(default=None, description="ID of item this depends on")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata")


class SmartScheduleRequest(BaseModel):
    """Request to schedule with smart timing.

    Example:
        ```json
        {
            "draft_id": "draft456",
            "contact_id": 1,
            "chat_id": "chat789",
            "message_text": "Hello!",
            "earliest": "2024-01-15T08:00:00Z",
            "latest": "2024-01-20T18:00:00Z"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "draft_id": "draft456",
                "contact_id": 1,
                "chat_id": "chat789",
                "message_text": "Hello!",
                "earliest": "2024-01-15T08:00:00Z",
                "latest": "2024-01-20T18:00:00Z",
            }
        }
    )

    draft_id: str = Field(..., description="ID of the draft")
    contact_id: int = Field(..., description="Contact ID")
    chat_id: str = Field(..., description="Chat ID")
    message_text: str = Field(..., min_length=1, description="Message content")
    earliest: datetime | None = Field(default=None, description="Earliest acceptable time")
    latest: datetime | None = Field(default=None, description="Latest acceptable time")
    priority: PriorityEnum = Field(default=PriorityEnum.NORMAL, description="Priority level")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata")


class SmartScheduleResponse(BaseModel):
    """Response for smart scheduling."""

    item: ScheduledItemResponse = Field(..., description="The scheduled item")
    suggestion: TimingSuggestionResponse = Field(..., description="Timing suggestion used")


class RescheduleRequest(BaseModel):
    """Request to reschedule an item."""

    send_at: datetime = Field(..., description="New send time")


class UpdateMessageRequest(BaseModel):
    """Request to update message text."""

    message_text: str = Field(..., min_length=1, description="New message content")


class ScheduledListResponse(BaseModel):
    """List of scheduled items response."""

    items: list[ScheduledItemResponse] = Field(default_factory=list)
    total: int = Field(..., ge=0)
    pending: int = Field(default=0, ge=0)
    sent: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)


class TimingSuggestRequest(BaseModel):
    """Request for timing suggestions."""

    earliest: datetime | None = Field(default=None, description="Earliest acceptable time")
    latest: datetime | None = Field(default=None, description="Latest acceptable time")
    num_suggestions: int = Field(default=3, ge=1, le=10, description="Number of suggestions")


class TimingSuggestionsResponse(BaseModel):
    """List of timing suggestions."""

    suggestions: list[TimingSuggestionResponse] = Field(default_factory=list)
    contact_id: int = Field(..., description="Contact ID")


class SchedulerStatsResponse(BaseModel):
    """Scheduler statistics response."""

    running: bool = Field(..., description="Whether scheduler is running")
    total: int = Field(..., ge=0, description="Total scheduled items")
    pending: int = Field(default=0, ge=0, description="Pending items")
    sent: int = Field(default=0, ge=0, description="Sent items")
    failed: int = Field(default=0, ge=0, description="Failed items")
    pending_in_undo_window: int = Field(default=0, ge=0, description="Items in undo window")
    next_due: datetime | None = Field(default=None, description="Next item due time")


# =============================================================================
# Helper Functions
# =============================================================================


def _item_to_response(item: ScheduledItem) -> ScheduledItemResponse:
    """Convert a ScheduledItem to API response."""
    result = None
    if item.result:
        result = SendResultResponse(
            success=item.result.success,
            sent_at=item.result.sent_at,
            error=item.result.error,
            attempts=item.result.attempts,
        )

    return ScheduledItemResponse(
        id=item.id,
        draft_id=item.draft_id,
        contact_id=item.contact_id,
        chat_id=item.chat_id,
        message_text=item.message_text,
        send_at=item.send_at,
        priority=PriorityEnum(item.priority.value),
        status=ScheduledStatusEnum(item.status.value),
        created_at=item.created_at,
        updated_at=item.updated_at,
        expires_at=item.expires_at,
        timezone=item.timezone,
        depends_on=item.depends_on,
        retry_count=item.retry_count,
        max_retries=item.max_retries,
        result=result,
    )


# =============================================================================
# API Endpoints
# =============================================================================


@router.post("/schedule", response_model=ScheduledItemResponse)
def schedule_draft(request: ScheduleDraftRequest) -> ScheduledItemResponse:
    """Schedule a draft for future sending.

    Args:
        request: Schedule request with draft details and send time.

    Returns:
        The created scheduled item.
    """
    scheduler = get_scheduler()

    # Validate send_at is in the future
    if request.send_at <= datetime.now(UTC):
        raise HTTPException(status_code=400, detail="send_at must be in the future")

    item = scheduler.schedule_draft(
        draft_id=request.draft_id,
        contact_id=request.contact_id,
        chat_id=request.chat_id,
        message_text=request.message_text,
        send_at=request.send_at,
        priority=Priority(request.priority.value),
        timezone=request.timezone,
        depends_on=request.depends_on,
        metadata=request.metadata,
    )

    return _item_to_response(item)


@router.post("/smart-schedule", response_model=SmartScheduleResponse)
def smart_schedule_draft(request: SmartScheduleRequest) -> SmartScheduleResponse:
    """Schedule a draft using smart timing analysis.

    Analyzes the contact's interaction patterns to suggest the optimal
    send time within the specified window.

    Args:
        request: Smart schedule request with time window.

    Returns:
        The scheduled item and timing suggestion used.
    """
    scheduler = get_scheduler()

    item, suggestion = scheduler.schedule_with_smart_timing(
        draft_id=request.draft_id,
        contact_id=request.contact_id,
        chat_id=request.chat_id,
        message_text=request.message_text,
        earliest=request.earliest,
        latest=request.latest,
        priority=Priority(request.priority.value),
        metadata=request.metadata,
    )

    return SmartScheduleResponse(
        item=_item_to_response(item),
        suggestion=TimingSuggestionResponse(
            suggested_time=suggestion.suggested_time,
            confidence=suggestion.confidence,
            reason=suggestion.reason,
            is_optimal=suggestion.is_optimal,
        ),
    )


@router.get("", response_model=ScheduledListResponse)
def list_scheduled(
    contact_id: int | None = None,
    status: ScheduledStatusEnum | None = None,
    limit: int = 50,
) -> ScheduledListResponse:
    """List scheduled items.

    Args:
        contact_id: Filter by contact.
        status: Filter by status.
        limit: Maximum items to return.

    Returns:
        List of scheduled items with counts.
    """
    scheduler = get_scheduler()

    status_filter = ScheduledStatus(status.value) if status else None
    items = scheduler.get_scheduled(
        contact_id=contact_id,
        status=status_filter,
        limit=limit,
    )

    # Get counts for all items
    all_items = scheduler.get_scheduled(limit=1000)
    pending = sum(1 for i in all_items if i.status == ScheduledStatus.PENDING)
    sent = sum(1 for i in all_items if i.status == ScheduledStatus.SENT)
    failed = sum(1 for i in all_items if i.status == ScheduledStatus.FAILED)

    return ScheduledListResponse(
        items=[_item_to_response(i) for i in items],
        total=len(all_items),
        pending=pending,
        sent=sent,
        failed=failed,
    )


@router.get("/stats", response_model=SchedulerStatsResponse)
def get_scheduler_stats() -> SchedulerStatsResponse:
    """Get scheduler statistics.

    Returns:
        Scheduler statistics including counts by status.
    """
    scheduler = get_scheduler()
    stats = scheduler.get_stats()

    return SchedulerStatsResponse(
        running=stats["running"],
        total=stats["total"],
        pending=stats.get("by_status", {}).get("pending", 0),
        sent=stats.get("by_status", {}).get("sent", 0),
        failed=stats.get("by_status", {}).get("failed", 0),
        pending_in_undo_window=stats.get("pending_in_undo_window", 0),
        next_due=(datetime.fromisoformat(stats["next_due"]) if stats.get("next_due") else None),
    )


@router.get("/{item_id}", response_model=ScheduledItemResponse)
def get_scheduled_item(item_id: str) -> ScheduledItemResponse:
    """Get a specific scheduled item.

    Args:
        item_id: The item identifier.

    Returns:
        The scheduled item details.
    """
    scheduler = get_scheduler()
    item = scheduler.get_item(item_id)

    if item is None:
        raise HTTPException(status_code=404, detail=f"Scheduled item not found: {item_id}")

    return _item_to_response(item)


@router.delete("/{item_id}")
def cancel_scheduled(item_id: str) -> dict[str, Any]:
    """Cancel a scheduled item.

    Args:
        item_id: The item identifier.

    Returns:
        Confirmation of cancellation.
    """
    scheduler = get_scheduler()
    item = scheduler.get_item(item_id)

    if item is None:
        raise HTTPException(status_code=404, detail=f"Scheduled item not found: {item_id}")

    if item.is_terminal:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel item with status '{item.status.value}'",
        )

    success = scheduler.cancel(item_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to cancel item")

    return {"success": True, "message": f"Scheduled item {item_id} cancelled"}


@router.put("/{item_id}/reschedule", response_model=ScheduledItemResponse)
def reschedule_item(item_id: str, request: RescheduleRequest) -> ScheduledItemResponse:
    """Reschedule an item to a new time.

    Args:
        item_id: The item identifier.
        request: Reschedule request with new send time.

    Returns:
        The updated scheduled item.
    """
    scheduler = get_scheduler()

    # Validate send_at is in the future
    if request.send_at <= datetime.now(UTC):
        raise HTTPException(status_code=400, detail="send_at must be in the future")

    item = scheduler.reschedule(item_id, request.send_at)

    if item is None:
        raise HTTPException(
            status_code=404, detail=f"Scheduled item not found or cannot be rescheduled: {item_id}"
        )

    return _item_to_response(item)


@router.put("/{item_id}/message", response_model=ScheduledItemResponse)
def update_scheduled_message(item_id: str, request: UpdateMessageRequest) -> ScheduledItemResponse:
    """Update the message text of a scheduled item.

    Args:
        item_id: The item identifier.
        request: Update request with new message text.

    Returns:
        The updated scheduled item.
    """
    scheduler = get_scheduler()
    item = scheduler.update_message(item_id, request.message_text)

    if item is None:
        raise HTTPException(
            status_code=404, detail=f"Scheduled item not found or cannot be updated: {item_id}"
        )

    return _item_to_response(item)


@router.get("/timing/suggest/{contact_id}", response_model=TimingSuggestionsResponse)
def suggest_timing(
    contact_id: int,
    earliest: datetime | None = None,
    latest: datetime | None = None,
    num_suggestions: int = 3,
) -> TimingSuggestionsResponse:
    """Get timing suggestions for a contact.

    Analyzes the contact's interaction history to suggest optimal
    send times.

    Args:
        contact_id: The contact ID.
        earliest: Earliest acceptable time.
        latest: Latest acceptable time.
        num_suggestions: Number of suggestions (1-10).

    Returns:
        List of timing suggestions.
    """
    scheduler = get_scheduler()
    suggestions = scheduler.suggest_time(
        contact_id,
        earliest=earliest,
        latest=latest,
        num_suggestions=min(10, max(1, num_suggestions)),
    )

    return TimingSuggestionsResponse(
        suggestions=[
            TimingSuggestionResponse(
                suggested_time=s.suggested_time,
                confidence=s.confidence,
                reason=s.reason,
                is_optimal=s.is_optimal,
            )
            for s in suggestions
        ],
        contact_id=contact_id,
    )


@router.post("/timing/prefs/{contact_id}")
def set_timing_preferences(
    contact_id: int,
    timezone: str | None = None,
    preferred_hours: list[int] | None = None,
    optimal_weekdays: list[int] | None = None,
) -> dict[str, Any]:
    """Set timing preferences for a contact.

    Args:
        contact_id: The contact ID.
        timezone: Contact's timezone (IANA format).
        preferred_hours: List of preferred hours (0-23).
        optimal_weekdays: List of preferred weekdays (0-6, Mon-Sun).

    Returns:
        Confirmation of update.
    """
    from jarvis.scheduler.models import ContactTimingPrefs

    analyzer = get_timing_analyzer()

    prefs = ContactTimingPrefs(
        contact_id=contact_id,
        timezone=timezone,
        preferred_hours=preferred_hours or [],
        optimal_weekdays=optimal_weekdays or [],
    )

    analyzer.set_contact_prefs(contact_id, prefs)

    return {"success": True, "message": f"Timing preferences set for contact {contact_id}"}


@router.post("/start")
def start_scheduler_endpoint() -> dict[str, Any]:
    """Start the background scheduler.

    Returns:
        Confirmation that scheduler was started.
    """
    scheduler = get_scheduler()
    if scheduler.is_running:
        return {"success": True, "message": "Scheduler is already running"}

    start_scheduler()
    return {"success": True, "message": "Scheduler started"}


@router.post("/stop")
def stop_scheduler_endpoint() -> dict[str, Any]:
    """Stop the background scheduler.

    Returns:
        Confirmation that scheduler was stopped.
    """
    scheduler = get_scheduler()
    if not scheduler.is_running:
        return {"success": True, "message": "Scheduler is not running"}

    stop_scheduler()
    return {"success": True, "message": "Scheduler stopped"}


@router.delete("/terminal/clear")
def clear_terminal_items() -> dict[str, Any]:
    """Clear all terminal (sent/cancelled/expired) items.

    Returns:
        Number of items removed.
    """
    from jarvis.scheduler.queue import get_scheduler_queue

    queue = get_scheduler_queue()
    count = queue.clear_terminal()

    return {"success": True, "items_removed": count}
