"""Calendar and event models.

Contains schemas for calendar events, event detection, and event creation.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DetectedEventResponse(BaseModel):
    """Event detected from message text.

    Represents a potential calendar event extracted from a message
    using NLP-based date/time parsing.

    Example:
        ```json
        {
            "title": "Dinner with John",
            "start": "2024-01-20T18:00:00",
            "end": "2024-01-20T19:00:00",
            "location": "Downtown Restaurant",
            "description": "Let's have dinner tomorrow at 6pm at Downtown Restaurant",
            "all_day": false,
            "confidence": 0.85,
            "source_text": "Let's have dinner tomorrow at 6pm at Downtown Restaurant",
            "message_id": 12345
        }
        ```
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "title": "Dinner with John",
                "start": "2024-01-20T18:00:00",
                "end": "2024-01-20T19:00:00",
                "location": "Downtown Restaurant",
                "description": "Let's have dinner tomorrow at 6pm",
                "all_day": False,
                "confidence": 0.85,
                "source_text": "Let's have dinner tomorrow at 6pm at Downtown Restaurant",
                "message_id": 12345,
            }
        },
    )

    title: str = Field(
        ...,
        description="Event title extracted from text",
        examples=["Dinner with John", "Team meeting"],
    )
    start: datetime = Field(
        ...,
        description="Event start time",
    )
    end: datetime | None = Field(
        default=None,
        description="Event end time (estimated if not specified)",
    )
    location: str | None = Field(
        default=None,
        description="Event location if detected",
        examples=["Downtown Restaurant", "Conference Room A"],
    )
    description: str | None = Field(
        default=None,
        description="Event description from source text",
    )
    all_day: bool = Field(
        default=False,
        description="Whether this is an all-day event",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score (0.0 to 1.0)",
        examples=[0.85, 0.7],
    )
    source_text: str = Field(
        default="",
        description="Original text that was parsed",
    )
    message_id: int | None = Field(
        default=None,
        description="iMessage ID if from a message",
    )


class CalendarResponse(BaseModel):
    """macOS Calendar summary.

    Represents an available calendar from the system.

    Example:
        ```json
        {
            "id": "calendar-123",
            "name": "Work",
            "color": "#FF5733",
            "is_editable": true
        }
        ```
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "calendar-123",
                "name": "Work",
                "color": "#FF5733",
                "is_editable": True,
            }
        },
    )

    id: str = Field(
        ...,
        description="Calendar unique identifier",
    )
    name: str = Field(
        ...,
        description="Calendar display name",
        examples=["Work", "Personal", "Family"],
    )
    color: str | None = Field(
        default=None,
        description="Calendar color (hex code)",
        examples=["#FF5733", "#3498DB"],
    )
    is_editable: bool = Field(
        default=True,
        description="Whether events can be added to this calendar",
    )


class CalendarEventResponse(BaseModel):
    """Calendar event from macOS Calendar.

    Represents an event from the macOS Calendar app.

    Example:
        ```json
        {
            "id": "event-456",
            "calendar_id": "calendar-123",
            "calendar_name": "Work",
            "title": "Team Meeting",
            "start": "2024-01-20T10:00:00",
            "end": "2024-01-20T11:00:00",
            "all_day": false,
            "location": "Conference Room A",
            "notes": "Weekly sync",
            "url": null,
            "attendees": ["john@example.com"],
            "status": "confirmed"
        }
        ```
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "event-456",
                "calendar_id": "calendar-123",
                "calendar_name": "Work",
                "title": "Team Meeting",
                "start": "2024-01-20T10:00:00",
                "end": "2024-01-20T11:00:00",
                "all_day": False,
                "location": "Conference Room A",
                "notes": "Weekly sync",
                "url": None,
                "attendees": ["john@example.com"],
                "status": "confirmed",
            }
        },
    )

    id: str = Field(
        ...,
        description="Event unique identifier",
    )
    calendar_id: str = Field(
        ...,
        description="Parent calendar ID",
    )
    calendar_name: str = Field(
        ...,
        description="Parent calendar name",
    )
    title: str = Field(
        ...,
        description="Event title",
        examples=["Team Meeting", "Lunch with Client"],
    )
    start: datetime = Field(
        ...,
        description="Event start time",
    )
    end: datetime = Field(
        ...,
        description="Event end time",
    )
    all_day: bool = Field(
        default=False,
        description="Whether this is an all-day event",
    )
    location: str | None = Field(
        default=None,
        description="Event location",
        examples=["Conference Room A", "123 Main St"],
    )
    notes: str | None = Field(
        default=None,
        description="Event notes/description",
    )
    url: str | None = Field(
        default=None,
        description="Event URL",
    )
    attendees: list[str] = Field(
        default_factory=list,
        description="List of attendee emails",
    )
    status: str = Field(
        default="confirmed",
        description="Event status: confirmed, tentative, cancelled",
        examples=["confirmed", "tentative", "cancelled"],
    )


class DetectEventsRequest(BaseModel):
    """Request to detect events from text.

    Example:
        ```json
        {
            "text": "Let's have dinner tomorrow at 6pm",
            "message_id": 12345
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Let's have dinner tomorrow at 6pm",
                "message_id": 12345,
            }
        }
    )

    text: str = Field(
        ...,
        min_length=1,
        description="Text to analyze for events",
    )
    message_id: int | None = Field(
        default=None,
        description="Optional message ID for tracking",
    )


class DetectEventsFromMessagesRequest(BaseModel):
    """Request to detect events from conversation messages.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "limit": 50
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "limit": 50,
            }
        }
    )

    chat_id: str = Field(
        ...,
        description="Conversation ID to analyze",
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of recent messages to analyze",
    )


class CreateEventRequest(BaseModel):
    """Request to create a calendar event.

    Example:
        ```json
        {
            "calendar_id": "calendar-123",
            "title": "Team Meeting",
            "start": "2024-01-20T10:00:00",
            "end": "2024-01-20T11:00:00",
            "all_day": false,
            "location": "Conference Room A",
            "notes": "Weekly sync"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "calendar_id": "calendar-123",
                "title": "Team Meeting",
                "start": "2024-01-20T10:00:00",
                "end": "2024-01-20T11:00:00",
                "all_day": False,
                "location": "Conference Room A",
                "notes": "Weekly sync",
            }
        }
    )

    calendar_id: str = Field(
        ...,
        description="Target calendar ID",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Event title",
    )
    start: datetime = Field(
        ...,
        description="Event start time",
    )
    end: datetime = Field(
        ...,
        description="Event end time",
    )
    all_day: bool = Field(
        default=False,
        description="Whether this is an all-day event",
    )
    location: str | None = Field(
        default=None,
        max_length=500,
        description="Event location",
    )
    notes: str | None = Field(
        default=None,
        max_length=2000,
        description="Event notes/description",
    )
    url: str | None = Field(
        default=None,
        max_length=500,
        description="Event URL",
    )


class CreateEventFromDetectedRequest(BaseModel):
    """Request to create event from a detected event.

    Example:
        ```json
        {
            "calendar_id": "calendar-123",
            "detected_event": {
                "title": "Dinner",
                "start": "2024-01-20T18:00:00",
                "end": "2024-01-20T19:00:00",
                "confidence": 0.85
            }
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "calendar_id": "calendar-123",
                "detected_event": {
                    "title": "Dinner",
                    "start": "2024-01-20T18:00:00",
                    "end": "2024-01-20T19:00:00",
                    "all_day": False,
                    "confidence": 0.85,
                },
            }
        }
    )

    calendar_id: str = Field(
        ...,
        description="Target calendar ID",
    )
    detected_event: DetectedEventResponse = Field(
        ...,
        description="Detected event to add to calendar",
    )


class CreateEventResponse(BaseModel):
    """Response after creating a calendar event.

    Example:
        ```json
        {
            "success": true,
            "event_id": "event-789",
            "error": null
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "event_id": "event-789",
                "error": None,
            }
        }
    )

    success: bool = Field(
        ...,
        description="Whether the event was created successfully",
    )
    event_id: str | None = Field(
        default=None,
        description="ID of created event (if successful)",
    )
    error: str | None = Field(
        default=None,
        description="Error message (if failed)",
    )
