"""Calendar API endpoints.

Provides endpoints for event detection from messages, listing calendars,
retrieving events, and creating calendar entries.
"""

from __future__ import annotations

import asyncio
import functools
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_imessage_reader
from api.schemas import (
    CalendarEventResponse,
    CalendarResponse,
    CreateEventFromDetectedRequest,
    CreateEventRequest,
    CreateEventResponse,
    DetectedEventResponse,
    DetectEventsFromMessagesRequest,
    DetectEventsRequest,
    ErrorResponse,
)
from contracts.calendar import DetectedEvent
from integrations.calendar import (
    get_calendar_reader,
    get_calendar_writer,
    get_event_detector,
)
from integrations.calendar.reader import CalendarReaderImpl
from integrations.calendar.writer import CalendarWriterImpl
from integrations.imessage import ChatDBReader

if TYPE_CHECKING:
    pass

router = APIRouter(prefix="/calendars", tags=["calendars"])


# Dependency for calendar reader
def get_calendar_reader_dep() -> CalendarReaderImpl:
    """Get calendar reader with access check."""
    reader = get_calendar_reader()
    if not reader.check_access():
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Permission denied",
                "message": "Calendar access is required.",
                "instructions": [
                    "Open System Settings",
                    "Go to Privacy & Security > Calendars",
                    "Enable access for your terminal application",
                    "Restart the JARVIS API server",
                ],
            },
        )
    return reader


# Dependency for calendar writer
def get_calendar_writer_dep() -> CalendarWriterImpl:
    """Get calendar writer with access check."""
    writer = get_calendar_writer()
    if not writer.check_access():
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Permission denied",
                "message": "Calendar write access is required.",
                "instructions": [
                    "Open System Settings",
                    "Go to Privacy & Security > Calendars",
                    "Enable access for your terminal application",
                    "Restart the JARVIS API server",
                ],
            },
        )
    return writer


@router.get(
    "",
    response_model=list[CalendarResponse],
    summary="List available calendars",
    responses={
        200: {"description": "List of available calendars"},
        403: {"description": "Calendar access denied", "model": ErrorResponse},
    },
)
async def list_calendars(
    reader: CalendarReaderImpl = Depends(get_calendar_reader_dep),
) -> list[CalendarResponse]:
    """List all available calendars from macOS Calendar.

    Returns calendars that the user has access to, including both
    local calendars and synced calendars (iCloud, Google, etc.).
    """
    # Run blocking calendar read in executor to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    calendars = await loop.run_in_executor(None, reader.get_calendars)
    return [CalendarResponse.model_validate(cal) for cal in calendars]


@router.get(
    "/events",
    response_model=list[CalendarEventResponse],
    summary="Get calendar events",
    responses={
        200: {"description": "List of calendar events"},
        403: {"description": "Calendar access denied", "model": ErrorResponse},
    },
)
def get_events(
    calendar_id: str | None = Query(
        default=None,
        description="Filter by calendar ID (all calendars if not specified)",
    ),
    days: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Number of days to look ahead",
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of events to return",
    ),
    reader: CalendarReaderImpl = Depends(get_calendar_reader_dep),
) -> list[CalendarEventResponse]:
    """Get upcoming calendar events.

    Returns events from the specified date range, sorted by start time.
    """
    start = datetime.now(tz=UTC)
    end = start + timedelta(days=days)

    events = reader.get_events(
        calendar_id=calendar_id,
        start=start,
        end=end,
        limit=limit,
    )

    return [CalendarEventResponse.model_validate(evt) for evt in events]


@router.get(
    "/events/search",
    response_model=list[CalendarEventResponse],
    summary="Search calendar events",
    responses={
        200: {"description": "List of matching events"},
        403: {"description": "Calendar access denied", "model": ErrorResponse},
    },
)
def search_events(
    query: str = Query(..., min_length=1, description="Search query"),
    calendar_id: str | None = Query(
        default=None,
        description="Filter by calendar ID",
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of results",
    ),
    reader: CalendarReaderImpl = Depends(get_calendar_reader_dep),
) -> list[CalendarEventResponse]:
    """Search for calendar events by title, location, or notes.

    Searches upcoming events (up to 1 year) for the query text.
    """
    events = reader.search_events(
        query=query,
        calendar_id=calendar_id,
        limit=limit,
    )

    return [CalendarEventResponse.model_validate(evt) for evt in events]


@router.post(
    "/events",
    response_model=CreateEventResponse,
    summary="Create a calendar event",
    responses={
        200: {"description": "Event created successfully"},
        400: {"description": "Invalid event data", "model": ErrorResponse},
        403: {"description": "Calendar access denied", "model": ErrorResponse},
    },
)
def create_event(
    request: CreateEventRequest,
    writer: CalendarWriterImpl = Depends(get_calendar_writer_dep),
) -> CreateEventResponse:
    """Create a new calendar event.

    Creates an event in the specified calendar with the provided details.
    """
    result = writer.create_event(
        calendar_id=request.calendar_id,
        title=request.title,
        start=request.start,
        end=request.end,
        all_day=request.all_day,
        location=request.location,
        notes=request.notes,
        url=request.url,
    )

    return CreateEventResponse(
        success=result.success,
        event_id=result.event_id,
        error=result.error,
    )


@router.post(
    "/detect",
    response_model=list[DetectedEventResponse],
    summary="Detect events in text",
    responses={
        200: {"description": "Detected events from text"},
    },
)
def detect_events_in_text(
    request: DetectEventsRequest,
) -> list[DetectedEventResponse]:
    """Detect potential calendar events in text using NLP.

    Analyzes text for date/time mentions and event indicators,
    returning detected events with confidence scores.
    """
    detector = get_event_detector()
    events = detector.detect_events(
        text=request.text,
        message_id=request.message_id,
    )

    return [
        DetectedEventResponse(
            title=evt.title,
            start=evt.start,
            end=evt.end,
            location=evt.location,
            description=evt.description,
            all_day=evt.all_day,
            confidence=evt.confidence,
            source_text=evt.source_text,
            message_id=evt.message_id,
        )
        for evt in events
    ]


@router.post(
    "/detect/messages",
    response_model=list[DetectedEventResponse],
    summary="Detect events in conversation messages",
    responses={
        200: {"description": "Detected events from messages"},
        403: {"description": "iMessage access denied", "model": ErrorResponse},
        404: {"description": "Conversation not found", "model": ErrorResponse},
    },
)
def detect_events_in_messages(
    request: DetectEventsFromMessagesRequest,
    imessage_reader: ChatDBReader = Depends(get_imessage_reader),
) -> list[DetectedEventResponse]:
    """Detect potential calendar events in conversation messages.

    Analyzes recent messages from a conversation for date/time mentions
    and event indicators, returning detected events with confidence scores.
    """
    # Get recent messages
    messages = imessage_reader.get_messages(
        chat_id=request.chat_id,
        limit=request.limit,
    )

    if not messages:
        return []

    # Detect events in each message
    detector = get_event_detector()
    all_events: list[DetectedEventResponse] = []

    for msg in messages:
        if not msg.text:
            continue

        events = detector.detect_events(
            text=msg.text,
            reference_date=msg.date,
            message_id=msg.id,
        )

        for evt in events:
            all_events.append(
                DetectedEventResponse(
                    title=evt.title,
                    start=evt.start,
                    end=evt.end,
                    location=evt.location,
                    description=evt.description,
                    all_day=evt.all_day,
                    confidence=evt.confidence,
                    source_text=evt.source_text,
                    message_id=evt.message_id,
                )
            )

    # Sort by confidence (highest first) and deduplicate by title/start
    all_events.sort(key=lambda e: e.confidence, reverse=True)

    # Deduplicate: keep highest confidence for similar events
    seen: set[tuple[str, str]] = set()
    unique_events: list[DetectedEventResponse] = []

    for evt in all_events:  # type: ignore[assignment]
        # evt is DetectedEventResponse from all_events list
        key = (evt.title.lower(), evt.start.isoformat()[:10])
        if key not in seen:
            seen.add(key)
            unique_events.append(evt)  # type: ignore[arg-type]

    return unique_events


@router.post(
    "/events/from-detected",
    response_model=CreateEventResponse,
    summary="Create event from detected event",
    responses={
        200: {"description": "Event created successfully"},
        400: {"description": "Invalid event data", "model": ErrorResponse},
        403: {"description": "Calendar access denied", "model": ErrorResponse},
    },
)
def create_event_from_detected(
    request: CreateEventFromDetectedRequest,
    writer: CalendarWriterImpl = Depends(get_calendar_writer_dep),
) -> CreateEventResponse:
    """Create a calendar event from a detected event.

    Takes a detected event and adds it to the specified calendar.
    This is the action triggered by "Add to Calendar" buttons.
    """
    detected = request.detected_event

    # Convert to DetectedEvent dataclass
    event = DetectedEvent(
        title=detected.title,
        start=detected.start,
        end=detected.end,
        location=detected.location,
        description=detected.description,
        all_day=detected.all_day,
        confidence=detected.confidence,
        source_text=detected.source_text,
        message_id=detected.message_id,
    )

    result = writer.create_event_from_detected(
        calendar_id=request.calendar_id,
        event=event,
    )

    return CreateEventResponse(
        success=result.success,
        event_id=result.event_id,
        error=result.error,
    )
