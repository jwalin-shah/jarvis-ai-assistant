"""Calendar integration interfaces.

Provides contracts for calendar event detection and macOS Calendar integration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass
class DetectedEvent:
    """Event detected from message text using NLP.

    Represents a potential calendar event extracted from a message,
    with parsed date/time information and confidence score.
    """

    title: str
    start: datetime
    end: datetime | None = None
    location: str | None = None
    description: str | None = None
    all_day: bool = False
    confidence: float = 0.0
    source_text: str = ""  # Original text that was parsed
    message_id: int | None = None  # iMessage ID if from a message


@dataclass
class CalendarEvent:
    """Calendar event from macOS Calendar.

    Represents an event retrieved from or to be added to macOS Calendar.
    """

    id: str
    calendar_id: str
    calendar_name: str
    title: str
    start: datetime
    end: datetime
    all_day: bool = False
    location: str | None = None
    notes: str | None = None
    url: str | None = None
    attendees: list[str] = field(default_factory=list)
    status: str = "confirmed"  # confirmed, tentative, cancelled


@dataclass
class Calendar:
    """macOS Calendar summary."""

    id: str
    name: str
    color: str | None = None
    is_editable: bool = True


@dataclass
class CreateEventResult:
    """Result of creating a calendar event."""

    success: bool
    event_id: str | None = None
    error: str | None = None


class EventDetector(Protocol):
    """Interface for detecting events in message text."""

    def detect_events(
        self,
        text: str,
        reference_date: datetime | None = None,
        message_id: int | None = None,
    ) -> list[DetectedEvent]:
        """Detect potential calendar events in text.

        Args:
            text: Message text to analyze.
            reference_date: Reference date for relative date parsing.
                           Defaults to current datetime.
            message_id: Optional message ID for tracking source.

        Returns:
            List of detected events with confidence scores.
        """
        ...

    def detect_events_batch(
        self,
        texts: list[tuple[str, int | None]],
        reference_date: datetime | None = None,
    ) -> list[list[DetectedEvent]]:
        """Detect events in multiple texts.

        Args:
            texts: List of (text, message_id) tuples to analyze.
            reference_date: Reference date for relative date parsing.

        Returns:
            List of detected events for each input text.
        """
        ...


class CalendarReader(Protocol):
    """Interface for reading from macOS Calendar."""

    def check_access(self) -> bool:
        """Check if we have permission to access Calendar.

        Returns:
            True if calendar access is available.
        """
        ...

    def get_calendars(self) -> list[Calendar]:
        """Get available calendars.

        Returns:
            List of available calendars.
        """
        ...

    def get_events(
        self,
        calendar_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 50,
    ) -> list[CalendarEvent]:
        """Get events from calendar(s).

        Args:
            calendar_id: Optional calendar ID to filter by.
            start: Start of date range (defaults to now).
            end: End of date range (defaults to 30 days from now).
            limit: Maximum number of events to return.

        Returns:
            List of calendar events.
        """
        ...

    def search_events(
        self,
        query: str,
        calendar_id: str | None = None,
        limit: int = 50,
    ) -> list[CalendarEvent]:
        """Search events by text query.

        Args:
            query: Search text.
            calendar_id: Optional calendar ID to filter by.
            limit: Maximum number of results.

        Returns:
            List of matching events.
        """
        ...


class CalendarWriter(Protocol):
    """Interface for writing to macOS Calendar."""

    def check_access(self) -> bool:
        """Check if we have permission to write to Calendar.

        Returns:
            True if write access is available.
        """
        ...

    def create_event(
        self,
        calendar_id: str,
        title: str,
        start: datetime,
        end: datetime,
        all_day: bool = False,
        location: str | None = None,
        notes: str | None = None,
        url: str | None = None,
    ) -> CreateEventResult:
        """Create a new calendar event.

        Args:
            calendar_id: Target calendar ID.
            title: Event title.
            start: Event start time.
            end: Event end time.
            all_day: Whether this is an all-day event.
            location: Optional location.
            notes: Optional notes/description.
            url: Optional URL.

        Returns:
            Result with event ID on success or error message.
        """
        ...

    def create_event_from_detected(
        self,
        calendar_id: str,
        event: DetectedEvent,
    ) -> CreateEventResult:
        """Create a calendar event from a detected event.

        Args:
            calendar_id: Target calendar ID.
            event: Detected event to add.

        Returns:
            Result with event ID on success or error message.
        """
        ...
