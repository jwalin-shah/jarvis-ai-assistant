"""Calendar integration for JARVIS.

Provides event detection from messages and macOS Calendar integration.

Usage:
    from integrations.calendar import (
        EventDetectorImpl,
        CalendarReaderImpl,
        CalendarWriterImpl,
        get_event_detector,
        get_calendar_reader,
        get_calendar_writer,
    )

    # Detect events in message text
    detector = get_event_detector()
    events = detector.detect_events("Let's meet tomorrow at 3pm")

    # Read calendar events
    reader = get_calendar_reader()
    if reader.check_access():
        calendars = reader.get_calendars()
        events = reader.get_events()

    # Create calendar events
    writer = get_calendar_writer()
    result = writer.create_event(
        calendar_id="...",
        title="Meeting",
        start=datetime.now(),
        end=datetime.now() + timedelta(hours=1),
    )
"""

from integrations.calendar.detector import (
    EventDetectorImpl,
    get_event_detector,
    reset_event_detector,
)
from integrations.calendar.reader import (
    CalendarReaderImpl,
    get_calendar_reader,
    reset_calendar_reader,
)
from integrations.calendar.writer import (
    CalendarWriterImpl,
    get_calendar_writer,
    reset_calendar_writer,
)

__all__ = [
    # Event Detection
    "EventDetectorImpl",
    "get_event_detector",
    "reset_event_detector",
    # Calendar Reading
    "CalendarReaderImpl",
    "get_calendar_reader",
    "reset_calendar_reader",
    # Calendar Writing
    "CalendarWriterImpl",
    "get_calendar_writer",
    "reset_calendar_writer",
]
