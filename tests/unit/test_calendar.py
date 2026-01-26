"""Unit tests for calendar integration.

Tests for event detection, calendar errors, and API schemas.
"""

from datetime import datetime, timedelta

import pytest

from contracts.calendar import CalendarEvent, CreateEventResult, DetectedEvent
from integrations.calendar import (
    get_event_detector,
    reset_event_detector,
)
from jarvis.errors import (
    CalendarAccessError,
    CalendarCreateError,
    CalendarError,
    ErrorCode,
    EventParseError,
    calendar_permission_denied,
)


class TestEventDetector:
    """Tests for EventDetectorImpl."""

    @pytest.fixture(autouse=True)
    def reset_detector(self):
        """Reset singleton detector before each test."""
        reset_event_detector()
        yield
        reset_event_detector()

    def test_detect_events_empty_text(self):
        """Returns empty list for empty text."""
        detector = get_event_detector()
        events = detector.detect_events("")
        assert events == []

    def test_detect_events_no_events(self):
        """Returns empty list when no events detected."""
        detector = get_event_detector()
        events = detector.detect_events("Hello, how are you?")
        assert events == []

    def test_detect_meeting_tomorrow(self):
        """Detects meeting mention with 'tomorrow'."""
        detector = get_event_detector()
        reference = datetime(2024, 1, 15, 10, 0, 0)
        events = detector.detect_events(
            "Let's have a meeting tomorrow at 3pm",
            reference_date=reference,
        )

        assert len(events) >= 1
        event = events[0]
        assert "meeting" in event.title.lower()
        assert event.start.date() == (reference + timedelta(days=1)).date()
        assert event.start.hour == 15  # 3pm
        assert event.confidence > 0.5

    def test_detect_dinner_with_time(self):
        """Detects dinner with specific time."""
        detector = get_event_detector()
        reference = datetime(2024, 1, 15, 10, 0, 0)
        events = detector.detect_events(
            "Let's have dinner tomorrow at 7pm at the restaurant",
            reference_date=reference,
        )

        assert len(events) >= 1
        event = events[0]
        assert "dinner" in event.title.lower()
        assert event.start.hour == 19  # 7pm
        assert event.all_day is False

    def test_detect_event_with_location(self):
        """Detects event with location."""
        detector = get_event_detector()
        reference = datetime(2024, 1, 15, 10, 0, 0)
        events = detector.detect_events(
            "Meeting tomorrow at Conference Room A",
            reference_date=reference,
        )

        assert len(events) >= 1
        event = events[0]
        assert event.location is not None
        assert "conference" in event.location.lower()

    def test_detect_day_of_week(self):
        """Detects events on specific day of week."""
        detector = get_event_detector()
        reference = datetime(2024, 1, 15, 10, 0, 0)  # Monday
        events = detector.detect_events(
            "Let's have lunch on Friday at noon",
            reference_date=reference,
        )

        assert len(events) >= 1
        event = events[0]
        # Should be the next Friday (Jan 19, 2024)
        assert event.start.weekday() == 4  # Friday

    def test_detect_all_day_event(self):
        """Detects all-day event without time."""
        detector = get_event_detector()
        reference = datetime(2024, 1, 15, 10, 0, 0)
        events = detector.detect_events(
            "Birthday party next Saturday",
            reference_date=reference,
        )

        assert len(events) >= 1
        event = events[0]
        assert event.start.weekday() == 5  # Saturday
        # Without specific time, should be all-day
        assert event.all_day is True

    def test_detect_events_batch(self):
        """Detects events in multiple texts."""
        detector = get_event_detector()
        reference = datetime(2024, 1, 15, 10, 0, 0)
        texts = [
            ("Meeting tomorrow at 2pm", 1),
            ("Hello there!", 2),
            ("Dinner on Friday", 3),
        ]
        results = detector.detect_events_batch(texts, reference_date=reference)

        assert len(results) == 3
        assert len(results[0]) >= 1  # Meeting
        assert len(results[1]) == 0  # No event
        assert len(results[2]) >= 1  # Dinner

    def test_detect_event_with_message_id(self):
        """Preserves message_id in detected event."""
        detector = get_event_detector()
        reference = datetime(2024, 1, 15, 10, 0, 0)
        events = detector.detect_events(
            "Meeting tomorrow at 2pm",
            reference_date=reference,
            message_id=12345,
        )

        assert len(events) >= 1
        assert events[0].message_id == 12345

    def test_detect_event_source_text(self):
        """Preserves source text in detected event."""
        detector = get_event_detector()
        text = "Let's have a meeting tomorrow at 3pm"
        events = detector.detect_events(text)

        assert len(events) >= 1
        assert events[0].source_text in text

    def test_confidence_with_time_higher(self):
        """Events with specific time have higher confidence."""
        detector = get_event_detector()
        reference = datetime(2024, 1, 15, 10, 0, 0)

        events_with_time = detector.detect_events(
            "Meeting tomorrow at 3pm",
            reference_date=reference,
        )
        events_without_time = detector.detect_events(
            "Meeting tomorrow",
            reference_date=reference,
        )

        if events_with_time and events_without_time:
            assert events_with_time[0].confidence >= events_without_time[0].confidence

    def test_singleton_pattern(self):
        """get_event_detector returns same instance."""
        detector1 = get_event_detector()
        detector2 = get_event_detector()
        assert detector1 is detector2


class TestCalendarErrors:
    """Tests for calendar-related errors."""

    def test_calendar_error_default(self):
        """CalendarError has sensible defaults."""
        error = CalendarError()
        assert "calendar" in error.message.lower()
        assert error.code == ErrorCode.CAL_NOT_AVAILABLE

    def test_calendar_access_error_default(self):
        """CalendarAccessError has appropriate defaults."""
        error = CalendarAccessError()
        assert "access" in error.message.lower() or "calendar" in error.message.lower()
        assert error.code == ErrorCode.CAL_ACCESS_DENIED

    def test_calendar_access_error_with_permission(self):
        """CalendarAccessError can include permission instructions."""
        error = CalendarAccessError("Permission denied", requires_permission=True)
        assert error.details["requires_permission"] is True
        assert "permission_instructions" in error.details
        assert len(error.details["permission_instructions"]) > 0

    def test_calendar_create_error_default(self):
        """CalendarCreateError has appropriate defaults."""
        error = CalendarCreateError()
        assert "create" in error.message.lower() or "event" in error.message.lower()
        assert error.code == ErrorCode.CAL_CREATE_FAILED

    def test_calendar_create_error_with_details(self):
        """CalendarCreateError can include event details."""
        error = CalendarCreateError(
            "Failed to create",
            calendar_id="cal-123",
            event_title="Team Meeting",
        )
        assert error.details["calendar_id"] == "cal-123"
        assert error.details["event_title"] == "Team Meeting"

    def test_event_parse_error_default(self):
        """EventParseError has appropriate defaults."""
        error = EventParseError()
        assert error.code == ErrorCode.CAL_PARSE_FAILED

    def test_event_parse_error_truncates_long_text(self):
        """EventParseError truncates long source text."""
        long_text = "x" * 300
        error = EventParseError("Failed to parse", source_text=long_text)
        assert len(error.details["source_text"]) < 300
        assert error.details["source_text"].endswith("...")

    def test_calendar_permission_denied_convenience(self):
        """calendar_permission_denied creates appropriate error."""
        error = calendar_permission_denied()
        assert isinstance(error, CalendarAccessError)
        assert error.details["requires_permission"] is True
        assert "permission_instructions" in error.details

    def test_calendar_errors_inheritance(self):
        """Calendar errors inherit from CalendarError and JarvisError."""
        from jarvis.errors import JarvisError

        errors = [
            CalendarError(),
            CalendarAccessError(),
            CalendarCreateError(),
            EventParseError(),
        ]

        for error in errors:
            assert isinstance(error, CalendarError)
            assert isinstance(error, JarvisError)

    def test_calendar_error_to_dict(self):
        """Calendar errors can be converted to dict."""
        error = CalendarCreateError(
            "Failed to create event",
            calendar_id="cal-123",
            event_title="Meeting",
        )
        result = error.to_dict()

        assert result["error"] == "CalendarCreateError"
        assert result["code"] == ErrorCode.CAL_CREATE_FAILED.value
        assert result["detail"] == "Failed to create event"
        assert "details" in result
        assert result["details"]["calendar_id"] == "cal-123"


class TestCalendarAPIErrorMapping:
    """Tests for API error status code mapping."""

    def test_calendar_access_error_is_403(self):
        """CalendarAccessError maps to 403."""
        from api.errors import get_status_code_for_error

        error = CalendarAccessError()
        assert get_status_code_for_error(error) == 403

    def test_calendar_create_error_is_500(self):
        """CalendarCreateError maps to 500."""
        from api.errors import get_status_code_for_error

        error = CalendarCreateError()
        assert get_status_code_for_error(error) == 500

    def test_event_parse_error_is_400(self):
        """EventParseError maps to 400."""
        from api.errors import get_status_code_for_error

        error = EventParseError()
        assert get_status_code_for_error(error) == 400


class TestDetectedEventDataclass:
    """Tests for DetectedEvent dataclass."""

    def test_detected_event_required_fields(self):
        """DetectedEvent requires title and start."""
        event = DetectedEvent(
            title="Meeting",
            start=datetime(2024, 1, 15, 10, 0, 0),
        )
        assert event.title == "Meeting"
        assert event.start == datetime(2024, 1, 15, 10, 0, 0)

    def test_detected_event_default_fields(self):
        """DetectedEvent has sensible defaults."""
        event = DetectedEvent(
            title="Meeting",
            start=datetime(2024, 1, 15, 10, 0, 0),
        )
        assert event.end is None
        assert event.location is None
        assert event.description is None
        assert event.all_day is False
        assert event.confidence == 0.0
        assert event.source_text == ""
        assert event.message_id is None

    def test_detected_event_all_fields(self):
        """DetectedEvent can have all fields set."""
        start = datetime(2024, 1, 15, 10, 0, 0)
        end = datetime(2024, 1, 15, 11, 0, 0)
        event = DetectedEvent(
            title="Meeting",
            start=start,
            end=end,
            location="Conference Room A",
            description="Weekly sync",
            all_day=False,
            confidence=0.85,
            source_text="Meeting tomorrow at 10am",
            message_id=12345,
        )

        assert event.title == "Meeting"
        assert event.start == start
        assert event.end == end
        assert event.location == "Conference Room A"
        assert event.description == "Weekly sync"
        assert event.all_day is False
        assert event.confidence == 0.85
        assert event.source_text == "Meeting tomorrow at 10am"
        assert event.message_id == 12345


class TestCalendarEventDataclass:
    """Tests for CalendarEvent dataclass."""

    def test_calendar_event_required_fields(self):
        """CalendarEvent requires id, calendar_id, calendar_name, title, start, end."""
        event = CalendarEvent(
            id="event-123",
            calendar_id="cal-456",
            calendar_name="Work",
            title="Meeting",
            start=datetime(2024, 1, 15, 10, 0, 0),
            end=datetime(2024, 1, 15, 11, 0, 0),
        )
        assert event.id == "event-123"
        assert event.calendar_id == "cal-456"
        assert event.calendar_name == "Work"
        assert event.title == "Meeting"

    def test_calendar_event_default_fields(self):
        """CalendarEvent has sensible defaults."""
        event = CalendarEvent(
            id="event-123",
            calendar_id="cal-456",
            calendar_name="Work",
            title="Meeting",
            start=datetime(2024, 1, 15, 10, 0, 0),
            end=datetime(2024, 1, 15, 11, 0, 0),
        )
        assert event.all_day is False
        assert event.location is None
        assert event.notes is None
        assert event.url is None
        assert event.attendees == []
        assert event.status == "confirmed"


class TestCreateEventResult:
    """Tests for CreateEventResult dataclass."""

    def test_success_result(self):
        """CreateEventResult for successful creation."""
        result = CreateEventResult(
            success=True,
            event_id="event-123",
        )
        assert result.success is True
        assert result.event_id == "event-123"
        assert result.error is None

    def test_failure_result(self):
        """CreateEventResult for failed creation."""
        result = CreateEventResult(
            success=False,
            error="Calendar not found",
        )
        assert result.success is False
        assert result.event_id is None
        assert result.error == "Calendar not found"
