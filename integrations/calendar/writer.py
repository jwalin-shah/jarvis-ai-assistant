"""macOS Calendar writer via AppleScript.

Provides write access to macOS Calendar for creating events.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timedelta

from jarvis.contracts.calendar import CreateEventResult, DetectedEvent
from jarvis.core.exceptions import CalendarAccessError

logger = logging.getLogger(__name__)

# AppleScript template to create an event
# Note: Line length in AppleScript templates is unavoidable
CREATE_EVENT_SCRIPT_TEMPLATE = """
tell application "Calendar"
    set targetCalendar to first calendar whose id is "{calendar_id}"
    set eventProps to {{summary:"{title}", start date:date "{start_date}", ¬
        end date:date "{end_date}"{extra_properties}}}
    set newEvent to make new event at end of events of targetCalendar with properties eventProps
    return uid of newEvent
end tell
"""

# AppleScript template for all-day events
CREATE_ALLDAY_EVENT_SCRIPT_TEMPLATE = """
tell application "Calendar"
    set targetCalendar to first calendar whose id is "{calendar_id}"
    set eventProps to {{summary:"{title}", start date:date "{start_date}", ¬
        end date:date "{end_date}", allday event:true{extra_properties}}}
    set newEvent to make new event at end of events of targetCalendar with properties eventProps
    return uid of newEvent
end tell
"""


class CalendarWriterImpl:
    """Writes events to macOS Calendar.

    Uses AppleScript to interact with the Calendar app.
    """

    def __init__(self) -> None:
        """Initialize the calendar writer."""
        self._access_checked = False
        self._has_access = False

    def check_access(self) -> bool:
        """Check if we have permission to write to Calendar.

        Returns:
            True if write access is available.
        """
        if self._access_checked:
            return self._has_access

        try:
            # Try a simple AppleScript command that requires write access
            # We just check if Calendar is accessible; actual write permission
            # is granted when creating events
            result = subprocess.run(
                ["osascript", "-e", 'tell application "Calendar" to name of calendars'],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self._has_access = result.returncode == 0
            self._access_checked = True
            return self._has_access
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning("Calendar write access check failed: %s", e)
            self._has_access = False
            self._access_checked = True
            return False

    def require_access(self) -> None:
        """Require calendar access, raising error if unavailable.

        Raises:
            CalendarAccessError: If calendar access is not available.
        """
        if not self.check_access():
            raise CalendarAccessError(
                "Calendar write access is required. Please grant Calendar permissions.",
                requires_permission=True,
            )

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
        self.require_access()

        # Escape special characters in strings
        title = self._escape_applescript(title)

        # Format dates for AppleScript
        start_str = start.strftime("%B %d, %Y %I:%M:%S %p")
        end_str = end.strftime("%B %d, %Y %I:%M:%S %p")

        # Build extra properties
        extra_props = []
        if location:
            extra_props.append(f'location:"{self._escape_applescript(location)}"')
        if notes:
            extra_props.append(f'description:"{self._escape_applescript(notes)}"')
        if url:
            extra_props.append(f'url:"{self._escape_applescript(url)}"')

        extra_properties = ""
        if extra_props:
            extra_properties = ", " + ", ".join(extra_props)

        # Select template based on all-day flag
        template = CREATE_ALLDAY_EVENT_SCRIPT_TEMPLATE if all_day else CREATE_EVENT_SCRIPT_TEMPLATE

        # Build script - escape calendar_id for AppleScript safety
        safe_calendar_id = self._escape_applescript(calendar_id)
        script = template.format(
            calendar_id=safe_calendar_id,
            title=title,
            start_date=start_str,
            end_date=end_str,
            extra_properties=extra_properties,
        )

        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or "Unknown error"
                logger.error("Failed to create event: %s", error_msg)
                return CreateEventResult(
                    success=False,
                    error=f"Failed to create event: {error_msg}",
                )

            event_id = result.stdout.strip()
            logger.info("Created event with ID: %s", event_id)

            return CreateEventResult(
                success=True,
                event_id=event_id,
            )

        except subprocess.TimeoutExpired:
            return CreateEventResult(
                success=False,
                error="Calendar request timed out",
            )
        except (subprocess.SubprocessError, OSError) as e:
            logger.exception("Failed to create event")
            return CreateEventResult(
                success=False,
                error=str(e),
            )

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
        return self.create_event(
            calendar_id=calendar_id,
            title=event.title,
            start=event.start,
            end=event.end or (event.start + timedelta(hours=1)),
            all_day=event.all_day,
            location=event.location,
            notes=event.description,
        )

    def _escape_applescript(self, text: str) -> str:
        """Escape special characters for AppleScript strings.

        Args:
            text: Text to escape.

        Returns:
            Escaped text.
        """
        # Escape backslashes first, then quotes
        text = text.replace("\\", "\\\\")
        text = text.replace('"', '\\"')
        text = text.replace("\n", "\\n")
        text = text.replace("\r", "\\r")
        return text


# Module-level singleton
_writer: CalendarWriterImpl | None = None


def get_calendar_writer() -> CalendarWriterImpl:
    """Get the singleton calendar writer.

    Returns:
        CalendarWriterImpl instance.
    """
    global _writer
    if _writer is None:
        _writer = CalendarWriterImpl()
    return _writer


def reset_calendar_writer() -> None:
    """Reset the singleton calendar writer."""
    global _writer
    _writer = None
