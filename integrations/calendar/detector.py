"""Event detection from message text using NLP.

Uses regex patterns and dateutil for extracting dates, times, and event mentions
from natural language text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from dateutil import parser as dateutil_parser

from contracts.calendar import DetectedEvent

if TYPE_CHECKING:
    pass


# Common event indicator words/phrases
EVENT_INDICATORS = [
    r"meeting",
    r"appointment",
    r"call",
    r"interview",
    r"dinner",
    r"lunch",
    r"breakfast",
    r"coffee",
    r"party",
    r"event",
    r"conference",
    r"webinar",
    r"session",
    r"class",
    r"lesson",
    r"workout",
    r"gym",
    r"doctor",
    r"dentist",
    r"flight",
    r"trip",
    r"vacation",
    r"birthday",
    r"anniversary",
    r"wedding",
    r"concert",
    r"show",
    r"game",
    r"match",
    r"reservation",
]

# Time patterns
TIME_PATTERN = re.compile(
    r"""
    (?:
        (?P<hour>\d{1,2})
        (?::(?P<minute>\d{2}))?
        \s*
        (?P<ampm>am|pm|a\.m\.|p\.m\.)?
    |
        (?P<hour24>\d{1,2}):(?P<minute24>\d{2})
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Relative date patterns
RELATIVE_DATE_PATTERNS = {
    r"\btoday\b": 0,
    r"\btomorrow\b": 1,
    r"\bthe day after tomorrow\b": 2,
    r"\bnext week\b": 7,
    r"\bthis weekend\b": None,  # Special handling
    r"\bnext weekend\b": None,  # Special handling
}

# Day of week patterns
DAYS_OF_WEEK = {
    r"\bmonday\b": 0,
    r"\btuesday\b": 1,
    r"\bwednesday\b": 2,
    r"\bthursday\b": 3,
    r"\bfriday\b": 4,
    r"\bsaturday\b": 5,
    r"\bsunday\b": 6,
}

# Location indicators
LOCATION_PATTERNS = [
    r"at\s+(?P<location>[A-Z][^,.\n]+)",
    r"in\s+(?P<location>[A-Z][^,.\n]+)",
    r"@\s*(?P<location>[^\s,.\n]+)",
]

# Pre-compiled patterns at module level (avoids recompilation per instance)
_COMPILED_EVENT_PATTERN = re.compile(
    r"\b(" + "|".join(EVENT_INDICATORS) + r")\b",
    re.IGNORECASE,
)
_COMPILED_LOCATION_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in LOCATION_PATTERNS)


@dataclass
class DateTimeMatch:
    """Represents a matched date/time in text."""

    datetime_value: datetime
    original_text: str
    confidence: float
    is_all_day: bool = False


class EventDetectorImpl:
    """Detects calendar events from natural language text.

    Uses a combination of:
    - Regex patterns for common event types and time expressions
    - dateutil for flexible date parsing
    - Confidence scoring based on pattern matches
    """

    def __init__(self, default_event_duration_minutes: int = 60) -> None:
        """Initialize the event detector.

        Args:
            default_event_duration_minutes: Default duration for detected events.
        """
        self.default_duration = timedelta(minutes=default_event_duration_minutes)
        # Use pre-compiled module-level patterns (avoids recompilation per instance)
        self._event_pattern = _COMPILED_EVENT_PATTERN
        self._location_patterns = _COMPILED_LOCATION_PATTERNS

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
            message_id: Optional message ID for tracking source.

        Returns:
            List of detected events with confidence scores.
        """
        if not text or not text.strip():
            return []

        reference = reference_date or datetime.now()
        events: list[DetectedEvent] = []

        # Split into sentences for better context
        sentences = re.split(r"[.!?\n]+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check for event indicators
            event_match = self._event_pattern.search(sentence)
            if not event_match:
                continue

            # Try to extract date/time
            datetime_matches = self._extract_datetimes(sentence, reference)
            if not datetime_matches:
                continue

            # Use the first (most confident) datetime match
            dt_match = datetime_matches[0]

            # Extract location if present
            location = self._extract_location(sentence)

            # Build event title from context
            title = self._build_title(sentence, event_match.group(0))

            # Calculate confidence
            confidence = self._calculate_confidence(
                has_event_word=True,
                has_time=not dt_match.is_all_day,
                has_location=location is not None,
                datetime_confidence=dt_match.confidence,
            )

            # Determine end time
            end_time = dt_match.datetime_value + self.default_duration
            if dt_match.is_all_day:
                end_time = dt_match.datetime_value.replace(hour=23, minute=59)

            events.append(
                DetectedEvent(
                    title=title,
                    start=dt_match.datetime_value,
                    end=end_time,
                    location=location,
                    description=sentence,
                    all_day=dt_match.is_all_day,
                    confidence=confidence,
                    source_text=sentence,
                    message_id=message_id,
                )
            )

        # Sort by confidence (highest first)
        events.sort(key=lambda e: e.confidence, reverse=True)
        return events

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
        return [self.detect_events(text, reference_date, message_id) for text, message_id in texts]

    def _extract_datetimes(
        self,
        text: str,
        reference: datetime,
    ) -> list[DateTimeMatch]:
        """Extract date/time expressions from text.

        Args:
            text: Text to parse.
            reference: Reference date for relative expressions.

        Returns:
            List of matched date/times, sorted by confidence.
        """
        matches: list[DateTimeMatch] = []
        text_lower = text.lower()

        # Check for relative dates
        for pattern, days_offset in RELATIVE_DATE_PATTERNS.items():
            match = re.search(pattern, text_lower)
            if match:
                original_text = match.group()
                if days_offset is not None:
                    dt = reference + timedelta(days=days_offset)
                    # Try to find a time
                    time_match = TIME_PATTERN.search(text)
                    if time_match:
                        dt = self._apply_time_match(dt, time_match)
                        is_all_day = False
                        confidence = 0.9
                    else:
                        dt = dt.replace(hour=9, minute=0, second=0, microsecond=0)
                        is_all_day = True
                        confidence = 0.7

                    matches.append(
                        DateTimeMatch(
                            datetime_value=dt,
                            original_text=original_text,
                            confidence=confidence,
                            is_all_day=is_all_day,
                        )
                    )
                else:
                    # Handle weekend patterns
                    dt = self._get_next_weekend(reference, "next" in pattern)
                    matches.append(
                        DateTimeMatch(
                            datetime_value=dt,
                            original_text=original_text,
                            confidence=0.6,
                            is_all_day=True,
                        )
                    )

        # Check for days of week
        for pattern, weekday in DAYS_OF_WEEK.items():
            match = re.search(pattern, text_lower)
            if match:
                original_text = match.group()
                dt = self._get_next_weekday(reference, weekday)
                # Try to find a time
                time_match = TIME_PATTERN.search(text)
                if time_match:
                    dt = self._apply_time_match(dt, time_match)
                    is_all_day = False
                    confidence = 0.85
                else:
                    dt = dt.replace(hour=9, minute=0, second=0, microsecond=0)
                    is_all_day = True
                    confidence = 0.65

                matches.append(
                    DateTimeMatch(
                        datetime_value=dt,
                        original_text=original_text,
                        confidence=confidence,
                        is_all_day=is_all_day,
                    )
                )

        # Try dateutil parser for more complex expressions
        try:
            parsed_dt = dateutil_parser.parse(text, fuzzy=True, default=reference)
            # Only use if it's different from reference (actually found something)
            if parsed_dt.date() != reference.date() or parsed_dt.time() != reference.time():
                confidence = 0.75
                is_all_day = parsed_dt.hour == 0 and parsed_dt.minute == 0

                # Check if we didn't already match this date
                already_matched = any(m.datetime_value.date() == parsed_dt.date() for m in matches)

                if not already_matched:
                    matches.append(
                        DateTimeMatch(
                            datetime_value=parsed_dt,
                            original_text=text,
                            confidence=confidence,
                            is_all_day=is_all_day,
                        )
                    )
        except (ValueError, OverflowError):
            pass

        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    def _apply_time_match(self, dt: datetime, match: re.Match) -> datetime:  # type: ignore[type-arg]
        """Apply a time regex match to a datetime.

        Args:
            dt: Base datetime.
            match: Time pattern match.

        Returns:
            Datetime with time applied.
        """
        groups = match.groupdict()

        if groups.get("hour24"):
            hour = int(groups["hour24"])
            minute = int(groups["minute24"])
        else:
            hour = int(groups.get("hour", 0) or 0)
            minute = int(groups.get("minute", 0) or 0)
            ampm = (groups.get("ampm") or "").lower().replace(".", "")

            if ampm == "pm" and hour < 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0

        return dt.replace(hour=hour, minute=minute, second=0, microsecond=0)

    def _get_next_weekday(self, reference: datetime, target_weekday: int) -> datetime:
        """Get the next occurrence of a weekday.

        Args:
            reference: Reference date.
            target_weekday: Target weekday (0=Monday, 6=Sunday).

        Returns:
            Next occurrence of the target weekday.
        """
        days_ahead = target_weekday - reference.weekday()
        if days_ahead < 0:  # Target day already happened this week
            days_ahead += 7
        # If days_ahead == 0, it's today, so return today
        return reference + timedelta(days=days_ahead)

    def _get_next_weekend(self, reference: datetime, next_week: bool) -> datetime:
        """Get the next Saturday.

        Args:
            reference: Reference date.
            next_week: If True, get next week's weekend.

        Returns:
            Next Saturday date.
        """
        saturday = self._get_next_weekday(reference, 5)  # Saturday
        if next_week and saturday - reference < timedelta(days=7):
            saturday += timedelta(days=7)
        return saturday.replace(hour=9, minute=0, second=0, microsecond=0)

    def _extract_location(self, text: str) -> str | None:
        """Extract location from text.

        Args:
            text: Text to search.

        Returns:
            Location string if found, None otherwise.
        """
        for pattern in self._location_patterns:
            match = pattern.search(text)
            if match:
                location = match.group("location").strip()
                # Clean up common trailing words
                location = re.sub(r"\s+(at|on|for|to)\s*$", "", location, flags=re.IGNORECASE)
                if location:
                    return location
        return None

    def _build_title(self, sentence: str, event_word: str) -> str:
        """Build an event title from context.

        Args:
            sentence: Full sentence.
            event_word: The matched event indicator word.

        Returns:
            Event title.
        """
        # Try to find a more specific title around the event word
        # Look for patterns like "dinner with John" or "meeting about project"
        patterns = [
            rf"({event_word}\s+(?:with|about|for|re:?)\s+[A-Za-z\s]+)",
            rf"([A-Za-z]+\s+{event_word})",
        ]

        for pattern in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                # Capitalize first letter
                return title[0].upper() + title[1:]

        # Fall back to just the event word, capitalized
        return event_word.capitalize()

    def _calculate_confidence(
        self,
        has_event_word: bool,
        has_time: bool,
        has_location: bool,
        datetime_confidence: float,
    ) -> float:
        """Calculate overall confidence score.

        Args:
            has_event_word: Whether an event indicator was found.
            has_time: Whether a specific time was found.
            has_location: Whether a location was found.
            datetime_confidence: Confidence from datetime parsing.

        Returns:
            Overall confidence score (0.0 to 1.0).
        """
        score = 0.0

        if has_event_word:
            score += 0.3

        if has_time:
            score += 0.25
        else:
            score += 0.1  # All-day events are less certain

        if has_location:
            score += 0.15

        # Blend with datetime parsing confidence
        score = (score + datetime_confidence) / 2

        return min(1.0, score)


# Module-level singleton
_detector: EventDetectorImpl | None = None


def get_event_detector() -> EventDetectorImpl:
    """Get the singleton event detector.

    Returns:
        EventDetectorImpl instance.
    """
    global _detector
    if _detector is None:
        _detector = EventDetectorImpl()
    return _detector


def reset_event_detector() -> None:
    """Reset the singleton event detector."""
    global _detector
    _detector = None
