"""Smart timing analysis for optimal message send times.

Analyzes contact interaction history to suggest optimal send times,
respecting quiet hours and contact preferences.

Usage:
    from jarvis.scheduler.timing import get_timing_analyzer, suggest_send_time

    analyzer = get_timing_analyzer()
    suggestion = analyzer.suggest_time(contact_id=1)
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from jarvis.scheduler.models import (
    ContactTimingPrefs,
    QuietHours,
    TimingSuggestion,
)

logger = logging.getLogger(__name__)

# Default quiet hours (10 PM to 8 AM)
DEFAULT_QUIET_HOURS = QuietHours(start_hour=22, end_hour=8, enabled=True)

# Default timezone
DEFAULT_TIMEZONE = "America/Los_Angeles"


class TimingAnalyzer:
    """Analyzes interaction patterns to suggest optimal send times.

    Uses historical message data to learn when contacts are most responsive
    and suggests send times that maximize engagement while respecting
    quiet hours and preferences.
    """

    def __init__(
        self,
        default_quiet_hours: QuietHours | None = None,
        default_timezone: str = DEFAULT_TIMEZONE,
    ) -> None:
        """Initialize the timing analyzer.

        Args:
            default_quiet_hours: Default quiet hours configuration.
            default_timezone: Default timezone for contacts without one set.
        """
        self._default_quiet_hours = default_quiet_hours or DEFAULT_QUIET_HOURS
        self._default_timezone = default_timezone
        self._contact_prefs: dict[int, ContactTimingPrefs] = {}
        self._interaction_cache: dict[int, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()

    def set_contact_prefs(self, contact_id: int, prefs: ContactTimingPrefs) -> None:
        """Set timing preferences for a contact.

        Args:
            contact_id: The contact ID.
            prefs: The timing preferences.
        """
        with self._lock:
            self._contact_prefs[contact_id] = prefs

    def get_contact_prefs(self, contact_id: int) -> ContactTimingPrefs | None:
        """Get timing preferences for a contact.

        Args:
            contact_id: The contact ID.

        Returns:
            The preferences if set, None otherwise.
        """
        with self._lock:
            return self._contact_prefs.get(contact_id)

    def cache_interactions(self, contact_id: int, interactions: list[dict[str, Any]]) -> None:
        """Cache interaction history for a contact.

        Each interaction should have:
        - timestamp: datetime of the interaction
        - is_from_me: bool indicating if I sent the message
        - response_time_mins: optional float for response time

        Args:
            contact_id: The contact ID.
            interactions: List of interaction records.
        """
        with self._lock:
            self._interaction_cache[contact_id] = interactions

    def analyze_patterns(self, contact_id: int) -> dict[str, Any]:
        """Analyze interaction patterns for a contact.

        Args:
            contact_id: The contact ID.

        Returns:
            Dictionary with pattern analysis:
            - preferred_hours: List of hours with highest response rates
            - preferred_days: List of weekdays with highest engagement
            - avg_response_time_mins: Average response time
            - total_interactions: Total number of interactions
        """
        with self._lock:
            interactions = self._interaction_cache.get(contact_id, [])

        if not interactions:
            return {
                "preferred_hours": [],
                "preferred_days": [],
                "avg_response_time_mins": None,
                "total_interactions": 0,
            }

        # Count responses by hour and day
        hour_responses: Counter[int] = Counter()
        day_responses: Counter[int] = Counter()
        response_times: list[float] = []

        for interaction in interactions:
            ts = interaction.get("timestamp")
            if ts is None:
                continue

            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)

            # Count their messages (responses to me)
            if not interaction.get("is_from_me", False):
                hour_responses[ts.hour] += 1
                day_responses[ts.weekday()] += 1

            # Track response times
            if interaction.get("response_time_mins") is not None:
                response_times.append(interaction["response_time_mins"])

        # Find top hours (those with above-average response count)
        if hour_responses:
            avg_hour = sum(hour_responses.values()) / 24
            preferred_hours = sorted(
                [h for h, c in hour_responses.items() if c > avg_hour],
                key=lambda h: hour_responses[h],
                reverse=True,
            )[:5]
        else:
            preferred_hours = []

        # Find top days
        if day_responses:
            avg_day = sum(day_responses.values()) / 7
            preferred_days = sorted(
                [d for d, c in day_responses.items() if c > avg_day],
                key=lambda d: day_responses[d],
                reverse=True,
            )
        else:
            preferred_days = []

        # Calculate average response time
        avg_response_time = None
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)

        return {
            "preferred_hours": preferred_hours,
            "preferred_days": preferred_days,
            "avg_response_time_mins": avg_response_time,
            "total_interactions": len(interactions),
            "hour_distribution": dict(hour_responses),
            "day_distribution": dict(day_responses),
        }

    def suggest_time(
        self,
        contact_id: int,
        earliest: datetime | None = None,
        latest: datetime | None = None,
        respect_quiet_hours: bool = True,
        num_suggestions: int = 3,
    ) -> list[TimingSuggestion]:
        """Suggest optimal send times for a contact.

        Args:
            contact_id: The contact ID.
            earliest: Earliest acceptable time (defaults to now).
            latest: Latest acceptable time (defaults to 7 days from now).
            respect_quiet_hours: Whether to respect quiet hours.
            num_suggestions: Number of suggestions to return.

        Returns:
            List of TimingSuggestion objects, best first.
        """
        now = datetime.now(UTC)
        earliest = earliest or now
        latest = latest or (now + timedelta(days=7))

        # Get contact preferences
        prefs = self.get_contact_prefs(contact_id)
        timezone = prefs.timezone if prefs else self._default_timezone
        quiet_hours = (
            prefs.quiet_hours if prefs and prefs.quiet_hours else self._default_quiet_hours
        )

        # Get pattern analysis
        patterns = self.analyze_patterns(contact_id)
        preferred_hours = patterns.get("preferred_hours", []) or list(range(9, 18))
        preferred_days = patterns.get("preferred_days", []) or list(range(5))  # Mon-Fri

        # Override with contact-specific preferences if set
        if prefs:
            if prefs.preferred_hours:
                preferred_hours = prefs.preferred_hours
            if prefs.optimal_weekdays:
                preferred_days = prefs.optimal_weekdays

        # Generate candidate times
        candidates: list[tuple[datetime, float, dict[str, Any]]] = []

        try:
            tz = ZoneInfo(timezone) if timezone else ZoneInfo(self._default_timezone)
        except Exception:
            tz = ZoneInfo("UTC")

        # Iterate through time slots
        current = earliest
        while current < latest and len(candidates) < 100:  # Limit iterations
            local_time = current.astimezone(tz)

            # Calculate score for this time slot
            score = 0.0
            factors: dict[str, Any] = {}

            # Check quiet hours
            if respect_quiet_hours and quiet_hours.is_quiet_time(local_time):
                current += timedelta(hours=1)
                continue

            # Score based on hour preference
            hour = local_time.hour
            if hour in preferred_hours:
                hour_rank = preferred_hours.index(hour)
                hour_score = 1.0 - (hour_rank * 0.1)  # Top hour gets 1.0, next gets 0.9, etc.
                score += hour_score * 0.4
                factors["preferred_hour"] = True
                factors["hour_rank"] = hour_rank
            else:
                # Still give some score if within business hours
                if 9 <= hour <= 17:
                    score += 0.2
                    factors["business_hours"] = True

            # Score based on day preference
            weekday = local_time.weekday()
            if weekday in preferred_days:
                day_rank = preferred_days.index(weekday)
                day_score = 1.0 - (day_rank * 0.15)
                score += day_score * 0.3
                factors["preferred_day"] = True
                factors["day_rank"] = day_rank
            else:
                # Give small score for weekdays
                if weekday < 5:
                    score += 0.1
                    factors["weekday"] = True

            # Boost for being soon (prefer earlier times)
            time_diff = (current - now).total_seconds() / 3600  # Hours from now
            if time_diff < 24:
                score += 0.2 * (1 - time_diff / 24)
                factors["soon_boost"] = True
            elif time_diff < 48:
                score += 0.1
                factors["within_48h"] = True

            # Add to candidates
            if score > 0:
                candidates.append((current, score, factors))

            # Move to next hour
            current += timedelta(hours=1)

        # Sort by score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Build suggestions
        suggestions: list[TimingSuggestion] = []
        seen_hours: set[int] = set()

        for send_time, score, factors in candidates:
            local_time = send_time.astimezone(tz)

            # Diversify suggestions (don't suggest same hour multiple times)
            if local_time.hour in seen_hours:
                continue
            seen_hours.add(local_time.hour)

            # Build reason string
            reasons: list[str] = []
            if factors.get("preferred_hour"):
                reasons.append(f"high engagement at {local_time.strftime('%I %p').lstrip('0')}")
            if factors.get("preferred_day"):
                reasons.append(f"{local_time.strftime('%A')}s work well")
            if factors.get("soon_boost"):
                reasons.append("sending soon")

            reason = "; ".join(reasons) if reasons else "general availability"

            suggestion = TimingSuggestion(
                suggested_time=send_time,
                confidence=min(1.0, score),
                reason=reason,
                is_optimal=(len(suggestions) == 0),  # First is optimal
                factors=factors,
            )
            suggestions.append(suggestion)

            if len(suggestions) >= num_suggestions:
                break

        # If no suggestions, return immediate send
        if not suggestions:
            next_allowed = earliest
            if respect_quiet_hours:
                next_allowed = quiet_hours.next_allowed_time(earliest.astimezone(tz))
                if not isinstance(next_allowed, datetime):
                    next_allowed = earliest

            suggestions.append(
                TimingSuggestion(
                    suggested_time=next_allowed,
                    confidence=0.3,
                    reason="next available time",
                    is_optimal=True,
                    factors={"fallback": True},
                )
            )

        return suggestions

    def is_good_time(
        self,
        contact_id: int,
        send_time: datetime | None = None,
        respect_quiet_hours: bool = True,
    ) -> tuple[bool, str]:
        """Check if a given time is good for sending.

        Args:
            contact_id: The contact ID.
            send_time: The time to check (defaults to now).
            respect_quiet_hours: Whether to check quiet hours.

        Returns:
            Tuple of (is_good, reason).
        """
        if send_time is None:
            send_time = datetime.now(UTC)

        # Get contact preferences
        prefs = self.get_contact_prefs(contact_id)
        timezone = prefs.timezone if prefs else self._default_timezone
        quiet_hours = (
            prefs.quiet_hours if prefs and prefs.quiet_hours else self._default_quiet_hours
        )

        try:
            tz = ZoneInfo(timezone) if timezone else ZoneInfo(self._default_timezone)
        except Exception:
            tz = ZoneInfo("UTC")

        local_time = send_time.astimezone(tz)

        # Check quiet hours
        if respect_quiet_hours and quiet_hours.is_quiet_time(local_time):
            return (
                False,
                f"Within quiet hours ({quiet_hours.start_hour}:00-{quiet_hours.end_hour}:00)",
            )

        # Get pattern analysis
        patterns = self.analyze_patterns(contact_id)
        preferred_hours = patterns.get("preferred_hours", [])

        hour = local_time.hour
        weekday = local_time.weekday()

        # Build assessment
        reasons_good: list[str] = []
        reasons_bad: list[str] = []

        if preferred_hours and hour in preferred_hours:
            reasons_good.append("historically responsive at this hour")
        elif hour < 8 or hour > 21:
            reasons_bad.append("outside typical hours")

        preferred_days = patterns.get("preferred_days", [])
        if preferred_days and weekday in preferred_days:
            reasons_good.append("historically responsive on this day")
        elif weekday >= 5:
            reasons_bad.append("weekend")

        if reasons_bad and not reasons_good:
            return (False, "; ".join(reasons_bad))
        elif reasons_good:
            return (True, "; ".join(reasons_good))
        else:
            return (True, "acceptable time")


# Module-level singleton
_analyzer: TimingAnalyzer | None = None
_analyzer_lock = threading.Lock()


def get_timing_analyzer() -> TimingAnalyzer:
    """Get the singleton timing analyzer instance.

    Returns:
        Shared TimingAnalyzer instance.
    """
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                _analyzer = TimingAnalyzer()
    return _analyzer


def reset_timing_analyzer() -> None:
    """Reset the singleton timing analyzer (for testing)."""
    global _analyzer
    with _analyzer_lock:
        _analyzer = None


def suggest_send_time(
    contact_id: int,
    earliest: datetime | None = None,
    latest: datetime | None = None,
) -> TimingSuggestion:
    """Convenience function to get the best send time suggestion.

    Args:
        contact_id: The contact ID.
        earliest: Earliest acceptable time.
        latest: Latest acceptable time.

    Returns:
        The best TimingSuggestion.
    """
    analyzer = get_timing_analyzer()
    suggestions = analyzer.suggest_time(contact_id, earliest, latest, num_suggestions=1)
    return (
        suggestions[0]
        if suggestions
        else TimingSuggestion(
            suggested_time=earliest or datetime.now(UTC),
            confidence=0.1,
            reason="no data available",
        )
    )


# Export all public symbols
__all__ = [
    "TimingAnalyzer",
    "get_timing_analyzer",
    "reset_timing_analyzer",
    "suggest_send_time",
    "DEFAULT_QUIET_HOURS",
    "DEFAULT_TIMEZONE",
]
