"""Date and time utilities.

Provides specialized parsers and converters, specifically for
Apple-related epoch timestamps and other common assistant formats.
"""

from __future__ import annotations

from datetime import datetime, timezone

# Apple epoch starts at Jan 1, 2001
APPLE_EPOCH_OFFSET = 978307200


def apple_to_unix(apple_ts: float | int) -> float:
    """Convert Apple timestamp (seconds since 2001-01-01) to Unix timestamp.

    Note: iMessage often uses nanoseconds. This function expects seconds.
    If you have nanoseconds, divide by 1,000,000,000 first.
    """
    return float(apple_ts) + APPLE_EPOCH_OFFSET


def unix_to_apple(unix_ts: float) -> float:
    """Convert Unix timestamp to Apple timestamp (seconds since 2001-01-01)."""
    return unix_ts - APPLE_EPOCH_OFFSET


def parse_apple_timestamp(ns: int) -> datetime:
    """Parse Apple nanosecond timestamp to aware datetime object in UTC.

    Args:
        ns: Nanoseconds since Jan 1, 2001.

    Returns:
        A timezone-aware datetime object in UTC.
    """
    seconds = ns / 1_000_000_000
    unix_ts = seconds + APPLE_EPOCH_OFFSET
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc)


def now_utc() -> datetime:
    """Get current time as UTC-aware datetime."""
    return datetime.now(timezone.utc)
