"""Time-series aggregation for analytics.

Provides functions for aggregating message data by various time periods
with support for pre-computation and caching.

Uses pandas for vectorized operations when available (10-50x faster),
falls back to pure Python for minimal dependencies.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from jarvis.observability.insights import analyze_sentiment

if TYPE_CHECKING:
    from jarvis.contracts.imessage import Message

# Optional pandas import for vectorized operations
try:
    import pandas as pd  # noqa: F401

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass(slots=True)
class DailyAggregate:
    """Pre-computed daily aggregate for efficient querying."""

    date: str  # YYYY-MM-DD format
    total_messages: int = 0
    sent_count: int = 0
    received_count: int = 0
    avg_sentiment: float = 0.0
    unique_contacts: int = 0
    response_times: list[float] = field(default_factory=list)
    hourly_breakdown: dict[int, int] = field(default_factory=dict)
    emoji_count: int = 0
    attachment_count: int = 0


@dataclass(slots=True)
class HourlyAggregate:
    """Hourly message aggregate."""

    hour: int  # 0-23
    count: int = 0
    sent: int = 0
    received: int = 0
    avg_length: float = 0.0


@dataclass(slots=True)
class WeeklyAggregate:
    """Weekly message aggregate."""

    week: str  # YYYY-WNN format
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    total_messages: int = 0
    sent_count: int = 0
    received_count: int = 0
    avg_sentiment: float = 0.0
    daily_breakdown: dict[str, int] = field(default_factory=dict)
    active_contacts: int = 0


@dataclass(slots=True)
class MonthlyAggregate:
    """Monthly message aggregate."""

    month: str  # YYYY-MM format
    total_messages: int = 0
    sent_count: int = 0
    received_count: int = 0
    avg_sentiment: float = 0.0
    weekly_breakdown: dict[str, int] = field(default_factory=dict)
    active_contacts: int = 0
    avg_messages_per_day: float = 0.0


@dataclass(slots=True)
class _HourlyBucket:
    count: int = 0
    sent: int = 0
    received: int = 0
    total_length: int = 0


@dataclass(slots=True)
class _DailyBucket:
    messages: list[Message] = field(default_factory=list)
    sent: int = 0
    received: int = 0
    contacts: set[str] = field(default_factory=set)
    hourly: dict[int, int] = field(default_factory=dict)
    attachments: int = 0


@dataclass(slots=True)
class _WeeklyBucket:
    messages: list[Message] = field(default_factory=list)
    sent: int = 0
    received: int = 0
    contacts: set[str] = field(default_factory=set)
    daily: dict[str, int] = field(default_factory=dict)
    dates: list[datetime] = field(default_factory=list)


@dataclass(slots=True)
class _MonthlyBucket:
    messages: list[Message] = field(default_factory=list)
    sent: int = 0
    received: int = 0
    contacts: set[str] = field(default_factory=set)
    weekly: dict[str, int] = field(default_factory=dict)
    days: set[str] = field(default_factory=set)


def _aggregate_by_hour_vectorized(messages: list[Message]) -> list[HourlyAggregate]:
    """Vectorized implementation using pandas (10-50x faster)."""
    import pandas as pd

    # Build DataFrame from messages
    df = pd.DataFrame(
        [
            {
                "hour": msg.date.hour,
                "is_from_me": msg.is_from_me,
                "length": len(msg.text) if msg.text else 0,
            }
            for msg in messages
        ]
    )

    if df.empty:
        return [HourlyAggregate(hour=h) for h in range(24)]

    # Vectorized aggregation using groupby
    grouped = df.groupby("hour").agg(
        {
            "is_from_me": ["count", "sum"],
            "length": "sum",
        }
    )

    # Flatten column names
    grouped.columns = ["count", "sent", "total_length"]
    grouped["received"] = grouped["count"] - grouped["sent"]

    # Build results for all 24 hours
    result: list[HourlyAggregate] = []
    for hour in range(24):
        if hour in grouped.index:
            row = grouped.loc[hour]
            count = int(row["count"])
            avg_length = row["total_length"] / count if count > 0 else 0.0
            result.append(
                HourlyAggregate(
                    hour=hour,
                    count=count,
                    sent=int(row["sent"]),
                    received=int(row["received"]),
                    avg_length=round(avg_length, 1),
                )
            )
        else:
            result.append(HourlyAggregate(hour=hour))

    return result


def aggregate_by_hour(messages: list[Message]) -> list[HourlyAggregate]:
    """Aggregate messages by hour of day.

    Args:
        messages: List of messages to aggregate

    Returns:
        List of 24 HourlyAggregate objects (0-23)
    """
    # Use vectorized implementation if pandas available and enough messages
    if PANDAS_AVAILABLE and len(messages) > 100:
        return _aggregate_by_hour_vectorized(messages)

    # Fallback to pure Python for small datasets
    hourly_data: dict[int, _HourlyBucket] = {}

    for msg in messages:
        hour = msg.date.hour
        bucket = hourly_data.setdefault(hour, _HourlyBucket())
        bucket.count += 1
        if msg.is_from_me:
            bucket.sent += 1
        else:
            bucket.received += 1
        if msg.text:
            bucket.total_length += len(msg.text)

    result: list[HourlyAggregate] = []
    for hour in range(24):
        data = hourly_data.get(hour, _HourlyBucket())
        count = data.count
        avg_length = data.total_length / count if count > 0 else 0.0
        result.append(
            HourlyAggregate(
                hour=hour,
                count=data.count,
                sent=data.sent,
                received=data.received,
                avg_length=round(avg_length, 1),
            )
        )

    return result


def _aggregate_by_day_vectorized(
    messages: list[Message],
    include_sentiment: bool = False,
) -> list[DailyAggregate]:
    """Vectorized implementation using pandas (10-50x faster)."""
    import pandas as pd

    # Build DataFrame from messages
    df = pd.DataFrame(
        [
            {
                "date": msg.date.strftime("%Y-%m-%d"),
                "is_from_me": msg.is_from_me,
                "sender": msg.sender,
                "hour": msg.date.hour,
                "attachments": len(msg.attachments) if msg.attachments else 0,
                "text": msg.text,
            }
            for msg in messages
        ]
    )

    if df.empty:
        return []

    # Vectorized aggregation
    grouped = df.groupby("date").agg(
        {
            "is_from_me": ["count", "sum"],
            "sender": lambda x: x[~df.loc[x.index, "is_from_me"]].nunique(),
            "attachments": "sum",
        }
    )
    grouped.columns = ["total", "sent", "unique_contacts", "attachments"]
    grouped["received"] = grouped["total"] - grouped["sent"]

    # Calculate hourly breakdown per day
    hourly_pivot = df.groupby(["date", "hour"]).size().unstack(fill_value=0)

    # Build results
    result: list[DailyAggregate] = []
    for date_key in sorted(grouped.index):
        row = grouped.loc[date_key]
        hourly = hourly_pivot.loc[date_key].to_dict() if date_key in hourly_pivot.index else {}

        # Sentiment (still expensive, but we batch it)
        avg_sentiment = 0.0
        if include_sentiment:
            day_texts = df[df["date"] == date_key]["text"].dropna()
            if not day_texts.empty:
                sentiments = [analyze_sentiment(t).score for t in day_texts]
                avg_sentiment = sum(sentiments) / len(sentiments)

        result.append(
            DailyAggregate(
                date=date_key,
                total_messages=int(row["total"]),
                sent_count=int(row["sent"]),
                received_count=int(row["received"]),
                avg_sentiment=round(avg_sentiment, 3),
                unique_contacts=int(row["unique_contacts"]),
                hourly_breakdown=hourly,
                emoji_count=0,
                attachment_count=int(row["attachments"]),
            )
        )

    return result


def aggregate_by_day(
    messages: list[Message],
    include_sentiment: bool = False,
) -> list[DailyAggregate]:
    """Aggregate messages by day.

    Args:
        messages: List of messages to aggregate
        include_sentiment: Whether to compute sentiment (slower)

    Returns:
        List of DailyAggregate objects sorted by date
    """
    # Use vectorized implementation if pandas available and enough messages
    if PANDAS_AVAILABLE and len(messages) > 100:
        return _aggregate_by_day_vectorized(messages, include_sentiment)

    # Fallback to pure Python
    daily_data: dict[str, _DailyBucket] = {}

    for msg in messages:
        date_key = msg.date.strftime("%Y-%m-%d")
        data = daily_data.setdefault(date_key, _DailyBucket())
        data.messages.append(msg)
        if msg.is_from_me:
            data.sent += 1
        else:
            data.received += 1
            data.contacts.add(msg.sender)
        data.hourly[msg.date.hour] = data.hourly.get(msg.date.hour, 0) + 1
        if msg.attachments:
            data.attachments += len(msg.attachments)

    result: list[DailyAggregate] = []
    for date_key, data in sorted(daily_data.items()):
        msgs = data.messages
        total = len(msgs)

        # Sentiment (optional, expensive)
        avg_sentiment = 0.0
        if include_sentiment and msgs:
            sentiments = [analyze_sentiment(m.text).score for m in msgs if m.text]
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)

        result.append(
            DailyAggregate(
                date=date_key,
                total_messages=total,
                sent_count=data.sent,
                received_count=data.received,
                avg_sentiment=round(avg_sentiment, 3),
                unique_contacts=len(data.contacts),
                hourly_breakdown=dict(data.hourly),
                emoji_count=0,
                attachment_count=data.attachments,
            )
        )

    return result


def aggregate_by_week(
    messages: list[Message],
    include_sentiment: bool = False,
) -> list[WeeklyAggregate]:
    """Aggregate messages by week.

    Args:
        messages: List of messages to aggregate
        include_sentiment: Whether to compute sentiment

    Returns:
        List of WeeklyAggregate objects sorted by week
    """
    weekly_data: dict[str, _WeeklyBucket] = {}

    for msg in messages:
        week_key = msg.date.strftime("%Y-W%W")
        date_key = msg.date.strftime("%Y-%m-%d")
        data = weekly_data.setdefault(week_key, _WeeklyBucket())
        data.messages.append(msg)
        if msg.is_from_me:
            data.sent += 1
        else:
            data.received += 1
            data.contacts.add(msg.sender)
        data.daily[date_key] = data.daily.get(date_key, 0) + 1
        data.dates.append(msg.date)

    result: list[WeeklyAggregate] = []
    for week_key, data in sorted(weekly_data.items()):
        msgs = data.messages
        total = len(msgs)
        dates = sorted(data.dates)

        # Sentiment
        avg_sentiment = 0.0
        if include_sentiment and msgs:
            sentiments = [analyze_sentiment(m.text).score for m in msgs if m.text]
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)

        # Week start/end dates
        start_date = dates[0].strftime("%Y-%m-%d") if dates else ""
        end_date = dates[-1].strftime("%Y-%m-%d") if dates else ""

        result.append(
            WeeklyAggregate(
                week=week_key,
                start_date=start_date,
                end_date=end_date,
                total_messages=total,
                sent_count=data.sent,
                received_count=data.received,
                avg_sentiment=round(avg_sentiment, 3),
                daily_breakdown=dict(data.daily),
                active_contacts=len(data.contacts),
            )
        )

    return result


def aggregate_by_month(
    messages: list[Message],
    include_sentiment: bool = False,
) -> list[MonthlyAggregate]:
    """Aggregate messages by month.

    Args:
        messages: List of messages to aggregate
        include_sentiment: Whether to compute sentiment

    Returns:
        List of MonthlyAggregate objects sorted by month
    """
    monthly_data: dict[str, _MonthlyBucket] = {}

    for msg in messages:
        month_key = msg.date.strftime("%Y-%m")
        week_key = msg.date.strftime("%Y-W%W")
        date_key = msg.date.strftime("%Y-%m-%d")
        data = monthly_data.setdefault(month_key, _MonthlyBucket())
        data.messages.append(msg)
        if msg.is_from_me:
            data.sent += 1
        else:
            data.received += 1
            data.contacts.add(msg.sender)
        data.weekly[week_key] = data.weekly.get(week_key, 0) + 1
        data.days.add(date_key)

    result: list[MonthlyAggregate] = []
    for month_key, data in sorted(monthly_data.items()):
        msgs = data.messages
        total = len(msgs)
        days_active = len(data.days)

        # Sentiment
        avg_sentiment = 0.0
        if include_sentiment and msgs:
            sentiments = [analyze_sentiment(m.text).score for m in msgs if m.text]
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)

        # Messages per day average
        avg_per_day = total / days_active if days_active > 0 else 0.0

        result.append(
            MonthlyAggregate(
                month=month_key,
                total_messages=total,
                sent_count=data.sent,
                received_count=data.received,
                avg_sentiment=round(avg_sentiment, 3),
                weekly_breakdown=dict(data.weekly),
                active_contacts=len(data.contacts),
                avg_messages_per_day=round(avg_per_day, 2),
            )
        )

    return result


class TimeSeriesAggregator:
    """Time-series aggregator with caching support.

    Provides efficient aggregation with support for:
    - Pre-computed daily aggregates
    - Incremental updates
    - Date range queries
    """

    def __init__(self) -> None:
        """Initialize the aggregator."""
        self._daily_cache: dict[str, DailyAggregate] = {}
        self._last_update: datetime | None = None

    def update_daily_aggregates(
        self,
        messages: list[Message],
        include_sentiment: bool = False,
    ) -> None:
        """Update daily aggregate cache with new messages.

        Args:
            messages: New messages to aggregate
            include_sentiment: Whether to compute sentiment
        """
        aggregates = aggregate_by_day(messages, include_sentiment)
        for agg in aggregates:
            self._daily_cache[agg.date] = agg
        self._last_update = datetime.now(UTC)

    def get_daily_aggregates(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[DailyAggregate]:
        """Get daily aggregates within a date range.

        Args:
            start_date: Start date (YYYY-MM-DD), inclusive
            end_date: End date (YYYY-MM-DD), inclusive

        Returns:
            List of DailyAggregate objects in the range
        """
        result = []
        for date_key, agg in sorted(self._daily_cache.items()):
            if start_date and date_key < start_date:
                continue
            if end_date and date_key > end_date:
                continue
            result.append(agg)
        return result

    def get_activity_heatmap_data(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, object]]:
        """Get data formatted for activity heatmap visualization.

        Returns GitHub-style contribution data format.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of dicts with date and count for heatmap
        """
        aggregates = self.get_daily_aggregates(start_date, end_date)
        return [
            {
                "date": agg.date,
                "count": agg.total_messages,
                "level": self._get_activity_level(agg.total_messages),
            }
            for agg in aggregates
        ]

    def get_timeline_data(
        self,
        granularity: str = "day",
        start_date: str | None = None,
        end_date: str | None = None,
        messages: list[Message] | None = None,
    ) -> list[dict[str, object]]:
        """Get time-series data for chart visualization.

        Args:
            granularity: "hour", "day", "week", or "month"
            start_date: Start date filter
            end_date: End date filter
            messages: Optional messages for direct aggregation

        Returns:
            List of data points for time series chart
        """
        if messages is None:
            # Use cached daily aggregates
            aggregates = self.get_daily_aggregates(start_date, end_date)
            if granularity == "day":
                return [
                    {
                        "date": agg.date,
                        "total": agg.total_messages,
                        "sent": agg.sent_count,
                        "received": agg.received_count,
                    }
                    for agg in aggregates
                ]
            # Roll up to week/month if needed
            elif granularity == "week":
                weekly: dict[str, dict[str, int]] = {}
                for agg in aggregates:
                    dt = datetime.strptime(agg.date, "%Y-%m-%d")
                    week_key = dt.strftime("%Y-W%W")
                    bucket = weekly.setdefault(
                        week_key,
                        {"total": 0, "sent": 0, "received": 0},
                    )
                    bucket["total"] += agg.total_messages
                    bucket["sent"] += agg.sent_count
                    bucket["received"] += agg.received_count
                return [{"date": k, **v} for k, v in sorted(weekly.items())]
            elif granularity == "month":
                monthly: dict[str, dict[str, int]] = {}
                for agg in aggregates:
                    month_key = agg.date[:7]  # YYYY-MM
                    bucket = monthly.setdefault(
                        month_key,
                        {"total": 0, "sent": 0, "received": 0},
                    )
                    bucket["total"] += agg.total_messages
                    bucket["sent"] += agg.sent_count
                    bucket["received"] += agg.received_count
                return [{"date": k, **v} for k, v in sorted(monthly.items())]
            else:
                # hour not supported from daily cache
                return []
        else:
            # Direct aggregation from messages
            if granularity == "hour":
                hourly = aggregate_by_hour(messages)
                return [
                    {
                        "hour": h.hour,
                        "total": h.count,
                        "sent": h.sent,
                        "received": h.received,
                    }
                    for h in hourly
                ]
            elif granularity == "day":
                daily = aggregate_by_day(messages)
                return [
                    {
                        "date": d.date,
                        "total": d.total_messages,
                        "sent": d.sent_count,
                        "received": d.received_count,
                    }
                    for d in daily
                ]
            elif granularity == "week":
                weekly_agg = aggregate_by_week(messages)
                return [
                    {
                        "date": w.week,
                        "total": w.total_messages,
                        "sent": w.sent_count,
                        "received": w.received_count,
                    }
                    for w in weekly_agg
                ]
            elif granularity == "month":
                monthly_agg = aggregate_by_month(messages)
                return [
                    {
                        "date": m.month,
                        "total": m.total_messages,
                        "sent": m.sent_count,
                        "received": m.received_count,
                    }
                    for m in monthly_agg
                ]
            return []

    def _get_activity_level(self, count: int) -> int:
        """Get activity level (0-4) for heatmap coloring.

        Args:
            count: Message count for the day

        Returns:
            Level 0-4 (0=none, 4=very high)
        """
        if count == 0:
            return 0
        elif count <= 5:
            return 1
        elif count <= 15:
            return 2
        elif count <= 30:
            return 3
        return 4

    @property
    def last_update(self) -> datetime | None:
        """Get the timestamp of the last cache update."""
        return self._last_update

    @property
    def cached_date_range(self) -> tuple[str | None, str | None]:
        """Get the date range covered by the cache."""
        if not self._daily_cache:
            return None, None
        dates = sorted(self._daily_cache.keys())
        return dates[0], dates[-1]

    def clear_cache(self) -> None:
        """Clear all cached aggregates."""
        self._daily_cache.clear()
        self._last_update = None


# Global aggregator instance
_aggregator: TimeSeriesAggregator | None = None
_aggregator_lock = threading.Lock()


def get_aggregator() -> TimeSeriesAggregator:
    """Get the global time series aggregator instance.

    Uses double-checked locking pattern for thread safety.
    """
    global _aggregator
    if _aggregator is None:
        with _aggregator_lock:
            if _aggregator is None:
                _aggregator = TimeSeriesAggregator()
    return _aggregator
