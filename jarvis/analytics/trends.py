"""Trend detection algorithms for analytics.

Provides statistical analysis for detecting:
- Overall trends (increasing, decreasing, stable)
- Anomalies (unusual activity spikes/drops)
- Peak periods (most active times)
- Seasonal patterns
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contracts.imessage import Message


@dataclass
class TrendResult:
    """Result of trend analysis."""

    direction: str  # "increasing", "decreasing", "stable"
    percentage_change: float
    confidence: float  # 0-1 confidence in the trend
    start_value: float
    end_value: float
    period_count: int


@dataclass
class AnomalyResult:
    """Detected anomaly in time series data."""

    date: str
    value: float
    expected_value: float
    deviation: float  # Standard deviations from mean
    anomaly_type: str  # "spike" or "drop"


@dataclass
class PeakPeriod:
    """Detected peak activity period."""

    period_type: str  # "hour", "day", "week"
    period_value: str | int  # e.g., 14 for hour, "Monday" for day
    count: int
    percentage_of_total: float


@dataclass
class TrendAnalysis:
    """Complete trend analysis results."""

    overall_trend: TrendResult
    weekly_trend: TrendResult | None
    sentiment_trend: TrendResult | None
    anomalies: list[AnomalyResult]
    peak_hours: list[PeakPeriod]
    peak_days: list[PeakPeriod]
    seasonality_detected: bool = False
    seasonality_period: str | None = None


def detect_trend(
    values: list[float],
    window_size: int = 3,
) -> TrendResult:
    """Detect trend in a series of values.

    Uses moving average comparison to determine trend direction.

    Args:
        values: List of numeric values (chronologically ordered)
        window_size: Window size for moving average

    Returns:
        TrendResult with trend direction and statistics
    """
    if len(values) < window_size * 2:
        return TrendResult(
            direction="stable",
            percentage_change=0.0,
            confidence=0.0,
            start_value=values[0] if values else 0.0,
            end_value=values[-1] if values else 0.0,
            period_count=len(values),
        )

    # Compute moving averages for start and end periods
    start_avg = sum(values[:window_size]) / window_size
    end_avg = sum(values[-window_size:]) / window_size

    # Calculate percentage change
    if start_avg == 0:
        pct_change = 100.0 if end_avg > 0 else 0.0
    else:
        pct_change = ((end_avg - start_avg) / start_avg) * 100

    # Determine direction
    if pct_change > 10:
        direction = "increasing"
    elif pct_change < -10:
        direction = "decreasing"
    else:
        direction = "stable"

    # Calculate confidence based on consistency of trend
    # Count how many periods show the same direction
    increases = sum(1 for i in range(1, len(values)) if values[i] > values[i - 1])
    decreases = sum(1 for i in range(1, len(values)) if values[i] < values[i - 1])
    total_changes = len(values) - 1

    if total_changes == 0:
        confidence = 0.0
    elif direction == "increasing":
        confidence = increases / total_changes
    elif direction == "decreasing":
        confidence = decreases / total_changes
    else:
        # For stable, confidence is based on low variance
        stables = total_changes - increases - decreases  # noqa: F841
        confidence = max(
            0.5,
            1.0 - (abs(pct_change) / 100),
        )

    return TrendResult(
        direction=direction,
        percentage_change=round(pct_change, 2),
        confidence=round(confidence, 2),
        start_value=start_avg,
        end_value=end_avg,
        period_count=len(values),
    )


def detect_anomalies(
    data: list[tuple[str, float]],
    threshold_std: float = 2.0,
) -> list[AnomalyResult]:
    """Detect anomalies in time series data.

    Uses standard deviation-based detection.

    Args:
        data: List of (date_key, value) tuples
        threshold_std: Number of standard deviations for anomaly threshold

    Returns:
        List of detected anomalies
    """
    if len(data) < 5:
        return []

    values = [v for _, v in data]

    # Calculate mean and standard deviation
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std_dev = variance**0.5

    if std_dev == 0:
        return []

    anomalies: list[AnomalyResult] = []
    for date_key, value in data:
        deviation = (value - mean) / std_dev
        if abs(deviation) >= threshold_std:
            anomaly_type = "spike" if deviation > 0 else "drop"
            anomalies.append(
                AnomalyResult(
                    date=date_key,
                    value=value,
                    expected_value=round(mean, 2),
                    deviation=round(deviation, 2),
                    anomaly_type=anomaly_type,
                )
            )

    return anomalies


def detect_peak_periods(
    messages: list[Message],
    top_n: int = 3,
) -> tuple[list[PeakPeriod], list[PeakPeriod]]:
    """Detect peak activity hours and days.

    Args:
        messages: List of messages to analyze
        top_n: Number of top periods to return

    Returns:
        Tuple of (peak_hours, peak_days)
    """
    if not messages:
        return [], []

    total = len(messages)

    # Count by hour
    hour_counts: Counter[int] = Counter(m.date.hour for m in messages)
    peak_hours = [
        PeakPeriod(
            period_type="hour",
            period_value=hour,
            count=count,
            percentage_of_total=round((count / total) * 100, 1),
        )
        for hour, count in hour_counts.most_common(top_n)
    ]

    # Count by day of week
    day_counts: Counter[str] = Counter(m.date.strftime("%A") for m in messages)
    peak_days = [
        PeakPeriod(
            period_type="day",
            period_value=day,
            count=count,
            percentage_of_total=round((count / total) * 100, 1),
        )
        for day, count in day_counts.most_common(top_n)
    ]

    return peak_hours, peak_days


def detect_seasonality(
    daily_counts: dict[str, int],
    min_weeks: int = 4,
) -> tuple[bool, str | None]:
    """Detect weekly seasonality in message patterns.

    Args:
        daily_counts: Dict of date -> message count
        min_weeks: Minimum weeks of data needed

    Returns:
        Tuple of (seasonality_detected, period_description)
    """
    if len(daily_counts) < min_weeks * 7:
        return False, None

    # Group by day of week
    day_of_week_counts: dict[int, list[int]] = {i: [] for i in range(7)}

    for date_str, count in daily_counts.items():
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        day_of_week_counts[dt.weekday()].append(count)

    # Calculate average for each day of week
    day_averages = {
        day: sum(counts) / len(counts) if counts else 0
        for day, counts in day_of_week_counts.items()
    }

    # Check for significant difference between weekdays and weekends
    weekday_avg = sum(day_averages[i] for i in range(5)) / 5
    weekend_avg = sum(day_averages[i] for i in [5, 6]) / 2

    if weekday_avg == 0 and weekend_avg == 0:
        return False, None

    total_avg = (weekday_avg * 5 + weekend_avg * 2) / 7

    # Significant if weekday/weekend difference is > 30%
    if total_avg > 0:
        weekday_diff = abs(weekday_avg - total_avg) / total_avg
        weekend_diff = abs(weekend_avg - total_avg) / total_avg

        if weekday_diff > 0.3 or weekend_diff > 0.3:
            if weekday_avg > weekend_avg:
                return True, "Higher weekday activity"
            else:
                return True, "Higher weekend activity"

    return False, None


class TrendAnalyzer:
    """Comprehensive trend analyzer for conversation analytics.

    Provides trend detection, anomaly detection, and pattern analysis
    for message data.
    """

    def __init__(
        self,
        anomaly_threshold: float = 2.0,
        trend_window_size: int = 3,
    ) -> None:
        """Initialize trend analyzer.

        Args:
            anomaly_threshold: Standard deviations for anomaly detection
            trend_window_size: Window size for trend calculation
        """
        self.anomaly_threshold = anomaly_threshold
        self.trend_window_size = trend_window_size

    def analyze_message_trends(self, messages: list[Message]) -> TrendAnalysis:
        """Perform comprehensive trend analysis on messages.

        Args:
            messages: List of messages to analyze

        Returns:
            TrendAnalysis with all trend metrics
        """
        if not messages:
            return TrendAnalysis(
                overall_trend=TrendResult(
                    direction="stable",
                    percentage_change=0.0,
                    confidence=0.0,
                    start_value=0.0,
                    end_value=0.0,
                    period_count=0,
                ),
                weekly_trend=None,
                sentiment_trend=None,
                anomalies=[],
                peak_hours=[],
                peak_days=[],
            )

        # Sort messages chronologically
        sorted_msgs = sorted(messages, key=lambda m: m.date)

        # Group by day for daily counts
        daily_counts: Counter[str] = Counter(m.date.strftime("%Y-%m-%d") for m in sorted_msgs)
        daily_data = sorted(daily_counts.items())
        daily_values = [v for _, v in daily_data]

        # Overall trend from daily counts
        overall_trend = detect_trend(daily_values, self.trend_window_size)

        # Weekly trend (aggregate by week)
        weekly_counts: Counter[str] = Counter(m.date.strftime("%Y-W%W") for m in sorted_msgs)
        weekly_values = [v for _, v in sorted(weekly_counts.items())]
        weekly_trend = (
            detect_trend(weekly_values, self.trend_window_size) if len(weekly_values) >= 4 else None
        )

        # Anomaly detection
        anomalies = detect_anomalies(list(daily_data), self.anomaly_threshold)

        # Peak periods
        peak_hours, peak_days = detect_peak_periods(sorted_msgs)

        # Seasonality
        seasonality_detected, seasonality_period = detect_seasonality(dict(daily_counts))

        return TrendAnalysis(
            overall_trend=overall_trend,
            weekly_trend=weekly_trend,
            sentiment_trend=None,  # Can be computed separately if needed
            anomalies=anomalies,
            peak_hours=peak_hours,
            peak_days=peak_days,
            seasonality_detected=seasonality_detected,
            seasonality_period=seasonality_period,
        )

    def analyze_contact_activity_trend(
        self,
        messages: list[Message],
    ) -> dict[str, TrendResult]:
        """Analyze activity trends per contact.

        Args:
            messages: Messages to analyze

        Returns:
            Dict mapping contact_id to TrendResult
        """
        # Group by contact
        contact_messages: dict[str, list[Message]] = {}
        for msg in messages:
            if not msg.is_from_me:
                if msg.chat_id not in contact_messages:
                    contact_messages[msg.chat_id] = []
                contact_messages[msg.chat_id].append(msg)

        results: dict[str, TrendResult] = {}
        for contact_id, msgs in contact_messages.items():
            daily_counts: Counter[str] = Counter(m.date.strftime("%Y-%m-%d") for m in msgs)
            daily_values = [v for _, v in sorted(daily_counts.items())]
            if len(daily_values) >= 4:
                results[contact_id] = detect_trend(daily_values, self.trend_window_size)

        return results

    def get_trending_contacts(
        self,
        messages: list[Message],
        top_n: int = 5,
    ) -> list[tuple[str, str | None, TrendResult]]:
        """Get contacts with the most significant activity trends.

        Args:
            messages: Messages to analyze
            top_n: Number of top trending contacts

        Returns:
            List of (contact_id, contact_name, trend) tuples
        """
        # Group messages by contact
        contact_data: dict[str, tuple[str | None, list[Message]]] = {}
        for msg in messages:
            chat_id = msg.chat_id
            if chat_id not in contact_data:
                name = msg.sender_name if not msg.is_from_me else None
                contact_data[chat_id] = (name, [])
            contact_data[chat_id][1].append(msg)

        # Analyze trends
        trending: list[tuple[str, str | None, TrendResult]] = []
        for contact_id, (name, msgs) in contact_data.items():
            daily_counts: Counter[str] = Counter(m.date.strftime("%Y-%m-%d") for m in msgs)
            daily_values = [v for _, v in sorted(daily_counts.items())]
            if len(daily_values) >= 4:
                trend = detect_trend(daily_values, self.trend_window_size)
                # Only include significant trends
                if abs(trend.percentage_change) > 15 and trend.confidence > 0.5:
                    trending.append((contact_id, name, trend))

        # Sort by absolute percentage change
        trending.sort(key=lambda x: abs(x[2].percentage_change), reverse=True)
        return trending[:top_n]

    def compare_periods(
        self,
        messages: list[Message],
        current_days: int = 7,
        previous_days: int = 7,
    ) -> dict[str, float]:
        """Compare metrics between current and previous period.

        Args:
            messages: Messages to analyze
            current_days: Days in current period
            previous_days: Days in previous period

        Returns:
            Dict with percentage changes for various metrics
        """
        sorted_msgs = sorted(messages, key=lambda m: m.date, reverse=True)
        if not sorted_msgs:
            return {}

        latest_date = sorted_msgs[0].date

        # Split into periods
        from datetime import timedelta

        current_start = latest_date - timedelta(days=current_days)
        previous_start = current_start - timedelta(days=previous_days)

        current_msgs = [m for m in sorted_msgs if m.date >= current_start]
        previous_msgs = [m for m in sorted_msgs if previous_start <= m.date < current_start]

        # Calculate metrics
        def safe_pct_change(curr: float, prev: float) -> float:
            if prev == 0:
                return 100.0 if curr > 0 else 0.0
            return round(((curr - prev) / prev) * 100, 1)

        current_total = len(current_msgs)
        previous_total = len(previous_msgs)

        current_sent = sum(1 for m in current_msgs if m.is_from_me)
        previous_sent = sum(1 for m in previous_msgs if m.is_from_me)

        current_contacts = len({m.chat_id for m in current_msgs})
        previous_contacts = len({m.chat_id for m in previous_msgs})

        return {
            "total_messages_change": safe_pct_change(current_total, previous_total),
            "sent_messages_change": safe_pct_change(current_sent, previous_sent),
            "active_contacts_change": safe_pct_change(current_contacts, previous_contacts),
            "current_period_total": current_total,
            "previous_period_total": previous_total,
        }


# Global analyzer instance
_trend_analyzer: TrendAnalyzer | None = None
_trend_analyzer_lock = threading.Lock()


def get_trend_analyzer() -> TrendAnalyzer:
    """Get the global trend analyzer instance.

    Uses double-checked locking pattern for thread safety.
    """
    global _trend_analyzer
    if _trend_analyzer is None:
        with _trend_analyzer_lock:
            if _trend_analyzer is None:
                _trend_analyzer = TrendAnalyzer()
    return _trend_analyzer
