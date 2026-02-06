"""Core analytics computation engine.

Provides centralized analytics computation with caching and
pre-computation support for dashboard performance.
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from jarvis.observability.insights import (
    EMOJI_PATTERN,
    analyze_sentiment,
    analyze_sentiment_trends,
)
from jarvis.metrics import TTLCache

if TYPE_CHECKING:
    from contracts.imessage import Message


@dataclass
class OverviewMetrics:
    """Dashboard overview metrics."""

    total_messages: int = 0
    total_sent: int = 0
    total_received: int = 0
    active_conversations: int = 0
    avg_messages_per_day: float = 0.0
    avg_response_time_minutes: float | None = None
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    peak_hour: int | None = None
    peak_day: str | None = None
    date_range_start: datetime | None = None
    date_range_end: datetime | None = None


@dataclass
class ContactAnalytics:
    """Analytics for a specific contact."""

    contact_id: str
    contact_name: str | None
    total_messages: int = 0
    sent_count: int = 0
    received_count: int = 0
    avg_response_time_minutes: float | None = None
    sentiment_score: float = 0.0
    last_message_date: datetime | None = None
    message_trend: str = "stable"  # increasing, decreasing, stable
    engagement_score: float = 0.0  # 0-100


@dataclass
class EmojiStats:
    """Emoji usage statistics."""

    total_count: int = 0
    unique_count: int = 0
    top_emojis: dict[str, int] = field(default_factory=dict)
    emojis_per_message: float = 0.0
    emoji_sentiment_ratio: float = 0.0  # positive / (positive + negative)


@dataclass
class MessageLengthStats:
    """Message length statistics."""

    avg_length: float = 0.0
    median_length: float = 0.0
    min_length: int = 0
    max_length: int = 0
    short_count: int = 0  # <= 20 chars
    medium_count: int = 0  # 21-100 chars
    long_count: int = 0  # 101-300 chars
    very_long_count: int = 0  # > 300 chars


@dataclass
class ReplyTypeDistribution:
    """Distribution of AI reply suggestion types."""

    agree_count: int = 0
    decline_count: int = 0
    question_count: int = 0
    neutral_count: int = 0
    custom_count: int = 0
    total_suggestions: int = 0
    acceptance_rate: float = 0.0


@dataclass
class AnalyticsResult:
    """Complete analytics result."""

    overview: OverviewMetrics
    contacts: list[ContactAnalytics]
    emoji_stats: EmojiStats
    message_length_stats: MessageLengthStats
    hourly_distribution: dict[int, int]
    daily_distribution: dict[str, int]
    monthly_counts: dict[str, int]
    weekly_counts: dict[str, int]
    sentiment_trends: list[dict[str, object]]
    reply_distribution: ReplyTypeDistribution | None = None
    computed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class AnalyticsEngine:
    """Core analytics computation engine.

    Provides efficient analytics computation with caching support
    for dashboard and reporting use cases.
    """

    def __init__(self, cache_ttl_seconds: float = 300.0) -> None:
        """Initialize analytics engine.

        Args:
            cache_ttl_seconds: TTL for cached results (default 5 minutes)
        """
        self._cache = TTLCache(ttl_seconds=cache_ttl_seconds, maxsize=50)

    def compute_overview(
        self,
        messages: list[Message],
        time_range_days: int | None = None,
    ) -> OverviewMetrics:
        """Compute overview metrics for messages.

        Args:
            messages: List of messages to analyze
            time_range_days: Optional time range filter in days

        Returns:
            OverviewMetrics with computed statistics
        """
        if not messages:
            return OverviewMetrics()

        # Filter by time range if specified
        if time_range_days is not None:
            cutoff = datetime.now(UTC) - timedelta(days=time_range_days)
            messages = [m for m in messages if m.date >= cutoff]

        if not messages:
            return OverviewMetrics()

        # Sort chronologically
        sorted_msgs = sorted(messages, key=lambda m: m.date)

        # Basic counts
        total = len(sorted_msgs)
        sent = sum(1 for m in sorted_msgs if m.is_from_me)
        received = total - sent

        # Unique conversations
        unique_chats = len({m.chat_id for m in sorted_msgs})

        # Date range
        first_date = sorted_msgs[0].date
        last_date = sorted_msgs[-1].date
        days_span = max((last_date - first_date).days, 1)

        # Messages per day
        msg_per_day = total / days_span

        # Response time calculation
        avg_response = self._compute_avg_response_time(sorted_msgs)

        # Sentiment
        all_text = " ".join(m.text for m in sorted_msgs if m.text)
        sentiment = analyze_sentiment(all_text)

        # Peak times
        hour_counts: Counter[int] = Counter(m.date.hour for m in sorted_msgs)
        day_counts: Counter[str] = Counter(m.date.strftime("%A") for m in sorted_msgs)
        peak_hour = hour_counts.most_common(1)[0][0] if hour_counts else None
        peak_day = day_counts.most_common(1)[0][0] if day_counts else None

        return OverviewMetrics(
            total_messages=total,
            total_sent=sent,
            total_received=received,
            active_conversations=unique_chats,
            avg_messages_per_day=round(msg_per_day, 2),
            avg_response_time_minutes=avg_response,
            sentiment_score=sentiment.score,
            sentiment_label=sentiment.label,
            peak_hour=peak_hour,
            peak_day=peak_day,
            date_range_start=first_date,
            date_range_end=last_date,
        )

    def compute_contact_analytics(
        self,
        messages: list[Message],
        contact_id: str,
        contact_name: str | None = None,
    ) -> ContactAnalytics:
        """Compute analytics for a specific contact.

        Args:
            messages: Messages for this contact
            contact_id: Contact identifier (chat_id)
            contact_name: Optional display name

        Returns:
            ContactAnalytics with computed statistics
        """
        if not messages:
            return ContactAnalytics(
                contact_id=contact_id,
                contact_name=contact_name,
            )

        sorted_msgs = sorted(messages, key=lambda m: m.date)

        total = len(sorted_msgs)
        sent = sum(1 for m in sorted_msgs if m.is_from_me)
        received = total - sent

        # Response time
        avg_response = self._compute_avg_response_time(sorted_msgs)

        # Sentiment
        all_text = " ".join(m.text for m in sorted_msgs if m.text)
        sentiment = analyze_sentiment(all_text)

        # Message trend (compare first half to second half)
        trend = self._compute_message_trend(sorted_msgs)

        # Engagement score (balance + frequency)
        engagement = self._compute_engagement_score(
            sent_count=sent,
            received_count=received,
            total_messages=total,
            days_active=len({m.date.date() for m in sorted_msgs}),
        )

        return ContactAnalytics(
            contact_id=contact_id,
            contact_name=contact_name,
            total_messages=total,
            sent_count=sent,
            received_count=received,
            avg_response_time_minutes=avg_response,
            sentiment_score=sentiment.score,
            last_message_date=sorted_msgs[-1].date if sorted_msgs else None,
            message_trend=trend,
            engagement_score=engagement,
        )

    def compute_emoji_stats(self, messages: list[Message]) -> EmojiStats:
        """Compute emoji usage statistics.

        Args:
            messages: Messages to analyze

        Returns:
            EmojiStats with emoji usage data
        """
        if not messages:
            return EmojiStats()

        emoji_counts: Counter[str] = Counter()
        messages_with_emojis = 0

        for msg in messages:
            if not msg.text:
                continue
            emojis = EMOJI_PATTERN.findall(msg.text)
            if emojis:
                messages_with_emojis += 1
                for emoji_group in emojis:
                    for char in emoji_group:
                        if EMOJI_PATTERN.match(char):
                            emoji_counts[char] += 1

        total_emojis = sum(emoji_counts.values())
        unique_emojis = len(emoji_counts)

        # Top 10 emojis
        top_emojis = dict(emoji_counts.most_common(10))

        # Emojis per message
        emojis_per_msg = total_emojis / len(messages) if messages else 0.0

        return EmojiStats(
            total_count=total_emojis,
            unique_count=unique_emojis,
            top_emojis=top_emojis,
            emojis_per_message=round(emojis_per_msg, 3),
        )

    def compute_message_length_stats(self, messages: list[Message]) -> MessageLengthStats:
        """Compute message length statistics.

        Args:
            messages: Messages to analyze

        Returns:
            MessageLengthStats with length distribution
        """
        if not messages:
            return MessageLengthStats()

        lengths = [len(m.text) for m in messages if m.text]
        if not lengths:
            return MessageLengthStats()

        sorted_lengths = sorted(lengths)
        n = len(sorted_lengths)

        # Median
        mid = n // 2
        if n % 2 == 0:
            median = (sorted_lengths[mid - 1] + sorted_lengths[mid]) / 2
        else:
            median = sorted_lengths[mid]

        # Distribution buckets
        short = sum(1 for l in lengths if l <= 20)
        medium = sum(1 for l in lengths if 20 < l <= 100)
        long = sum(1 for l in lengths if 100 < l <= 300)
        very_long = sum(1 for l in lengths if l > 300)

        return MessageLengthStats(
            avg_length=round(sum(lengths) / n, 1),
            median_length=round(median, 1),
            min_length=min(lengths),
            max_length=max(lengths),
            short_count=short,
            medium_count=medium,
            long_count=long,
            very_long_count=very_long,
        )

    def compute_time_distributions(
        self, messages: list[Message]
    ) -> tuple[dict[int, int], dict[str, int], dict[str, int], dict[str, int]]:
        """Compute time-based message distributions.

        Args:
            messages: Messages to analyze

        Returns:
            Tuple of (hourly, daily, weekly, monthly) distributions
        """
        hourly: Counter[int] = Counter()
        daily: Counter[str] = Counter()
        weekly: Counter[str] = Counter()
        monthly: Counter[str] = Counter()

        for msg in messages:
            hourly[msg.date.hour] += 1
            daily[msg.date.strftime("%A")] += 1
            weekly[msg.date.strftime("%Y-W%W")] += 1
            monthly[msg.date.strftime("%Y-%m")] += 1

        # Ensure all hours are present
        hourly_dict = {h: hourly.get(h, 0) for h in range(24)}

        # Sort weekly and monthly
        weekly_dict = dict(sorted(weekly.items()))
        monthly_dict = dict(sorted(monthly.items()))

        return hourly_dict, dict(daily), weekly_dict, monthly_dict

    def compute_full_analytics(
        self,
        messages: list[Message],
        contact_messages: dict[str, list[Message]] | None = None,
        time_range_days: int | None = None,
        cache_key: str | None = None,
    ) -> AnalyticsResult:
        """Compute comprehensive analytics.

        Args:
            messages: All messages to analyze
            contact_messages: Optional pre-grouped messages by contact
            time_range_days: Optional time range filter
            cache_key: Optional cache key for result caching

        Returns:
            AnalyticsResult with all computed analytics
        """
        # Check cache
        if cache_key:
            found, cached = self._cache.get(cache_key)
            if found:
                return cached  # type: ignore[return-value]

        # Overview
        overview = self.compute_overview(messages, time_range_days)

        # Contact analytics
        contacts: list[ContactAnalytics] = []
        if contact_messages:
            for contact_id, msgs in contact_messages.items():
                # Get contact name from first message if available
                name = None
                for m in msgs:
                    if not m.is_from_me and m.sender_name:
                        name = m.sender_name
                        break
                contacts.append(self.compute_contact_analytics(msgs, contact_id, name))
            # Sort by total messages descending
            contacts.sort(key=lambda c: c.total_messages, reverse=True)

        # Emoji stats
        emoji_stats = self.compute_emoji_stats(messages)

        # Message length stats
        length_stats = self.compute_message_length_stats(messages)

        # Time distributions
        hourly, daily, weekly, monthly = self.compute_time_distributions(messages)

        # Sentiment trends (weekly granularity)
        trends = analyze_sentiment_trends(messages, granularity="week")
        sentiment_trends = [
            {"date": t.date, "score": t.score, "message_count": t.message_count} for t in trends
        ]

        result = AnalyticsResult(
            overview=overview,
            contacts=contacts,
            emoji_stats=emoji_stats,
            message_length_stats=length_stats,
            hourly_distribution=hourly,
            daily_distribution=daily,
            monthly_counts=monthly,
            weekly_counts=weekly,
            sentiment_trends=sentiment_trends,
        )

        # Cache result
        if cache_key:
            self._cache.set(cache_key, result)

        return result

    def _compute_avg_response_time(self, messages: list[Message]) -> float | None:
        """Compute average response time in minutes.

        Only counts responses within 24 hours.
        """
        if len(messages) < 2:
            return None

        response_times: list[float] = []
        prev_msg = None

        for msg in messages:
            if prev_msg is not None and msg.is_from_me != prev_msg.is_from_me:
                time_diff = (msg.date - prev_msg.date).total_seconds()
                if 0 < time_diff < 86400:  # Within 24 hours
                    response_times.append(time_diff / 60.0)
            prev_msg = msg

        if not response_times:
            return None

        return round(sum(response_times) / len(response_times), 1)

    def _compute_message_trend(self, messages: list[Message]) -> str:
        """Compute message trend direction."""
        if len(messages) < 10:
            return "stable"

        mid = len(messages) // 2
        first_half = messages[:mid]
        second_half = messages[mid:]

        # Get date ranges
        first_days = (first_half[-1].date - first_half[0].date).days or 1
        second_days = (second_half[-1].date - second_half[0].date).days or 1

        first_rate = len(first_half) / first_days
        second_rate = len(second_half) / second_days

        if first_rate == 0:
            return "increasing" if second_rate > 0 else "stable"

        change_pct = ((second_rate - first_rate) / first_rate) * 100

        if change_pct > 15:
            return "increasing"
        elif change_pct < -15:
            return "decreasing"
        return "stable"

    def _compute_engagement_score(
        self,
        sent_count: int,
        received_count: int,
        total_messages: int,
        days_active: int,
    ) -> float:
        """Compute engagement score (0-100).

        Based on:
        - Message balance (60% weight)
        - Activity frequency (40% weight)
        """
        if total_messages == 0:
            return 0.0

        # Balance score: closer to 50/50 is better
        balance_ratio = min(sent_count, received_count) / max(sent_count, received_count, 1)
        balance_score = balance_ratio * 100

        # Frequency score: based on messages per active day
        msgs_per_day = total_messages / max(days_active, 1)
        # Cap at 10 messages/day for max score
        freq_score = min(msgs_per_day * 10, 100)

        # Weighted average
        engagement = balance_score * 0.6 + freq_score * 0.4
        return round(engagement, 1)

    def invalidate_cache(self, cache_key: str | None = None) -> None:
        """Invalidate cached results.

        Args:
            cache_key: Specific key to invalidate, or None to clear all
        """
        if cache_key:
            self._cache.invalidate(cache_key)
        # Note: TTLCache doesn't have a clear_all method, so individual
        # keys must be invalidated


# Global engine instance
_analytics_engine: AnalyticsEngine | None = None
_analytics_engine_lock = threading.Lock()


def get_analytics_engine() -> AnalyticsEngine:
    """Get the global analytics engine instance.

    Uses double-checked locking pattern for thread safety.
    """
    global _analytics_engine
    if _analytics_engine is None:
        with _analytics_engine_lock:
            if _analytics_engine is None:
                _analytics_engine = AnalyticsEngine()
    return _analytics_engine
