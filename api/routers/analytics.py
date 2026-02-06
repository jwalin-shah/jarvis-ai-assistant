"""Comprehensive analytics API endpoints.

Provides endpoints for:
- Dashboard overview metrics
- Time-series data for charts
- Per-contact statistics
- Trending patterns detection
- Data export (CSV/JSON)
"""

from __future__ import annotations

import threading
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from api.dependencies import get_imessage_reader
from api.ratelimit import RATE_LIMIT_READ, limiter
from api.schemas.stats import TimeRangeEnum
from integrations.imessage import ChatDBReader
from jarvis.analytics import (
    get_analytics_engine,
)
from jarvis.analytics.reports import get_report_generator
from jarvis.analytics.trends import (
    PeakPeriod,
    detect_anomalies,
    detect_seasonality,
    detect_trend,
    get_trend_analyzer,
)
from jarvis.metrics import TTLCache

router = APIRouter(prefix="/analytics", tags=["analytics"])

# Cache for analytics data with 5-minute TTL
_analytics_cache: TTLCache | None = None
_analytics_cache_lock = threading.Lock()


def get_analytics_cache() -> TTLCache:
    """Get analytics cache with 5-minute TTL.

    Uses double-checked locking pattern for thread safety.
    """
    global _analytics_cache
    if _analytics_cache is None:
        with _analytics_cache_lock:
            if _analytics_cache is None:
                _analytics_cache = TTLCache(ttl_seconds=300.0, maxsize=100)
    return _analytics_cache


def _get_time_range_start(time_range: TimeRangeEnum) -> datetime | None:
    """Get start datetime for time range."""
    now = datetime.now(UTC)
    if time_range == TimeRangeEnum.WEEK:
        return now - timedelta(days=7)
    elif time_range == TimeRangeEnum.MONTH:
        return now - timedelta(days=30)
    elif time_range == TimeRangeEnum.THREE_MONTHS:
        return now - timedelta(days=90)
    return None  # all_time


def _iter_filtered_messages(
    reader: ChatDBReader,
    time_range_start: datetime | None,
    per_conversation_limit: int,
) -> Iterator[tuple[Any, list[Any]]]:
    conversations = reader.get_conversations(limit=200)
    for conv in conversations:
        messages = reader.get_messages(conv.chat_id, limit=per_conversation_limit)
        if time_range_start:
            messages = [m for m in messages if m.date >= time_range_start]
        if messages:
            yield conv, messages


def _get_activity_level(count: int) -> int:
    if count == 0:
        return 0
    if count <= 5:
        return 1
    if count <= 15:
        return 2
    if count <= 30:
        return 3
    return 4


def _build_timeline_from_counts(
    granularity: str,
    daily_counts: dict[str, dict[str, int]],
    hourly_counts: dict[int, dict[str, int]],
) -> list[dict[str, object]]:
    if granularity == "hour":
        return [
            {
                "hour": hour,
                "total": counts["total"],
                "sent": counts["sent"],
                "received": counts["received"],
            }
            for hour, counts in sorted(hourly_counts.items())
        ]

    if granularity == "day":
        return [
            {
                "date": date_key,
                "total": counts["total"],
                "sent": counts["sent"],
                "received": counts["received"],
            }
            for date_key, counts in sorted(daily_counts.items())
        ]

    if granularity == "week":
        weekly: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "sent": 0, "received": 0}
        )
        for date_key, counts in daily_counts.items():
            dt = datetime.strptime(date_key, "%Y-%m-%d")
            week_key = dt.strftime("%Y-W%W")
            weekly[week_key]["total"] += counts["total"]
            weekly[week_key]["sent"] += counts["sent"]
            weekly[week_key]["received"] += counts["received"]
        return [{"date": week_key, **counts} for week_key, counts in sorted(weekly.items())]

    if granularity == "month":
        monthly: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "sent": 0, "received": 0}
        )
        for date_key, counts in daily_counts.items():
            month_key = date_key[:7]
            monthly[month_key]["total"] += counts["total"]
            monthly[month_key]["sent"] += counts["sent"]
            monthly[month_key]["received"] += counts["received"]
        return [{"date": month_key, **counts} for month_key, counts in sorted(monthly.items())]

    return []


def _fetch_overview_data(
    reader: ChatDBReader,
    time_range: TimeRangeEnum,
    time_range_start: datetime | None,
) -> dict[str, Any]:
    """Fetch and compute overview analytics data (blocking I/O).

    This is run in a threadpool to avoid blocking the event loop.
    """
    # Fetch all conversations and messages in batches to limit memory
    conversations = reader.get_conversations(limit=200)
    all_messages = []

    for conv in conversations:
        messages = reader.get_messages(conv.chat_id, limit=500)
        if time_range_start:
            messages = [m for m in messages if m.date >= time_range_start]
        all_messages.extend(messages)

    # Compute analytics
    engine = get_analytics_engine()
    overview = engine.compute_overview(all_messages)

    # Trend comparison
    trend_analyzer = get_trend_analyzer()
    days = 7 if time_range == TimeRangeEnum.WEEK else 30
    comparison = trend_analyzer.compare_periods(all_messages, days, days)

    return {
        "total_messages": overview.total_messages,
        "sent_messages": overview.total_sent,
        "received_messages": overview.total_received,
        "active_conversations": overview.active_conversations,
        "avg_messages_per_day": overview.avg_messages_per_day,
        "avg_response_time_minutes": overview.avg_response_time_minutes,
        "sentiment": {
            "score": overview.sentiment_score,
            "label": overview.sentiment_label,
        },
        "peak_hour": overview.peak_hour,
        "peak_day": overview.peak_day,
        "date_range": {
            "start": overview.date_range_start.isoformat() if overview.date_range_start else None,
            "end": overview.date_range_end.isoformat() if overview.date_range_end else None,
        },
        "period_comparison": {
            "total_change_percent": comparison.get("total_messages_change", 0),
            "sent_change_percent": comparison.get("sent_messages_change", 0),
            "contacts_change_percent": comparison.get("active_contacts_change", 0),
        },
        "time_range": time_range.value,
    }


@router.get(
    "/overview",
    summary="Get analytics dashboard overview",
    response_description="Dashboard overview metrics",
)
@limiter.limit(RATE_LIMIT_READ)
async def get_analytics_overview(
    request: Request,
    time_range: TimeRangeEnum = Query(
        default=TimeRangeEnum.MONTH,
        description="Time range for analytics",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> dict[str, Any]:
    """Get comprehensive overview metrics for the analytics dashboard.

    Returns aggregated statistics including:
    - Total messages (sent/received)
    - Active conversations count
    - Average messages per day
    - Average response time
    - Overall sentiment
    - Peak activity times
    - Period comparison (vs previous period)

    **Example Response:**
    ```json
    {
        "total_messages": 5000,
        "sent_messages": 2500,
        "received_messages": 2500,
        "active_conversations": 45,
        "avg_messages_per_day": 50.5,
        "avg_response_time_minutes": 12.5,
        "sentiment": {"score": 0.35, "label": "positive"},
        "peak_hour": 14,
        "peak_day": "Wednesday",
        "period_comparison": {
            "total_change_percent": 15.5,
            "sent_change_percent": 10.2,
            "contacts_change_percent": 5.0
        }
    }
    ```
    """
    cache_key = f"overview:{time_range.value}"
    cache = get_analytics_cache()

    found, cached = cache.get(cache_key)
    if found:
        return cached  # type: ignore[no-any-return]

    # Get time range filter
    time_range_start = _get_time_range_start(time_range)

    # Run blocking I/O in threadpool
    result = await run_in_threadpool(_fetch_overview_data, reader, time_range, time_range_start)

    cache.set(cache_key, result)
    return result


def _fetch_timeline_data(
    reader: ChatDBReader,
    granularity: str,
    metric: str,
    time_range: TimeRangeEnum,
    time_range_start: datetime | None,
) -> dict[str, Any]:
    """Fetch and compute timeline data (blocking I/O)."""
    daily_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "sent": 0, "received": 0}
    )
    hourly_counts: dict[int, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "sent": 0, "received": 0}
    )

    for _, messages in _iter_filtered_messages(
        reader, time_range_start, per_conversation_limit=500
    ):
        for msg in messages:
            date_key = msg.date.strftime("%Y-%m-%d")
            daily_counts[date_key]["total"] += 1
            if msg.is_from_me:
                daily_counts[date_key]["sent"] += 1
            else:
                daily_counts[date_key]["received"] += 1

            hour = msg.date.hour
            hourly_counts[hour]["total"] += 1
            if msg.is_from_me:
                hourly_counts[hour]["sent"] += 1
            else:
                hourly_counts[hour]["received"] += 1

    timeline_data = _build_timeline_from_counts(
        granularity=granularity,
        daily_counts=daily_counts,
        hourly_counts=hourly_counts,
    )

    return {
        "granularity": granularity,
        "metric": metric,
        "time_range": time_range.value,
        "data": timeline_data,
        "total_points": len(timeline_data),
    }


@router.get(
    "/timeline",
    summary="Get time-series data for charts",
    response_description="Time-series data points",
)
@limiter.limit(RATE_LIMIT_READ)
async def get_analytics_timeline(
    request: Request,
    granularity: str = Query(
        default="day",
        pattern="^(hour|day|week|month)$",
        description="Time granularity (hour, day, week, month)",
    ),
    time_range: TimeRangeEnum = Query(
        default=TimeRangeEnum.MONTH,
        description="Time range for data",
    ),
    metric: str = Query(
        default="messages",
        pattern="^(messages|sentiment|response_time)$",
        description="Metric to track",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> dict[str, Any]:
    """Get time-series data for chart visualization.

    Supports multiple granularities and metrics for flexible charting.

    **Granularities:**
    - `hour`: Hourly breakdown (24 data points)
    - `day`: Daily counts
    - `week`: Weekly aggregates
    - `month`: Monthly aggregates

    **Metrics:**
    - `messages`: Message counts (total, sent, received)
    - `sentiment`: Average sentiment scores
    - `response_time`: Average response times

    **Example Response:**
    ```json
    {
        "granularity": "day",
        "metric": "messages",
        "data": [
            {"date": "2024-01-15", "total": 45, "sent": 22, "received": 23},
            {"date": "2024-01-16", "total": 52, "sent": 25, "received": 27}
        ]
    }
    ```
    """
    cache_key = f"timeline:{granularity}:{time_range.value}:{metric}"
    cache = get_analytics_cache()

    found, cached = cache.get(cache_key)
    if found:
        return cached  # type: ignore[no-any-return]

    time_range_start = _get_time_range_start(time_range)

    result = await run_in_threadpool(
        _fetch_timeline_data, reader, granularity, metric, time_range, time_range_start
    )

    cache.set(cache_key, result)
    return result


def _fetch_heatmap_data(
    reader: ChatDBReader,
    time_range: TimeRangeEnum,
    time_range_start: datetime | None,
) -> dict[str, Any]:
    """Fetch and compute heatmap data (blocking I/O)."""
    daily_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "sent": 0, "received": 0}
    )

    for _, messages in _iter_filtered_messages(
        reader, time_range_start, per_conversation_limit=1000
    ):
        for msg in messages:
            date_key = msg.date.strftime("%Y-%m-%d")
            daily_counts[date_key]["total"] += 1
            if msg.is_from_me:
                daily_counts[date_key]["sent"] += 1
            else:
                daily_counts[date_key]["received"] += 1

    heatmap_data = [
        {
            "date": date_key,
            "count": counts["total"],
            "level": _get_activity_level(counts["total"]),
        }
        for date_key, counts in sorted(daily_counts.items())
    ]

    # Calculate stats
    counts: list[int] = [int(d["count"]) for d in heatmap_data]  # type: ignore[call-overload]
    active_days = sum(1 for c in counts if c > 0)

    return {
        "data": heatmap_data,
        "stats": {
            "total_days": len(heatmap_data),
            "active_days": active_days,
            "max_count": max(counts) if counts else 0,
            "avg_count": round(sum(counts) / len(counts), 1) if counts else 0,
        },
        "time_range": time_range.value,
    }


@router.get(
    "/heatmap",
    summary="Get activity heatmap data",
    response_description="Heatmap data for activity calendar",
)
@limiter.limit(RATE_LIMIT_READ)
async def get_activity_heatmap(
    request: Request,
    time_range: TimeRangeEnum = Query(
        default=TimeRangeEnum.THREE_MONTHS,
        description="Time range for heatmap",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> dict[str, Any]:
    """Get GitHub-style activity heatmap data.

    Returns daily activity levels for calendar visualization.

    **Activity Levels:**
    - 0: No activity
    - 1: Low (1-5 messages)
    - 2: Medium (6-15 messages)
    - 3: High (16-30 messages)
    - 4: Very high (31+ messages)

    **Example Response:**
    ```json
    {
        "data": [
            {"date": "2024-01-15", "count": 45, "level": 4},
            {"date": "2024-01-16", "count": 12, "level": 2}
        ],
        "stats": {
            "total_days": 90,
            "active_days": 75,
            "max_count": 85,
            "avg_count": 25.5
        }
    }
    ```
    """
    cache_key = f"heatmap:{time_range.value}"
    cache = get_analytics_cache()

    found, cached = cache.get(cache_key)
    if found:
        return cached  # type: ignore[no-any-return]

    time_range_start = _get_time_range_start(time_range)

    result = await run_in_threadpool(_fetch_heatmap_data, reader, time_range, time_range_start)

    cache.set(cache_key, result)
    return result


def _fetch_contact_stats(
    reader: ChatDBReader,
    chat_id: str,
    time_range: TimeRangeEnum,
    time_range_start: datetime | None,
) -> dict[str, Any] | None:
    """Fetch and compute contact statistics (blocking I/O).

    Returns None if no messages found.
    """
    # Fetch messages for this contact
    messages = reader.get_messages(chat_id, limit=1000)
    if time_range_start:
        messages = [m for m in messages if m.date >= time_range_start]

    if not messages:
        return None

    # Get contact name
    contact_name = None
    for m in messages:
        if not m.is_from_me and m.sender_name:
            contact_name = m.sender_name
            break

    # Compute analytics
    engine = get_analytics_engine()
    contact_analytics = engine.compute_contact_analytics(messages, chat_id, contact_name)
    emoji_stats = engine.compute_emoji_stats(messages)
    hourly, daily, weekly, monthly = engine.compute_time_distributions(messages)

    return {
        "contact_id": chat_id,
        "contact_name": contact_name,
        "total_messages": contact_analytics.total_messages,
        "sent_count": contact_analytics.sent_count,
        "received_count": contact_analytics.received_count,
        "avg_response_time_minutes": contact_analytics.avg_response_time_minutes,
        "sentiment_score": contact_analytics.sentiment_score,
        "engagement_score": contact_analytics.engagement_score,
        "message_trend": contact_analytics.message_trend,
        "last_message_date": contact_analytics.last_message_date.isoformat()
        if contact_analytics.last_message_date
        else None,
        "emoji_usage": {
            "total": emoji_stats.total_count,
            "per_message": emoji_stats.emojis_per_message,
            "top_emojis": emoji_stats.top_emojis,
        },
        "hourly_distribution": hourly,
        "daily_distribution": daily,
        "weekly_counts": weekly,
        "time_range": time_range.value,
    }


@router.get(
    "/contacts/{chat_id}/stats",
    summary="Get per-contact statistics",
    response_description="Contact-specific analytics",
)
@limiter.limit(RATE_LIMIT_READ)
async def get_contact_stats(
    request: Request,
    chat_id: str,
    time_range: TimeRangeEnum = Query(
        default=TimeRangeEnum.MONTH,
        description="Time range for statistics",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> dict[str, Any]:
    """Get detailed analytics for a specific contact.

    Returns comprehensive statistics including:
    - Message counts and balance
    - Response time metrics
    - Sentiment analysis
    - Activity patterns
    - Engagement score

    **Example Response:**
    ```json
    {
        "contact_id": "chat123456",
        "contact_name": "John Doe",
        "total_messages": 500,
        "sent_count": 245,
        "received_count": 255,
        "avg_response_time_minutes": 8.5,
        "sentiment_score": 0.42,
        "engagement_score": 78.5,
        "message_trend": "increasing",
        "hourly_distribution": {...},
        "daily_distribution": {...}
    }
    ```
    """
    cache_key = f"contact:{chat_id}:{time_range.value}"
    cache = get_analytics_cache()

    found, cached = cache.get(cache_key)
    if found:
        return cached  # type: ignore[no-any-return]

    time_range_start = _get_time_range_start(time_range)

    result = await run_in_threadpool(
        _fetch_contact_stats, reader, chat_id, time_range, time_range_start
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No messages found for contact {chat_id}",
        )

    cache.set(cache_key, result)
    return result


def _fetch_leaderboard_data(
    reader: ChatDBReader,
    time_range: TimeRangeEnum,
    time_range_start: datetime | None,
    result_limit: int,
    sort_by: str,
) -> dict[str, Any]:
    """Fetch and compute leaderboard data (blocking I/O)."""
    # Fetch all conversations
    conversations = reader.get_conversations(limit=200)
    contact_data = []

    engine = get_analytics_engine()

    for conv in conversations:
        messages = reader.get_messages(conv.chat_id, limit=500)
        if time_range_start:
            messages = [m for m in messages if m.date >= time_range_start]
        if not messages:
            continue

        analytics = engine.compute_contact_analytics(messages, conv.chat_id, conv.display_name)
        contact_data.append(analytics)

    # Sort by criteria
    if sort_by == "messages":
        contact_data.sort(key=lambda c: c.total_messages, reverse=True)
    elif sort_by == "engagement":
        contact_data.sort(key=lambda c: c.engagement_score, reverse=True)
    elif sort_by == "response_time":
        # Filter out None values and sort by response time (fastest first)
        contact_data = [c for c in contact_data if c.avg_response_time_minutes is not None]
        contact_data.sort(key=lambda c: c.avg_response_time_minutes or float("inf"))

    # Build response
    contacts = [
        {
            "rank": i + 1,
            "contact_id": c.contact_id,
            "contact_name": c.contact_name,
            "total_messages": c.total_messages,
            "sent_count": c.sent_count,
            "received_count": c.received_count,
            "engagement_score": c.engagement_score,
            "avg_response_time_minutes": c.avg_response_time_minutes,
            "sentiment_score": c.sentiment_score,
            "trend": c.message_trend,
        }
        for i, c in enumerate(contact_data[:result_limit])
    ]

    return {
        "contacts": contacts,
        "total_contacts": len(contact_data),
        "sort_by": sort_by,
        "time_range": time_range.value,
    }


@router.get(
    "/contacts/leaderboard",
    summary="Get top contacts ranking",
    response_description="Ranked list of most active contacts",
)
@limiter.limit(RATE_LIMIT_READ)
async def get_contacts_leaderboard(
    request: Request,
    time_range: TimeRangeEnum = Query(
        default=TimeRangeEnum.MONTH,
        description="Time range for ranking",
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
        description="Number of contacts to return",
    ),
    sort_by: str = Query(
        default="messages",
        pattern="^(messages|engagement|response_time)$",
        description="Sort criteria",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> dict[str, Any]:
    """Get ranked list of top contacts by various criteria.

    **Sort Options:**
    - `messages`: By total message count
    - `engagement`: By engagement score
    - `response_time`: By average response time (fastest first)

    **Example Response:**
    ```json
    {
        "contacts": [
            {
                "rank": 1,
                "contact_id": "chat123",
                "contact_name": "John Doe",
                "total_messages": 500,
                "engagement_score": 85.5,
                "trend": "increasing"
            }
        ],
        "total_contacts": 45
    }
    ```
    """
    cache_key = f"leaderboard:{time_range.value}:{limit}:{sort_by}"
    cache = get_analytics_cache()

    found, cached = cache.get(cache_key)
    if found:
        return cached  # type: ignore[no-any-return]

    time_range_start = _get_time_range_start(time_range)

    result = await run_in_threadpool(
        _fetch_leaderboard_data, reader, time_range, time_range_start, limit, sort_by
    )

    cache.set(cache_key, result)
    return result


def _fetch_trends_data(
    reader: ChatDBReader,
    time_range: TimeRangeEnum,
    time_range_start: datetime | None,
) -> dict[str, Any]:
    """Fetch and compute trends data (blocking I/O)."""
    trend_analyzer = get_trend_analyzer()
    daily_counts: Counter[str] = Counter()
    weekly_counts: Counter[str] = Counter()
    hour_counts: Counter[int] = Counter()
    day_counts: Counter[str] = Counter()
    contact_daily_counts: dict[str, Counter[str]] = defaultdict(Counter)
    contact_names: dict[str, str | None] = {}

    total_messages = 0

    for _, messages in _iter_filtered_messages(
        reader, time_range_start, per_conversation_limit=500
    ):
        for msg in messages:
            total_messages += 1
            date_key = msg.date.strftime("%Y-%m-%d")
            daily_counts[date_key] += 1
            week_key = msg.date.strftime("%Y-W%W")
            weekly_counts[week_key] += 1
            hour_counts[msg.date.hour] += 1
            day_counts[msg.date.strftime("%A")] += 1

            chat_id = msg.chat_id
            contact_daily_counts[chat_id][date_key] += 1
            if chat_id not in contact_names and not msg.is_from_me:
                contact_names[chat_id] = msg.sender_name

    daily_data = sorted(daily_counts.items())
    daily_data_float = [(k, float(v)) for k, v in daily_data]
    daily_values = [v for _, v in daily_data_float]
    overall_trend = detect_trend(daily_values, trend_analyzer.trend_window_size)

    weekly_values = [float(v) for _, v in sorted(weekly_counts.items())]
    weekly_trend = (
        detect_trend(weekly_values, trend_analyzer.trend_window_size)
        if len(weekly_values) >= 4
        else None
    )

    anomalies = detect_anomalies(daily_data_float, trend_analyzer.anomaly_threshold)

    peak_hours = [
        PeakPeriod(
            period_type="hour",
            period_value=hour,
            count=count,
            percentage_of_total=(
                round((count / total_messages) * 100, 1) if total_messages else 0.0
            ),
        )
        for hour, count in hour_counts.most_common(3)
    ]

    peak_days = [
        PeakPeriod(
            period_type="day",
            period_value=day,
            count=count,
            percentage_of_total=(
                round((count / total_messages) * 100, 1) if total_messages else 0.0
            ),
        )
        for day, count in day_counts.most_common(3)
    ]

    seasonality_detected, seasonality_period = detect_seasonality(dict(daily_counts))

    trending_contacts: list[tuple[str, str | None, Any]] = []
    for contact_id, counts in contact_daily_counts.items():
        contact_values = [float(v) for _, v in sorted(counts.items())]
        if len(contact_values) < 4:
            continue
        trend = detect_trend(contact_values, trend_analyzer.trend_window_size)
        if abs(trend.percentage_change) > 15 and trend.confidence > 0.5:
            trending_contacts.append((contact_id, contact_names.get(contact_id), trend))
    trending_contacts.sort(key=lambda x: abs(x[2].percentage_change), reverse=True)
    trending_contacts = trending_contacts[:5]

    return {
        "overall_trend": {
            "direction": overall_trend.direction,
            "percentage_change": overall_trend.percentage_change,
            "confidence": overall_trend.confidence,
        },
        "weekly_trend": {
            "direction": weekly_trend.direction,
            "percentage_change": weekly_trend.percentage_change,
            "confidence": weekly_trend.confidence,
        }
        if weekly_trend
        else None,
        "trending_contacts": [
            {
                "contact_id": cid,
                "contact_name": name,
                "trend": trend.direction,
                "change_percent": trend.percentage_change,
                "confidence": trend.confidence,
            }
            for cid, name, trend in trending_contacts
        ],
        "anomalies": [
            {
                "date": a.date,
                "type": a.anomaly_type,
                "value": a.value,
                "expected": a.expected_value,
                "deviation": a.deviation,
            }
            for a in anomalies
        ],
        "peak_hours": [
            {
                "hour": p.period_value,
                "count": p.count,
                "percentage": p.percentage_of_total,
            }
            for p in peak_hours
        ],
        "peak_days": [
            {
                "day": p.period_value,
                "count": p.count,
                "percentage": p.percentage_of_total,
            }
            for p in peak_days
        ],
        "seasonality": {
            "detected": seasonality_detected,
            "pattern": seasonality_period,
        },
        "time_range": time_range.value,
    }


@router.get(
    "/trends",
    summary="Get trending patterns",
    response_description="Detected trends and anomalies",
)
@limiter.limit(RATE_LIMIT_READ)
async def get_trending_patterns(
    request: Request,
    time_range: TimeRangeEnum = Query(
        default=TimeRangeEnum.MONTH,
        description="Time range for trend analysis",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> dict[str, Any]:
    """Get detected trends, anomalies, and patterns.

    Returns:
    - Overall activity trend
    - Trending contacts (significant changes)
    - Detected anomalies (unusual activity)
    - Seasonality patterns

    **Example Response:**
    ```json
    {
        "overall_trend": {
            "direction": "increasing",
            "percentage_change": 25.5,
            "confidence": 0.85
        },
        "trending_contacts": [
            {
                "contact_id": "chat123",
                "contact_name": "John",
                "trend": "increasing",
                "change_percent": 45.5
            }
        ],
        "anomalies": [
            {
                "date": "2024-01-20",
                "type": "spike",
                "value": 150,
                "expected": 45
            }
        ]
    }
    ```
    """
    cache_key = f"trends:{time_range.value}"
    cache = get_analytics_cache()

    found, cached = cache.get(cache_key)
    if found:
        return cached  # type: ignore[no-any-return]

    time_range_start = _get_time_range_start(time_range)

    result = await run_in_threadpool(_fetch_trends_data, reader, time_range, time_range_start)

    cache.set(cache_key, result)
    return result


def _fetch_export_data(
    reader: ChatDBReader,
    export_format: str,
    time_range: TimeRangeEnum,
    time_range_start: datetime | None,
) -> tuple[str, str, str]:
    """Fetch and generate export data (blocking I/O).

    Returns tuple of (content, media_type, filename).
    """
    # Fetch messages
    conversations = reader.get_conversations(limit=200)
    all_messages = []
    contact_messages: dict[str, list[Any]] = defaultdict(list)

    for conv in conversations:
        messages = reader.get_messages(conv.chat_id, limit=500)
        if time_range_start:
            messages = [m for m in messages if m.date >= time_range_start]
        all_messages.extend(messages)
        contact_messages[conv.chat_id].extend(messages)

    report_gen = get_report_generator()

    if export_format == "json":
        content = report_gen.export_to_json(all_messages, contact_messages)
        return content, "application/json", f"analytics_{time_range.value}.json"
    else:  # csv
        csv_exports = report_gen.export_to_csv(all_messages)
        daily_csv = csv_exports.get("daily_analytics.csv", "")
        return daily_csv, "text/csv", f"daily_analytics_{time_range.value}.csv"


@router.get(
    "/export",
    summary="Export analytics data",
    response_description="Exported analytics in requested format",
)
@limiter.limit(RATE_LIMIT_READ)
async def export_analytics(
    request: Request,
    format: str = Query(
        default="json",
        pattern="^(json|csv)$",
        description="Export format",
    ),
    time_range: TimeRangeEnum = Query(
        default=TimeRangeEnum.MONTH,
        description="Time range for export",
    ),
    reader: ChatDBReader = Depends(get_imessage_reader),
) -> Response:
    """Export analytics data in CSV or JSON format.

    **Formats:**
    - `json`: Complete analytics as JSON
    - `csv`: Daily/weekly/monthly aggregates as CSV files (zip archive)

    Returns the file content with appropriate headers for download.
    """
    time_range_start = _get_time_range_start(time_range)

    content, media_type, filename = await run_in_threadpool(
        _fetch_export_data, reader, format, time_range, time_range_start
    )

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.delete(
    "/cache",
    summary="Clear analytics cache",
    response_description="Cache invalidation confirmation",
)
@limiter.limit(RATE_LIMIT_READ)
async def clear_analytics_cache(request: Request) -> dict[str, str]:
    """Clear the analytics cache to force fresh computation.

    Use this after significant data changes or when troubleshooting
    stale data issues.
    """
    global _analytics_cache
    _analytics_cache = None
    return {"status": "ok", "message": "Analytics cache cleared"}
