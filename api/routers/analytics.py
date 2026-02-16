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
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from api.dependencies import get_imessage_reader
from api.ratelimit import RATE_LIMIT_READ, limiter
from api.schemas.stats import TimeRangeEnum
from api.services.analytics_service import (
    fetch_contact_stats,
    fetch_export_data,
    fetch_heatmap_data,
    fetch_leaderboard_data,
    fetch_overview_data,
    fetch_timeline_data,
    fetch_trends_data,
)
from integrations.imessage import ChatDBReader
from jarvis.infrastructure.cache import TTLCache
from jarvis.metrics import get_template_analytics
from models.templates import _load_templates

router = APIRouter(prefix="/analytics", tags=["analytics"])


class TemplateAnalyticsResponse(BaseModel):
    """Response model for template analytics summary."""

    total_queries: int
    template_hits: int
    model_fallbacks: int
    hit_rate_percent: float
    cache_hit_rate: float
    unique_templates_matched: int
    queries_per_second: float
    uptime_seconds: float


class TopTemplateItem(BaseModel):
    """A single template in the top templates list."""

    template_name: str
    match_count: int


class MissedQueryItem(BaseModel):
    """A single missed query entry."""

    query_hash: str
    similarity: float
    best_template: str | None
    timestamp: str


class CategoryAverageItem(BaseModel):
    """Average similarity for a template category."""

    category: str
    average_similarity: float


class TemplateInfo(BaseModel):
    """Information about a single template."""

    name: str
    pattern_count: int
    sample_patterns: list[str]


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

    time_range_start = _get_time_range_start(time_range)

    result = await run_in_threadpool(fetch_overview_data, reader, time_range, time_range_start)

    cache.set(cache_key, result)
    return result


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
        fetch_timeline_data, reader, granularity, metric, time_range, time_range_start
    )

    cache.set(cache_key, result)
    return result


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

    result = await run_in_threadpool(fetch_heatmap_data, reader, time_range, time_range_start)

    cache.set(cache_key, result)
    return result


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
        fetch_contact_stats, reader, chat_id, time_range, time_range_start
    )

    if result is None:
        from jarvis.core.exceptions import GraphContactNotFoundError

        raise GraphContactNotFoundError(
            f"No messages found for contact {chat_id}", contact_id=chat_id
        )

    cache.set(cache_key, result)
    return result


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
        fetch_leaderboard_data, reader, time_range, time_range_start, limit, sort_by
    )

    cache.set(cache_key, result)
    return result


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

    result = await run_in_threadpool(fetch_trends_data, reader, time_range, time_range_start)

    cache.set(cache_key, result)
    return result


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
        fetch_export_data, reader, format, time_range, time_range_start
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
    with _analytics_cache_lock:
        _analytics_cache = None
    return {"status": "ok", "message": "Analytics cache cleared"}


# --- Template Analytics Endpoints ---


@router.get("/templates", response_model=TemplateAnalyticsResponse)
def get_template_analytics_summary() -> TemplateAnalyticsResponse:
    """Get template analytics summary."""
    analytics = get_template_analytics()
    stats = analytics.get_stats()
    return TemplateAnalyticsResponse(**stats)


@router.get("/templates/top")
def get_top_templates(limit: int = 20) -> list[TopTemplateItem]:
    """Get most frequently matched templates."""
    analytics = get_template_analytics()
    top = analytics.get_top_templates(limit=limit)
    return [TopTemplateItem(**item) for item in top]


@router.get("/templates/missed")
def get_missed_queries(limit: int = 50) -> list[MissedQueryItem]:
    """Get queries that fell through to model generation."""
    analytics = get_template_analytics()
    missed = analytics.get_missed_queries(limit=limit)
    return [MissedQueryItem(**item) for item in missed]


@router.get("/templates/categories")
def get_category_averages() -> list[CategoryAverageItem]:
    """Get average similarity scores per template category."""
    analytics = get_template_analytics()
    averages = analytics.get_category_averages()
    return [
        CategoryAverageItem(category=cat, average_similarity=round(avg, 4))
        for cat, avg in sorted(averages.items(), key=lambda x: x[1], reverse=True)
    ]


@router.get("/templates/list")
def list_available_templates() -> list[TemplateInfo]:
    """List all available templates."""
    templates = _load_templates()
    return [
        TemplateInfo(
            name=t.name,
            pattern_count=len(t.patterns),
            sample_patterns=t.patterns[:3],
        )
        for t in templates
    ]


@router.get("/templates/coverage")
def get_template_coverage() -> dict[str, Any]:
    """Get overall template coverage statistics."""
    analytics = get_template_analytics()
    stats = analytics.get_stats()
    templates = _load_templates()
    total_patterns = sum(len(t.patterns) for t in templates)

    return {
        "total_templates": len(templates),
        "total_patterns": total_patterns,
        "responses_from_templates": stats["template_hits"],
        "responses_from_model": stats["model_fallbacks"],
        "coverage_percent": stats["hit_rate_percent"],
        "template_efficiency": (
            round(stats["template_hits"] / len(templates), 2)
            if len(templates) > 0 and stats["template_hits"] > 0
            else 0.0
        ),
    }


@router.get("/templates/export")
def export_raw_template_analytics() -> JSONResponse:
    """Export raw template analytics data as JSON."""
    analytics = get_template_analytics()
    raw_data = analytics.export_raw()
    return JSONResponse(
        content=raw_data,
        headers={"Content-Disposition": "attachment; filename=template_analytics.json"},
    )


@router.post("/templates/reset")
def reset_template_analytics() -> dict[str, str]:
    """Reset all template analytics data."""
    analytics = get_template_analytics()
    analytics.reset()
    return {"status": "ok", "message": "Template analytics reset successfully"}


@router.get("/templates/dashboard")
def get_template_dashboard_data() -> dict[str, Any]:
    """Get all data needed for the template analytics dashboard."""
    analytics = get_template_analytics()
    stats = analytics.get_stats()
    templates = _load_templates()

    return {
        "summary": stats,
        "top_templates": analytics.get_top_templates(limit=20),
        "missed_queries": analytics.get_missed_queries(limit=20),
        "category_averages": [
            {"category": cat, "average_similarity": round(avg, 4)}
            for cat, avg in analytics.get_category_averages().items()
        ],
        "coverage": {
            "total_templates": len(templates),
            "total_patterns": sum(len(t.patterns) for t in templates),
            "responses_from_templates": stats["template_hits"],
            "responses_from_model": stats["model_fallbacks"],
            "coverage_percent": stats["hit_rate_percent"],
        },
        "pie_chart_data": {
            "template_responses": stats["template_hits"],
            "model_responses": stats["model_fallbacks"],
        },
    }
