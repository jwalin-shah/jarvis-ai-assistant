"""Tests for analytics API router.

Tests comprehensive coverage of all analytics endpoints including:
- Overview dashboard metrics
- Timeline data with multiple granularities
- Activity heatmaps
- Contact-specific statistics
- Leaderboard rankings
- Trend detection
- Data export (JSON/CSV)
- Cache management
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from api.schemas.stats import TimeRangeEnum
from contracts.imessage import Message
from jarvis.analytics import (
    ContactAnalytics,
    EmojiStats,
    OverviewMetrics,
)
from jarvis.analytics.trends import TrendResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def app():
    """Get the FastAPI application instance."""
    from api.main import app as main_app

    return main_app


@pytest.fixture
def analytics_router():
    """Get analytics router functions."""
    from api.routers.analytics import (
        _get_time_range_start,
        get_analytics_cache,
    )
    from api.services.analytics_service import (
        build_timeline_from_counts,
        get_activity_level,
    )

    return {
        "build_timeline": build_timeline_from_counts,
        "get_activity_level": get_activity_level,
        "get_time_range_start": _get_time_range_start,
        "get_cache": get_analytics_cache,
    }


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_reader():
    """Create mock ChatDBReader."""
    reader = MagicMock()
    reader.check_access = MagicMock(return_value=True)
    reader.close = MagicMock()
    return reader


@pytest.fixture
def mock_request():
    """Create mock FastAPI request."""
    request = MagicMock(spec=Request)
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    return request


@pytest.fixture
def sample_message():
    """Create a sample message."""

    def _create(
        text: str = "Test message",
        is_from_me: bool = False,
        date: datetime | None = None,
        msg_id: int = 1,
        chat_id: str = "chat123",
    ) -> Message:
        if date is None:
            date = datetime.now(UTC)
        return Message(
            id=msg_id,
            chat_id=chat_id,
            sender="+15551234567" if not is_from_me else "me",
            sender_name="Test User" if not is_from_me else None,
            text=text,
            date=date,
            is_from_me=is_from_me,
            attachments=[],
            reply_to_id=None,
            reactions=[],
        )

    return _create


@pytest.fixture
def sample_conversation():
    """Create sample conversation."""
    conv = MagicMock()
    conv.chat_id = "chat123"
    conv.display_name = "Test User"
    return conv


@pytest.fixture(autouse=True)
def clear_cache(analytics_router):
    """Clear analytics cache before each test."""
    cache = analytics_router["get_cache"]()
    cache.invalidate()
    yield
    cache.invalidate()


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_time_range_start_week(self, analytics_router):
        """WEEK returns 7 days ago."""
        get_time_range_start = analytics_router["get_time_range_start"]
        result = get_time_range_start(TimeRangeEnum.WEEK)
        assert result is not None
        expected = datetime.now(UTC) - timedelta(days=7)
        assert abs((result - expected).total_seconds()) < 5

    def test_get_time_range_start_month(self, analytics_router):
        """MONTH returns 30 days ago."""
        get_time_range_start = analytics_router["get_time_range_start"]
        result = get_time_range_start(TimeRangeEnum.MONTH)
        assert result is not None
        expected = datetime.now(UTC) - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 5

    def test_get_time_range_start_three_months(self, analytics_router):
        """THREE_MONTHS returns 90 days ago."""
        get_time_range_start = analytics_router["get_time_range_start"]
        result = get_time_range_start(TimeRangeEnum.THREE_MONTHS)
        assert result is not None
        expected = datetime.now(UTC) - timedelta(days=90)
        assert abs((result - expected).total_seconds()) < 5

    def test_get_time_range_start_all_time(self, analytics_router):
        """ALL_TIME returns None."""
        get_time_range_start = analytics_router["get_time_range_start"]
        result = get_time_range_start(TimeRangeEnum.ALL_TIME)
        assert result is None

    def test_get_activity_level_zero(self, analytics_router):
        """Zero messages returns level 0."""
        get_activity_level = analytics_router["get_activity_level"]
        assert get_activity_level(0) == 0

    def test_get_activity_level_low(self, analytics_router):
        """1-5 messages returns level 1."""
        get_activity_level = analytics_router["get_activity_level"]
        assert get_activity_level(1) == 1
        assert get_activity_level(5) == 1

    def test_get_activity_level_medium(self, analytics_router):
        """6-15 messages returns level 2."""
        get_activity_level = analytics_router["get_activity_level"]
        assert get_activity_level(6) == 2
        assert get_activity_level(15) == 2

    def test_get_activity_level_high(self, analytics_router):
        """16-30 messages returns level 3."""
        get_activity_level = analytics_router["get_activity_level"]
        assert get_activity_level(16) == 3
        assert get_activity_level(30) == 3

    def test_get_activity_level_very_high(self, analytics_router):
        """31+ messages returns level 4."""
        get_activity_level = analytics_router["get_activity_level"]
        assert get_activity_level(31) == 4
        assert get_activity_level(100) == 4

    def test_build_timeline_hour_granularity(self, analytics_router):
        """Hour granularity builds correct timeline."""
        build_timeline = analytics_router["build_timeline"]
        hourly = {
            9: {"total": 10, "sent": 5, "received": 5},
            14: {"total": 20, "sent": 12, "received": 8},
        }
        result = build_timeline("hour", {}, hourly)
        assert len(result) == 2
        assert result[0]["hour"] == 9
        assert result[0]["total"] == 10
        assert result[1]["hour"] == 14

    def test_build_timeline_day_granularity(self, analytics_router):
        """Day granularity builds correct timeline."""
        build_timeline = analytics_router["build_timeline"]
        daily = {
            "2024-01-15": {"total": 50, "sent": 25, "received": 25},
            "2024-01-16": {"total": 60, "sent": 30, "received": 30},
        }
        result = build_timeline("day", daily, {})
        assert len(result) == 2
        assert result[0]["date"] == "2024-01-15"
        assert result[0]["total"] == 50

    def test_build_timeline_week_granularity(self, analytics_router):
        """Week granularity aggregates correctly."""
        build_timeline = analytics_router["build_timeline"]
        daily = {
            "2024-01-15": {"total": 10, "sent": 5, "received": 5},
            "2024-01-16": {"total": 20, "sent": 10, "received": 10},
        }
        result = build_timeline("week", daily, {})
        assert len(result) >= 1
        assert "date" in result[0]
        assert result[0]["total"] == 30

    def test_build_timeline_month_granularity(self, analytics_router):
        """Month granularity aggregates correctly."""
        build_timeline = analytics_router["build_timeline"]
        daily = {
            "2024-01-15": {"total": 10, "sent": 5, "received": 5},
            "2024-01-20": {"total": 20, "sent": 10, "received": 10},
        }
        result = build_timeline("month", daily, {})
        assert len(result) == 1
        assert result[0]["date"] == "2024-01"
        assert result[0]["total"] == 30


# =============================================================================
# Overview Endpoint Tests
# =============================================================================


class TestOverviewEndpoint:
    """Tests for GET /analytics/overview."""

    @patch("api.routers.analytics.run_in_threadpool")
    @patch("api.routers.analytics.get_imessage_reader")
    def test_overview_success(
        self, mock_get_reader, mock_threadpool, client, mock_reader, mock_request
    ):
        """Overview returns aggregated metrics."""
        mock_get_reader.return_value.__enter__.return_value = mock_reader

        overview_data = {
            "total_messages": 1000,
            "sent_messages": 500,
            "received_messages": 500,
            "active_conversations": 10,
            "avg_messages_per_day": 50.5,
            "avg_response_time_minutes": 12.5,
            "sentiment": {"score": 0.35, "label": "positive"},
            "peak_hour": 14,
            "peak_day": "Wednesday",
            "date_range": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-31T23:59:59Z"},
            "period_comparison": {
                "total_change_percent": 15.5,
                "sent_change_percent": 10.2,
                "contacts_change_percent": 5.0,
            },
            "time_range": "month",
        }

        async def mock_fetch(*args):
            return overview_data

        mock_threadpool.side_effect = mock_fetch

        response = client.get("/analytics/overview?time_range=month")

        assert response.status_code == 200
        data = response.json()
        assert data["total_messages"] == 1000
        assert data["active_conversations"] == 10
        assert data["sentiment"]["score"] == 0.35

    @patch("api.routers.analytics.get_imessage_reader")
    def test_overview_uses_cache(self, mock_get_reader, client, mock_reader):
        """Second request uses cached data."""
        mock_get_reader.return_value.__enter__.return_value = mock_reader

        overview_data = {"total_messages": 100, "time_range": "week"}

        with patch("api.routers.analytics.run_in_threadpool") as mock_threadpool:

            async def mock_fetch(*args):
                return overview_data

            mock_threadpool.side_effect = mock_fetch

            # First request
            client.get("/analytics/overview?time_range=week")
            assert mock_threadpool.call_count == 1

            # Second request - should use cache
            response = client.get("/analytics/overview?time_range=week")
            assert response.status_code == 200
            # Still called once (cache hit)
            assert mock_threadpool.call_count == 1

    def test_overview_invalid_time_range(self, client):
        """Invalid time_range returns 422."""
        response = client.get("/analytics/overview?time_range=invalid")
        assert response.status_code == 422


# =============================================================================
# Timeline Endpoint Tests
# =============================================================================


class TestTimelineEndpoint:
    """Tests for GET /analytics/timeline."""

    @patch("api.routers.analytics.run_in_threadpool")
    @patch("api.routers.analytics.get_imessage_reader")
    def test_timeline_day_granularity(self, mock_get_reader, mock_threadpool, client, mock_reader):
        """Timeline with day granularity returns daily data."""
        mock_get_reader.return_value.__enter__.return_value = mock_reader

        timeline_data = {
            "granularity": "day",
            "metric": "messages",
            "time_range": "month",
            "data": [
                {"date": "2024-01-15", "total": 45, "sent": 22, "received": 23},
                {"date": "2024-01-16", "total": 52, "sent": 25, "received": 27},
            ],
            "total_points": 2,
        }

        async def mock_fetch(*args):
            return timeline_data

        mock_threadpool.side_effect = mock_fetch

        response = client.get("/analytics/timeline?granularity=day&time_range=month")

        assert response.status_code == 200
        data = response.json()
        assert data["granularity"] == "day"
        assert len(data["data"]) == 2

    @patch("api.routers.analytics.run_in_threadpool")
    @patch("api.routers.analytics.get_imessage_reader")
    def test_timeline_hour_granularity(self, mock_get_reader, mock_threadpool, client, mock_reader):
        """Timeline with hour granularity returns hourly data."""
        mock_get_reader.return_value.__enter__.return_value = mock_reader

        timeline_data = {
            "granularity": "hour",
            "metric": "messages",
            "time_range": "week",
            "data": [
                {"hour": 9, "total": 10, "sent": 5, "received": 5},
                {"hour": 14, "total": 20, "sent": 12, "received": 8},
            ],
            "total_points": 2,
        }

        async def mock_fetch(*args):
            return timeline_data

        mock_threadpool.side_effect = mock_fetch

        response = client.get("/analytics/timeline?granularity=hour&time_range=week")

        assert response.status_code == 200
        data = response.json()
        assert data["granularity"] == "hour"
        assert len(data["data"]) == 2

    def test_timeline_invalid_granularity(self, client):
        """Invalid granularity returns 422."""
        response = client.get("/analytics/timeline?granularity=invalid")
        assert response.status_code == 422

    def test_timeline_invalid_metric(self, client):
        """Invalid metric returns 422."""
        response = client.get("/analytics/timeline?metric=invalid")
        assert response.status_code == 422


# =============================================================================
# Heatmap Endpoint Tests
# =============================================================================


class TestHeatmapEndpoint:
    """Tests for GET /analytics/heatmap."""

    @patch("api.routers.analytics.run_in_threadpool")
    @patch("api.routers.analytics.get_imessage_reader")
    def test_heatmap_success(self, mock_get_reader, mock_threadpool, client, mock_reader):
        """Heatmap returns activity data."""
        mock_get_reader.return_value.__enter__.return_value = mock_reader

        heatmap_data = {
            "data": [
                {"date": "2024-01-15", "count": 45, "level": 4},
                {"date": "2024-01-16", "count": 12, "level": 2},
            ],
            "stats": {
                "total_days": 90,
                "active_days": 75,
                "max_count": 85,
                "avg_count": 25.5,
            },
            "time_range": "three_months",
        }

        async def mock_fetch(*args):
            return heatmap_data

        mock_threadpool.side_effect = mock_fetch

        response = client.get("/analytics/heatmap?time_range=three_months")

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2
        assert data["stats"]["total_days"] == 90
        assert data["stats"]["active_days"] == 75

    @patch("api.routers.analytics.run_in_threadpool")
    @patch("api.routers.analytics.get_imessage_reader")
    def test_heatmap_empty_data(self, mock_get_reader, mock_threadpool, client, mock_reader):
        """Heatmap with no messages returns empty data."""
        mock_get_reader.return_value.__enter__.return_value = mock_reader

        heatmap_data = {
            "data": [],
            "stats": {"total_days": 0, "active_days": 0, "max_count": 0, "avg_count": 0},
            "time_range": "month",
        }

        async def mock_fetch(*args):
            return heatmap_data

        mock_threadpool.side_effect = mock_fetch

        response = client.get("/analytics/heatmap?time_range=month")

        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []
        assert data["stats"]["active_days"] == 0


# =============================================================================
# Contact Stats Endpoint Tests
# =============================================================================


class TestContactStatsEndpoint:
    """Tests for GET /analytics/contacts/{chat_id}/stats."""

    @patch("api.routers.analytics.run_in_threadpool")
    @patch("api.routers.analytics.get_imessage_reader")
    def test_contact_stats_success(self, mock_get_reader, mock_threadpool, client, mock_reader):
        """Contact stats returns detailed analytics."""
        mock_get_reader.return_value.__enter__.return_value = mock_reader

        stats_data = {
            "contact_id": "chat123",
            "contact_name": "John Doe",
            "total_messages": 500,
            "sent_count": 245,
            "received_count": 255,
            "avg_response_time_minutes": 8.5,
            "sentiment_score": 0.42,
            "engagement_score": 78.5,
            "message_trend": "increasing",
            "last_message_date": "2024-01-31T10:30:00Z",
            "emoji_usage": {"total": 50, "per_message": 0.1, "top_emojis": [("😊", 10)]},
            "hourly_distribution": {},
            "daily_distribution": {},
            "weekly_counts": {},
            "time_range": "month",
        }

        async def mock_fetch(*args):
            return stats_data

        mock_threadpool.side_effect = mock_fetch

        response = client.get("/analytics/contacts/chat123/stats?time_range=month")

        assert response.status_code == 200
        data = response.json()
        assert data["contact_id"] == "chat123"
        assert data["total_messages"] == 500
        assert data["engagement_score"] == 78.5

    @patch("api.routers.analytics.run_in_threadpool")
    @patch("api.routers.analytics.get_imessage_reader")
    def test_contact_stats_not_found(self, mock_get_reader, mock_threadpool, client, mock_reader):
        """Contact with no messages returns 404."""
        mock_get_reader.return_value.__enter__.return_value = mock_reader

        async def mock_fetch(*args):
            return None

        mock_threadpool.side_effect = mock_fetch

        response = client.get("/analytics/contacts/nonexistent/stats")

        assert response.status_code == 404
        assert "No messages found" in response.json()["detail"]


# =============================================================================
# Leaderboard Endpoint Tests
# =============================================================================


class TestLeaderboardEndpoint:
    """Tests for GET /analytics/contacts/leaderboard."""

    @patch("api.routers.analytics.run_in_threadpool")
    @patch("api.routers.analytics.get_imessage_reader")
    def test_leaderboard_by_messages(self, mock_get_reader, mock_threadpool, client, mock_reader):
        """Leaderboard sorts by message count."""
        mock_get_reader.return_value.__enter__.return_value = mock_reader

        leaderboard_data = {
            "contacts": [
                {
                    "rank": 1,
                    "contact_id": "chat1",
                    "contact_name": "Alice",
                    "total_messages": 1000,
                    "sent_count": 500,
                    "received_count": 500,
                    "engagement_score": 85.0,
                    "avg_response_time_minutes": 5.0,
                    "sentiment_score": 0.5,
                    "trend": "stable",
                },
                {
                    "rank": 2,
                    "contact_id": "chat2",
                    "contact_name": "Bob",
                    "total_messages": 800,
                    "sent_count": 400,
                    "received_count": 400,
                    "engagement_score": 80.0,
                    "avg_response_time_minutes": 7.0,
                    "sentiment_score": 0.4,
                    "trend": "increasing",
                },
            ],
            "total_contacts": 2,
            "sort_by": "messages",
            "time_range": "month",
        }

        async def mock_fetch(*args):
            return leaderboard_data

        mock_threadpool.side_effect = mock_fetch

        response = client.get("/analytics/contacts/leaderboard?sort_by=messages&limit=10")

        assert response.status_code == 200
        data = response.json()
        assert len(data["contacts"]) == 2
        assert data["contacts"][0]["rank"] == 1
        assert data["contacts"][0]["total_messages"] == 1000

    @patch("api.routers.analytics.run_in_threadpool")
    @patch("api.routers.analytics.get_imessage_reader")
    def test_leaderboard_respects_limit(
        self, mock_get_reader, mock_threadpool, client, mock_reader
    ):
        """Leaderboard respects limit parameter."""
        mock_get_reader.return_value.__enter__.return_value = mock_reader

        leaderboard_data = {
            "contacts": [{"rank": i + 1, "contact_id": f"chat{i}"} for i in range(5)],
            "total_contacts": 10,
            "sort_by": "messages",
            "time_range": "month",
        }

        async def mock_fetch(*args):
            return leaderboard_data

        mock_threadpool.side_effect = mock_fetch

        response = client.get("/analytics/contacts/leaderboard?limit=5")

        assert response.status_code == 200
        data = response.json()
        assert len(data["contacts"]) == 5

    def test_leaderboard_invalid_sort_by(self, client):
        """Invalid sort_by returns 422."""
        response = client.get("/analytics/contacts/leaderboard?sort_by=invalid")
        assert response.status_code == 422

    def test_leaderboard_limit_too_large(self, client):
        """Limit above 50 returns 422."""
        response = client.get("/analytics/contacts/leaderboard?limit=100")
        assert response.status_code == 422


# =============================================================================
# Trends Endpoint Tests
# =============================================================================


class TestTrendsEndpoint:
    """Tests for GET /analytics/trends."""

    @patch("api.routers.analytics.run_in_threadpool")
    @patch("api.routers.analytics.get_imessage_reader")
    def test_trends_success(self, mock_get_reader, mock_threadpool, client, mock_reader):
        """Trends returns detected patterns."""
        mock_get_reader.return_value.__enter__.return_value = mock_reader

        trends_data = {
            "overall_trend": {
                "direction": "increasing",
                "percentage_change": 25.5,
                "confidence": 0.85,
            },
            "weekly_trend": {"direction": "stable", "percentage_change": 2.0, "confidence": 0.6},
            "trending_contacts": [
                {
                    "contact_id": "chat123",
                    "contact_name": "John",
                    "trend": "increasing",
                    "change_percent": 45.5,
                    "confidence": 0.9,
                }
            ],
            "anomalies": [
                {
                    "date": "2024-01-20",
                    "type": "spike",
                    "value": 150,
                    "expected": 45,
                    "deviation": 105,
                }
            ],
            "peak_hours": [{"hour": 14, "count": 500, "percentage": 25.0}],
            "peak_days": [{"day": "Monday", "count": 1000, "percentage": 20.0}],
            "seasonality": {"detected": True, "pattern": "weekly"},
            "time_range": "month",
        }

        async def mock_fetch(*args):
            return trends_data

        mock_threadpool.side_effect = mock_fetch

        response = client.get("/analytics/trends?time_range=month")

        assert response.status_code == 200
        data = response.json()
        assert data["overall_trend"]["direction"] == "increasing"
        assert len(data["anomalies"]) == 1
        assert data["seasonality"]["detected"] is True


# =============================================================================
# Export Endpoint Tests
# =============================================================================


class TestExportEndpoint:
    """Tests for GET /analytics/export."""

    @patch("api.routers.analytics.run_in_threadpool")
    @patch("api.routers.analytics.get_imessage_reader")
    def test_export_json_format(self, mock_get_reader, mock_threadpool, client, mock_reader):
        """Export in JSON format returns JSON content."""
        mock_get_reader.return_value.__enter__.return_value = mock_reader

        async def mock_fetch(*args):
            return ('{"data": "test"}', "application/json", "analytics_month.json")

        mock_threadpool.side_effect = mock_fetch

        response = client.get("/analytics/export?format=json&time_range=month")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert "analytics_month.json" in response.headers["content-disposition"]

    @patch("api.routers.analytics.run_in_threadpool")
    @patch("api.routers.analytics.get_imessage_reader")
    def test_export_csv_format(self, mock_get_reader, mock_threadpool, client, mock_reader):
        """Export in CSV format returns CSV content."""
        mock_get_reader.return_value.__enter__.return_value = mock_reader

        async def mock_fetch(*args):
            return ("date,count\n2024-01-15,50\n", "text/csv", "daily_analytics_month.csv")

        mock_threadpool.side_effect = mock_fetch

        response = client.get("/analytics/export?format=csv&time_range=month")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "daily_analytics_month.csv" in response.headers["content-disposition"]

    def test_export_invalid_format(self, client):
        """Invalid format returns 422."""
        response = client.get("/analytics/export?format=xml")
        assert response.status_code == 422


# =============================================================================
# Cache Management Tests
# =============================================================================


class TestCacheManagement:
    """Tests for cache management."""

    @patch("api.routers.analytics.get_imessage_reader")
    def test_clear_cache_success(self, mock_get_reader, client, mock_reader):
        """Clear cache endpoint resets cache."""
        mock_get_reader.return_value.__enter__.return_value = mock_reader

        response = client.delete("/analytics/cache")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "cleared" in data["message"].lower()

    def test_cache_isolation_between_time_ranges(self, analytics_router):
        """Different time ranges use separate cache keys."""
        cache = analytics_router["get_cache"]()

        cache.set("overview:week", {"data": "week"})
        cache.set("overview:month", {"data": "month"})

        found_week, data_week = cache.get("overview:week")
        found_month, data_month = cache.get("overview:month")

        assert found_week
        assert found_month
        assert data_week["data"] == "week"
        assert data_month["data"] == "month"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_no_full_disk_access(self, app, client):
        """Without Full Disk Access, returns 403."""
        from api.dependencies import get_imessage_reader

        def fake_reader():
            raise HTTPException(status_code=403, detail="Permission denied")

        app.dependency_overrides[get_imessage_reader] = fake_reader
        try:
            response = client.get("/analytics/overview")
            assert response.status_code == 403
        finally:
            app.dependency_overrides.pop(get_imessage_reader, None)

    @patch("api.routers.analytics.run_in_threadpool")
    @patch("api.routers.analytics.get_imessage_reader")
    def test_database_error_handling(self, mock_get_reader, mock_threadpool, client, mock_reader):
        """Database errors are handled gracefully."""
        mock_get_reader.return_value.__enter__.return_value = mock_reader

        async def mock_fetch(*args):
            raise RuntimeError("Database connection failed")

        mock_threadpool.side_effect = mock_fetch

        response = client.get("/analytics/overview")

        # Should return 500 or handle gracefully
        assert response.status_code >= 400
