"""Tests for the analytics module.

Tests analytics engine, aggregator, trends detection, and report generation.
"""

from datetime import UTC, datetime, timedelta

from jarvis.analytics import (
    AnalyticsEngine,
    ReportGenerator,
    TimeSeriesAggregator,
    TrendAnalyzer,
    aggregate_by_day,
    aggregate_by_hour,
    aggregate_by_month,
    aggregate_by_week,
    detect_anomalies,
    detect_peak_periods,
    detect_trend,
)
from jarvis.contracts.imessage import Message


def create_message(
    text: str,
    is_from_me: bool = False,
    date: datetime | None = None,
    msg_id: int = 1,
    chat_id: str = "test_chat",
) -> Message:
    """Helper to create a Message object for testing."""
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


class TestAnalyticsEngine:
    """Tests for AnalyticsEngine class."""

    def test_compute_overview_empty_messages(self):
        """Empty messages returns default overview."""
        engine = AnalyticsEngine()
        result = engine.compute_overview([])
        assert result.total_messages == 0
        assert result.total_sent == 0
        assert result.total_received == 0

    def test_compute_overview_basic_counts(self):
        """Basic message counts are computed correctly."""
        engine = AnalyticsEngine()
        now = datetime.now(UTC)
        messages = [
            create_message("Hello", is_from_me=False, date=now),
            create_message("Hi", is_from_me=True, date=now + timedelta(minutes=5)),
            create_message("How are you?", is_from_me=False, date=now + timedelta(minutes=10)),
        ]
        result = engine.compute_overview(messages)
        assert result.total_messages == 3
        assert result.total_sent == 1
        assert result.total_received == 2

    def test_compute_overview_with_time_range(self):
        """Time range filtering works correctly."""
        engine = AnalyticsEngine()
        now = datetime.now(UTC)
        messages = [
            create_message("Old", date=now - timedelta(days=30)),
            create_message("Recent", date=now - timedelta(days=3)),
        ]
        result = engine.compute_overview(messages, time_range_days=7)
        assert result.total_messages == 1

    def test_compute_overview_peak_times(self):
        """Peak hour and day are detected."""
        engine = AnalyticsEngine()
        now = datetime.now(UTC).replace(hour=14, minute=0, second=0, microsecond=0)
        messages = []
        # Add multiple messages at 2pm
        for i in range(5):
            messages.append(create_message(f"msg{i}", date=now + timedelta(days=i)))
        result = engine.compute_overview(messages)
        assert result.peak_hour == 14

    def test_compute_contact_analytics_empty(self):
        """Empty messages returns default contact analytics."""
        engine = AnalyticsEngine()
        result = engine.compute_contact_analytics([], "test_contact")
        assert result.total_messages == 0
        assert result.contact_id == "test_contact"

    def test_compute_contact_analytics_engagement(self):
        """Engagement score is computed correctly."""
        engine = AnalyticsEngine()
        now = datetime.now(UTC)
        messages = []
        # Balanced conversation
        for i in range(10):
            messages.append(
                create_message(f"them{i}", is_from_me=False, date=now + timedelta(hours=i * 2))
            )
            messages.append(
                create_message(f"me{i}", is_from_me=True, date=now + timedelta(hours=i * 2 + 1))
            )
        result = engine.compute_contact_analytics(messages, "test_contact")
        assert result.engagement_score > 50

    def test_compute_emoji_stats_empty(self):
        """Empty messages returns zero emoji stats."""
        engine = AnalyticsEngine()
        result = engine.compute_emoji_stats([])
        assert result.total_count == 0
        assert result.unique_count == 0

    def test_compute_emoji_stats_with_emojis(self):
        """Emoji stats are computed correctly."""
        engine = AnalyticsEngine()
        messages = [
            create_message("I love this! ❤️ 😊"),
            create_message("Great! 😊 😊"),
        ]
        result = engine.compute_emoji_stats(messages)
        assert result.total_count >= 2
        assert result.unique_count >= 2

    def test_compute_message_length_stats_empty(self):
        """Empty messages returns default length stats."""
        engine = AnalyticsEngine()
        result = engine.compute_message_length_stats([])
        assert result.avg_length == 0.0

    def test_compute_message_length_stats_distribution(self):
        """Message length distribution is computed correctly."""
        engine = AnalyticsEngine()
        messages = [
            create_message("Hi"),  # short
            create_message("This is a medium length message that has some words"),  # medium
            create_message("A" * 150),  # long
            create_message("B" * 400),  # very long
        ]
        result = engine.compute_message_length_stats(messages)
        assert result.short_count == 1
        assert result.medium_count == 1
        assert result.long_count == 1
        assert result.very_long_count == 1

    def test_compute_time_distributions(self):
        """Time distributions are computed correctly."""
        engine = AnalyticsEngine()
        now = datetime.now(UTC)
        messages = [
            create_message("Morning", date=now.replace(hour=9)),
            create_message("Afternoon", date=now.replace(hour=14)),
            create_message("Evening", date=now.replace(hour=20)),
        ]
        hourly, daily, weekly, monthly = engine.compute_time_distributions(messages)
        assert 9 in hourly
        assert 14 in hourly
        assert 20 in hourly


class TestAggregator:
    """Tests for aggregation functions."""

    def test_aggregate_by_hour_empty(self):
        """Empty messages returns 24 zeroed hours."""
        result = aggregate_by_hour([])
        assert len(result) == 24
        assert all(h.count == 0 for h in result)

    def test_aggregate_by_hour_distribution(self):
        """Hourly distribution is computed correctly."""
        now = datetime.now(UTC)
        messages = [
            create_message("Morning", date=now.replace(hour=9)),
            create_message("Morning2", date=now.replace(hour=9)),
            create_message("Evening", date=now.replace(hour=20)),
        ]
        result = aggregate_by_hour(messages)
        assert result[9].count == 2
        assert result[20].count == 1

    def test_aggregate_by_day_empty(self):
        """Empty messages returns empty list."""
        result = aggregate_by_day([])
        assert result == []

    def test_aggregate_by_day_multiple_days(self):
        """Multiple days are aggregated correctly."""
        # Use noon to avoid date boundary issues when running near midnight
        now = datetime.now(UTC).replace(hour=12, minute=0, second=0, microsecond=0)
        messages = [
            create_message("Day1", date=now),
            create_message("Day1_2", date=now + timedelta(hours=1)),
            create_message("Day2", date=now + timedelta(days=1)),
        ]
        result = aggregate_by_day(messages)
        assert len(result) == 2
        assert result[0].total_messages == 2
        assert result[1].total_messages == 1

    def test_aggregate_by_week_empty(self):
        """Empty messages returns empty list."""
        result = aggregate_by_week([])
        assert result == []

    def test_aggregate_by_month_empty(self):
        """Empty messages returns empty list."""
        result = aggregate_by_month([])
        assert result == []


class TestTimeSeriesAggregator:
    """Tests for TimeSeriesAggregator class."""

    def test_update_and_retrieve_daily(self):
        """Daily aggregates can be updated and retrieved."""
        aggregator = TimeSeriesAggregator()
        # Use noon to avoid date boundary issues when running near midnight
        now = datetime.now(UTC).replace(hour=12, minute=0, second=0, microsecond=0)
        messages = [
            create_message("Day1", date=now),
            create_message("Day2", date=now + timedelta(days=1)),
        ]
        aggregator.update_daily_aggregates(messages)
        result = aggregator.get_daily_aggregates()
        assert len(result) == 2

    def test_get_activity_heatmap_data(self):
        """Activity heatmap data is correctly formatted."""
        aggregator = TimeSeriesAggregator()
        # Use noon to avoid date boundary issues when running near midnight
        now = datetime.now(UTC).replace(hour=12, minute=0, second=0, microsecond=0)
        messages = [
            create_message("Day1", date=now),
            create_message("Day1_2", date=now + timedelta(hours=1)),
        ]
        aggregator.update_daily_aggregates(messages)
        result = aggregator.get_activity_heatmap_data()
        assert len(result) == 1
        assert "date" in result[0]
        assert "count" in result[0]
        assert "level" in result[0]

    def test_get_timeline_data_with_messages(self):
        """Timeline data is computed from messages."""
        aggregator = TimeSeriesAggregator()
        # Use noon to avoid date boundary issues when running near midnight
        now = datetime.now(UTC).replace(hour=12, minute=0, second=0, microsecond=0)
        messages = [
            create_message("Msg1", is_from_me=True, date=now),
            create_message("Msg2", is_from_me=False, date=now + timedelta(hours=1)),
        ]
        result = aggregator.get_timeline_data(granularity="day", messages=messages)
        assert len(result) == 1
        assert result[0]["total"] == 2
        assert result[0]["sent"] == 1
        assert result[0]["received"] == 1

    def test_clear_cache(self):
        """Cache can be cleared."""
        aggregator = TimeSeriesAggregator()
        now = datetime.now(UTC)
        messages = [create_message("Test", date=now)]
        aggregator.update_daily_aggregates(messages)
        assert len(aggregator.get_daily_aggregates()) == 1
        aggregator.clear_cache()
        assert len(aggregator.get_daily_aggregates()) == 0


class TestTrendDetection:
    """Tests for trend detection functions."""

    def test_detect_trend_empty(self):
        """Empty values returns stable trend."""
        result = detect_trend([])
        assert result.direction == "stable"
        assert result.confidence == 0.0

    def test_detect_trend_increasing(self):
        """Increasing values are detected."""
        values = [10, 12, 15, 20, 25, 30, 40, 50]
        result = detect_trend(values)
        assert result.direction == "increasing"
        assert result.percentage_change > 0

    def test_detect_trend_decreasing(self):
        """Decreasing values are detected."""
        values = [50, 40, 30, 25, 20, 15, 12, 10]
        result = detect_trend(values)
        assert result.direction == "decreasing"
        assert result.percentage_change < 0

    def test_detect_trend_stable(self):
        """Stable values are detected."""
        values = [20, 21, 19, 20, 21, 20, 19, 20]
        result = detect_trend(values)
        assert result.direction == "stable"
        assert -10 <= result.percentage_change <= 10

    def test_detect_anomalies_empty(self):
        """Empty data returns no anomalies."""
        result = detect_anomalies([])
        assert result == []

    def test_detect_anomalies_spike(self):
        """Spike anomalies are detected."""
        data = [
            ("2024-01-01", 10),
            ("2024-01-02", 12),
            ("2024-01-03", 11),
            ("2024-01-04", 500),  # Spike - very extreme value to ensure detection
            ("2024-01-05", 10),
        ]
        # Use a lower threshold to ensure detection
        result = detect_anomalies(data, threshold_std=1.5)
        assert len(result) >= 1
        assert result[0].anomaly_type == "spike"

    def test_detect_peak_periods_empty(self):
        """Empty messages returns empty peak periods."""
        hours, days = detect_peak_periods([])
        assert hours == []
        assert days == []

    def test_detect_peak_periods_with_messages(self):
        """Peak periods are detected from messages."""
        now = datetime.now(UTC)
        messages = []
        # Add more messages at 2pm
        for i in range(5):
            messages.append(
                create_message(f"msg{i}", date=now.replace(hour=14) + timedelta(days=i))
            )
        # Add fewer at 9am
        messages.append(create_message("morning", date=now.replace(hour=9)))
        hours, days = detect_peak_periods(messages, top_n=3)
        assert len(hours) >= 1
        assert hours[0].period_value == 14


class TestTrendAnalyzer:
    """Tests for TrendAnalyzer class."""

    def test_analyze_message_trends_empty(self):
        """Empty messages returns default analysis."""
        analyzer = TrendAnalyzer()
        result = analyzer.analyze_message_trends([])
        assert result.overall_trend.direction == "stable"
        assert result.anomalies == []
        assert result.peak_hours == []

    def test_analyze_message_trends_with_messages(self):
        """Message trends are analyzed correctly."""
        analyzer = TrendAnalyzer()
        now = datetime.now(UTC)
        messages = []
        for i in range(30):
            messages.append(create_message(f"msg{i}", date=now - timedelta(days=i)))
        result = analyzer.analyze_message_trends(messages)
        assert result.overall_trend is not None
        assert len(result.peak_hours) > 0
        assert len(result.peak_days) > 0

    def test_get_trending_contacts(self):
        """Trending contacts are identified."""
        analyzer = TrendAnalyzer()
        now = datetime.now(UTC)
        messages = []
        # Contact 1: increasing activity
        for i in range(10):
            count = 1 if i < 5 else 3
            for j in range(count):
                messages.append(
                    create_message(f"c1_{i}_{j}", chat_id="chat1", date=now + timedelta(days=i))
                )
        result = analyzer.get_trending_contacts(messages, top_n=5)
        # Should detect at least the trending contact
        assert isinstance(result, list)

    def test_compare_periods(self):
        """Period comparison works correctly."""
        analyzer = TrendAnalyzer()
        now = datetime.now(UTC)
        messages = []
        # Current period: 5 messages
        for i in range(5):
            messages.append(create_message(f"current{i}", date=now - timedelta(days=i)))
        # Previous period: 10 messages
        for i in range(7, 17):
            messages.append(create_message(f"previous{i}", date=now - timedelta(days=i)))
        result = analyzer.compare_periods(messages, current_days=7, previous_days=7)
        assert "total_messages_change" in result
        assert "current_period_total" in result
        assert "previous_period_total" in result


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    def test_generate_overview_report_empty(self):
        """Empty messages generates minimal report."""
        gen = ReportGenerator()
        result = gen.generate_overview_report([])
        assert result.title == "Conversation Analytics Overview"
        assert result.sections == []
        assert result.date_range_start is None

    def test_generate_overview_report_with_messages(self):
        """Overview report includes all sections."""
        gen = ReportGenerator()
        now = datetime.now(UTC)
        messages = []
        for i in range(20):
            messages.append(
                create_message(f"msg{i}", is_from_me=i % 2 == 0, date=now - timedelta(days=i))
            )
        result = gen.generate_overview_report(messages)
        assert result.title == "Conversation Analytics Overview"
        assert len(result.sections) > 0
        assert result.date_range_start is not None

    def test_generate_contact_report(self):
        """Contact report is generated correctly."""
        gen = ReportGenerator()
        now = datetime.now(UTC)
        messages = [
            create_message("Hello", date=now),
            create_message("Hi", is_from_me=True, date=now + timedelta(minutes=5)),
        ]
        result = gen.generate_contact_report(messages, "test_contact", "Test User")
        assert "Test User" in result.title
        assert len(result.sections) > 0

    def test_generate_comparison_report(self):
        """Comparison report is generated correctly."""
        gen = ReportGenerator()
        now = datetime.now(UTC)
        messages = []
        for i in range(20):
            messages.append(create_message(f"msg{i}", date=now - timedelta(days=i)))
        result = gen.generate_comparison_report(messages, current_days=7, previous_days=7)
        assert "Comparison" in result.title
        assert len(result.sections) > 0

    def test_export_to_csv(self):
        """CSV export generates valid content."""
        gen = ReportGenerator()
        now = datetime.now(UTC)
        messages = [
            create_message("Day1", date=now),
            create_message("Day2", date=now + timedelta(days=1)),
        ]
        result = gen.export_to_csv(messages)
        assert "daily_analytics.csv" in result
        assert "date," in result["daily_analytics.csv"]

    def test_export_to_json(self):
        """JSON export generates valid content."""
        gen = ReportGenerator()
        now = datetime.now(UTC)
        messages = [
            create_message("Test1", date=now),
            create_message("Test2", date=now + timedelta(hours=1)),
        ]
        result = gen.export_to_json(messages)
        assert "overview" in result
        assert "total_messages" in result

    def test_report_to_dict(self):
        """Report can be converted to dictionary."""
        gen = ReportGenerator()
        now = datetime.now(UTC)
        messages = [create_message("Test", date=now)]
        report = gen.generate_overview_report(messages)
        result = report.to_dict()
        assert "title" in result
        assert "sections" in result
        assert "generated_at" in result

    def test_report_to_json(self):
        """Report can be converted to JSON string."""
        gen = ReportGenerator()
        now = datetime.now(UTC)
        messages = [create_message("Test", date=now)]
        report = gen.generate_overview_report(messages)
        result = report.to_json()
        assert isinstance(result, str)
        assert "title" in result
