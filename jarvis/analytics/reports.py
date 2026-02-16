"""Report generation for analytics.

Provides structured report generation for analytics data,
supporting multiple output formats (JSON, CSV, etc.).
"""

from __future__ import annotations

import csv
import io
import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from jarvis.analytics.aggregator import (
    aggregate_by_day,
    aggregate_by_month,
    aggregate_by_week,
)
from jarvis.analytics.engine import AnalyticsEngine, get_analytics_engine
from jarvis.analytics.trends import TrendAnalyzer, get_trend_analyzer

if TYPE_CHECKING:
    from contracts.imessage import Message


@dataclass
class ReportSection:
    """A section within an analytics report."""

    title: str
    description: str
    data: dict[str, Any]
    chart_type: str | None = None  # "line", "bar", "pie", "heatmap", etc.


@dataclass
class AnalyticsReport:
    """Complete analytics report."""

    title: str
    description: str
    generated_at: datetime
    date_range_start: datetime | None
    date_range_end: datetime | None
    sections: list[ReportSection] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "title": self.title,
            "description": self.description,
            "generated_at": self.generated_at.isoformat(),
            "date_range_start": (
                self.date_range_start.isoformat() if self.date_range_start else None
            ),
            "date_range_end": (self.date_range_end.isoformat() if self.date_range_end else None),
            "sections": [
                {
                    "title": s.title,
                    "description": s.description,
                    "data": s.data,
                    "chart_type": s.chart_type,
                }
                for s in self.sections
            ],
            "summary": self.summary,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class ReportGenerator:
    """Generator for analytics reports.

    Creates comprehensive analytics reports from message data
    with support for various output formats.
    """

    def __init__(
        self,
        engine: AnalyticsEngine | None = None,
        trend_analyzer: TrendAnalyzer | None = None,
    ) -> None:
        """Initialize report generator.

        Args:
            engine: Analytics engine instance (uses global if None)
            trend_analyzer: Trend analyzer instance (uses global if None)
        """
        self._engine = engine or get_analytics_engine()
        self._trend_analyzer = trend_analyzer or get_trend_analyzer()

    def generate_overview_report(
        self,
        messages: list[Message],
        title: str = "Conversation Analytics Overview",
    ) -> AnalyticsReport:
        """Generate a comprehensive overview report.

        Args:
            messages: Messages to analyze
            title: Report title

        Returns:
            AnalyticsReport with overview statistics
        """
        if not messages:
            return AnalyticsReport(
                title=title,
                description="No messages to analyze",
                generated_at=datetime.now(UTC),
                date_range_start=None,
                date_range_end=None,
            )

        # Compute analytics
        overview = self._engine.compute_overview(messages)
        emoji_stats = self._engine.compute_emoji_stats(messages)
        length_stats = self._engine.compute_message_length_stats(messages)
        hourly, daily, weekly, monthly = self._engine.compute_time_distributions(messages)

        # Trend analysis
        trend_analysis = self._trend_analyzer.analyze_message_trends(messages)

        sections = [
            # Overview section
            ReportSection(
                title="Overview",
                description="High-level messaging statistics",
                data={
                    "total_messages": overview.total_messages,
                    "sent_messages": overview.total_sent,
                    "received_messages": overview.total_received,
                    "active_conversations": overview.active_conversations,
                    "avg_messages_per_day": overview.avg_messages_per_day,
                    "avg_response_time_minutes": overview.avg_response_time_minutes,
                },
            ),
            # Sentiment section
            ReportSection(
                title="Sentiment Analysis",
                description="Overall conversation sentiment",
                data={
                    "sentiment_score": overview.sentiment_score,
                    "sentiment_label": overview.sentiment_label,
                },
                chart_type="gauge",
            ),
            # Activity patterns
            ReportSection(
                title="Activity Patterns",
                description="When you communicate most",
                data={
                    "peak_hour": overview.peak_hour,
                    "peak_day": overview.peak_day,
                    "hourly_distribution": hourly,
                    "daily_distribution": daily,
                },
                chart_type="heatmap",
            ),
            # Time series
            ReportSection(
                title="Message Volume Over Time",
                description="Weekly message counts",
                data={
                    "weekly_counts": weekly,
                    "monthly_counts": monthly,
                    "trend": {
                        "direction": trend_analysis.overall_trend.direction,
                        "percentage_change": trend_analysis.overall_trend.percentage_change,
                        "confidence": trend_analysis.overall_trend.confidence,
                    },
                },
                chart_type="line",
            ),
            # Emoji usage
            ReportSection(
                title="Emoji Usage",
                description="Most frequently used emojis",
                data={
                    "total_emojis": emoji_stats.total_count,
                    "unique_emojis": emoji_stats.unique_count,
                    "emojis_per_message": emoji_stats.emojis_per_message,
                    "top_emojis": emoji_stats.top_emojis,
                },
                chart_type="bar",
            ),
            # Message length
            ReportSection(
                title="Message Length Distribution",
                description="How long your messages typically are",
                data={
                    "avg_length": length_stats.avg_length,
                    "median_length": length_stats.median_length,
                    "distribution": {
                        "short": length_stats.short_count,
                        "medium": length_stats.medium_count,
                        "long": length_stats.long_count,
                        "very_long": length_stats.very_long_count,
                    },
                },
                chart_type="pie",
            ),
        ]

        # Add anomalies if any detected
        if trend_analysis.anomalies:
            sections.append(
                ReportSection(
                    title="Detected Anomalies",
                    description="Unusual activity patterns",
                    data={
                        "anomalies": [
                            {
                                "date": a.date,
                                "value": a.value,
                                "expected": a.expected_value,
                                "type": a.anomaly_type,
                            }
                            for a in trend_analysis.anomalies
                        ]
                    },
                )
            )

        # Summary
        summary = {
            "total_messages": overview.total_messages,
            "date_range": f"{overview.date_range_start} to {overview.date_range_end}"
            if overview.date_range_start
            else "N/A",
            "overall_trend": trend_analysis.overall_trend.direction,
            "sentiment": overview.sentiment_label,
            "most_active_hour": overview.peak_hour,
            "most_active_day": overview.peak_day,
        }

        return AnalyticsReport(
            title=title,
            description="Comprehensive analytics report for your conversations",
            generated_at=datetime.now(UTC),
            date_range_start=overview.date_range_start,
            date_range_end=overview.date_range_end,
            sections=sections,
            summary=summary,
            metadata={
                "messages_analyzed": len(messages),
                "report_version": "1.0",
            },
        )

    def generate_contact_report(
        self,
        messages: list[Message],
        contact_id: str,
        contact_name: str | None = None,
    ) -> AnalyticsReport:
        """Generate a report for a specific contact.

        Args:
            messages: Messages with this contact
            contact_id: Contact identifier
            contact_name: Display name

        Returns:
            AnalyticsReport with contact-specific analytics
        """
        if not messages:
            return AnalyticsReport(
                title=f"Analytics: {contact_name or contact_id}",
                description="No messages to analyze",
                generated_at=datetime.now(UTC),
                date_range_start=None,
                date_range_end=None,
            )

        contact_analytics = self._engine.compute_contact_analytics(
            messages, contact_id, contact_name
        )
        emoji_stats = self._engine.compute_emoji_stats(messages)  # noqa: F841
        hourly, daily, weekly, monthly = self._engine.compute_time_distributions(messages)

        sections = [
            ReportSection(
                title="Contact Overview",
                description=f"Communication statistics with {contact_name or contact_id}",
                data={
                    "total_messages": contact_analytics.total_messages,
                    "sent": contact_analytics.sent_count,
                    "received": contact_analytics.received_count,
                    "avg_response_time": contact_analytics.avg_response_time_minutes,
                    "engagement_score": contact_analytics.engagement_score,
                    "message_trend": contact_analytics.message_trend,
                },
            ),
            ReportSection(
                title="Sentiment",
                description="Communication tone",
                data={
                    "sentiment_score": contact_analytics.sentiment_score,
                },
                chart_type="gauge",
            ),
            ReportSection(
                title="Activity Timeline",
                description="Communication frequency over time",
                data={
                    "weekly_counts": weekly,
                    "monthly_counts": monthly,
                },
                chart_type="line",
            ),
            ReportSection(
                title="Communication Patterns",
                description="When you typically communicate",
                data={
                    "hourly_distribution": hourly,
                    "daily_distribution": daily,
                },
                chart_type="heatmap",
            ),
        ]

        return AnalyticsReport(
            title=f"Analytics: {contact_name or contact_id}",
            description=f"Detailed analytics for conversations with {contact_name or contact_id}",
            generated_at=datetime.now(UTC),
            date_range_start=contact_analytics.last_message_date,
            date_range_end=datetime.now(UTC),
            sections=sections,
            summary={
                "total_messages": contact_analytics.total_messages,
                "engagement": contact_analytics.engagement_score,
                "trend": contact_analytics.message_trend,
            },
        )

    def generate_comparison_report(
        self,
        messages: list[Message],
        current_days: int = 7,
        previous_days: int = 7,
    ) -> AnalyticsReport:
        """Generate a period comparison report.

        Args:
            messages: Messages to analyze
            current_days: Days in current period
            previous_days: Days in previous period

        Returns:
            AnalyticsReport comparing two time periods
        """
        comparison = self._trend_analyzer.compare_periods(messages, current_days, previous_days)

        sections = [
            ReportSection(
                title="Period Comparison",
                description=f"Comparing last {current_days} days vs previous {previous_days} days",
                data={
                    "total_change": comparison.get("total_messages_change", 0),
                    "sent_change": comparison.get("sent_messages_change", 0),
                    "contacts_change": comparison.get("active_contacts_change", 0),
                    "current_total": comparison.get("current_period_total", 0),
                    "previous_total": comparison.get("previous_period_total", 0),
                },
                chart_type="comparison",
            )
        ]

        return AnalyticsReport(
            title="Period Comparison Report",
            description="Comparing messaging activity between periods",
            generated_at=datetime.now(UTC),
            date_range_start=None,
            date_range_end=datetime.now(UTC),
            sections=sections,
            summary=comparison,
        )

    def export_to_csv(
        self,
        messages: list[Message],
        include_daily: bool = True,
        include_weekly: bool = True,
        include_monthly: bool = True,
    ) -> dict[str, str]:
        """Export analytics data to CSV format.

        Args:
            messages: Messages to analyze
            include_daily: Include daily aggregates
            include_weekly: Include weekly aggregates
            include_monthly: Include monthly aggregates

        Returns:
            Dict mapping filename to CSV content
        """
        exports: dict[str, str] = {}

        if include_daily:
            daily_aggs = aggregate_by_day(messages)
            buffer = io.StringIO()
            writer = csv.DictWriter(
                buffer,
                fieldnames=[
                    "date",
                    "total_messages",
                    "sent",
                    "received",
                    "unique_contacts",
                ],
            )
            writer.writeheader()
            for agg in daily_aggs:
                writer.writerow(
                    {
                        "date": agg.date,
                        "total_messages": agg.total_messages,
                        "sent": agg.sent_count,
                        "received": agg.received_count,
                        "unique_contacts": agg.unique_contacts,
                    }
                )
            exports["daily_analytics.csv"] = buffer.getvalue()

        if include_weekly:
            weekly_aggs = aggregate_by_week(messages)
            buffer = io.StringIO()
            writer = csv.DictWriter(
                buffer,
                fieldnames=[
                    "week",
                    "start_date",
                    "end_date",
                    "total_messages",
                    "sent",
                    "received",
                    "active_contacts",
                ],
            )
            writer.writeheader()
            for wagg in weekly_aggs:
                writer.writerow(
                    {
                        "week": wagg.week,
                        "start_date": wagg.start_date,
                        "end_date": wagg.end_date,
                        "total_messages": wagg.total_messages,
                        "sent": wagg.sent_count,
                        "received": wagg.received_count,
                        "active_contacts": wagg.active_contacts,
                    }
                )
            exports["weekly_analytics.csv"] = buffer.getvalue()

        if include_monthly:
            monthly_aggs = aggregate_by_month(messages)
            buffer = io.StringIO()
            writer = csv.DictWriter(
                buffer,
                fieldnames=[
                    "month",
                    "total_messages",
                    "sent",
                    "received",
                    "active_contacts",
                    "avg_per_day",
                ],
            )
            writer.writeheader()
            for magg in monthly_aggs:
                writer.writerow(
                    {
                        "month": magg.month,
                        "total_messages": magg.total_messages,
                        "sent": magg.sent_count,
                        "received": magg.received_count,
                        "active_contacts": magg.active_contacts,
                        "avg_per_day": magg.avg_messages_per_day,
                    }
                )
            exports["monthly_analytics.csv"] = buffer.getvalue()

        return exports

    def export_to_json(
        self,
        messages: list[Message],
        contact_messages: dict[str, list[Message]] | None = None,
    ) -> str:
        """Export full analytics to JSON.

        Args:
            messages: All messages
            contact_messages: Optional grouped messages by contact

        Returns:
            JSON string with complete analytics
        """
        result = self._engine.compute_full_analytics(messages, contact_messages)

        # Convert dataclasses to dicts for JSON serialization
        export_data = {
            "overview": {
                "total_messages": result.overview.total_messages,
                "total_sent": result.overview.total_sent,
                "total_received": result.overview.total_received,
                "active_conversations": result.overview.active_conversations,
                "avg_messages_per_day": result.overview.avg_messages_per_day,
                "avg_response_time_minutes": result.overview.avg_response_time_minutes,
                "sentiment_score": result.overview.sentiment_score,
                "sentiment_label": result.overview.sentiment_label,
                "peak_hour": result.overview.peak_hour,
                "peak_day": result.overview.peak_day,
            },
            "contacts": [
                {
                    "contact_id": c.contact_id,
                    "contact_name": c.contact_name,
                    "total_messages": c.total_messages,
                    "sent_count": c.sent_count,
                    "received_count": c.received_count,
                    "avg_response_time": c.avg_response_time_minutes,
                    "sentiment_score": c.sentiment_score,
                    "message_trend": c.message_trend,
                    "engagement_score": c.engagement_score,
                }
                for c in result.contacts
            ],
            "emoji_stats": {
                "total_count": result.emoji_stats.total_count,
                "unique_count": result.emoji_stats.unique_count,
                "top_emojis": result.emoji_stats.top_emojis,
                "emojis_per_message": result.emoji_stats.emojis_per_message,
            },
            "message_length": {
                "avg_length": result.message_length_stats.avg_length,
                "median_length": result.message_length_stats.median_length,
                "distribution": {
                    "short": result.message_length_stats.short_count,
                    "medium": result.message_length_stats.medium_count,
                    "long": result.message_length_stats.long_count,
                    "very_long": result.message_length_stats.very_long_count,
                },
            },
            "time_distributions": {
                "hourly": result.hourly_distribution,
                "daily": result.daily_distribution,
                "weekly": result.weekly_counts,
                "monthly": result.monthly_counts,
            },
            "sentiment_trends": result.sentiment_trends,
            "computed_at": result.computed_at.isoformat(),
        }

        return json.dumps(export_data, indent=2, default=str)


# Global report generator instance
_report_generator: ReportGenerator | None = None
_report_generator_lock = threading.Lock()


def get_report_generator() -> ReportGenerator:
    """Get the global report generator instance."""
    global _report_generator
    if _report_generator is None:
        with _report_generator_lock:
            if _report_generator is None:
                _report_generator = ReportGenerator()
    return _report_generator
