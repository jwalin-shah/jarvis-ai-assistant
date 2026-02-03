"""Analytics module for conversation insights.

Provides comprehensive analytics including:
- Time-series aggregation
- Trend detection
- Report generation
- Pre-computed daily aggregates
"""

from __future__ import annotations

from jarvis.analytics.aggregator import (
    DailyAggregate,
    TimeSeriesAggregator,
    aggregate_by_day,
    aggregate_by_hour,
    aggregate_by_month,
    aggregate_by_week,
)
from jarvis.analytics.engine import (
    AnalyticsEngine,
    AnalyticsResult,
    ContactAnalytics,
    OverviewMetrics,
)
from jarvis.analytics.reports import (
    AnalyticsReport,
    ReportGenerator,
    ReportSection,
)
from jarvis.analytics.trends import (
    TrendAnalyzer,
    TrendResult,
    detect_anomalies,
    detect_peak_periods,
    detect_trend,
)

__all__ = [
    # Engine
    "AnalyticsEngine",
    "AnalyticsResult",
    "ContactAnalytics",
    "OverviewMetrics",
    # Aggregator
    "DailyAggregate",
    "TimeSeriesAggregator",
    "aggregate_by_day",
    "aggregate_by_hour",
    "aggregate_by_month",
    "aggregate_by_week",
    # Trends
    "TrendAnalyzer",
    "TrendResult",
    "detect_anomalies",
    "detect_peak_periods",
    "detect_trend",
    # Reports
    "AnalyticsReport",
    "ReportGenerator",
    "ReportSection",
]
