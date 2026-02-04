"""Quality dashboard for tracking metrics over time.

Provides quality metrics tracking, per-model comparison,
regression detection, and alerting on quality degradation.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from statistics import mean
from typing import Any

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Severity levels for quality alerts."""

    INFO = "info"  # Informational
    WARNING = "warning"  # Quality declining
    CRITICAL = "critical"  # Quality below acceptable threshold


class AlertType(str, Enum):
    """Types of quality alerts."""

    REGRESSION = "regression"  # Quality score regression
    THRESHOLD_BREACH = "threshold_breach"  # Score below threshold
    TREND_NEGATIVE = "trend_negative"  # Sustained negative trend
    MODEL_DEGRADATION = "model_degradation"  # Specific model degraded
    LATENCY_INCREASE = "latency_increase"  # Quality check latency increased


@dataclass
class QualityAlert:
    """A quality alert notification."""

    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metric_name: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    model_name: str | None = None
    acknowledged: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "current_value": round(self.current_value, 4),
            "threshold_value": round(self.threshold_value, 4),
            "model_name": self.model_name,
            "acknowledged": self.acknowledged,
        }


@dataclass
class QualityMetricPoint:
    """A single quality metric data point."""

    timestamp: datetime
    value: float
    model_name: str | None = None
    dimension: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityTrend:
    """Quality trend analysis."""

    dimension: str
    current_value: float
    previous_value: float
    change_percent: float
    trend_direction: str  # "up", "down", "stable"
    is_improving: bool
    period_days: int = 7
    data_points: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension,
            "current_value": round(self.current_value, 4),
            "previous_value": round(self.previous_value, 4),
            "change_percent": round(self.change_percent, 2),
            "trend_direction": self.trend_direction,
            "is_improving": self.is_improving,
            "period_days": self.period_days,
            "data_points": self.data_points,
        }


@dataclass
class ModelQualityComparison:
    """Quality comparison between models."""

    model_name: str
    overall_score: float
    dimension_scores: dict[str, float]
    sample_count: int
    avg_latency_ms: float
    ranking: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "overall_score": round(self.overall_score, 4),
            "dimension_scores": {k: round(v, 4) for k, v in self.dimension_scores.items()},
            "sample_count": self.sample_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "ranking": self.ranking,
        }


class QualityDashboard:
    """Dashboard for tracking and analyzing quality metrics.

    Features:
    - Time-series tracking of quality metrics
    - Per-model quality comparison
    - Regression detection
    - Alert generation for quality degradation
    """

    # Configuration
    MAX_DATA_POINTS = 10000
    MAX_ALERTS = 100
    REGRESSION_THRESHOLD = 0.1  # 10% drop triggers alert
    TREND_WINDOW_DAYS = 7
    CRITICAL_THRESHOLD = 0.4  # Below this is critical

    # Metric dimensions
    DIMENSIONS = [
        "hallucination",
        "factuality",
        "consistency",
        "grounding",
        "coherence",
        "relevance",
        "overall",
    ]

    def __init__(self) -> None:
        """Initialize the quality dashboard."""
        self._lock = threading.Lock()
        self._start_time = datetime.now(UTC)

        # Time-series data storage (per dimension)
        self._metrics: dict[str, deque[QualityMetricPoint]] = {
            dim: deque(maxlen=self.MAX_DATA_POINTS) for dim in self.DIMENSIONS
        }

        # Per-model metrics
        self._model_metrics: dict[str, dict[str, deque[QualityMetricPoint]]] = {}

        # Alerts
        self._alerts: deque[QualityAlert] = deque(maxlen=self.MAX_ALERTS)

        # Running aggregates
        self._dimension_totals: dict[str, float] = {dim: 0.0 for dim in self.DIMENSIONS}
        self._dimension_counts: dict[str, int] = {dim: 0 for dim in self.DIMENSIONS}

    def record_quality_check(
        self,
        dimension_scores: dict[str, float],
        overall_score: float,
        model_name: str | None = None,
        latency_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a quality check result.

        Args:
            dimension_scores: Scores for each dimension
            overall_score: Overall quality score
            model_name: Optional model name for per-model tracking
            latency_ms: Quality check latency
            metadata: Optional additional metadata
        """
        timestamp = datetime.now(UTC)
        metadata = metadata or {}

        with self._lock:
            # Record overall score
            point = QualityMetricPoint(
                timestamp=timestamp,
                value=overall_score,
                model_name=model_name,
                dimension="overall",
                latency_ms=latency_ms,
                metadata=metadata,
            )
            self._metrics["overall"].append(point)
            self._dimension_totals["overall"] += overall_score
            self._dimension_counts["overall"] += 1

            # Record dimension scores
            for dim, score in dimension_scores.items():
                if dim in self._metrics:
                    dim_point = QualityMetricPoint(
                        timestamp=timestamp,
                        value=score,
                        model_name=model_name,
                        dimension=dim,
                        latency_ms=latency_ms,
                    )
                    self._metrics[dim].append(dim_point)
                    self._dimension_totals[dim] += score
                    self._dimension_counts[dim] += 1

            # Record per-model metrics
            if model_name:
                if model_name not in self._model_metrics:
                    self._model_metrics[model_name] = {
                        dim: deque(maxlen=1000) for dim in self.DIMENSIONS
                    }

                self._model_metrics[model_name]["overall"].append(point)
                for dim, score in dimension_scores.items():
                    if dim in self._model_metrics[model_name]:
                        dim_point = QualityMetricPoint(
                            timestamp=timestamp,
                            value=score,
                            model_name=model_name,
                            dimension=dim,
                        )
                        self._model_metrics[model_name][dim].append(dim_point)

            # Check for alerts
            self._check_alerts(overall_score, dimension_scores, model_name)

    def get_summary(self) -> dict[str, Any]:
        """Get current quality summary.

        Returns:
            Summary of current quality metrics
        """
        with self._lock:
            elapsed = (datetime.now(UTC) - self._start_time).total_seconds()

            summary = {
                "uptime_seconds": round(elapsed, 2),
                "total_checks": self._dimension_counts["overall"],
                "dimensions": {},
                "recent_alerts": len([a for a in self._alerts if not a.acknowledged]),
            }

            for dim in self.DIMENSIONS:
                count = self._dimension_counts[dim]
                if count > 0:
                    avg = self._dimension_totals[dim] / count
                    summary["dimensions"][dim] = {
                        "average": round(avg, 4),
                        "count": count,
                    }

                    # Add recent average (last 100 points)
                    recent = list(self._metrics[dim])[-100:]
                    if recent:
                        recent_avg = mean(p.value for p in recent)
                        summary["dimensions"][dim]["recent_average"] = round(recent_avg, 4)

            return summary

    def get_trends(self, days: int = 7) -> list[QualityTrend]:
        """Get quality trends over time.

        Args:
            days: Number of days to analyze

        Returns:
            List of QualityTrend objects for each dimension
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)
        half_cutoff = datetime.now(UTC) - timedelta(days=days // 2)
        trends: list[QualityTrend] = []

        with self._lock:
            for dim in self.DIMENSIONS:
                points = [p for p in self._metrics[dim] if p.timestamp >= cutoff]

                if len(points) < 2:
                    continue

                # Split into two periods
                first_half = [p for p in points if p.timestamp < half_cutoff]
                second_half = [p for p in points if p.timestamp >= half_cutoff]

                if not first_half or not second_half:
                    continue

                prev_avg = mean(p.value for p in first_half)
                curr_avg = mean(p.value for p in second_half)

                if prev_avg == 0:
                    change_pct = 0.0
                else:
                    change_pct = ((curr_avg - prev_avg) / prev_avg) * 100

                if change_pct > 5:
                    direction = "up"
                elif change_pct < -5:
                    direction = "down"
                else:
                    direction = "stable"

                # For most dimensions, higher is better (except hallucination)
                if dim != "hallucination":
                    is_improving = direction == "up"
                else:
                    is_improving = direction == "down"

                trends.append(
                    QualityTrend(
                        dimension=dim,
                        current_value=curr_avg,
                        previous_value=prev_avg,
                        change_percent=change_pct,
                        trend_direction=direction,
                        is_improving=is_improving,
                        period_days=days,
                        data_points=len(points),
                    )
                )

        return trends

    def get_model_comparison(self) -> list[ModelQualityComparison]:
        """Get quality comparison across models.

        Returns:
            List of ModelQualityComparison objects ranked by overall score
        """
        comparisons: list[ModelQualityComparison] = []

        with self._lock:
            for model_name, metrics in self._model_metrics.items():
                overall_points = list(metrics["overall"])

                if not overall_points:
                    continue

                overall_score = mean(p.value for p in overall_points)
                latencies = [p.latency_ms for p in overall_points if p.latency_ms > 0]
                avg_latency = mean(latencies) if latencies else 0.0

                dimension_scores = {}
                for dim in self.DIMENSIONS:
                    if dim == "overall":
                        continue
                    dim_points = list(metrics.get(dim, []))
                    if dim_points:
                        dimension_scores[dim] = mean(p.value for p in dim_points)

                comparisons.append(
                    ModelQualityComparison(
                        model_name=model_name,
                        overall_score=overall_score,
                        dimension_scores=dimension_scores,
                        sample_count=len(overall_points),
                        avg_latency_ms=avg_latency if avg_latency else 0.0,
                    )
                )

        # Sort by overall score (descending) and assign rankings
        comparisons.sort(key=lambda c: c.overall_score, reverse=True)
        for i, comp in enumerate(comparisons):
            comp.ranking = i + 1

        return comparisons

    def get_alerts(
        self,
        include_acknowledged: bool = False,
        severity: AlertSeverity | None = None,
    ) -> list[QualityAlert]:
        """Get quality alerts.

        Args:
            include_acknowledged: Include acknowledged alerts
            severity: Filter by severity level

        Returns:
            List of QualityAlert objects
        """
        with self._lock:
            alerts = list(self._alerts)

        if not include_acknowledged:
            alerts = [a for a in alerts if not a.acknowledged]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        # Sort by timestamp (most recent first)
        alerts.sort(key=lambda a: a.timestamp, reverse=True)

        return alerts

    def acknowledge_alert(self, index: int) -> bool:
        """Acknowledge an alert by index.

        Args:
            index: Alert index

        Returns:
            True if acknowledged successfully
        """
        with self._lock:
            alerts = list(self._alerts)
            if 0 <= index < len(alerts):
                alerts[index].acknowledged = True
                return True
        return False

    def detect_regression(
        self,
        dimension: str = "overall",
        window_size: int = 100,
    ) -> tuple[bool, float]:
        """Detect quality regression.

        Args:
            dimension: Dimension to check
            window_size: Number of points to compare

        Returns:
            Tuple of (regression_detected, change_percentage)
        """
        with self._lock:
            points = list(self._metrics.get(dimension, []))

        if len(points) < window_size * 2:
            return False, 0.0

        # Compare recent window to previous window
        recent = points[-window_size:]
        previous = points[-window_size * 2 : -window_size]

        recent_avg = mean(p.value for p in recent)
        previous_avg = mean(p.value for p in previous)

        if previous_avg == 0:
            return False, 0.0

        change_pct = ((recent_avg - previous_avg) / previous_avg) * 100

        # For most dimensions, negative change is regression
        # For hallucination, positive change is regression
        if dimension == "hallucination":
            regression = change_pct > self.REGRESSION_THRESHOLD * 100
        else:
            regression = change_pct < -self.REGRESSION_THRESHOLD * 100

        return regression, change_pct

    def get_time_series(
        self,
        dimension: str = "overall",
        hours: int = 24,
        resolution_minutes: int = 60,
    ) -> list[dict[str, Any]]:
        """Get time series data for a dimension.

        Args:
            dimension: Dimension to get
            hours: Number of hours of history
            resolution_minutes: Time bucket size in minutes

        Returns:
            List of time-series data points
        """
        cutoff = datetime.now(UTC) - timedelta(hours=hours)

        with self._lock:
            points = [p for p in self._metrics.get(dimension, []) if p.timestamp >= cutoff]

        if not points:
            return []

        # Bucket by time
        buckets: dict[datetime, list[float]] = {}
        for point in points:
            # Round to resolution
            bucket_time = point.timestamp.replace(
                minute=(point.timestamp.minute // resolution_minutes) * resolution_minutes,
                second=0,
                microsecond=0,
            )
            if bucket_time not in buckets:
                buckets[bucket_time] = []
            buckets[bucket_time].append(point.value)

        # Build time series
        series = []
        for bucket_time in sorted(buckets.keys()):
            values = buckets[bucket_time]
            series.append(
                {
                    "timestamp": bucket_time.isoformat(),
                    "value": round(mean(values), 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "count": len(values),
                }
            )

        return series

    def _check_alerts(
        self,
        overall_score: float,
        dimension_scores: dict[str, float],
        model_name: str | None,
    ) -> None:
        """Check for alert conditions (called with lock held)."""
        # Check overall threshold
        if overall_score < self.CRITICAL_THRESHOLD:
            alert = QualityAlert(
                alert_type=AlertType.THRESHOLD_BREACH,
                severity=AlertSeverity.CRITICAL,
                message=f"Overall quality score {overall_score:.2f} below critical threshold",
                metric_name="overall",
                current_value=overall_score,
                threshold_value=self.CRITICAL_THRESHOLD,
                model_name=model_name,
            )
            self._alerts.append(alert)
            logger.warning("Quality alert: %s", alert.message)

        # Check for regression (need enough data)
        if self._dimension_counts["overall"] >= 200:
            points = list(self._metrics["overall"])[-100:]
            recent_avg = mean(p.value for p in points)

            all_avg = self._dimension_totals["overall"] / self._dimension_counts["overall"]

            if all_avg > 0:
                change = (recent_avg - all_avg) / all_avg
                if change < -self.REGRESSION_THRESHOLD:
                    alert = QualityAlert(
                        alert_type=AlertType.REGRESSION,
                        severity=AlertSeverity.WARNING,
                        message=f"Quality regression detected: {change * 100:.1f}% decline",
                        metric_name="overall",
                        current_value=recent_avg,
                        threshold_value=all_avg,
                        model_name=model_name,
                    )
                    # Only add if not duplicate recent alert
                    recent_alerts = [
                        a
                        for a in self._alerts
                        if a.alert_type == AlertType.REGRESSION
                        and (datetime.now(UTC) - a.timestamp).seconds < 3600
                    ]
                    if not recent_alerts:
                        self._alerts.append(alert)
                        logger.warning("Quality alert: %s", alert.message)

    def reset(self) -> None:
        """Reset all dashboard data."""
        with self._lock:
            self._start_time = datetime.now(UTC)
            for dim in self.DIMENSIONS:
                self._metrics[dim].clear()
                self._dimension_totals[dim] = 0.0
                self._dimension_counts[dim] = 0
            self._model_metrics.clear()
            self._alerts.clear()


# Global singleton
_dashboard: QualityDashboard | None = None
_dashboard_lock = threading.Lock()


def get_quality_dashboard() -> QualityDashboard:
    """Get the global quality dashboard instance.

    Returns:
        Shared QualityDashboard instance
    """
    global _dashboard
    if _dashboard is None:
        with _dashboard_lock:
            if _dashboard is None:
                _dashboard = QualityDashboard()
    return _dashboard


def reset_quality_dashboard() -> None:
    """Reset the global quality dashboard instance."""
    global _dashboard
    with _dashboard_lock:
        _dashboard = None
