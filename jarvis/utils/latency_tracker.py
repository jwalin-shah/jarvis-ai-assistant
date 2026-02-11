"""Latency tracking for performance monitoring.

Tracks operation timings and flags suspicious patterns (N+1, slow queries).
Integrates with metrics system for dashboard reporting.
"""

import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

LATENCY_THRESHOLDS = {
    "conversations_fetch": 100,  # Should be <100ms (was 1400ms)
    "message_load": 100,  # Should be <100ms (was 500ms with N+1)
    "fact_save": 50,  # Should be <50ms (was 150ms with N+1)
    "search_filter": 100,  # Should be <100ms (was wasted 1000 msg load)
    "graph_build": 100,  # Should be <100ms (was 200ms with loops)
    "socket_startup": 500,  # Socket should be ready in <500ms
    "db_query": 200,  # Single query should be <200ms
}


@dataclass
class LatencyRecord:
    """Single operation latency measurement."""

    operation: str
    elapsed_ms: float
    timestamp: float
    threshold_ms: float | None = None
    exceeded: bool = False
    metadata: dict = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            **asdict(self),
            "metadata": self.metadata or {},
        }


class LatencyTracker:
    """Track operation latencies and detect performance regressions."""

    def __init__(self):
        self._records: deque[LatencyRecord] = deque(maxlen=10000)
        self._thresholds = LATENCY_THRESHOLDS.copy()

    @contextmanager
    def track(self, operation: str, threshold_ms: float | None = None, **metadata):
        """Context manager for tracking operation latency.

        Usage:
            with tracker.track("conversations_fetch", message_count=50):
                conversations = get_conversations(50)
        """
        threshold = threshold_ms or self._thresholds.get(operation)
        start = time.perf_counter()

        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            exceeded = threshold and elapsed_ms > threshold

            record = LatencyRecord(
                operation=operation,
                elapsed_ms=elapsed_ms,
                timestamp=time.time(),
                threshold_ms=threshold,
                exceeded=exceeded,
                metadata=metadata,
            )
            self._records.append(record)

            # Log warnings for exceeded thresholds
            if exceeded:
                logger.warning(
                    f"[LATENCY] {operation} took {elapsed_ms:.1f}ms "
                    f"(threshold: {threshold}ms) - possible N+1 pattern. "
                    f"Metadata: {metadata}"
                )
            else:
                logger.debug(f"[LATENCY] {operation} took {elapsed_ms:.1f}ms (ok)")

    def get_records(self) -> list[LatencyRecord]:
        """Get all recorded latencies."""
        return self._records

    def get_slow_operations(self) -> list[LatencyRecord]:
        """Get operations that exceeded thresholds."""
        return [r for r in self._records if r.exceeded]

    def summary(self) -> dict:
        """Get summary statistics."""
        if not self._records:
            return {}

        slow = self.get_slow_operations()
        return {
            "total_operations": len(self._records),
            "slow_operations": len(slow),
            "slow_ops_pct": (len(slow) / len(self._records) * 100) if self._records else 0,
            "average_ms": sum(r.elapsed_ms for r in self._records) / len(self._records),
            "slow_operations_detail": [asdict(r) for r in slow[:10]],  # Top 10
        }


# Global tracker instance
_tracker = LatencyTracker()


def get_tracker() -> LatencyTracker:
    """Get global latency tracker."""
    return _tracker


@contextmanager
def track_latency(operation: str, **metadata):
    """Convenience function for tracking latency.

    Usage:
        with track_latency("query_execute", query_type="SELECT"):
            result = db.execute(...)
    """
    with _tracker.track(operation, **metadata):
        yield
