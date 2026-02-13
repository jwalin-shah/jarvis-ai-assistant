"""Latency tracking for performance monitoring.

Tracks operation timings and flags suspicious patterns (N+1, slow queries).
Integrates with metrics system for dashboard reporting.

Budget tiers define SLO targets:
- INSTANT: <100ms (DB queries, cache lookups)
- FAST: <500ms (search, embeddings)
- ASYNC: <5s (LLM generation)
- BACKGROUND: no limit (model loads, prefetch)
"""

import functools
import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class BudgetTier(Enum):
    """Performance budget tiers with target latencies (ms)."""

    INSTANT = 100
    FAST = 500
    ASYNC = 5000
    BACKGROUND = 0  # No limit


# Operation -> (tier, budget_ms) mapping
OPERATION_BUDGETS: dict[str, tuple[BudgetTier, int]] = {
    # Database operations (INSTANT)
    "conversations_fetch": (BudgetTier.INSTANT, 100),
    "message_load": (BudgetTier.INSTANT, 100),
    "fact_save": (BudgetTier.INSTANT, 50),
    "db_query": (BudgetTier.INSTANT, 200),
    "graph_build": (BudgetTier.INSTANT, 100),
    # Search/embedding operations (FAST)
    "search_filter": (BudgetTier.FAST, 100),
    "semantic_search": (BudgetTier.FAST, 500),
    "embedding_encode": (BudgetTier.FAST, 500),
    "classify_intent": (BudgetTier.FAST, 200),
    "smart_replies": (BudgetTier.FAST, 500),
    # LLM generation (ASYNC)
    "generate_draft": (BudgetTier.ASYNC, 5000),
    "summarize": (BudgetTier.ASYNC, 5000),
    "rpc.generate_draft": (BudgetTier.ASYNC, 5000),
    "rpc.summarize": (BudgetTier.ASYNC, 5000),
    # Infrastructure (various)
    "socket_startup": (BudgetTier.FAST, 500),
    "rpc.ping": (BudgetTier.INSTANT, 100),
    "rpc.get_conversations": (BudgetTier.INSTANT, 100),
    "rpc.get_messages": (BudgetTier.INSTANT, 100),
    "rpc.get_health": (BudgetTier.INSTANT, 100),
    "rpc.classify_intent": (BudgetTier.FAST, 200),
    "rpc.semantic_search": (BudgetTier.FAST, 500),
    "rpc.get_smart_replies": (BudgetTier.FAST, 500),
    # Background (no limit)
    "model_load": (BudgetTier.BACKGROUND, 0),
    "prefetch": (BudgetTier.BACKGROUND, 0),
}

# Backward compat: derive LATENCY_THRESHOLDS from OPERATION_BUDGETS
LATENCY_THRESHOLDS = {
    op: budget_ms for op, (_, budget_ms) in OPERATION_BUDGETS.items() if budget_ms > 0
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

    def get_slo_compliance(self, operation: str | None = None) -> dict:
        """Get SLO compliance stats.

        Args:
            operation: Filter to specific operation, or None for all budgeted ops.

        Returns:
            Dict with total, compliant, compliance_pct, p95_ms.
        """
        if operation:
            records = [r for r in self._records if r.operation == operation]
        else:
            # Only include records that have a threshold (budgeted operations)
            records = [r for r in self._records if r.threshold_ms is not None]

        if not records:
            return {"total": 0, "compliant": 0, "compliance_pct": 100.0, "p95_ms": 0.0}

        compliant = sum(1 for r in records if not r.exceeded)
        sorted_ms = sorted(r.elapsed_ms for r in records)
        p95_idx = min(int(len(sorted_ms) * 0.95), len(sorted_ms) - 1)

        return {
            "total": len(records),
            "compliant": compliant,
            "compliance_pct": round(compliant / len(records) * 100, 2),
            "p95_ms": round(sorted_ms[p95_idx], 2),
        }

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


def perf_budget(budget: BudgetTier | int):
    """Decorator that tracks function execution against a performance budget.

    Usage:
        @perf_budget(BudgetTier.INSTANT)
        def get_conversations(limit):
            ...

        @perf_budget(200)  # explicit ms budget
        def custom_query():
            ...
    """
    if isinstance(budget, BudgetTier):
        threshold_ms = budget.value if budget.value > 0 else None
    else:
        threshold_ms = budget if budget > 0 else None

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with _tracker.track(func.__qualname__, threshold_ms=threshold_ms):
                return func(*args, **kwargs)

        return wrapper

    return decorator
