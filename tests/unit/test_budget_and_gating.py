"""Tests for performance budget system and draft confidence gating."""

from __future__ import annotations

import time


class TestBudgetTiers:
    """Verify budget tier definitions are correct."""

    def test_budget_tier_values(self) -> None:
        from jarvis.utils.latency_tracker import BudgetTier

        assert BudgetTier.INSTANT.value == 100
        assert BudgetTier.FAST.value == 500
        assert BudgetTier.ASYNC.value == 5000
        assert BudgetTier.BACKGROUND.value == 0

    def test_operation_budgets_cover_key_operations(self) -> None:
        from jarvis.utils.latency_tracker import OPERATION_BUDGETS

        required_ops = [
            "conversations_fetch",
            "message_load",
            "db_query",
            "semantic_search",
            "generate_draft",
        ]
        for op in required_ops:
            assert op in OPERATION_BUDGETS, f"Missing budget for {op}"

    def test_rpc_budgets_exist(self) -> None:
        from jarvis.utils.latency_tracker import OPERATION_BUDGETS

        rpc_ops = [k for k in OPERATION_BUDGETS if k.startswith("rpc.")]
        assert len(rpc_ops) >= 5, f"Expected >=5 RPC budgets, got {len(rpc_ops)}"

    def test_latency_thresholds_backward_compat(self) -> None:
        from jarvis.utils.latency_tracker import LATENCY_THRESHOLDS

        assert "conversations_fetch" in LATENCY_THRESHOLDS
        assert LATENCY_THRESHOLDS["conversations_fetch"] == 100
        assert "model_load" not in LATENCY_THRESHOLDS
        assert "prefetch" not in LATENCY_THRESHOLDS


class TestSLOCompliance:
    """Test SLO compliance tracking."""

    def test_empty_tracker_compliance(self) -> None:
        from jarvis.utils.latency_tracker import LatencyTracker

        tracker = LatencyTracker()
        result = tracker.get_slo_compliance()
        assert result["total"] == 0
        assert result["compliance_pct"] == 100.0

    def test_all_compliant(self) -> None:
        from jarvis.utils.latency_tracker import LatencyTracker

        tracker = LatencyTracker()
        for _ in range(10):
            with tracker.track("conversations_fetch"):
                pass
        result = tracker.get_slo_compliance("conversations_fetch")
        assert result["total"] == 10
        assert result["compliant"] == 10
        assert result["compliance_pct"] == 100.0

    def test_mixed_compliance(self) -> None:
        from jarvis.utils.latency_tracker import LatencyRecord, LatencyTracker

        tracker = LatencyTracker()
        with tracker.track("db_query"):
            pass
        tracker._records.append(
            LatencyRecord(
                operation="db_query",
                elapsed_ms=500.0,
                timestamp=time.time(),
                threshold_ms=200,
                exceeded=True,
            )
        )
        result = tracker.get_slo_compliance("db_query")
        assert result["total"] == 2
        assert result["compliant"] == 1
        assert result["compliance_pct"] == 50.0

    def test_p95_computation(self) -> None:
        from jarvis.utils.latency_tracker import LatencyRecord, LatencyTracker

        tracker = LatencyTracker()
        for _ in range(95):
            tracker._records.append(
                LatencyRecord(
                    operation="db_query",
                    elapsed_ms=10.0,
                    timestamp=time.time(),
                    threshold_ms=200,
                    exceeded=False,
                )
            )
        for _ in range(5):
            tracker._records.append(
                LatencyRecord(
                    operation="db_query",
                    elapsed_ms=300.0,
                    timestamp=time.time(),
                    threshold_ms=200,
                    exceeded=True,
                )
            )
        result = tracker.get_slo_compliance("db_query")
        assert result["p95_ms"] >= 10.0

    def test_operation_filter(self) -> None:
        from jarvis.utils.latency_tracker import LatencyTracker

        tracker = LatencyTracker()
        with tracker.track("conversations_fetch"):
            pass
        with tracker.track("db_query"):
            pass
        result = tracker.get_slo_compliance("conversations_fetch")
        assert result["total"] == 1


class TestPerfBudgetDecorator:
    """Test the @perf_budget decorator."""

    def test_decorator_with_tier(self) -> None:
        from jarvis.utils.latency_tracker import BudgetTier, perf_budget

        @perf_budget(BudgetTier.INSTANT)
        def fast_func():
            return 42

        assert fast_func() == 42

    def test_decorator_with_ms(self) -> None:
        from jarvis.utils.latency_tracker import perf_budget

        @perf_budget(200)
        def custom_func():
            return "ok"

        assert custom_func() == "ok"

    def test_decorator_records_latency(self) -> None:
        from jarvis.utils.latency_tracker import BudgetTier, get_tracker, perf_budget

        tracker = get_tracker()
        initial_count = len(tracker._records)

        @perf_budget(BudgetTier.FAST)
        def tracked_func():
            return True

        tracked_func()
        assert len(tracker._records) > initial_count


class TestDraftMetrics:
    """Test draft confidence gating metrics."""

    def test_record_shown(self) -> None:
        from jarvis.metrics import DraftMetrics

        metrics = DraftMetrics()
        metrics.record(0.8, gated=False)
        stats = metrics.get_stats()
        assert stats["total_requests"] == 1
        assert stats["shown_count"] == 1
        assert stats["gated_count"] == 0

    def test_record_gated(self) -> None:
        from jarvis.metrics import DraftMetrics

        metrics = DraftMetrics()
        metrics.record(0.3, gated=True)
        stats = metrics.get_stats()
        assert stats["total_requests"] == 1
        assert stats["gated_count"] == 1
        assert stats["gate_rate_pct"] == 100.0

    def test_confidence_histogram(self) -> None:
        from jarvis.metrics import DraftMetrics

        metrics = DraftMetrics()
        metrics.record(0.85, gated=False)
        metrics.record(0.35, gated=True)
        metrics.record(0.55, gated=False)
        stats = metrics.get_stats()
        histogram = stats["confidence_histogram"]
        assert histogram["0.8-0.9"] == 1
        assert histogram["0.3-0.4"] == 1
        assert histogram["0.5-0.6"] == 1

    def test_reset(self) -> None:
        from jarvis.metrics import DraftMetrics

        metrics = DraftMetrics()
        metrics.record(0.5, gated=False)
        metrics.reset()
        assert metrics.get_stats()["total_requests"] == 0

    def test_global_singleton(self) -> None:
        from jarvis.metrics import get_draft_metrics, reset_metrics

        reset_metrics()
        m1 = get_draft_metrics()
        m2 = get_draft_metrics()
        assert m1 is m2


class TestConfidenceGatingThreshold:
    """Test that the confidence threshold constant is properly defined."""

    def test_threshold_value(self) -> None:
        from jarvis.socket_server import DRAFT_CONFIDENCE_THRESHOLD

        assert DRAFT_CONFIDENCE_THRESHOLD == 0.4

    def test_router_includes_confidence_score(self) -> None:
        """Verify _to_legacy_response includes confidence_score float."""
        import inspect

        from jarvis.router import ReplyRouter

        source = inspect.getsource(ReplyRouter._to_legacy_response)
        assert "confidence_score" in source
