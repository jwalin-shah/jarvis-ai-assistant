"""Unit tests for the graceful degradation controller.

Tests the GracefulDegradationController class in core/health/degradation.py
for degradation level tracking, recovery logic, and fallback behavior.
"""

from unittest.mock import MagicMock

import pytest

from contracts.health import DegradationPolicy, FeatureState
from core.health.degradation import (
    FeatureRegistration,
    GracefulDegradationController,
    get_degradation_controller,
    reset_degradation_controller,
)


@pytest.fixture
def controller():
    """Create a fresh degradation controller for each test."""
    return GracefulDegradationController()


@pytest.fixture
def basic_policy():
    """Create a basic degradation policy for testing."""
    return DegradationPolicy(
        feature_name="test_feature",
        health_check=lambda: True,
        degraded_behavior=lambda *args, **kwargs: "degraded",
        fallback_behavior=lambda *args, **kwargs: "fallback",
        recovery_check=lambda: True,
        max_failures=3,
    )


class TestFeatureRegistration:
    """Tests for FeatureRegistration class."""

    def test_stores_policy(self):
        """Stores the provided policy."""
        policy = MagicMock()
        circuit = MagicMock()
        registration = FeatureRegistration(policy, circuit)

        assert registration.policy is policy

    def test_stores_circuit_breaker(self):
        """Stores the provided circuit breaker."""
        policy = MagicMock()
        circuit = MagicMock()
        registration = FeatureRegistration(policy, circuit)

        assert registration.circuit_breaker is circuit

    def test_primary_callable_initially_none(self):
        """Primary callable is initially None."""
        policy = MagicMock()
        circuit = MagicMock()
        registration = FeatureRegistration(policy, circuit)

        assert registration.primary_callable is None


class TestGracefulDegradationControllerInit:
    """Tests for controller initialization."""

    def test_initializes_empty_features(self):
        """Controller starts with no features registered."""
        controller = GracefulDegradationController()
        health = controller.get_health()
        assert len(health) == 0


class TestRegisterFeature:
    """Tests for feature registration."""

    def test_registers_feature(self, controller, basic_policy):
        """Can register a feature."""
        controller.register_feature(basic_policy)
        health = controller.get_health()

        assert "test_feature" in health

    def test_registered_feature_is_healthy(self, controller, basic_policy):
        """Newly registered feature is healthy."""
        controller.register_feature(basic_policy)
        health = controller.get_health()

        assert health["test_feature"] == FeatureState.HEALTHY

    def test_replaces_existing_feature(self, controller, basic_policy):
        """Replaces policy if feature already registered."""
        controller.register_feature(basic_policy)

        new_policy = DegradationPolicy(
            feature_name="test_feature",
            health_check=lambda: False,
            degraded_behavior=lambda: "new_degraded",
            fallback_behavior=lambda: "new_fallback",
            recovery_check=lambda: False,
            max_failures=10,
        )
        controller.register_feature(new_policy)

        # Should still have one feature
        health = controller.get_health()
        assert len(health) == 1

    def test_multiple_features(self, controller):
        """Can register multiple features."""
        policy1 = DegradationPolicy(
            feature_name="feature1",
            health_check=lambda: True,
            degraded_behavior=lambda: "d1",
            fallback_behavior=lambda: "f1",
            recovery_check=lambda: True,
        )
        policy2 = DegradationPolicy(
            feature_name="feature2",
            health_check=lambda: True,
            degraded_behavior=lambda: "d2",
            fallback_behavior=lambda: "f2",
            recovery_check=lambda: True,
        )

        controller.register_feature(policy1)
        controller.register_feature(policy2)

        health = controller.get_health()
        assert len(health) == 2
        assert "feature1" in health
        assert "feature2" in health


class TestExecute:
    """Tests for execute method."""

    def test_raises_on_unregistered_feature(self, controller):
        """Raises KeyError for unregistered feature."""
        with pytest.raises(KeyError) as exc_info:
            controller.execute("nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_executes_primary_on_healthy(self, controller, basic_policy):
        """Executes primary function when healthy."""
        controller.register_feature(basic_policy)

        result = controller.execute("test_feature", primary=lambda: "primary_result")

        assert result == "primary_result"

    def test_falls_back_on_primary_failure(self, controller):
        """Uses degraded behavior when primary fails."""
        policy = DegradationPolicy(
            feature_name="flaky",
            health_check=lambda: True,
            degraded_behavior=lambda: "degraded_result",
            fallback_behavior=lambda: "fallback_result",
            recovery_check=lambda: True,
            max_failures=3,  # Need multiple failures to open circuit
        )
        controller.register_feature(policy)

        def failing_primary():
            raise RuntimeError("Primary failed")

        # First call fails and uses degraded (circuit still closed)
        result1 = controller.execute("flaky", failing_primary)
        assert result1 == "degraded_result"

    def test_passes_args_to_primary(self, controller, basic_policy):
        """Passes arguments to primary function."""
        controller.register_feature(basic_policy)

        def add(a, b):
            return a + b

        result = controller.execute("test_feature", add, 3, 4)

        assert result == 7

    def test_passes_kwargs_to_primary(self, controller, basic_policy):
        """Passes keyword arguments to primary function."""
        controller.register_feature(basic_policy)

        def multiply(x, factor=1):
            return x * factor

        result = controller.execute("test_feature", multiply, 5, factor=3)

        assert result == 15


class TestExecuteDegradedBehavior:
    """Tests for degraded behavior execution."""

    def test_uses_degraded_on_failure(self, controller):
        """Uses degraded behavior after failure."""
        policy = DegradationPolicy(
            feature_name="test",
            health_check=lambda: True,
            degraded_behavior=lambda: "degraded_mode",
            fallback_behavior=lambda: "fallback_mode",
            recovery_check=lambda: True,
            max_failures=5,  # Won't open circuit yet
        )
        controller.register_feature(policy)

        def failing_primary():
            raise ValueError("Oops")

        result = controller.execute("test", primary=failing_primary)

        assert result == "degraded_mode"

    def test_degraded_receives_args(self, controller):
        """Degraded behavior receives arguments."""
        received_args = []

        def degraded_fn(*args, **kwargs):
            received_args.extend(args)
            return "degraded"

        policy = DegradationPolicy(
            feature_name="test",
            health_check=lambda: True,
            degraded_behavior=degraded_fn,
            fallback_behavior=lambda: "fallback",
            recovery_check=lambda: True,
        )
        controller.register_feature(policy)

        def failing(*args, **kwargs):
            raise RuntimeError("fail")

        controller.execute("test", failing, 1, 2, 3)

        assert received_args == [1, 2, 3]


class TestExecuteFallbackBehavior:
    """Tests for fallback behavior execution."""

    def test_uses_fallback_when_circuit_open(self, controller):
        """Uses fallback when circuit is open."""
        policy = DegradationPolicy(
            feature_name="test",
            health_check=lambda: True,
            degraded_behavior=lambda: "degraded",
            fallback_behavior=lambda: "fallback_only",
            recovery_check=lambda: True,
            max_failures=1,
        )
        controller.register_feature(policy)

        def failing():
            raise RuntimeError("fail")

        # First call trips the circuit
        controller.execute("test", primary=failing)

        # Second call should use fallback directly
        result = controller.execute("test", primary=failing)

        assert result == "fallback_only"

    def test_fallback_receives_args(self, controller):
        """Fallback behavior receives arguments."""
        received_kwargs = {}

        def fallback_fn(*args, **kwargs):
            received_kwargs.update(kwargs)
            return "fallback"

        def degraded_fails(*a, **kw):
            raise RuntimeError("degraded fails")

        policy = DegradationPolicy(
            feature_name="test",
            health_check=lambda: True,
            degraded_behavior=degraded_fails,
            fallback_behavior=fallback_fn,
            recovery_check=lambda: True,
            max_failures=1,
        )
        controller.register_feature(policy)

        def failing(*args, **kwargs):
            raise RuntimeError("primary fails")

        controller.execute("test", failing, key="value")

        # After circuit opens, fallback receives kwargs
        controller.execute("test", failing, key="value2")
        assert received_kwargs.get("key") == "value2"


class TestGetHealth:
    """Tests for get_health method."""

    def test_empty_when_no_features(self, controller):
        """Returns empty dict when no features registered."""
        health = controller.get_health()
        assert health == {}

    def test_shows_healthy_state(self, controller, basic_policy):
        """Shows HEALTHY for closed circuit."""
        controller.register_feature(basic_policy)
        health = controller.get_health()

        assert health["test_feature"] == FeatureState.HEALTHY

    def test_shows_failed_state(self, controller):
        """Shows FAILED for open circuit."""
        policy = DegradationPolicy(
            feature_name="test",
            health_check=lambda: True,
            degraded_behavior=lambda: "degraded",
            fallback_behavior=lambda: "fallback",
            recovery_check=lambda: True,
            max_failures=1,
        )
        controller.register_feature(policy)

        def failing():
            raise RuntimeError("fail")

        # Trip the circuit
        controller.execute("test", primary=failing)

        health = controller.get_health()
        assert health["test"] == FeatureState.FAILED


class TestResetFeature:
    """Tests for reset_feature method."""

    def test_raises_on_unregistered(self, controller):
        """Raises KeyError for unregistered feature."""
        with pytest.raises(KeyError):
            controller.reset_feature("nonexistent")

    def test_resets_circuit_breaker(self, controller):
        """Resets circuit breaker to closed state."""
        policy = DegradationPolicy(
            feature_name="test",
            health_check=lambda: True,
            degraded_behavior=lambda: "degraded",
            fallback_behavior=lambda: "fallback",
            recovery_check=lambda: True,
            max_failures=1,
        )
        controller.register_feature(policy)

        def failing():
            raise RuntimeError("fail")

        # Trip the circuit
        controller.execute("test", primary=failing)
        assert controller.get_health()["test"] == FeatureState.FAILED

        # Reset
        controller.reset_feature("test")
        assert controller.get_health()["test"] == FeatureState.HEALTHY


class TestUnregisterFeature:
    """Tests for unregister_feature method."""

    def test_raises_on_unregistered(self, controller):
        """Raises KeyError for unregistered feature."""
        with pytest.raises(KeyError):
            controller.unregister_feature("nonexistent")

    def test_removes_feature(self, controller, basic_policy):
        """Removes feature from controller."""
        controller.register_feature(basic_policy)
        assert "test_feature" in controller.get_health()

        controller.unregister_feature("test_feature")
        assert "test_feature" not in controller.get_health()


class TestGetFeatureStats:
    """Tests for get_feature_stats method."""

    def test_raises_on_unregistered(self, controller):
        """Raises KeyError for unregistered feature."""
        with pytest.raises(KeyError):
            controller.get_feature_stats("nonexistent")

    def test_returns_stats_dict(self, controller, basic_policy):
        """Returns dictionary with stats."""
        controller.register_feature(basic_policy)
        stats = controller.get_feature_stats("test_feature")

        assert "feature_name" in stats
        assert "state" in stats
        assert "failure_count" in stats
        assert "success_count" in stats
        assert "total_executions" in stats

    def test_stats_reflect_operations(self, controller, basic_policy):
        """Stats reflect executed operations."""
        controller.register_feature(basic_policy)

        controller.execute("test_feature", primary=lambda: "ok")
        controller.execute("test_feature", primary=lambda: "ok")

        stats = controller.get_feature_stats("test_feature")
        assert stats["total_successes"] >= 2


class TestSingleton:
    """Tests for singleton controller access."""

    def test_get_degradation_controller_returns_instance(self):
        """get_degradation_controller returns a controller instance."""
        reset_degradation_controller()
        controller = get_degradation_controller()

        assert isinstance(controller, GracefulDegradationController)

    def test_get_degradation_controller_same_instance(self):
        """get_degradation_controller returns same instance."""
        reset_degradation_controller()
        controller1 = get_degradation_controller()
        controller2 = get_degradation_controller()

        assert controller1 is controller2

    def test_reset_degradation_controller(self):
        """reset_degradation_controller creates new instance."""
        reset_degradation_controller()
        controller1 = get_degradation_controller()

        reset_degradation_controller()
        controller2 = get_degradation_controller()

        assert controller1 is not controller2


class TestHealthCheckBehavior:
    """Tests for health check execution."""

    def test_unhealthy_check_uses_fallback(self, controller):
        """Uses fallback when health check returns False."""
        policy = DegradationPolicy(
            feature_name="test",
            health_check=lambda: False,
            degraded_behavior=lambda: "degraded",
            fallback_behavior=lambda: "fallback_due_to_health",
            recovery_check=lambda: True,
        )
        controller.register_feature(policy)

        # Execute without primary - should check health
        result = controller.execute("test", primary=None)

        assert result == "fallback_due_to_health"

    def test_health_check_exception_returns_false(self, controller):
        """Health check that raises exception is treated as unhealthy."""
        policy = DegradationPolicy(
            feature_name="test",
            health_check=lambda: (_ for _ in ()).throw(RuntimeError("health error")),
            degraded_behavior=lambda: "degraded",
            fallback_behavior=lambda: "fallback",
            recovery_check=lambda: True,
        )
        controller.register_feature(policy)

        # Should not raise, should use fallback
        result = controller.execute("test", primary=None)

        assert result == "fallback"
