"""Unit tests for WS5 Memory Controller implementation.

Tests memory monitoring, controller protocol, mode determination,
pressure callbacks, and singleton management.
"""

import threading
from unittest.mock import MagicMock

import pytest

from contracts.memory import MemoryMode, MemoryState
from core.memory.controller import (
    DefaultMemoryController,
    MemoryThresholds,
    get_memory_controller,
    reset_memory_controller,
)
from core.memory.monitor import MemoryMonitor, SystemMemoryInfo


class TestMemoryMonitor:
    """Tests for MemoryMonitor."""

    def test_get_system_memory_returns_dataclass(self, monkeypatch):
        """Verify get_system_memory returns SystemMemoryInfo."""
        mock_mem = MagicMock()
        mock_mem.total = 16 * 1024 * 1024 * 1024  # 16GB
        mock_mem.available = 8 * 1024 * 1024 * 1024  # 8GB
        mock_mem.used = 8 * 1024 * 1024 * 1024  # 8GB
        mock_mem.percent = 50.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        monitor = MemoryMonitor()
        info = monitor.get_system_memory()

        assert isinstance(info, SystemMemoryInfo)
        assert info.total_mb == pytest.approx(16384.0, rel=0.01)
        assert info.available_mb == pytest.approx(8192.0, rel=0.01)
        assert info.used_mb == pytest.approx(8192.0, rel=0.01)
        assert info.percent_used == 50.0

    def test_get_available_mb(self, monkeypatch):
        """Test get_available_mb returns correct value."""
        mock_mem = MagicMock()
        mock_mem.available = 4 * 1024 * 1024 * 1024  # 4GB

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        monitor = MemoryMonitor()
        assert monitor.get_available_mb() == pytest.approx(4096.0, rel=0.01)

    def test_get_used_mb(self, monkeypatch):
        """Test get_used_mb returns correct value."""
        mock_mem = MagicMock()
        mock_mem.used = 6 * 1024 * 1024 * 1024  # 6GB

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        monitor = MemoryMonitor()
        assert monitor.get_used_mb() == pytest.approx(6144.0, rel=0.01)

    def test_get_percent_used(self, monkeypatch):
        """Test get_percent_used returns correct value."""
        mock_mem = MagicMock()
        mock_mem.percent = 75.5

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        monitor = MemoryMonitor()
        assert monitor.get_percent_used() == 75.5


class TestPressureLevels:
    """Tests for memory pressure level determination."""

    @pytest.mark.parametrize(
        "percent,expected",
        [
            (50.0, "green"),
            (69.9, "green"),
            (70.0, "yellow"),
            (80.0, "yellow"),
            (84.9, "yellow"),
            (85.0, "red"),
            (90.0, "red"),
            (94.9, "red"),
            (95.0, "critical"),
            (99.0, "critical"),
            (100.0, "critical"),
        ],
    )
    def test_pressure_level_thresholds(self, monkeypatch, percent, expected):
        """Test pressure level determination at boundary values."""
        mock_mem = MagicMock()
        mock_mem.percent = percent

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        monitor = MemoryMonitor()
        assert monitor.get_pressure_level() == expected

    def test_pressure_green_low_usage(self, monkeypatch):
        """Test green pressure at very low usage."""
        mock_mem = MagicMock()
        mock_mem.percent = 10.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        monitor = MemoryMonitor()
        assert monitor.get_pressure_level() == "green"


class TestMemoryThresholds:
    """Tests for MemoryThresholds configuration."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = MemoryThresholds()
        assert thresholds.full_mode_mb == 8000.0
        assert thresholds.lite_mode_mb == 4000.0
        assert thresholds.memory_buffer_multiplier == 1.2

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = MemoryThresholds(
            full_mode_mb=10000.0,
            lite_mode_mb=5000.0,
            memory_buffer_multiplier=1.5,
        )
        assert thresholds.full_mode_mb == 10000.0
        assert thresholds.lite_mode_mb == 5000.0
        assert thresholds.memory_buffer_multiplier == 1.5


class TestDefaultMemoryController:
    """Tests for DefaultMemoryController."""

    def test_get_state_returns_memory_state(self, monkeypatch):
        """Verify get_state returns MemoryState dataclass."""
        mock_mem = MagicMock()
        mock_mem.total = 16 * 1024 * 1024 * 1024
        mock_mem.available = 10 * 1024 * 1024 * 1024  # 10GB
        mock_mem.used = 6 * 1024 * 1024 * 1024
        mock_mem.percent = 37.5

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        state = controller.get_state()

        assert isinstance(state, MemoryState)
        assert state.available_mb == pytest.approx(10240.0, rel=0.01)
        assert state.used_mb == pytest.approx(6144.0, rel=0.01)
        assert state.model_loaded is False
        assert state.current_mode == MemoryMode.FULL
        assert state.pressure_level == "green"

    def test_get_state_with_model_loaded(self, monkeypatch):
        """Test get_state reflects model_loaded flag."""
        mock_mem = MagicMock()
        mock_mem.total = 16 * 1024 * 1024 * 1024
        mock_mem.available = 10 * 1024 * 1024 * 1024
        mock_mem.used = 6 * 1024 * 1024 * 1024
        mock_mem.percent = 37.5

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController(model_loaded=True)
        state = controller.get_state()

        assert state.model_loaded is True

    def test_set_model_loaded(self, monkeypatch):
        """Test set_model_loaded updates state."""
        mock_mem = MagicMock()
        mock_mem.total = 16 * 1024 * 1024 * 1024
        mock_mem.available = 10 * 1024 * 1024 * 1024
        mock_mem.used = 6 * 1024 * 1024 * 1024
        mock_mem.percent = 37.5

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        assert controller.get_state().model_loaded is False

        controller.set_model_loaded(True)
        assert controller.get_state().model_loaded is True

        controller.set_model_loaded(False)
        assert controller.get_state().model_loaded is False


class TestMemoryModes:
    """Tests for memory mode determination."""

    def test_full_mode_above_8gb(self, monkeypatch):
        """Test FULL mode when >8GB available."""
        mock_mem = MagicMock()
        mock_mem.available = 10 * 1024 * 1024 * 1024  # 10GB
        mock_mem.percent = 40.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        assert controller.get_mode() == MemoryMode.FULL

    def test_lite_mode_between_4_and_8gb(self, monkeypatch):
        """Test LITE mode when 4-8GB available."""
        mock_mem = MagicMock()
        mock_mem.available = 6 * 1024 * 1024 * 1024  # 6GB
        mock_mem.percent = 60.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        assert controller.get_mode() == MemoryMode.LITE

    def test_minimal_mode_below_4gb(self, monkeypatch):
        """Test MINIMAL mode when <4GB available."""
        mock_mem = MagicMock()
        mock_mem.available = 3 * 1024 * 1024 * 1024  # 3GB
        mock_mem.percent = 80.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        assert controller.get_mode() == MemoryMode.MINIMAL

    def test_mode_boundary_at_8gb(self, monkeypatch):
        """Test mode at exactly 8GB (should be LITE, not FULL)."""
        mock_mem = MagicMock()
        # 8000MB exactly is NOT > 8000, so should be LITE
        mock_mem.available = 8000 * 1024 * 1024
        mock_mem.percent = 50.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        assert controller.get_mode() == MemoryMode.LITE

    def test_mode_boundary_at_4gb(self, monkeypatch):
        """Test mode at exactly 4GB (should be MINIMAL, not LITE)."""
        mock_mem = MagicMock()
        # 4000MB exactly is NOT > 4000, so should be MINIMAL
        mock_mem.available = 4000 * 1024 * 1024
        mock_mem.percent = 75.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        assert controller.get_mode() == MemoryMode.MINIMAL

    def test_mode_with_custom_thresholds(self, monkeypatch):
        """Test mode determination with custom thresholds."""
        mock_mem = MagicMock()
        mock_mem.available = 6 * 1024 * 1024 * 1024  # 6GB
        mock_mem.percent = 60.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        # Custom thresholds that would make 6GB count as FULL
        thresholds = MemoryThresholds(full_mode_mb=5000.0, lite_mode_mb=2000.0)
        controller = DefaultMemoryController(thresholds=thresholds)

        assert controller.get_mode() == MemoryMode.FULL


class TestCanLoadModel:
    """Tests for can_load_model method."""

    def test_can_load_with_sufficient_memory(self, monkeypatch):
        """Test can_load_model returns True with sufficient memory."""
        mock_mem = MagicMock()
        mock_mem.available = 10 * 1024 * 1024 * 1024  # 10GB
        mock_mem.percent = 40.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        # Request 1GB, with 1.2x buffer = 1.2GB needed, have 10GB
        assert controller.can_load_model(1000) is True

    def test_cannot_load_with_insufficient_memory(self, monkeypatch):
        """Test can_load_model returns False with insufficient memory."""
        mock_mem = MagicMock()
        mock_mem.available = 1 * 1024 * 1024 * 1024  # 1GB
        mock_mem.percent = 90.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        # Request 2GB, with 1.2x buffer = 2.4GB needed, have 1GB
        assert controller.can_load_model(2000) is False

    def test_can_load_considers_buffer_multiplier(self, monkeypatch):
        """Test can_load_model applies buffer multiplier."""
        mock_mem = MagicMock()
        mock_mem.available = 1200 * 1024 * 1024  # 1200MB
        mock_mem.percent = 85.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        # Default buffer is 1.2x
        controller = DefaultMemoryController()
        # Request 1000MB * 1.2 = 1200MB needed, have exactly 1200MB
        assert controller.can_load_model(1000) is True

    def test_can_load_fails_just_under_buffer(self, monkeypatch):
        """Test can_load_model fails when just under buffered requirement."""
        mock_mem = MagicMock()
        mock_mem.available = 1199 * 1024 * 1024  # 1199MB
        mock_mem.percent = 85.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        # Request 1000MB * 1.2 = 1200MB needed, have 1199MB
        assert controller.can_load_model(1000) is False


class TestRequestMemory:
    """Tests for request_memory method."""

    def test_request_memory_succeeds_with_available(self, monkeypatch):
        """Test request_memory succeeds when memory is available."""
        mock_mem = MagicMock()
        mock_mem.available = 10 * 1024 * 1024 * 1024  # 10GB
        mock_mem.percent = 40.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        assert controller.request_memory(1000, priority=1) is True

    def test_request_memory_fails_in_critical_pressure(self, monkeypatch):
        """Test request_memory fails in critical pressure when unavailable."""
        mock_mem = MagicMock()
        mock_mem.available = 500 * 1024 * 1024  # 500MB
        mock_mem.percent = 97.0  # Critical

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        assert controller.request_memory(1000, priority=1) is False

    def test_request_memory_notifies_callbacks_on_critical(self, monkeypatch):
        """Test request_memory notifies callbacks in critical pressure."""
        mock_mem = MagicMock()
        mock_mem.available = 500 * 1024 * 1024  # 500MB
        mock_mem.percent = 97.0  # Critical

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        callback_calls = []
        controller.register_pressure_callback(lambda p: callback_calls.append(p))

        controller.request_memory(1000, priority=1)

        assert "critical" in callback_calls


class TestPressureCallbacks:
    """Tests for pressure callback registration and notification."""

    def test_register_pressure_callback(self, monkeypatch):
        """Test registering a pressure callback."""
        mock_mem = MagicMock()
        mock_mem.available = 10 * 1024 * 1024 * 1024
        mock_mem.percent = 40.0
        mock_mem.total = 16 * 1024 * 1024 * 1024
        mock_mem.used = 6 * 1024 * 1024 * 1024

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        callback = MagicMock()
        controller.register_pressure_callback(callback)

        assert callback in controller._callbacks

    def test_unregister_pressure_callback(self, monkeypatch):
        """Test unregistering a pressure callback."""
        mock_mem = MagicMock()
        mock_mem.available = 10 * 1024 * 1024 * 1024
        mock_mem.percent = 40.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        callback = MagicMock()
        controller.register_pressure_callback(callback)
        controller.unregister_pressure_callback(callback)

        assert callback not in controller._callbacks

    def test_callback_not_duplicated(self, monkeypatch):
        """Test callback is not registered twice."""
        mock_mem = MagicMock()
        mock_mem.available = 10 * 1024 * 1024 * 1024
        mock_mem.percent = 40.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        callback = MagicMock()
        controller.register_pressure_callback(callback)
        controller.register_pressure_callback(callback)

        assert controller._callbacks.count(callback) == 1

    def test_callback_invoked_on_pressure_change(self, monkeypatch):
        """Test callbacks are invoked when pressure level changes."""
        state_count = [0]
        pressure_values = []

        def mock_virtual_memory():
            mock = MagicMock()
            mock.total = 16 * 1024 * 1024 * 1024
            mock.used = 8 * 1024 * 1024 * 1024
            # Use state_count to track which get_state() call we're in
            # get_state() calls virtual_memory() multiple times per call
            if state_count[0] < 2:
                # First get_state() - green
                mock.available = 10 * 1024 * 1024 * 1024
                mock.percent = 40.0
            else:
                # Second get_state() - yellow
                mock.available = 3 * 1024 * 1024 * 1024
                mock.percent = 80.0
            return mock

        # Patch at the module level where psutil is imported
        import core.memory.monitor

        monkeypatch.setattr(core.memory.monitor.psutil, "virtual_memory", mock_virtual_memory)

        controller = DefaultMemoryController()
        controller.register_pressure_callback(lambda p: pressure_values.append(p))

        # First call establishes baseline (green)
        controller.get_state()
        state_count[0] = 2  # Switch to yellow for next get_state()
        # Second call should detect change to yellow
        controller.get_state()

        assert "yellow" in pressure_values

    def test_callback_error_does_not_break_others(self, monkeypatch):
        """Test callback error doesn't prevent other callbacks."""
        mock_mem = MagicMock()
        mock_mem.available = 500 * 1024 * 1024
        mock_mem.percent = 97.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()

        def bad_callback(pressure):
            raise RuntimeError("Callback error")

        good_callback = MagicMock()

        controller.register_pressure_callback(bad_callback)
        controller.register_pressure_callback(good_callback)

        # Should not raise, and good_callback should still be called
        controller.request_memory(1000, priority=1)

        good_callback.assert_called()


class TestControllerSingleton:
    """Tests for memory controller singleton management."""

    def test_get_memory_controller_returns_same_instance(self, monkeypatch):
        """Verify get_memory_controller returns singleton."""
        mock_mem = MagicMock()
        mock_mem.available = 10 * 1024 * 1024 * 1024
        mock_mem.percent = 40.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        reset_memory_controller()  # Clean state
        ctrl1 = get_memory_controller()
        ctrl2 = get_memory_controller()
        assert ctrl1 is ctrl2
        reset_memory_controller()  # Clean up

    def test_reset_memory_controller_clears_singleton(self, monkeypatch):
        """Verify reset_memory_controller clears the singleton."""
        mock_mem = MagicMock()
        mock_mem.available = 10 * 1024 * 1024 * 1024
        mock_mem.percent = 40.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        ctrl1 = get_memory_controller()
        reset_memory_controller()
        ctrl2 = get_memory_controller()
        assert ctrl1 is not ctrl2
        reset_memory_controller()  # Clean up

    def test_reset_when_none_is_safe(self):
        """Verify reset when no controller exists doesn't raise."""
        reset_memory_controller()  # Should not raise
        reset_memory_controller()  # Should not raise


class TestControllerThreadSafety:
    """Tests for thread safety of memory controller."""

    def test_concurrent_callback_registration(self, monkeypatch):
        """Test callbacks can be registered from multiple threads."""
        mock_mem = MagicMock()
        mock_mem.available = 10 * 1024 * 1024 * 1024
        mock_mem.percent = 40.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        callbacks = [MagicMock() for _ in range(10)]
        threads = []

        def register_callback(cb):
            controller.register_pressure_callback(cb)

        for cb in callbacks:
            t = threading.Thread(target=register_callback, args=(cb,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(controller._callbacks) == 10

    def test_concurrent_get_state(self, monkeypatch):
        """Test get_state is thread-safe."""
        mock_mem = MagicMock()
        mock_mem.available = 10 * 1024 * 1024 * 1024
        mock_mem.used = 6 * 1024 * 1024 * 1024
        mock_mem.total = 16 * 1024 * 1024 * 1024
        mock_mem.percent = 40.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        results = []

        def get_state():
            for _ in range(100):
                state = controller.get_state()
                results.append(state)

        threads = [threading.Thread(target=get_state) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 500
        assert all(isinstance(r, MemoryState) for r in results)


class TestProtocolCompliance:
    """Verify DefaultMemoryController implements MemoryController protocol."""

    def test_has_get_state_method(self):
        """Verify get_state method exists."""
        controller = DefaultMemoryController()
        assert hasattr(controller, "get_state")
        assert callable(controller.get_state)

    def test_has_get_mode_method(self):
        """Verify get_mode method exists."""
        controller = DefaultMemoryController()
        assert hasattr(controller, "get_mode")
        assert callable(controller.get_mode)

    def test_has_can_load_model_method(self):
        """Verify can_load_model method exists."""
        controller = DefaultMemoryController()
        assert hasattr(controller, "can_load_model")
        assert callable(controller.can_load_model)

    def test_has_request_memory_method(self):
        """Verify request_memory method exists."""
        controller = DefaultMemoryController()
        assert hasattr(controller, "request_memory")
        assert callable(controller.request_memory)

    def test_has_register_pressure_callback_method(self):
        """Verify register_pressure_callback method exists."""
        controller = DefaultMemoryController()
        assert hasattr(controller, "register_pressure_callback")
        assert callable(controller.register_pressure_callback)

    def test_get_state_returns_correct_type(self, monkeypatch):
        """Verify get_state returns MemoryState."""
        mock_mem = MagicMock()
        mock_mem.available = 10 * 1024 * 1024 * 1024
        mock_mem.used = 6 * 1024 * 1024 * 1024
        mock_mem.total = 16 * 1024 * 1024 * 1024
        mock_mem.percent = 40.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        state = controller.get_state()
        assert isinstance(state, MemoryState)

    def test_get_mode_returns_correct_type(self, monkeypatch):
        """Verify get_mode returns MemoryMode."""
        mock_mem = MagicMock()
        mock_mem.available = 10 * 1024 * 1024 * 1024
        mock_mem.percent = 40.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        mode = controller.get_mode()
        assert isinstance(mode, MemoryMode)


class TestCustomMonitor:
    """Tests for controller with custom monitor."""

    def test_controller_uses_custom_monitor(self):
        """Test controller can use a custom monitor."""
        mock_monitor = MagicMock(spec=MemoryMonitor)
        mock_monitor.get_available_mb.return_value = 12000.0
        mock_monitor.get_percent_used.return_value = 30.0
        mock_monitor.get_pressure_level.return_value = "green"
        mock_monitor.get_system_memory.return_value = SystemMemoryInfo(
            total_mb=16000.0,
            available_mb=12000.0,
            used_mb=4000.0,
            percent_used=30.0,
        )

        controller = DefaultMemoryController(monitor=mock_monitor)
        state = controller.get_state()

        assert state.available_mb == 12000.0
        mock_monitor.get_system_memory.assert_called_once()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_memory_available(self, monkeypatch):
        """Test handling of zero available memory."""
        mock_mem = MagicMock()
        mock_mem.available = 0
        mock_mem.used = 16 * 1024 * 1024 * 1024
        mock_mem.total = 16 * 1024 * 1024 * 1024
        mock_mem.percent = 100.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        assert controller.get_mode() == MemoryMode.MINIMAL
        assert controller.can_load_model(100) is False

    def test_very_large_memory_request(self, monkeypatch):
        """Test handling of very large memory request."""
        mock_mem = MagicMock()
        mock_mem.available = 10 * 1024 * 1024 * 1024
        mock_mem.percent = 40.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        # Request 100GB - should fail
        assert controller.can_load_model(100000) is False

    def test_zero_memory_request(self, monkeypatch):
        """Test handling of zero memory request."""
        mock_mem = MagicMock()
        mock_mem.available = 10 * 1024 * 1024 * 1024
        mock_mem.percent = 40.0

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        controller = DefaultMemoryController()
        # Zero request should always succeed
        assert controller.can_load_model(0) is True
        assert controller.request_memory(0, priority=1) is True
