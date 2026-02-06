"""Unit tests for the memory controller.

Tests the DefaultMemoryController class in core/memory/controller.py
for memory monitoring, threshold alerts, and mode determination.
"""

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


@pytest.fixture
def mock_monitor():
    """Create a mock memory monitor."""
    monitor = MagicMock(spec=MemoryMonitor)
    monitor.get_available_mb.return_value = 10000.0
    monitor.get_pressure_level.return_value = "green"
    monitor.get_system_memory.return_value = SystemMemoryInfo(
        total_mb=16000.0,
        available_mb=10000.0,
        used_mb=6000.0,
        percent_used=37.5,
    )
    return monitor


@pytest.fixture
def controller(mock_monitor):
    """Create a memory controller with mocked monitor."""
    return DefaultMemoryController(monitor=mock_monitor)


class TestMemoryThresholds:
    """Tests for MemoryThresholds dataclass."""

    def test_default_values(self):
        """Default thresholds have sensible values for 8GB systems."""
        thresholds = MemoryThresholds()
        # Sanity checks: thresholds are positive and ordered correctly
        assert thresholds.lite_mode_mb > 0
        assert thresholds.full_mode_mb > thresholds.lite_mode_mb
        assert thresholds.memory_buffer_multiplier >= 1.0

    def test_custom_values(self):
        """Can create thresholds with custom values."""
        thresholds = MemoryThresholds(
            full_mode_mb=12000.0,
            lite_mode_mb=6000.0,
            memory_buffer_multiplier=1.5,
        )
        assert thresholds.full_mode_mb == 12000.0
        assert thresholds.lite_mode_mb == 6000.0
        assert thresholds.memory_buffer_multiplier == 1.5


class TestDefaultMemoryControllerInit:
    """Tests for controller initialization."""

    def test_creates_default_monitor(self):
        """Creates default monitor if not provided."""
        controller = DefaultMemoryController()
        assert controller._monitor is not None

    def test_uses_provided_monitor(self, mock_monitor):
        """Uses provided monitor."""
        controller = DefaultMemoryController(monitor=mock_monitor)
        assert controller._monitor is mock_monitor

    def test_creates_default_thresholds(self):
        """Creates default thresholds if not provided."""
        controller = DefaultMemoryController()
        assert controller._thresholds is not None
        assert controller._thresholds.full_mode_mb == 8000.0

    def test_uses_provided_thresholds(self):
        """Uses provided thresholds."""
        thresholds = MemoryThresholds(full_mode_mb=10000.0)
        controller = DefaultMemoryController(thresholds=thresholds)
        assert controller._thresholds.full_mode_mb == 10000.0

    def test_model_loaded_default_false(self):
        """Model loaded defaults to False."""
        controller = DefaultMemoryController()
        assert controller._model_loaded is False

    def test_model_loaded_can_be_set(self):
        """Model loaded can be set on init."""
        controller = DefaultMemoryController(model_loaded=True)
        assert controller._model_loaded is True


class TestGetState:
    """Tests for get_state method."""

    def test_returns_memory_state(self, controller, mock_monitor):
        """Returns a MemoryState object."""
        state = controller.get_state()

        assert isinstance(state, MemoryState)

    def test_state_has_available_mb(self, controller, mock_monitor):
        """State includes available memory."""
        state = controller.get_state()

        assert state.available_mb == 10000.0

    def test_state_has_used_mb(self, controller, mock_monitor):
        """State includes used memory."""
        state = controller.get_state()

        assert state.used_mb == 6000.0

    def test_state_has_model_loaded(self, controller):
        """State includes model loaded status."""
        controller.set_model_loaded(True)
        state = controller.get_state()

        assert state.model_loaded is True

    def test_state_has_current_mode(self, controller, mock_monitor):
        """State includes current memory mode."""
        state = controller.get_state()

        assert state.current_mode == MemoryMode.FULL

    def test_state_has_pressure_level(self, controller, mock_monitor):
        """State includes pressure level."""
        state = controller.get_state()

        assert state.pressure_level == "green"


class TestGetMode:
    """Tests for get_mode method."""

    def test_full_mode_with_high_memory(self, mock_monitor):
        """Returns FULL mode when plenty of memory available."""
        mock_monitor.get_available_mb.return_value = 10000.0
        controller = DefaultMemoryController(monitor=mock_monitor)

        assert controller.get_mode() == MemoryMode.FULL

    def test_lite_mode_with_medium_memory(self, mock_monitor):
        """Returns LITE mode when moderate memory available."""
        mock_monitor.get_available_mb.return_value = 6000.0
        controller = DefaultMemoryController(monitor=mock_monitor)

        assert controller.get_mode() == MemoryMode.LITE

    def test_minimal_mode_with_low_memory(self, mock_monitor):
        """Returns MINIMAL mode when low memory available."""
        mock_monitor.get_available_mb.return_value = 400.0
        controller = DefaultMemoryController(monitor=mock_monitor)

        assert controller.get_mode() == MemoryMode.MINIMAL

    def test_mode_boundary_full_lite(self, mock_monitor):
        """Tests boundary between FULL and LITE modes."""
        thresholds = MemoryThresholds(full_mode_mb=8000.0)
        mock_monitor.get_available_mb.return_value = 8000.0
        controller = DefaultMemoryController(monitor=mock_monitor, thresholds=thresholds)

        # At exactly 8000, should be LITE (not greater than threshold)
        assert controller.get_mode() == MemoryMode.LITE

    def test_mode_boundary_lite_minimal(self, mock_monitor):
        """Tests boundary between LITE and MINIMAL modes."""
        thresholds = MemoryThresholds(lite_mode_mb=4000.0)
        mock_monitor.get_available_mb.return_value = 4000.0
        controller = DefaultMemoryController(monitor=mock_monitor, thresholds=thresholds)

        # At exactly 4000, should be MINIMAL (not greater than threshold)
        assert controller.get_mode() == MemoryMode.MINIMAL


class TestCanLoadModel:
    """Tests for can_load_model method."""

    def test_can_load_when_enough_memory(self, controller, mock_monitor):
        """Returns True when sufficient memory available."""
        mock_monitor.get_available_mb.return_value = 5000.0

        # Request 3000MB with 1.2x buffer = 3600MB needed, have 5000MB
        assert controller.can_load_model(3000.0) is True

    def test_cannot_load_when_insufficient_memory(self, controller, mock_monitor):
        """Returns False when insufficient memory available."""
        mock_monitor.get_available_mb.return_value = 3000.0

        # Request 3000MB with 1.2x buffer = 3600MB needed, have 3000MB
        assert controller.can_load_model(3000.0) is False

    def test_respects_buffer_multiplier(self, mock_monitor):
        """Respects the buffer multiplier in threshold config."""
        thresholds = MemoryThresholds(memory_buffer_multiplier=1.5)
        mock_monitor.get_available_mb.return_value = 4000.0
        controller = DefaultMemoryController(monitor=mock_monitor, thresholds=thresholds)

        # Request 3000MB with 1.5x buffer = 4500MB needed, have 4000MB
        assert controller.can_load_model(3000.0) is False

    def test_exact_match_can_load(self, controller, mock_monitor):
        """Can load when available exactly matches required with buffer."""
        mock_monitor.get_available_mb.return_value = 3600.0

        # Request 3000MB with 1.2x buffer = 3600MB needed, have 3600MB
        assert controller.can_load_model(3000.0) is True


class TestRequestMemory:
    """Tests for request_memory method."""

    def test_grants_when_enough_memory(self, controller, mock_monitor):
        """Returns True when memory can be satisfied."""
        mock_monitor.get_available_mb.return_value = 5000.0
        mock_monitor.get_pressure_level.return_value = "green"

        assert controller.request_memory(3000.0, priority=1) is True

    def test_denies_under_critical_pressure(self, controller, mock_monitor):
        """Returns False under critical pressure when insufficient."""
        mock_monitor.get_available_mb.return_value = 1000.0
        mock_monitor.get_pressure_level.return_value = "critical"

        assert controller.request_memory(5000.0, priority=1) is False

    def test_denies_under_red_pressure(self, controller, mock_monitor):
        """Returns False under red pressure when insufficient."""
        mock_monitor.get_available_mb.return_value = 1000.0
        mock_monitor.get_pressure_level.return_value = "red"

        assert controller.request_memory(5000.0, priority=1) is False

    def test_proceeds_under_yellow_pressure(self, controller, mock_monitor):
        """Proceeds (with warning) under yellow pressure if sufficient."""
        mock_monitor.get_available_mb.return_value = 5000.0
        mock_monitor.get_pressure_level.return_value = "yellow"

        assert controller.request_memory(3000.0, priority=1) is True


class TestPressureCallbacks:
    """Tests for pressure callback registration and notification."""

    def test_register_callback(self, controller):
        """Can register a pressure callback."""
        callback = MagicMock()
        controller.register_pressure_callback(callback)

        # Callback should be stored
        assert callback in controller._callbacks

    def test_unregister_callback(self, controller):
        """Can unregister a pressure callback."""
        callback = MagicMock()
        controller.register_pressure_callback(callback)
        controller.unregister_pressure_callback(callback)

        assert callback not in controller._callbacks

    def test_callback_not_registered_twice(self, controller):
        """Same callback not registered twice."""
        callback = MagicMock()
        controller.register_pressure_callback(callback)
        controller.register_pressure_callback(callback)

        assert controller._callbacks.count(callback) == 1

    def test_callback_called_on_pressure_change(self, mock_monitor):
        """Callbacks are called when pressure level changes."""
        controller = DefaultMemoryController(monitor=mock_monitor)
        callback = MagicMock()
        controller.register_pressure_callback(callback)

        # First call sets initial pressure
        mock_monitor.get_pressure_level.return_value = "green"
        mock_monitor.get_system_memory.return_value = SystemMemoryInfo(
            total_mb=16000.0, available_mb=10000.0, used_mb=6000.0, percent_used=37.5
        )
        controller.get_state()

        # Change pressure
        mock_monitor.get_pressure_level.return_value = "yellow"
        controller.get_state()

        callback.assert_called_once_with("yellow")

    def test_callback_not_called_when_pressure_same(self, mock_monitor):
        """Callbacks not called when pressure level unchanged."""
        controller = DefaultMemoryController(monitor=mock_monitor)
        callback = MagicMock()
        controller.register_pressure_callback(callback)

        mock_monitor.get_pressure_level.return_value = "green"
        mock_monitor.get_system_memory.return_value = SystemMemoryInfo(
            total_mb=16000.0, available_mb=10000.0, used_mb=6000.0, percent_used=37.5
        )

        # Multiple calls with same pressure
        controller.get_state()
        controller.get_state()
        controller.get_state()

        # Callback should not be called (same pressure)
        callback.assert_not_called()

    def test_callback_exception_handled(self, mock_monitor):
        """Callback exceptions don't crash the controller."""
        controller = DefaultMemoryController(monitor=mock_monitor)

        def bad_callback(level):
            raise RuntimeError("Callback error")

        controller.register_pressure_callback(bad_callback)

        # First call
        mock_monitor.get_pressure_level.return_value = "green"
        mock_monitor.get_system_memory.return_value = SystemMemoryInfo(
            total_mb=16000.0, available_mb=10000.0, used_mb=6000.0, percent_used=37.5
        )
        controller.get_state()

        # Should not raise when pressure changes
        mock_monitor.get_pressure_level.return_value = "yellow"
        controller.get_state()  # Should not raise


class TestSetModelLoaded:
    """Tests for set_model_loaded method."""

    def test_sets_model_loaded_true(self, controller):
        """Can set model loaded to True."""
        controller.set_model_loaded(True)
        assert controller._model_loaded is True

    def test_sets_model_loaded_false(self, controller):
        """Can set model loaded to False."""
        controller.set_model_loaded(True)
        controller.set_model_loaded(False)
        assert controller._model_loaded is False

    def test_state_reflects_model_loaded(self, controller, mock_monitor):
        """get_state reflects model loaded status."""
        controller.set_model_loaded(True)
        state = controller.get_state()
        assert state.model_loaded is True

        controller.set_model_loaded(False)
        state = controller.get_state()
        assert state.model_loaded is False


class TestSingleton:
    """Tests for singleton controller access."""

    def test_get_memory_controller_returns_instance(self):
        """get_memory_controller returns a controller instance."""
        reset_memory_controller()
        controller = get_memory_controller()

        assert isinstance(controller, DefaultMemoryController)

    def test_get_memory_controller_same_instance(self):
        """get_memory_controller returns same instance."""
        reset_memory_controller()
        controller1 = get_memory_controller()
        controller2 = get_memory_controller()

        assert controller1 is controller2

    def test_reset_memory_controller(self):
        """reset_memory_controller creates new instance."""
        reset_memory_controller()
        controller1 = get_memory_controller()

        reset_memory_controller()
        controller2 = get_memory_controller()

        assert controller1 is not controller2


class TestMemoryModeValues:
    """Tests for MemoryMode enum integration."""

    def test_full_mode_value(self):
        """FULL mode has correct value."""
        assert MemoryMode.FULL.value == "full"

    def test_lite_mode_value(self):
        """LITE mode has correct value."""
        assert MemoryMode.LITE.value == "lite"

    def test_minimal_mode_value(self):
        """MINIMAL mode has correct value."""
        assert MemoryMode.MINIMAL.value == "minimal"


class TestPressureLevels:
    """Tests for pressure level handling."""

    def test_notifies_on_critical_request_denial(self, mock_monitor):
        """Notifies callbacks when request denied under critical pressure."""
        controller = DefaultMemoryController(monitor=mock_monitor)
        callback = MagicMock()
        controller.register_pressure_callback(callback)

        mock_monitor.get_available_mb.return_value = 1000.0
        mock_monitor.get_pressure_level.return_value = "critical"

        controller.request_memory(5000.0, priority=1)

        callback.assert_called_with("critical")

    def test_notifies_on_red_request_denial(self, mock_monitor):
        """Notifies callbacks when request denied under red pressure."""
        controller = DefaultMemoryController(monitor=mock_monitor)
        callback = MagicMock()
        controller.register_pressure_callback(callback)

        mock_monitor.get_available_mb.return_value = 1000.0
        mock_monitor.get_pressure_level.return_value = "red"

        controller.request_memory(5000.0, priority=1)

        callback.assert_called_with("red")
