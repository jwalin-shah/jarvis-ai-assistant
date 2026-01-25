"""Memory controller implementing the MemoryController protocol.

Provides adaptive memory management for JARVIS, enabling operation
across different memory configurations (8GB, 16GB, etc.).
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

from contracts.memory import MemoryMode, MemoryState
from core.memory.monitor import MemoryMonitor

logger = logging.getLogger(__name__)


@dataclass
class MemoryThresholds:
    """Configuration for memory mode thresholds.

    Attributes:
        full_mode_mb: Minimum available MB for FULL mode (16GB+ systems)
        lite_mode_mb: Minimum available MB for LITE mode (8-16GB systems)
        memory_buffer_multiplier: Safety buffer when checking if model can load
    """

    full_mode_mb: float = 8000.0
    lite_mode_mb: float = 4000.0
    memory_buffer_multiplier: float = 1.2


class DefaultMemoryController:
    """Memory controller for adaptive operation across memory configurations.

    Implements the MemoryController protocol to provide:
    - Memory state monitoring
    - Mode determination (FULL/LITE/MINIMAL)
    - Model loading decisions
    - Pressure callback registration

    This controller does NOT load/unload models directly. It only
    provides state and recommendations for memory management.

    Thread-safe: Uses locks for callback registration.
    """

    def __init__(
        self,
        monitor: MemoryMonitor | None = None,
        thresholds: MemoryThresholds | None = None,
        model_loaded: bool = False,
    ) -> None:
        """Initialize the memory controller.

        Args:
            monitor: Memory monitor instance. Creates default if not provided.
            thresholds: Mode thresholds. Uses defaults if not provided.
            model_loaded: Initial model loaded state.
        """
        self._monitor = monitor or MemoryMonitor()
        self._thresholds = thresholds or MemoryThresholds()
        self._model_loaded = model_loaded
        self._callbacks: list[Callable[[str], None]] = []
        self._callback_lock = threading.Lock()
        self._last_pressure_level: str | None = None

    def get_state(self) -> MemoryState:
        """Get current memory state.

        Returns:
            MemoryState with current memory information.
        """
        info = self._monitor.get_system_memory()
        pressure_level = self._monitor.get_pressure_level()

        # Notify callbacks if pressure level changed
        self._check_pressure_change(pressure_level)

        return MemoryState(
            available_mb=info.available_mb,
            used_mb=info.used_mb,
            model_loaded=self._model_loaded,
            current_mode=self.get_mode(),
            pressure_level=pressure_level,
        )

    def get_mode(self) -> MemoryMode:
        """Determine appropriate mode based on available memory.

        Mode thresholds:
        - FULL: available_mb > 8000 (8GB+ available)
        - LITE: available_mb > 4000 (4-8GB available)
        - MINIMAL: available_mb <= 4000 (<4GB available)

        Returns:
            MemoryMode for current memory state.
        """
        available_mb = self._monitor.get_available_mb()

        if available_mb > self._thresholds.full_mode_mb:
            return MemoryMode.FULL
        elif available_mb > self._thresholds.lite_mode_mb:
            return MemoryMode.LITE
        else:
            return MemoryMode.MINIMAL

    def can_load_model(self, required_mb: float) -> bool:
        """Check if we have enough memory to load a model.

        Uses a buffer multiplier to ensure safe loading without
        causing memory pressure.

        Args:
            required_mb: Estimated memory requirement for the model.

        Returns:
            True if there's enough memory to safely load the model.
        """
        available_mb = self._monitor.get_available_mb()
        required_with_buffer = required_mb * self._thresholds.memory_buffer_multiplier

        can_load = available_mb >= required_with_buffer
        if not can_load:
            logger.warning(
                "Insufficient memory for model load: %.0fMB available, %.0fMB required "
                "(%.0fMB with buffer)",
                available_mb,
                required_mb,
                required_with_buffer,
            )

        return can_load

    def request_memory(self, required_mb: float, priority: int) -> bool:
        """Request memory, potentially signaling to unload lower-priority components.

        This method checks if memory can be satisfied and notifies pressure
        callbacks if memory is tight. It does NOT directly unload anything.

        Args:
            required_mb: Amount of memory needed.
            priority: Priority level (higher = more important).

        Returns:
            True if memory request can be satisfied, False otherwise.
        """
        available_mb = self._monitor.get_available_mb()
        pressure = self._monitor.get_pressure_level()

        # If we have enough memory, grant the request
        if available_mb >= required_mb:
            logger.debug(
                "Memory request granted: %.0fMB requested, %.0fMB available",
                required_mb,
                available_mb,
            )
            return True

        # If in critical pressure and can't satisfy, notify callbacks
        if pressure in ("red", "critical"):
            logger.warning(
                "Memory request cannot be satisfied: %.0fMB requested, %.0fMB available, "
                "pressure=%s, priority=%d",
                required_mb,
                available_mb,
                pressure,
                priority,
            )
            self._notify_callbacks(pressure)
            return False

        # Yellow pressure - try to proceed but warn
        if pressure == "yellow":
            logger.info(
                "Memory request proceeding under yellow pressure: %.0fMB requested, "
                "%.0fMB available, priority=%d",
                required_mb,
                available_mb,
                priority,
            )

        return available_mb >= required_mb

    def register_pressure_callback(self, callback: Callable[[str], None]) -> None:
        """Register callback for memory pressure events.

        Callbacks are invoked when pressure level changes or when
        memory requests cannot be satisfied under pressure.

        Args:
            callback: Function to call with pressure level string.
        """
        with self._callback_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
                logger.debug("Registered pressure callback: %s", callback)

    def unregister_pressure_callback(self, callback: Callable[[str], None]) -> None:
        """Unregister a previously registered pressure callback.

        Args:
            callback: Function to remove from callback list.
        """
        with self._callback_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
                logger.debug("Unregistered pressure callback: %s", callback)

    def set_model_loaded(self, loaded: bool) -> None:
        """Update model loaded state.

        Called by external components when model load/unload completes.

        Args:
            loaded: True if a model is now loaded, False otherwise.
        """
        self._model_loaded = loaded
        logger.debug("Model loaded state updated: %s", loaded)

    def _check_pressure_change(self, current_pressure: str) -> None:
        """Check if pressure level changed and notify callbacks if so.

        Args:
            current_pressure: Current pressure level string.
        """
        if self._last_pressure_level != current_pressure:
            old_pressure = self._last_pressure_level
            self._last_pressure_level = current_pressure

            if old_pressure is not None:
                logger.info(
                    "Memory pressure changed: %s -> %s",
                    old_pressure,
                    current_pressure,
                )
                self._notify_callbacks(current_pressure)

    def _notify_callbacks(self, pressure_level: str) -> None:
        """Notify all registered callbacks of pressure level.

        Args:
            pressure_level: Current pressure level string.
        """
        with self._callback_lock:
            callbacks = list(self._callbacks)

        for callback in callbacks:
            try:
                callback(pressure_level)
            except Exception:
                logger.exception("Error in pressure callback: %s", callback)


# Singleton instance with thread-safe initialization
_controller: DefaultMemoryController | None = None
_controller_lock = threading.Lock()


def get_memory_controller() -> DefaultMemoryController:
    """Get or create singleton memory controller instance.

    Thread-safe using double-check locking pattern.

    Returns:
        The shared DefaultMemoryController instance.
    """
    global _controller
    if _controller is None:
        with _controller_lock:
            # Double-check after acquiring lock
            if _controller is None:
                _controller = DefaultMemoryController()
    return _controller


def reset_memory_controller() -> None:
    """Reset the singleton memory controller instance.

    Use this to:
    - Clear state between tests
    - Force reinitialization of the controller
    """
    global _controller
    with _controller_lock:
        _controller = None
