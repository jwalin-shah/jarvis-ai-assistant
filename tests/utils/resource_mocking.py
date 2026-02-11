"""Mock resource constraints for testing edge cases.

Provides utilities to simulate memory pressure, GPU availability,
and other resource constraints without requiring actual hardware.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from types import TracebackType


class MemoryPressureSimulator:
    """Simulate memory pressure for testing OOM handling.

    Usage:
        with memory_pressure(available_mb=100):
            # Code that should handle low memory gracefully
            result = process_large_batch()
    """

    def __init__(
        self,
        available_mb: float,
        total_mb: float = 8192,
        used_mb: float | None = None,
    ):
        self.available_mb = available_mb
        self.total_mb = total_mb
        self.used_mb = used_mb or (total_mb - available_mb)

    def _create_mock_memory(self) -> MagicMock:
        """Create mock memory info object."""
        mock = MagicMock()
        mock.total = int(self.total_mb * 1024 * 1024)
        mock.available = int(self.available_mb * 1024 * 1024)
        mock.used = int(self.used_mb * 1024 * 1024)
        mock.free = int(self.available_mb * 1024 * 1024)
        mock.percent = 100 * (1 - self.available_mb / self.total_mb)
        mock.buffers = 0
        mock.cached = 0
        mock.shared = 0
        return mock

    @contextmanager
    def apply(self) -> Iterator[MemoryPressureSimulator]:
        """Apply memory pressure simulation."""
        mock_memory = self._create_mock_memory()

        with patch("psutil.virtual_memory", return_value=mock_memory):
            yield self

    def __enter__(self) -> MemoryPressureSimulator:
        """Context manager entry."""
        self._cm = self.apply()
        return self._cm.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self._cm.__exit__(exc_type, exc_val, exc_tb)


class GPUMock:
    """Mock GPU availability and memory for MLX-independent tests."""

    def __init__(
        self,
        is_available: bool = True,
        memory_gb: float = 8.0,
        device_name: str = "Apple M3",
        current_memory_gb: float | None = None,
    ):
        self.is_available = is_available
        self.memory_bytes = int(memory_gb * 1024 * 1024 * 1024)
        self.current_memory_bytes = int((current_memory_gb or memory_gb * 0.5) * 1024 * 1024 * 1024)
        self.device_name = device_name

    @contextmanager
    def mock_mlx(self) -> Iterator[GPUMock]:
        """Mock MLX Metal backend."""
        mock_metal = MagicMock()
        mock_metal.is_available.return_value = self.is_available
        mock_metal.get_active_memory.return_value = self.current_memory_bytes
        mock_metal.get_peak_memory.return_value = self.memory_bytes

        mock_mx = MagicMock()
        mock_mx.metal = mock_metal
        mock_mx.default_device.return_value = MagicMock(__str__=lambda _: self.device_name)

        # Store original if exists
        original_mlx = sys.modules.get("mlx")
        original_mlx_core = sys.modules.get("mlx.core")

        # Create mock modules
        mock_mlx_module = MagicMock()
        mock_mlx_module.core = mock_mx

        sys.modules["mlx"] = mock_mlx_module
        sys.modules["mlx.core"] = mock_mx

        try:
            yield self
        finally:
            # Restore original modules
            if original_mlx:
                sys.modules["mlx"] = original_mlx
            elif "mlx" in sys.modules:
                del sys.modules["mlx"]

            if original_mlx_core:
                sys.modules["mlx.core"] = original_mlx_core
            elif "mlx.core" in sys.modules:
                del sys.modules["mlx.core"]

    def __enter__(self) -> GPUMock:
        """Context manager entry."""
        self._cm = self.mock_mlx()
        return self._cm.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self._cm.__exit__(exc_type, exc_val, exc_tb)


class ResourceBudget:
    """Enforce resource limits during tests.

    Usage:
        with ResourceBudget(max_memory_mb=500):
            result = embedder.encode(large_batch)
            # Raises AssertionError if memory budget exceeded
    """

    def __init__(
        self,
        max_memory_mb: float | None = None,
        max_cpu_percent: float | None = None,
        max_duration_seconds: float | None = None,
    ):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.max_duration_seconds = max_duration_seconds
        self._initial_memory: float = 0.0
        self._start_time: float = 0.0

    def __enter__(self) -> ResourceBudget:
        """Start resource tracking."""
        import time

        try:
            import psutil

            self._process = psutil.Process()
            self._initial_memory = self._process.memory_info().rss / (1024 * 1024)
        except ImportError:
            self._process = None

        self._start_time = time.time()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Check resource usage against budget."""
        import time

        # Check memory
        if self.max_memory_mb and self._process:
            current_memory = self._process.memory_info().rss / (1024 * 1024)
            memory_used = current_memory - self._initial_memory
            if memory_used > self.max_memory_mb:
                raise AssertionError(
                    f"Memory budget exceeded: {memory_used:.1f}MB > {self.max_memory_mb}MB"
                )

        # Check duration
        if self.max_duration_seconds:
            duration = time.time() - self._start_time
            if duration > self.max_duration_seconds:
                raise AssertionError(
                    f"Duration budget exceeded: {duration:.2f}s > {self.max_duration_seconds}s"
                )

    def current_memory_mb(self) -> float:
        """Get current memory usage."""
        if self._process:
            return self._process.memory_info().rss / (1024 * 1024)
        return 0.0

    def memory_used_mb(self) -> float:
        """Get memory used since budget start."""
        return self.current_memory_mb() - self._initial_memory


@contextmanager
def low_memory(available_mb: float) -> Iterator[MemoryPressureSimulator]:
    """Convenience context manager for low memory simulation.

    Example:
        with low_memory(100):  # Only 100MB "available"
            # Test OOM handling
            with pytest.raises(MemoryError):
                allocate_large_buffer()
    """
    simulator = MemoryPressureSimulator(available_mb=available_mb)
    with simulator.apply():
        yield simulator


@contextmanager
def no_gpu() -> Iterator[GPUMock]:
    """Convenience context manager to simulate no GPU.

    Example:
        with no_gpu():
            # Test CPU fallback
            result = embedder.encode(text)
    """
    mock = GPUMock(is_available=False, memory_gb=0)
    with mock.mock_mlx():
        yield mock
