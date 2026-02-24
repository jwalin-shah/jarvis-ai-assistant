"""High-precision timing utilities for latency benchmarking.

Workstream 4: Latency Benchmark

Provides nanosecond-precision timing using time.perf_counter_ns()
for accurate measurement of model operations.
"""

import gc
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")

# MLX imports for GPU cache clearing
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@dataclass
class TimingResult:
    """Result of a timed operation."""

    elapsed_ns: int
    elapsed_ms: float

    @classmethod
    def from_ns(cls, elapsed_ns: int) -> "TimingResult":
        """Create TimingResult from nanosecond measurement."""
        return cls(elapsed_ns=elapsed_ns, elapsed_ms=elapsed_ns / 1_000_000)


class HighPrecisionTimer:
    """High-precision timer using time.perf_counter_ns().

    Provides nanosecond-level precision for timing model operations.
    """

    def __init__(self) -> None:
        """Initialize the timer."""
        self._start_ns: int | None = None
        self._end_ns: int | None = None

    def start(self) -> None:
        """Start the timer."""
        self._start_ns = time.perf_counter_ns()
        self._end_ns = None

    def stop(self) -> TimingResult:
        """Stop the timer and return elapsed time.

        Returns:
            TimingResult with elapsed time in nanoseconds and milliseconds.

        Raises:
            RuntimeError: If timer was not started.
        """
        if self._start_ns is None:
            msg = "Timer was not started. Call start() first."
            raise RuntimeError(msg)

        self._end_ns = time.perf_counter_ns()
        elapsed_ns = self._end_ns - self._start_ns
        return TimingResult.from_ns(elapsed_ns)

    def reset(self) -> None:
        """Reset the timer to initial state."""
        self._start_ns = None
        self._end_ns = None

    @property
    def elapsed_ns(self) -> int | None:
        """Return elapsed nanoseconds, or None if timer hasn't been stopped."""
        if self._start_ns is None or self._end_ns is None:
            return None
        return self._end_ns - self._start_ns

    @property
    def elapsed_ms(self) -> float | None:
        """Return elapsed milliseconds, or None if timer hasn't been stopped."""
        ns = self.elapsed_ns
        return None if ns is None else ns / 1_000_000


@contextmanager
def timed_operation() -> Generator[HighPrecisionTimer, None, None]:
    """Context manager for timing an operation.

    Example:
        with timed_operation() as timer:
            do_something()
        print(f"Took {timer.elapsed_ms:.2f}ms")

    Yields:
        HighPrecisionTimer instance with timing results.
    """
    timer = HighPrecisionTimer()
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


def measure_operation(func: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, TimingResult]:
    """Measure the execution time of a function.

    Args:
        func: Function to measure.
        *args: Positional arguments to pass to function.
        **kwargs: Keyword arguments to pass to function.

    Returns:
        Tuple of (function result, TimingResult).
    """
    timer = HighPrecisionTimer()
    timer.start()
    result = func(*args, **kwargs)
    timing = timer.stop()
    return result, timing


def force_model_unload() -> None:
    """Force complete model unload including GPU memory.

    Performs:
    1. Python garbage collection
    2. MLX Metal GPU cache clear (if available)
    3. Additional GC pass for thorough cleanup
    """
    # First GC pass
    gc.collect()

    # Clear MLX Metal GPU cache if available
    if HAS_MLX:
        try:
            mx.metal.clear_cache()
        except Exception:
            # Metal cache clear may not be available on all systems
            pass

    # Second GC pass for thorough cleanup
    gc.collect()


def warmup_timer() -> None:
    """Perform timer warmup to minimize JIT compilation effects.

    Runs several timing operations to warm up Python's perf_counter_ns.
    """
    for _ in range(10):
        timer = HighPrecisionTimer()
        timer.start()
        time.sleep(0.0001)  # 100 microseconds
        timer.stop()
