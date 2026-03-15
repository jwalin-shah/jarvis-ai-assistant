"""High-precision timing utilities for latency benchmarking.  # noqa: E501
  # noqa: E501
Workstream 4: Latency Benchmark  # noqa: E501
  # noqa: E501
Provides nanosecond-precision timing using time.perf_counter_ns()  # noqa: E501
for accurate measurement of model operations.  # noqa: E501
"""  # noqa: E501
  # noqa: E501
import gc  # noqa: E501
import time  # noqa: E501
from collections.abc import Callable, Generator  # noqa: E402  # noqa: E501
from contextlib import contextmanager  # noqa: E402  # noqa: E501
from dataclasses import dataclass  # noqa: E402  # noqa: E501
from typing import Any, TypeVar  # noqa: E402  # noqa: E501

  # noqa: E501
T = TypeVar("T")  # noqa: E501
  # noqa: E501
# MLX imports for GPU cache clearing  # noqa: E501
try:  # noqa: E501
    import mlx.core as mx  # noqa: E501
  # noqa: E501
    HAS_MLX = True  # noqa: E501
except ImportError:  # noqa: E501
    HAS_MLX = False  # noqa: E501
  # noqa: E501
  # noqa: E501
@dataclass  # noqa: E501
class TimingResult:  # noqa: E501
    """Result of a timed operation."""  # noqa: E501
  # noqa: E501
    elapsed_ns: int  # noqa: E501
    elapsed_ms: float  # noqa: E501
  # noqa: E501
    @classmethod  # noqa: E501
    def from_ns(cls, elapsed_ns: int) -> "TimingResult":  # noqa: E501
        """Create TimingResult from nanosecond measurement."""  # noqa: E501
        return cls(elapsed_ns=elapsed_ns, elapsed_ms=elapsed_ns / 1_000_000)  # noqa: E501
  # noqa: E501
  # noqa: E501
class HighPrecisionTimer:  # noqa: E501
    """High-precision timer using time.perf_counter_ns().  # noqa: E501
  # noqa: E501
    Provides nanosecond-level precision for timing model operations.  # noqa: E501
    """  # noqa: E501
  # noqa: E501
    def __init__(self) -> None:  # noqa: E501
        """Initialize the timer."""  # noqa: E501
        self._start_ns: int | None = None  # noqa: E501
        self._end_ns: int | None = None  # noqa: E501
  # noqa: E501
    def start(self) -> None:  # noqa: E501
        """Start the timer."""  # noqa: E501
        self._start_ns = time.perf_counter_ns()  # noqa: E501
        self._end_ns = None  # noqa: E501
  # noqa: E501
    def stop(self) -> TimingResult:  # noqa: E501
        """Stop the timer and return elapsed time.  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            TimingResult with elapsed time in nanoseconds and milliseconds.  # noqa: E501
  # noqa: E501
        Raises:  # noqa: E501
            RuntimeError: If timer was not started.  # noqa: E501
        """  # noqa: E501
        if self._start_ns is None:  # noqa: E501
            msg = "Timer was not started. Call start() first."  # noqa: E501
            raise RuntimeError(msg)  # noqa: E501
  # noqa: E501
        self._end_ns = time.perf_counter_ns()  # noqa: E501
        elapsed_ns = self._end_ns - self._start_ns  # noqa: E501
        return TimingResult.from_ns(elapsed_ns)  # noqa: E501
  # noqa: E501
    def reset(self) -> None:  # noqa: E501
        """Reset the timer to initial state."""  # noqa: E501
        self._start_ns = None  # noqa: E501
        self._end_ns = None  # noqa: E501
  # noqa: E501
    @property  # noqa: E501
    def elapsed_ns(self) -> int | None:  # noqa: E501
        """Return elapsed nanoseconds, or None if timer hasn't been stopped."""  # noqa: E501
        if self._start_ns is None or self._end_ns is None:  # noqa: E501
            return None  # noqa: E501
        return self._end_ns - self._start_ns  # noqa: E501
  # noqa: E501
    @property  # noqa: E501
    def elapsed_ms(self) -> float | None:  # noqa: E501
        """Return elapsed milliseconds, or None if timer hasn't been stopped."""  # noqa: E501
        ns = self.elapsed_ns  # noqa: E501
        return None if ns is None else ns / 1_000_000  # noqa: E501
  # noqa: E501
  # noqa: E501
@contextmanager  # noqa: E501
def timed_operation() -> Generator[HighPrecisionTimer, None, None]:  # noqa: E501
    """Context manager for timing an operation.  # noqa: E501
  # noqa: E501
    Example:  # noqa: E501
        with timed_operation() as timer:  # noqa: E501
            do_something()  # noqa: E501
        print(f"Took {timer.elapsed_ms:.2f}ms")  # noqa: E501
  # noqa: E501
    Yields:  # noqa: E501
        HighPrecisionTimer instance with timing results.  # noqa: E501
    """  # noqa: E501
    timer = HighPrecisionTimer()  # noqa: E501
    timer.start()  # noqa: E501
    try:  # noqa: E501
        yield timer  # noqa: E501
    finally:  # noqa: E501
        timer.stop()  # noqa: E501
  # noqa: E501
  # noqa: E501
def measure_operation(func: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, TimingResult]:  # noqa: E501
    """Measure the execution time of a function.  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        func: Function to measure.  # noqa: E501
        *args: Positional arguments to pass to function.  # noqa: E501
        **kwargs: Keyword arguments to pass to function.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        Tuple of (function result, TimingResult).  # noqa: E501
    """  # noqa: E501
    timer = HighPrecisionTimer()  # noqa: E501
    timer.start()  # noqa: E501
    result = func(*args, **kwargs)  # noqa: E501
    timing = timer.stop()  # noqa: E501
    return result, timing  # noqa: E501
  # noqa: E501
  # noqa: E501
def force_model_unload() -> None:  # noqa: E501
    """Force complete model unload including GPU memory.  # noqa: E501
  # noqa: E501
    Performs:  # noqa: E501
    1. Python garbage collection  # noqa: E501
    2. MLX Metal GPU cache clear (if available)  # noqa: E501
    3. Additional GC pass for thorough cleanup  # noqa: E501
    """  # noqa: E501
    # First GC pass  # noqa: E501
    gc.collect()  # noqa: E501
  # noqa: E501
    # Clear MLX Metal GPU cache if available  # noqa: E501
    if HAS_MLX:  # noqa: E501
        try:  # noqa: E501
            mx.metal.clear_cache()  # noqa: E501
        except Exception:  # noqa: E501
            # Metal cache clear may not be available on all systems  # noqa: E501
            pass  # noqa: E501
  # noqa: E501
    # Second GC pass for thorough cleanup  # noqa: E501
    gc.collect()  # noqa: E501
  # noqa: E501
  # noqa: E501
def warmup_timer() -> None:  # noqa: E501
    """Perform timer warmup to minimize JIT compilation effects.  # noqa: E501
  # noqa: E501
    Runs several timing operations to warm up Python's perf_counter_ns.  # noqa: E501
    """  # noqa: E501
    for _ in range(10):  # noqa: E501
        timer = HighPrecisionTimer()  # noqa: E501
        timer.start()  # noqa: E501
        time.sleep(0.0001)  # 100 microseconds  # noqa: E501
        timer.stop()  # noqa: E501
