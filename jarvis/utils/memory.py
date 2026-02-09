"""Memory and swap monitoring utilities for 8GB RAM constraint.

This module provides real-time tracking of memory usage and swap activity,
which is critical on memory-constrained systems where swapping causes 10-100x slowdowns.

On macOS, this module tracks:
- Memory pressure (0 = good, >50 = warning, >100 = critical)
- Compressed memory (RAM compression before swapping)
- Page-outs (actual swap activity)

macOS aggressively preemptive-swaps cold pages even with free RAM available,
so "swap used" alone is NOT a good indicator. Memory pressure + page-outs are the truth.
"""

import logging
import platform
import re
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)

IS_MACOS = platform.system() == "Darwin"


class SwapThresholdExceeded(Exception):
    """Raised when swap usage exceeds configured threshold.

    Note: Does not inherit from JarvisError to avoid circular import
    (this module is imported early in the boot sequence).
    """

    pass


@dataclass
class MacOSMemoryPressure:
    """macOS-specific memory pressure metrics."""

    pressure_level: int  # 0 = good, >50 = warning, >100 = critical
    compressed_mb: float  # Memory compressed in RAM (not swapped)
    pageouts: int  # Total pages swapped out since boot
    pageins: int  # Total pages swapped in since boot
    compressions: int  # Total compressions since boot
    decompressions: int  # Total decompressions since boot
    free_mb: float  # Actual free RAM
    compressed_ratio: float  # Compression ratio (compressed/compressor)

    def __str__(self) -> str:
        status = (
            "GOOD"
            if self.pressure_level == 0
            else ("WARNING" if self.pressure_level < 100 else "CRITICAL")
        )
        return (
            f"Pressure: {self.pressure_level} ({status}), "
            f"Compressed: {self.compressed_mb:.1f}MB (ratio {self.compressed_ratio:.1f}x), "
            f"Free: {self.free_mb:.1f}MB"
        )


@dataclass
class MemoryInfo:
    """Memory usage snapshot."""

    rss_mb: float  # Resident Set Size (actual physical RAM used)
    vms_mb: float  # Virtual Memory Size (allocated address space)
    percent: float  # Percentage of total system memory
    swap_used_mb: float  # System swap used (preemptive on macOS!)
    swap_percent: float  # Percentage of total swap
    timestamp: float
    macos_pressure: MacOSMemoryPressure | None = None  # macOS-specific metrics

    def __str__(self) -> str:
        base = (
            f"RAM: {self.rss_mb:.1f}MB ({self.percent:.1f}%), "
            f"VMS: {self.vms_mb:.1f}MB, "
            f"Swap: {self.swap_used_mb:.1f}MB ({self.swap_percent:.1f}%)"
        )
        if self.macos_pressure:
            base += f" | {self.macos_pressure}"
        return base


def get_macos_memory_pressure() -> MacOSMemoryPressure | None:
    """Get macOS memory pressure metrics from vm_stat and sysctl.

    Returns None on non-macOS systems.
    """
    if not IS_MACOS:
        return None

    try:
        # Parse vm_stat
        vm_stat = subprocess.check_output(["vm_stat"], text=True)
        stats = {}
        for line in vm_stat.splitlines():
            match = re.match(r'"?([^"]+)"?:\s+(\d+)', line)
            if match:
                key, value = match.groups()
                stats[key] = int(value)

        # Get page size (usually 16KB on Apple Silicon)
        page_size = stats.get("page size of", 16384)
        if "page size of" in stats:
            # Extract from "page size of 16384 bytes"
            for line in vm_stat.splitlines():
                if "page size" in line:
                    match = re.search(r"(\d+)\s+bytes", line)
                    if match:
                        page_size = int(match.group(1))
                        break

        # Get sysctl metrics
        sysctl = subprocess.check_output(["sysctl", "vm.memory_pressure"], text=True).strip()
        pressure_level = int(sysctl.split(":")[-1].strip())

        # Calculate metrics
        pages_compressed = stats.get("Pages stored in compressor", 0)
        pages_occupied = stats.get("Pages occupied by compressor", 0)
        pages_free = stats.get("Pages free", 0)

        compressed_mb = (pages_compressed * page_size) / 1024**2
        free_mb = (pages_free * page_size) / 1024**2
        compression_ratio = pages_compressed / pages_occupied if pages_occupied > 0 else 1.0

        return MacOSMemoryPressure(
            pressure_level=pressure_level,
            compressed_mb=compressed_mb,
            pageouts=stats.get("Pageouts", 0),
            pageins=stats.get("Pageins", 0),
            compressions=stats.get("Compressions", 0),
            decompressions=stats.get("Decompressions", 0),
            free_mb=free_mb,
            compressed_ratio=compression_ratio,
        )
    except (subprocess.CalledProcessError, KeyError, ValueError) as e:
        logger.debug(f"Failed to get macOS memory pressure: {e}")
        return None


def get_memory_info() -> MemoryInfo:
    """Get current memory usage for this process and system swap."""
    process = psutil.Process()
    mem = process.memory_info()
    swap = psutil.swap_memory()

    macos_pressure = get_macos_memory_pressure() if IS_MACOS else None

    return MemoryInfo(
        rss_mb=mem.rss / 1024**2,
        vms_mb=mem.vms / 1024**2,
        percent=process.memory_percent(),
        swap_used_mb=swap.used / 1024**2,
        swap_percent=swap.percent,
        timestamp=time.time(),
        macos_pressure=macos_pressure,
    )


def get_swap_info() -> dict[str, float]:
    """Get detailed swap information.

    Returns:
        Dict with keys: total_mb, used_mb, free_mb, percent
    """
    swap = psutil.swap_memory()
    return {
        "total_mb": swap.total / 1024**2,
        "used_mb": swap.used / 1024**2,
        "free_mb": swap.free / 1024**2,
        "percent": swap.percent,
    }


def log_memory_snapshot(label: str = "") -> None:
    """Log current memory state with optional label."""
    info = get_memory_info()
    prefix = f"[{label}] " if label else ""
    logger.info(f"{prefix}{info}")


@contextmanager
def track_memory(
    label: str = "",
    *,
    interval_sec: float = 5.0,
    swap_threshold_mb: float = 500.0,
    abort_on_threshold: bool = False,
) -> Iterator[list[MemoryInfo]]:
    """Context manager that tracks memory usage during an operation.

    Args:
        label: Optional label for logging
        interval_sec: How often to sample memory (0 = only start/end)
        swap_threshold_mb: Warn if swap exceeds this (default 500MB)
        abort_on_threshold: If True, raise SwapThresholdExceeded on threshold breach

    Yields:
        List that will be populated with MemoryInfo snapshots

    Example:
        with track_memory("training", interval_sec=10, swap_threshold_mb=500) as snapshots:
            train_model()
        # snapshots now contains periodic memory measurements
    """
    snapshots: list[MemoryInfo] = []
    start_info = get_memory_info()
    snapshots.append(start_info)

    prefix = f"[{label}] " if label else ""
    logger.info(f"{prefix}Memory tracking started: {start_info}")

    # Background monitoring if interval > 0
    monitor_active = interval_sec > 0
    last_check = time.time()

    try:
        yield snapshots
    finally:
        end_info = get_memory_info()
        snapshots.append(end_info)

        # Calculate deltas
        rss_delta = end_info.rss_mb - start_info.rss_mb
        swap_delta = end_info.swap_used_mb - start_info.swap_used_mb

        logger.info(
            f"{prefix}Memory tracking finished: {end_info}\n"
            f"  RSS delta: {rss_delta:+.1f}MB, Swap delta: {swap_delta:+.1f}MB"
        )


class MemoryMonitor:
    """Periodic memory monitor that logs swap usage.

    Use this for long-running operations like GridSearchCV where you want
    continuous monitoring without blocking the main thread.

    Example:
        monitor = MemoryMonitor(interval_sec=10, swap_threshold_mb=500)
        monitor.start("GridSearch training")

        # Do work...
        search.fit(X, y)

        monitor.stop()
        print(f"Peak swap: {monitor.peak_swap_mb:.1f}MB")
    """

    def __init__(
        self,
        interval_sec: float = 10.0,
        swap_threshold_mb: float = 500.0,
        log_level: int = logging.INFO,
    ):
        self.interval_sec = interval_sec
        self.swap_threshold_mb = swap_threshold_mb
        self.log_level = log_level

        self.snapshots: list[MemoryInfo] = []
        self.label: str = ""
        self.start_time: float = 0.0
        self.last_check: float = 0.0

        self._monitoring = False

    def start(self, label: str = "") -> None:
        """Start monitoring."""
        self.label = label
        self.start_time = time.time()
        self.last_check = self.start_time
        self._monitoring = True

        info = get_memory_info()
        self.snapshots = [info]

        prefix = f"[{self.label}] " if self.label else ""
        logger.log(self.log_level, f"{prefix}Memory monitor started: {info}")

    def check(self) -> MemoryInfo | None:
        """Check memory if interval has elapsed. Returns info if checked, None otherwise."""
        if not self._monitoring:
            return None

        elapsed = time.time() - self.last_check
        if elapsed < self.interval_sec:
            return None

        info = get_memory_info()
        self.snapshots.append(info)
        self.last_check = time.time()

        prefix = f"[{self.label}] " if self.label else ""
        log_msg = f"{prefix}Memory: {info}"

        # Track page-outs rate (actual swap activity)
        if IS_MACOS and len(self.snapshots) > 1 and info.macos_pressure:
            prev = self.snapshots[-2]
            if prev.macos_pressure:
                pageouts_delta = info.macos_pressure.pageouts - prev.macos_pressure.pageouts
                if pageouts_delta > 0:
                    time_delta = info.timestamp - prev.timestamp
                    pageouts_per_sec = pageouts_delta / time_delta if time_delta > 0 else 0
                    log_msg += f" | SWAPPING: {pageouts_delta} pages ({pageouts_per_sec:.1f}/s)"

        logger.log(self.log_level, log_msg)

        # Warn on memory pressure (more accurate than swap threshold on macOS)
        if IS_MACOS and info.macos_pressure:
            if info.macos_pressure.pressure_level >= 50:
                logger.warning(
                    f"{prefix}MEMORY PRESSURE: {info.macos_pressure.pressure_level} "
                    f"(compressed: {info.macos_pressure.compressed_mb:.1f}MB)"
                )
        elif info.swap_used_mb > self.swap_threshold_mb:
            logger.warning(
                f"{prefix}SWAP THRESHOLD EXCEEDED: "
                f"{info.swap_used_mb:.1f}MB > {self.swap_threshold_mb}MB"
            )

        return info

    def stop(self) -> MemoryInfo:
        """Stop monitoring and return final snapshot."""
        self._monitoring = False

        info = get_memory_info()
        self.snapshots.append(info)

        start_info = self.snapshots[0]
        rss_delta = info.rss_mb - start_info.rss_mb
        swap_delta = info.swap_used_mb - start_info.swap_used_mb
        duration = time.time() - self.start_time

        prefix = f"[{self.label}] " if self.label else ""
        logger.log(
            self.log_level,
            f"{prefix}Memory monitor stopped after {duration:.1f}s: {info}\n"
            f"  RSS delta: {rss_delta:+.1f}MB, Swap delta: {swap_delta:+.1f}MB",
        )

        return info

    @property
    def peak_swap_mb(self) -> float:
        """Get peak swap usage observed."""
        if not self.snapshots:
            return 0.0
        return max(s.swap_used_mb for s in self.snapshots)

    @property
    def peak_rss_mb(self) -> float:
        """Get peak RSS usage observed."""
        if not self.snapshots:
            return 0.0
        return max(s.rss_mb for s in self.snapshots)


def get_top_memory_processes(limit: int = 5) -> list[dict[str, str | float]]:
    """Get top memory-consuming processes.

    Useful for identifying what's causing swap activity.

    Returns:
        List of dicts with keys: pid, name, rss_mb, vms_mb, percent
    """
    processes = []
    for proc in psutil.process_iter(["pid", "name", "memory_info", "memory_percent"]):
        try:
            info = proc.info
            mem = info.get("memory_info")
            if mem is None:
                continue

            processes.append(
                {
                    "pid": info["pid"],
                    "name": info["name"] or "unknown",
                    "rss_mb": mem.rss / 1024**2,
                    "vms_mb": mem.vms / 1024**2,
                    "percent": info.get("memory_percent", 0.0) or 0.0,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            pass

    # Sort by RSS descending
    processes.sort(key=lambda x: x["rss_mb"], reverse=True)
    return processes[:limit]
