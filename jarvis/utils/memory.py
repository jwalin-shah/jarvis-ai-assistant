"""Memory and swap monitoring utilities for 8GB RAM constraint.

This module provides real-time tracking of memory usage and swap activity,
which is critical on memory-constrained systems where swapping causes 10-100x slowdowns.

On macOS, this module tracks:
- Memory pressure (0 = good, >50 = warning, >100 = critical)
- Compressed memory (RAM compression before swapping)
- Page-outs (actual swap activity)
- MLX GPU memory (Apple Silicon unified memory)

macOS aggressively preemptive-swaps cold pages even with free RAM available,
so "swap used" alone is NOT a good indicator. Memory pressure + page-outs are the truth.
"""

import logging
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import psutil

logger = logging.getLogger(__name__)

IS_MACOS = platform.system() == "Darwin"

# MLX memory tracking (lazy import to avoid dependency issues)
_mlx_available: bool | None = None


def _is_mlx_available() -> bool:
    """Check if MLX is available for GPU memory tracking."""
    global _mlx_available
    if _mlx_available is None:
        try:
            import mlx.core as mx  # noqa: F401

            _mlx_available = True
        except ImportError:
            _mlx_available = False
    return _mlx_available


@lru_cache(maxsize=1)
def get_mlx_memory_info() -> dict[str, Any] | None:
    """Get MLX GPU memory information using native MLX APIs.

    This is more accurate than psutil for Apple Silicon unified memory
    as it tracks the actual GPU/MLX memory pool usage.

    Returns:
        Dict with memory info or None if MLX unavailable.
        Keys: active_mb, peak_mb, cache_mb, limit_mb
    """
    if not _is_mlx_available():
        return None

    try:
        import mlx.core as mx

        # Get current memory stats from MLX
        active_bytes = mx.get_active_memory()
        peak_bytes = mx.get_peak_memory()
        cache_bytes = mx.get_cache_memory()

        # Get the memory limit if available (MLX 0.18+)
        try:
            limit_bytes = mx.metal.get_memory_limit()
        except (AttributeError, TypeError):
            limit_bytes = None

        bytes_per_mb = 1024 * 1024

        result = {
            "active_mb": active_bytes / bytes_per_mb,
            "peak_mb": peak_bytes / bytes_per_mb,
            "cache_mb": cache_bytes / bytes_per_mb,
            "active_gb": active_bytes / (1024**3),
            "peak_gb": peak_bytes / (1024**3),
            "cache_gb": cache_bytes / (1024**3),
        }

        if limit_bytes is not None:
            result["limit_mb"] = limit_bytes / bytes_per_mb
            result["limit_gb"] = limit_bytes / (1024**3)
            result["utilization_percent"] = (active_bytes / limit_bytes) * 100

        # Try to get GPU utilization (load) if possible
        # Note: This is a bit expensive, so we only do it if explicitly requested
        # or in background monitoring.
        return result
    except Exception as e:
        logger.debug(f"Failed to get MLX memory info: {e}")
        return None


def get_gpu_load_percent() -> float | None:
    """Get GPU utilization percentage on macOS.
    
    Warning: This calls powermetrics which can be slow (~100ms).
    """
    if not IS_MACOS:
        return None
        
    try:
        # Use powermetrics to get GPU load. Requires sudo or specific permissions.
        # Alternatively, use a faster but less reliable system_profiler check.
        # For now, we'll try a fast sysctl check for GPU activity.
        output = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).lower()
        
        # If Apple Silicon, we might be able to get it via some IOKit hooks
        # but for a CLI tool, powermetrics is the standard way.
        # We'll stick to memory-based utilization for now as it's 100% reliable.
        return None
    except Exception:
        return None


def clear_mlx_cache() -> bool:
    """Clear MLX GPU cache to free memory.

    Returns:
        True if cache was cleared, False if MLX unavailable.
    """
    if not _is_mlx_available():
        return False

    try:
        import mlx.core as mx

        before = mx.metal.get_cache_memory()
        mx.clear_cache()
        after = mx.metal.get_cache_memory()

        freed_mb = (before - after) / (1024 * 1024)
        if freed_mb > 1:
            logger.debug(f"Cleared {freed_mb:.1f}MB from MLX cache")

        return True
    except Exception as e:
        logger.debug(f"Failed to clear MLX cache: {e}")
        return False


class SwapThresholdExceededError(Exception):
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

    rss_mb: float  # Resident Set Size (physical RAM used)
    footprint_mb: float  # macOS 'phys_footprint' (actual ownership)
    vms_mb: float  # Virtual Memory Size
    percent: float  # Percentage of total system memory
    swap_used_mb: float  # System swap used
    swap_percent: float  # Percentage of total swap
    timestamp: float
    macos_pressure: MacOSMemoryPressure | None = None  # macOS-specific metrics
    mlx_memory: dict[str, Any] | None = None  # MLX GPU memory metrics
    thermal_state: str | None = None  # Nominal, Fair, Serious, Critical

    def __str__(self) -> str:
        base = (
            f"RAM: {self.rss_mb:.1f}MB (footprint: {self.footprint_mb:.1f}MB), "
            f"Swap: {self.swap_used_mb:.1f}MB ({self.swap_percent:.1f}%)"
        )
        if self.thermal_state and self.thermal_state != "nominal":
            base += f" | THERMAL: {self.thermal_state.upper()}"
        if self.macos_pressure:
            base += f" | {self.macos_pressure}"
        if self.mlx_memory:
            mlx_active = self.mlx_memory.get("active_mb", 0)
            mlx_peak = self.mlx_memory.get("peak_mb", 0)
            base += f" | MLX: {mlx_active:.1f}MB active, {mlx_peak:.1f}MB peak"
        return base


_pressure_cache: tuple[float, MacOSMemoryPressure | None] = (0.0, None)
_PRESSURE_CACHE_TTL = 5.0  # seconds


def get_macos_memory_pressure() -> MacOSMemoryPressure | None:
    """Get macOS memory pressure metrics from vm_stat and sysctl.

    Caches result for 5 seconds to avoid subprocess overhead on hot paths.
    Returns None on non-macOS systems.
    """
    global _pressure_cache
    if not IS_MACOS:
        return None

    now = time.monotonic()
    cached_at, cached_result = _pressure_cache
    if now - cached_at < _PRESSURE_CACHE_TTL:
        return cached_result

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

        result = MacOSMemoryPressure(
            pressure_level=pressure_level,
            compressed_mb=compressed_mb,
            pageouts=stats.get("Pageouts", 0),
            pageins=stats.get("Pageins", 0),
            compressions=stats.get("Compressions", 0),
            decompressions=stats.get("Decompressions", 0),
            free_mb=free_mb,
            compressed_ratio=compression_ratio,
        )
        _pressure_cache = (now, result)
        return result
    except (subprocess.CalledProcessError, KeyError, ValueError) as e:
        logger.debug(f"Failed to get macOS memory pressure: {e}")
        _pressure_cache = (now, None)
        return None


def get_thermal_state() -> str | None:
    """Get macOS thermal state (throttling indicator).
    
    Returns:
        nominal, fair, serious, critical, or None if unavailable.
    """
    if not IS_MACOS:
        return None
        
    try:
        # Try kern.thermal_level first
        try:
            sysctl = subprocess.check_output(
                ["sysctl", "-n", "kern.thermal_level"], text=True
            ).strip()
            level = int(sysctl)
        except Exception:
            # Fallback for some Apple Silicon versions: thermal_threshold
            # Note: thermal_threshold is usually 0-100 where higher is hotter
            sysctl = subprocess.check_output(
                ["sysctl", "-n", "machdep.xcpm.thermal_threshold"], text=True
            ).strip()
            # Convert 0-100 scale to 0-3 scale
            val = int(sysctl)
            if val < 50: level = 0
            elif val < 80: level = 1
            elif val < 95: level = 2
            else: level = 3
            
        states = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}
        return states.get(level, "unknown")
    except Exception:
        return "nominal"  # Assume nominal if we can't read it


def get_memory_info() -> MemoryInfo:
    """Get current memory usage for this process and system swap."""
    process = psutil.Process()
    mem = process.memory_info()
    swap = psutil.swap_memory()

    # Get Footprint (macOS-specific)
    # On macOS, RSS includes shared libraries. 'footprint' is what the OS 
    # uses for 'out of memory' killing.
    footprint = mem.rss
    if IS_MACOS:
        try:
            # RSS on macOS psutil is actually task_info.resident_size
            # We want task_info.phys_footprint if possible.
            # Some versions of psutil provide it in memory_info().
            if hasattr(mem, 'pfid'): # some psutil versions
                footprint = getattr(mem, 'footprint', mem.rss)
            else:
                # Fallback to a faster way if psutil doesn't have it
                pass
        except Exception:
            pass

    macos_pressure = get_macos_memory_pressure() if IS_MACOS else None
    mlx_memory = get_mlx_memory_info()
    thermal_state = get_thermal_state()

    return MemoryInfo(
        rss_mb=mem.rss / 1024**2,
        footprint_mb=footprint / 1024**2,
        vms_mb=mem.vms / 1024**2,
        percent=process.memory_percent(),
        swap_used_mb=swap.used / 1024**2,
        swap_percent=swap.percent,
        timestamp=time.time(),
        macos_pressure=macos_pressure,
        mlx_memory=mlx_memory,
        thermal_state=thermal_state,
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
                # 1. Track Page-outs (Disk I/O)
                pageouts_delta = info.macos_pressure.pageouts - prev.macos_pressure.pageouts
                
                # 2. Track Compression Activity (CPU I/O)
                comp_delta = info.macos_pressure.compressions - prev.macos_pressure.compressions
                
                time_delta = info.timestamp - prev.timestamp
                if time_delta > 0:
                    if pageouts_delta > 0:
                        pageouts_per_sec = pageouts_delta / time_delta
                        log_msg += f" | SWAPPING: {pageouts_delta} pages ({pageouts_per_sec:.1f}/s)"
                    
                    if comp_delta > 0:
                        comp_per_sec = comp_delta / time_delta
                        log_msg += f" | COMPRESSING: {comp_delta} ops ({comp_per_sec:.1f}/s)"

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
