"""Memory profiling and control interfaces.

Workstreams 1 and 5 implement against these contracts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    pass


@dataclass
class MemoryProfile:
    """Result of profiling a model's memory usage.

    Attributes:
        model_name: Name/identifier of the model.
        quantization: Quantization format (e.g., "4bit", "8bit", "fp16").
        context_length: Context length tested.
        rss_mb: Resident Set Size (actual RAM used) in megabytes.
        virtual_mb: Virtual memory allocated in megabytes.
        metal_mb: GPU memory (Apple Metal) in megabytes.
        load_time_seconds: Time to load model in seconds.
        timestamp: ISO format timestamp of profiling.
    """

    model_name: str
    quantization: str
    context_length: int
    rss_mb: float
    virtual_mb: float
    metal_mb: float
    load_time_seconds: float
    timestamp: str

    def __post_init__(self) -> None:
        """Validate field constraints."""
        if self.context_length < 0:
            msg = f"context_length must be >= 0, got {self.context_length}"
            raise ValueError(msg)
        if self.rss_mb < 0:
            msg = f"rss_mb must be >= 0, got {self.rss_mb}"
            raise ValueError(msg)
        if self.load_time_seconds < 0:
            msg = f"load_time_seconds must be >= 0, got {self.load_time_seconds}"
            raise ValueError(msg)


class MemoryMode(Enum):
    """Operating modes based on available memory."""

    FULL = "full"  # 16GB+ : All features, concurrent models
    LITE = "lite"  # 8-16GB : Sequential loading, reduced context
    MINIMAL = "minimal"  # <8GB : Templates only, cloud fallback


@dataclass
class MemoryState:
    """Current memory status of the system.

    Attributes:
        available_mb: Available RAM in megabytes.
        used_mb: Used RAM in megabytes.
        model_loaded: Whether a model is currently loaded.
        current_mode: Current operating mode (FULL/LITE/MINIMAL).
        pressure_level: Memory pressure level (green/yellow/red/critical).
    """

    available_mb: float
    used_mb: float
    model_loaded: bool
    current_mode: MemoryMode
    pressure_level: str

    def __post_init__(self) -> None:
        """Validate field constraints."""
        valid_pressure_levels = {"green", "yellow", "red", "critical"}
        if self.pressure_level not in valid_pressure_levels:
            msg = (
                f"pressure_level must be one of {valid_pressure_levels}, got {self.pressure_level}"
            )
            raise ValueError(msg)
        if self.available_mb < 0:
            msg = f"available_mb must be >= 0, got {self.available_mb}"
            raise ValueError(msg)
        if self.used_mb < 0:
            msg = f"used_mb must be >= 0, got {self.used_mb}"
            raise ValueError(msg)


class MemoryProfiler(Protocol):
    """Interface for memory profiling (Workstream 1)."""

    def profile_model(self, model_path: str, context_length: int) -> MemoryProfile:
        """Profile a model's memory usage. Must unload model after profiling.

        Args:
            model_path: Path to the model to profile.
            context_length: Context length to test with.

        Returns:
            Memory profile with RSS, virtual, and Metal GPU usage.

        Note:
            Implementation MUST unload the model after profiling to free memory.
        """
        ...


class MemoryController(Protocol):
    """Interface for memory management (Workstream 5)."""

    def get_state(self) -> MemoryState:
        """Get current memory state.

        Returns:
            Current system memory status including mode and pressure level.
        """
        ...

    def get_mode(self) -> MemoryMode:
        """Determine appropriate mode based on available memory.

        Returns:
            Operating mode (FULL/LITE/MINIMAL) based on available RAM.
        """
        ...

    def can_load_model(self, required_mb: float) -> bool:
        """Check if we have enough memory to load a model.

        Args:
            required_mb: Memory required in megabytes.

        Returns:
            True if sufficient memory is available.
        """
        ...

    def request_memory(self, required_mb: float, priority: int) -> bool:
        """Request memory, potentially unloading lower-priority components.

        Args:
            required_mb: Memory required in megabytes.
            priority: Priority level (higher = more important).

        Returns:
            True if memory was successfully allocated.
        """
        ...

    def register_pressure_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for memory pressure events.

        Args:
            callback: Function to call when memory pressure is detected.
        """
        ...
