"""Memory management (Workstream 5).

Provides adaptive memory monitoring and control for JARVIS,
enabling operation across 8GB, 16GB, and higher memory configurations.
"""

from core.memory.controller import (
    DefaultMemoryController,
    MemoryThresholds,
    get_memory_controller,
    reset_memory_controller,
)
from core.memory.monitor import MemoryMonitor, SystemMemoryInfo

__all__ = [
    "DefaultMemoryController",
    "MemoryMonitor",
    "MemoryThresholds",
    "SystemMemoryInfo",
    "get_memory_controller",
    "reset_memory_controller",
]
