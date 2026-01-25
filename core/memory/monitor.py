"""System memory monitoring using psutil.

Provides cross-platform memory usage information for the MemoryController.
"""

import logging
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)

# Constants
BYTES_PER_MB = 1024 * 1024


@dataclass
class SystemMemoryInfo:
    """Snapshot of system memory state."""

    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float


class MemoryMonitor:
    """Cross-platform system memory monitor using psutil.

    Provides real-time memory usage information for adaptive
    memory management decisions.
    """

    def get_system_memory(self) -> SystemMemoryInfo:
        """Get current system memory state.

        Returns:
            SystemMemoryInfo with current memory statistics.
        """
        mem = psutil.virtual_memory()
        return SystemMemoryInfo(
            total_mb=mem.total / BYTES_PER_MB,
            available_mb=mem.available / BYTES_PER_MB,
            used_mb=mem.used / BYTES_PER_MB,
            percent_used=mem.percent,
        )

    def get_available_mb(self) -> float:
        """Get available memory in MB.

        Returns:
            Available memory in megabytes.
        """
        return float(psutil.virtual_memory().available) / BYTES_PER_MB

    def get_used_mb(self) -> float:
        """Get used memory in MB.

        Returns:
            Used memory in megabytes.
        """
        return float(psutil.virtual_memory().used) / BYTES_PER_MB

    def get_percent_used(self) -> float:
        """Get memory usage percentage.

        Returns:
            Memory usage as a percentage (0-100).
        """
        return float(psutil.virtual_memory().percent)

    def get_pressure_level(self) -> str:
        """Determine memory pressure level based on usage percentage.

        Pressure levels:
        - green: <70% usage (normal operation)
        - yellow: 70-85% usage (moderate pressure)
        - red: 85-95% usage (high pressure)
        - critical: >95% usage (critical, should free memory)

        Returns:
            Pressure level string: "green", "yellow", "red", or "critical"
        """
        percent = self.get_percent_used()
        if percent >= 95:
            return "critical"
        elif percent >= 85:
            return "red"
        elif percent >= 70:
            return "yellow"
        else:
            return "green"
