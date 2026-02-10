"""Database reliability monitoring for JARVIS.

Provides health checks for the SQLite database including
connection testing, integrity verification, and disk space monitoring.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from jarvis.db.models import JARVIS_DB_PATH

logger = logging.getLogger(__name__)

BYTES_PER_MB = 1024 * 1024
MIN_DISK_SPACE_MB = 50


@dataclass
class HealthStatus:
    """Result of a database health check."""

    healthy: bool
    db_exists: bool = False
    db_size_mb: float = 0.0
    disk_free_mb: float = 0.0
    connection_ok: bool = False
    integrity_ok: bool = True
    last_check: float = field(default_factory=time.time)
    issues: list[str] = field(default_factory=list)


class ReliabilityMonitor:
    """Monitors database health and reliability.

    Performs periodic health checks including:
    - Database file existence and size
    - Connection testing
    - Disk space monitoring
    - SQLite integrity checks (on-demand)
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or JARVIS_DB_PATH
        self._last_status: HealthStatus | None = None
        self._check_interval = 60.0  # seconds

    def check_health(self, full_check: bool = False) -> HealthStatus:
        """Run a health check on the database.

        Args:
            full_check: If True, include integrity_check (slower).

        Returns:
            HealthStatus with current database state.
        """
        issues: list[str] = []
        db_exists = self.db_path.exists()
        db_size_mb = 0.0
        disk_free_mb = 0.0
        connection_ok = False
        integrity_ok = True

        if db_exists:
            db_size_mb = self.db_path.stat().st_size / BYTES_PER_MB

        # Check disk space
        try:
            stat = os.statvfs(str(self.db_path.parent))
            disk_free_mb = (stat.f_bavail * stat.f_frsize) / BYTES_PER_MB
            if disk_free_mb < MIN_DISK_SPACE_MB:
                issues.append(f"Low disk space: {disk_free_mb:.0f}MB free")
        except OSError:
            issues.append("Cannot check disk space")

        # Test connection
        if db_exists:
            try:
                conn = sqlite3.connect(str(self.db_path), timeout=5)
                conn.execute("SELECT 1")
                connection_ok = True

                if full_check:
                    result = conn.execute("PRAGMA integrity_check").fetchone()
                    integrity_ok = result[0] == "ok"
                    if not integrity_ok:
                        issues.append(f"Integrity check failed: {result[0]}")

                conn.close()
            except Exception as e:
                issues.append(f"Connection failed: {e}")
        else:
            issues.append("Database file not found")

        status = HealthStatus(
            healthy=db_exists and connection_ok and integrity_ok and not issues,
            db_exists=db_exists,
            db_size_mb=round(db_size_mb, 2),
            disk_free_mb=round(disk_free_mb, 0),
            connection_ok=connection_ok,
            integrity_ok=integrity_ok,
            issues=issues,
        )
        self._last_status = status
        return status

    @property
    def last_status(self) -> HealthStatus | None:
        """Get the most recent health status without running a new check."""
        return self._last_status

    def needs_check(self) -> bool:
        """Check if enough time has passed for a new health check."""
        if self._last_status is None:
            return True
        return (time.time() - self._last_status.last_check) > self._check_interval


_monitor: ReliabilityMonitor | None = None


def get_reliability_monitor() -> ReliabilityMonitor:
    """Get or create the singleton ReliabilityMonitor."""
    global _monitor
    if _monitor is None:
        _monitor = ReliabilityMonitor()
    return _monitor
