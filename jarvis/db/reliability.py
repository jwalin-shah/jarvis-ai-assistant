"""Database reliability monitoring, corruption detection, and recovery.

Provides continuous health monitoring, automated corruption detection,
and recovery procedures for JARVIS SQLite databases.

Usage:
    from jarvis.db.reliability import ReliabilityMonitor, RecoveryManager

    # Monitor database health
    monitor = ReliabilityMonitor()
    report = monitor.check_health()

    if not report.is_healthy:
        # Attempt recovery
        recovery = RecoveryManager()
        result = recovery.attempt_recovery()
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.db.models import JARVIS_DB_PATH

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Health check thresholds
FRAGMENTATION_WARNING_THRESHOLD = 0.2  # 20% freelist/pages
FRAGMENTATION_CRITICAL_THRESHOLD = 0.4  # 40% freelist/pages
WAL_SIZE_WARNING_THRESHOLD = 0.5  # WAL > 50% of DB size
MAX_BACKUP_AGE_HOURS = 6

# Recovery settings
MAX_RECOVERY_ATTEMPTS = 3
RECOVERY_BACKOFF_SECONDS = 5


class HealthStatus(Enum):
    """Overall health status of the database."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CORRUPT = "corrupt"
    UNKNOWN = "unknown"


class RecoveryLevel(Enum):
    """Levels of recovery intervention."""

    NONE = "none"
    OPTIMIZE = "optimize"  # VACUUM, REINDEX
    REPAIR = "repair"  # salvage data
    RESTORE = "restore"  # from backup
    FULL = "full"  # recreate from scratch


@dataclass
class CorruptionReport:
    """Report of database corruption findings."""

    timestamp: datetime
    corruption_detected: bool
    affected_tables: list[str] = field(default_factory=list)
    affected_pages: list[int] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    recoverable: bool = False
    suggested_action: RecoveryLevel = RecoveryLevel.NONE


@dataclass
class HealthReport:
    """Comprehensive database health report."""

    timestamp: datetime
    status: HealthStatus

    # Core metrics
    integrity_check_passed: bool
    page_count: int
    freelist_count: int
    journal_mode: str
    foreign_keys_ok: bool

    # File metrics
    db_path: Path
    db_size_bytes: int
    wal_size_bytes: int
    shm_size_bytes: int

    # Schema info
    schema_version: int | None = None
    table_count: int = 0
    index_count: int = 0

    # Backup info
    last_backup_timestamp: datetime | None = None
    last_backup_age_hours: float | None = None

    # Performance indicators
    fragmentation_ratio: float = 0.0
    wal_ratio: float = 0.0

    # Warnings and recommendations
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "integrity_check_passed": self.integrity_check_passed,
            "page_count": self.page_count,
            "freelist_count": self.freelist_count,
            "journal_mode": self.journal_mode,
            "foreign_keys_ok": self.foreign_keys_ok,
            "db_path": str(self.db_path),
            "db_size_bytes": self.db_size_bytes,
            "wal_size_bytes": self.wal_size_bytes,
            "shm_size_bytes": self.shm_size_bytes,
            "schema_version": self.schema_version,
            "table_count": self.table_count,
            "index_count": self.index_count,
            "fragmentation_ratio": self.fragmentation_ratio,
            "wal_ratio": self.wal_ratio,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }

    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class RecoveryResult:
    """Result of a recovery operation."""

    success: bool
    level: RecoveryLevel
    duration_seconds: float = 0.0
    error_message: str | None = None
    actions_taken: list[str] = field(default_factory=list)
    data_loss_estimate: str = "none"  # none, minimal, partial, significant


class ReliabilityMonitor:
    """Continuous database reliability monitoring.

    Provides comprehensive health checks, corruption detection,
    and performance monitoring for SQLite databases.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        backup_dir: Path | None = None,
    ) -> None:
        """Initialize reliability monitor.

        Args:
            db_path: Path to database. Uses default if None.
            backup_dir: Path to backup directory for age checks.
        """
        self.db_path = db_path or JARVIS_DB_PATH
        self.backup_dir = backup_dir or (Path.home() / ".jarvis" / "backups")

    def check_health(self, detailed: bool = True) -> HealthReport:
        """Perform comprehensive health check.

        Args:
            detailed: Whether to perform detailed checks (slower).

        Returns:
            HealthReport with complete health status.
        """
        timestamp = datetime.now()
        warnings: list[str] = []
        recommendations: list[str] = []

        # Basic file checks
        if not self.db_path.exists():
            return HealthReport(
                timestamp=timestamp,
                status=HealthStatus.CORRUPT,
                integrity_check_passed=False,
                page_count=0,
                freelist_count=0,
                journal_mode="unknown",
                foreign_keys_ok=False,
                db_path=self.db_path,
                db_size_bytes=0,
                wal_size_bytes=0,
                shm_size_bytes=0,
                warnings=["Database file does not exist"],
            )

        # Get file sizes
        db_size = self.db_path.stat().st_size
        wal_path = self.db_path.with_suffix(".db-wal")
        wal_size = wal_path.stat().st_size if wal_path.exists() else 0
        shm_path = self.db_path.with_suffix(".db-shm")
        shm_size = shm_path.stat().st_size if shm_path.exists() else 0

        # Connect and run checks
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row

                # Integrity check
                cursor = conn.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                integrity_passed = integrity_result == "ok"

                if not integrity_passed:
                    warnings.append(f"Integrity check failed: {integrity_result}")

                # Page information
                cursor = conn.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]

                cursor = conn.execute("PRAGMA freelist_count")
                freelist_count = cursor.fetchone()[0]

                # Calculate fragmentation
                fragmentation = freelist_count / page_count if page_count > 0 else 0.0

                if fragmentation > FRAGMENTATION_CRITICAL_THRESHOLD:
                    warnings.append(f"Critical fragmentation: {fragmentation:.1%}")
                    recommendations.append("Run VACUUM to defragment database")
                elif fragmentation > FRAGMENTATION_WARNING_THRESHOLD:
                    warnings.append(f"High fragmentation: {fragmentation:.1%}")
                    recommendations.append("Schedule VACUUM during maintenance window")

                # Journal mode
                cursor = conn.execute("PRAGMA journal_mode")
                journal_mode = cursor.fetchone()[0]

                if journal_mode != "wal":
                    warnings.append(f"Journal mode is {journal_mode}, expected WAL")

                # Foreign keys
                cursor = conn.execute("PRAGMA foreign_key_check")
                fk_violations = cursor.fetchall()
                foreign_keys_ok = len(fk_violations) == 0

                if not foreign_keys_ok:
                    warnings.append(f"Foreign key violations: {len(fk_violations)}")

                # Schema version
                try:
                    cursor = conn.execute(
                        "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
                    )
                    row = cursor.fetchone()
                    schema_version = row[0] if row else None
                except sqlite3.Error:
                    schema_version = None
                    warnings.append("Could not determine schema version")

                # Table and index counts
                cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")
                index_count = cursor.fetchone()[0]

        except sqlite3.Error as e:
            return HealthReport(
                timestamp=timestamp,
                status=HealthStatus.CORRUPT,
                integrity_check_passed=False,
                page_count=0,
                freelist_count=0,
                journal_mode="unknown",
                foreign_keys_ok=False,
                db_path=self.db_path,
                db_size_bytes=db_size,
                wal_size_bytes=wal_size,
                shm_size_bytes=shm_size,
                warnings=[f"Failed to open database: {e}"],
            )

        # Check WAL size
        wal_ratio = wal_size / db_size if db_size > 0 else 0.0
        if wal_ratio > WAL_SIZE_WARNING_THRESHOLD:
            warnings.append(f"Large WAL file: {self._format_bytes(wal_size)}")
            recommendations.append("Checkpoint WAL to reduce size")

        # Check backup age
        last_backup_time = self._get_last_backup_time()
        last_backup_age = None
        if last_backup_time:
            last_backup_age = (timestamp - last_backup_time).total_seconds() / 3600

            if last_backup_age > MAX_BACKUP_AGE_HOURS * 2:
                warnings.append(f"Last backup is {last_backup_age:.1f} hours old")
                recommendations.append("Create new backup immediately")
            elif last_backup_age > MAX_BACKUP_AGE_HOURS:
                recommendations.append("Consider creating a new backup")

        # Determine overall status
        if not integrity_passed:
            status = HealthStatus.CORRUPT
        elif warnings:
            if any("Critical" in w or "failed" in w.lower() for w in warnings):
                status = HealthStatus.UNHEALTHY
            else:
                status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        return HealthReport(
            timestamp=timestamp,
            status=status,
            integrity_check_passed=integrity_passed,
            page_count=page_count,
            freelist_count=freelist_count,
            journal_mode=journal_mode,
            foreign_keys_ok=foreign_keys_ok,
            db_path=self.db_path,
            db_size_bytes=db_size,
            wal_size_bytes=wal_size,
            shm_size_bytes=shm_size,
            schema_version=schema_version,
            table_count=table_count,
            index_count=index_count,
            last_backup_timestamp=last_backup_time,
            last_backup_age_hours=last_backup_age,
            fragmentation_ratio=fragmentation,
            wal_ratio=wal_ratio,
            warnings=warnings,
            recommendations=recommendations,
        )

    def detect_corruption(self) -> CorruptionReport:
        """Detect and report database corruption.

        Returns:
            CorruptionReport with detailed findings.
        """
        timestamp = datetime.now()
        affected_tables: list[str] = []
        affected_pages: list[int] = []
        errors: list[str] = []

        # Try to open database
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Run integrity check with extended output
                cursor = conn.execute("PRAGMA integrity_check")
                results = cursor.fetchall()

                corruption_detected = False
                for row in results:
                    message = row[0]
                    if message != "ok":
                        corruption_detected = True
                        errors.append(message)

                        # Parse affected pages from error messages
                        # Format: "rowid 123 missing from index idx_name"
                        if "rowid" in message:
                            parts = message.split()
                            for i, part in enumerate(parts):
                                if part == "rowid" and i + 1 < len(parts):
                                    try:
                                        page_num = int(parts[i + 1])
                                        affected_pages.append(page_num)
                                    except ValueError:
                                        pass

                # Check specific tables
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor]

                for table in tables:
                    try:
                        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                        cursor.fetchone()
                    except sqlite3.DatabaseError as e:
                        affected_tables.append(table)
                        errors.append(f"Table {table} error: {e}")
                        corruption_detected = True

        except sqlite3.DatabaseError as e:
            corruption_detected = True
            errors.append(f"Database error: {e}")

        # Determine recoverability
        recoverable = len(affected_tables) < len(tables) / 2

        # Suggest recovery level
        if not corruption_detected:
            suggested_action = RecoveryLevel.NONE
        elif recoverable and len(affected_pages) < 10:
            suggested_action = RecoveryLevel.OPTIMIZE
        elif recoverable:
            suggested_action = RecoveryLevel.REPAIR
        elif self._get_last_backup_time() is not None:
            suggested_action = RecoveryLevel.RESTORE
        else:
            suggested_action = RecoveryLevel.FULL

        return CorruptionReport(
            timestamp=timestamp,
            corruption_detected=corruption_detected,
            affected_tables=affected_tables,
            affected_pages=list(set(affected_pages)),  # Deduplicate
            error_messages=errors,
            recoverable=recoverable,
            suggested_action=suggested_action,
        )

    def continuous_monitor(
        self,
        interval_seconds: float = 60.0,
        callback: Callable[[HealthReport], None] | None = None,
    ) -> None:
        """Run continuous health monitoring.

        Args:
            interval_seconds: Time between checks.
            callback: Optional callback for each report.
        """
        logger.info(
            "Starting continuous monitoring (interval: %.0f seconds)",
            interval_seconds,
        )

        while True:
            try:
                report = self.check_health()

                if callback:
                    callback(report)
                else:
                    self._default_monitor_callback(report)

                if report.status == HealthStatus.CORRUPT:
                    logger.critical("Database corruption detected!")
                    # Could trigger automatic recovery here

            except Exception as e:
                logger.error("Monitor error: %s", e)

            time.sleep(interval_seconds)

    def _default_monitor_callback(self, report: HealthReport) -> None:
        """Default callback for continuous monitoring."""
        if report.status == HealthStatus.HEALTHY:
            logger.debug("Health check: %s", report.status.value)
        elif report.status == HealthStatus.DEGRADED:
            logger.warning("Health check: %s", report.status.value)
            for warning in report.warnings:
                logger.warning("  - %s", warning)
        else:
            logger.error("Health check: %s", report.status.value)
            for warning in report.warnings:
                logger.error("  - %s", warning)

    def _get_last_backup_time(self) -> datetime | None:
        """Get timestamp of most recent backup."""
        if not self.backup_dir.exists():
            return None

        backups = [
            f
            for f in self.backup_dir.iterdir()
            if f.suffix == ".db" and f.name.startswith("jarvis_")
        ]

        if not backups:
            return None

        latest = max(backups, key=lambda p: p.stat().st_mtime)
        return datetime.fromtimestamp(latest.stat().st_mtime)

    @staticmethod
    def _format_bytes(size: int) -> str:
        """Format byte size for human readability."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


class RecoveryManager:
    """Database recovery and repair operations.

    Provides multiple levels of recovery from optimization
    to full database rebuild.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        backup_dir: Path | None = None,
    ) -> None:
        """Initialize recovery manager.

        Args:
            db_path: Path to database. Uses default if None.
            backup_dir: Directory containing backups. Uses default if None.
        """
        self.db_path = db_path or JARVIS_DB_PATH
        self.backup_dir = backup_dir or (Path.home() / ".jarvis" / "backups")
        self.monitor = ReliabilityMonitor(self.db_path, self.backup_dir)

    def attempt_recovery(
        self,
        level: RecoveryLevel | None = None,
    ) -> RecoveryResult:
        """Attempt database recovery.

        Args:
            level: Recovery level to attempt. Auto-detected if None.

        Returns:
            RecoveryResult with outcome.
        """
        start_time = time.time()
        actions: list[str] = []

        # Auto-detect level if not specified
        if level is None:
            report = self.monitor.detect_corruption()
            level = report.suggested_action

        logger.info("Attempting recovery at level: %s", level.value)

        try:
            if level == RecoveryLevel.NONE:
                return RecoveryResult(
                    success=True,
                    level=level,
                    duration_seconds=time.time() - start_time,
                    actions_taken=["No recovery needed"],
                )

            elif level == RecoveryLevel.OPTIMIZE:
                return self._optimize_recovery(start_time, actions)

            elif level == RecoveryLevel.REPAIR:
                return self._repair_recovery(start_time, actions)

            elif level == RecoveryLevel.RESTORE:
                return self._restore_recovery(start_time, actions)

            elif level == RecoveryLevel.FULL:
                return self._full_recovery(start_time, actions)

            else:
                return RecoveryResult(
                    success=False,
                    level=level,
                    error_message=f"Unknown recovery level: {level}",
                )

        except Exception as e:
            logger.exception("Recovery failed")
            return RecoveryResult(
                success=False,
                level=level,
                duration_seconds=time.time() - start_time,
                error_message=str(e),
                actions_taken=actions,
            )

    def _optimize_recovery(
        self,
        start_time: float,
        actions: list[str],
    ) -> RecoveryResult:
        """Level 1: Optimization recovery (REINDEX, VACUUM)."""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Rebuild indices
            conn.execute("REINDEX")
            actions.append("Rebuilt all indices")

            # Vacuum to defragment
            conn.execute("VACUUM")
            actions.append("Ran VACUUM to defragment")

        # Verify
        report = self.monitor.check_health()
        success = report.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

        return RecoveryResult(
            success=success,
            level=RecoveryLevel.OPTIMIZE,
            duration_seconds=time.time() - start_time,
            actions_taken=actions,
            data_loss_estimate="none",
        )

    def _repair_recovery(
        self,
        start_time: float,
        actions: list[str],
    ) -> RecoveryResult:
        """Level 2: Data repair recovery."""
        # Create recovery database
        recovery_path = self.db_path.with_suffix(".recovery.db")

        # Use sqlite3 .recover command
        try:
            result = subprocess.run(
                ["sqlite3", str(self.db_path), ".recover"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Recovery command failed: {result.stderr}")

            # Apply recovery SQL to new database
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".sql",
                delete=False,
            ) as f:
                f.write(result.stdout)
                sql_path = Path(f.name)

            # Create new database from recovery SQL
            with sqlite3.connect(str(recovery_path)) as conn:
                with open(sql_path) as f:
                    conn.executescript(f.read())

            sql_path.unlink()
            actions.append("Recovered data using .recover")

            # Replace original with recovery
            backup_path = self.db_path.with_suffix(
                f".corrupt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            )
            shutil.move(self.db_path, backup_path)
            shutil.move(recovery_path, self.db_path)
            actions.append(f"Replaced corrupt database (saved to {backup_path})")

            return RecoveryResult(
                success=True,
                level=RecoveryLevel.REPAIR,
                duration_seconds=time.time() - start_time,
                actions_taken=actions,
                data_loss_estimate="minimal",
            )

        except Exception:
            if recovery_path.exists():
                recovery_path.unlink()
            raise

    def _restore_recovery(
        self,
        start_time: float,
        actions: list[str],
    ) -> RecoveryResult:
        """Level 3: Restore from backup."""
        from jarvis.db.backup import BackupManager

        manager = BackupManager(backup_dir=self.backup_dir)

        # Find most recent backup
        backups = manager.list_backups()
        if not backups:
            raise RuntimeError("No backups available for restore")

        latest_backup = backups[0]
        actions.append(f"Selected backup: {latest_backup}")

        # Save corrupt database
        corrupt_path = self.db_path.with_suffix(
            f".corrupt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        )
        shutil.copy2(self.db_path, corrupt_path)
        actions.append(f"Saved corrupt database to {corrupt_path}")

        # Restore from backup
        result = manager.restore_from_backup(latest_backup)

        if not result.success:
            raise RuntimeError(f"Restore failed: {result.error_message}")

        actions.append(f"Restored from {latest_backup}")

        return RecoveryResult(
            success=True,
            level=RecoveryLevel.RESTORE,
            duration_seconds=time.time() - start_time,
            actions_taken=actions,
            data_loss_estimate="depends on backup age",
        )

    def _full_recovery(
        self,
        start_time: float,
        actions: list[str],
    ) -> RecoveryResult:
        """Level 4: Full rebuild from scratch."""
        # Save corrupt database for forensics
        forensics_path = self.db_path.with_suffix(
            f".forensics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        )
        shutil.move(self.db_path, forensics_path)
        actions.append(f"Saved corrupt database for forensics: {forensics_path}")

        # Create fresh database with schema
        from jarvis.db.core import JarvisDBBase

        db = JarvisDBBase(self.db_path)
        db.init_schema()
        actions.append("Created fresh database with current schema")

        return RecoveryResult(
            success=True,
            level=RecoveryLevel.FULL,
            duration_seconds=time.time() - start_time,
            actions_taken=actions,
            data_loss_estimate="significant (all data lost)",
        )

    def checkpoint_wal(self) -> None:
        """Force WAL checkpoint to reduce WAL file size."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            logger.info("WAL checkpoint completed")

    def vacuum_into(self, target_path: Path) -> None:
        """Vacuum database into a new file.

        Args:
            target_path: Path for the vacuumed database.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(f"VACUUM INTO '{target_path}'")
            logger.info("Vacuumed database into %s", target_path)


def quick_health_check(db_path: Path | None = None) -> HealthStatus:
    """Quick health check returning only status.

    Args:
        db_path: Path to database. Uses default if None.

    Returns:
        HealthStatus enum value.
    """
    monitor = ReliabilityMonitor(db_path)
    report = monitor.check_health(detailed=False)
    return report.status


def run_health_report(db_path: Path | None = None) -> None:
    """Run and print a health report.

    Args:
        db_path: Path to database. Uses default if None.
    """
    monitor = ReliabilityMonitor(db_path)
    report = monitor.check_health()

    print("\n" + "=" * 60)
    print("DATABASE HEALTH REPORT")
    print("=" * 60)
    print(f"Timestamp: {report.timestamp}")
    print(f"Status: {report.status.value.upper()}")
    print(f"\nDatabase: {report.db_path}")
    print(f"Size: {ReliabilityMonitor._format_bytes(report.db_size_bytes)}")
    print(f"WAL Size: {ReliabilityMonitor._format_bytes(report.wal_size_bytes)}")
    print(f"\nSchema Version: {report.schema_version}")
    print(f"Tables: {report.table_count}, Indexes: {report.index_count}")
    print(f"\nIntegrity Check: {'PASS' if report.integrity_check_passed else 'FAIL'}")
    print(f"Foreign Keys: {'OK' if report.foreign_keys_ok else 'VIOLATIONS'}")
    print(f"Fragmentation: {report.fragmentation_ratio:.1%}")

    if report.warnings:
        print(f"\nWarnings ({len(report.warnings)}):")
        for warning in report.warnings:
            print(f"  ⚠ {warning}")

    if report.recommendations:
        print(f"\nRecommendations ({len(report.recommendations)}):")
        for rec in report.recommendations:
            print(f"  → {rec}")

    print("=" * 60 + "\n")
