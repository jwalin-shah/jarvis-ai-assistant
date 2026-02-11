"""Database backup and restore utilities for JARVIS.

Provides hot (online) backups, SQL exports, and point-in-time recovery
capabilities for SQLite databases. All operations respect the 8GB RAM
constraint through streaming and chunked processing.

Usage:
    from jarvis.db.backup import BackupManager

    manager = BackupManager()

    # Hot backup (online, no downtime)
    backup_path = manager.create_hot_backup()

    # Restore from backup
    manager.restore_from_backup(backup_path)

    # SQL export for portability
    manager.create_sql_export()
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from jarvis.db.models import JARVIS_DB_PATH

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Default backup locations
DEFAULT_BACKUP_DIR = Path.home() / ".jarvis" / "backups"
DEFAULT_EXPORT_DIR = Path.home() / ".jarvis" / "exports"
DEFAULT_MIGRATION_BACKUP_DIR = Path.home() / ".jarvis" / "backups" / "migrations"
DEFAULT_WAL_ARCHIVE_DIR = Path.home() / ".jarvis" / "wal_archive"

# Backup retention policies
HOT_BACKUP_RETENTION_DAYS = 7
EXPORT_RETENTION_DAYS = 30
WAL_ARCHIVE_RETENTION_HOURS = 72

# Performance tuning for 8GB RAM
BACKUP_PAGE_SIZE = 100  # Pages per backup chunk
MAX_BACKUP_MEMORY_MB = 100  # Memory limit for backup operations


@dataclass
class BackupResult:
    """Result of a backup operation."""

    success: bool
    backup_path: Path | None = None
    size_bytes: int = 0
    checksum: str = ""
    duration_seconds: float = 0.0
    error_message: str | None = None
    tables_backed_up: dict[str, int] = field(default_factory=dict)


@dataclass
class RestoreResult:
    """Result of a restore operation."""

    success: bool
    restored_path: Path | None = None
    backup_path: Path | None = None
    duration_seconds: float = 0.0
    error_message: str | None = None
    integrity_check_passed: bool = False
    schema_version: int | None = None


@dataclass
class HealthReport:
    """Database health check report."""

    timestamp: datetime
    integrity_check_passed: bool
    page_count: int
    freelist_count: int
    journal_mode: str
    foreign_keys_ok: bool
    size_bytes: int
    wal_size_bytes: int
    last_vacuum: datetime | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        """Check if database is in healthy state."""
        return (
            self.integrity_check_passed
            and self.foreign_keys_ok
            and self.journal_mode == "wal"
            and len(self.warnings) == 0
        )

    @property
    def fragmentation_ratio(self) -> float:
        """Calculate fragmentation ratio (freelist / total pages)."""
        if self.page_count == 0:
            return 0.0
        return self.freelist_count / self.page_count


class BackupManager:
    """Manager for database backup and restore operations.

    Thread-safe implementation with progress callbacks for long operations.
    All operations respect 8GB RAM constraints through streaming.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        backup_dir: Path | None = None,
        export_dir: Path | None = None,
        wal_archive_dir: Path | None = None,
    ) -> None:
        """Initialize backup manager.

        Args:
            db_path: Path to database file. Uses default if None.
            backup_dir: Directory for hot backups. Uses default if None.
            export_dir: Directory for SQL exports. Uses default if None.
            wal_archive_dir: Directory for WAL archiving. Uses default if None.
        """
        self.db_path = db_path or JARVIS_DB_PATH
        self.backup_dir = backup_dir or DEFAULT_BACKUP_DIR
        self.export_dir = export_dir or DEFAULT_EXPORT_DIR
        self.wal_archive_dir = wal_archive_dir or DEFAULT_WAL_ARCHIVE_DIR

        # Ensure directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.wal_archive_dir.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.RLock()
        self._backup_in_progress = False

    def create_hot_backup(
        self,
        name: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BackupResult:
        """Create a hot (online) backup of the database.

        Uses SQLite's online backup API for zero-downtime backups.
        Pages are copied in chunks to limit memory usage.

        Args:
            name: Optional name for the backup file. Auto-generated if None.
            progress_callback: Optional callback(current_page, total_pages).

        Returns:
            BackupResult with success status and metadata.
        """
        with self._lock:
            if self._backup_in_progress:
                return BackupResult(
                    success=False,
                    error_message="Another backup is already in progress",
                )

            self._backup_in_progress = True

        try:
            start_time = time.time()

            # Generate backup filename
            if name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"jarvis_{timestamp}.db"

            backup_path = self.backup_dir / name

            # Ensure parent directory exists
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info("Starting hot backup to %s", backup_path)

            # Perform online backup
            with sqlite3.connect(str(self.db_path)) as source:
                with sqlite3.connect(str(backup_path)) as dest:
                    # Use backup API with page-by-page copying
                    source.backup(
                        dest,
                        pages=BACKUP_PAGE_SIZE,
                        progress=progress_callback,
                    )

            # Calculate checksum
            checksum = self._calculate_checksum(backup_path)

            # Get table row counts
            tables = self._get_table_counts(backup_path)

            duration = time.time() - start_time
            size_bytes = backup_path.stat().st_size

            logger.info(
                "Hot backup completed in %.2f seconds (size: %s)",
                duration,
                self._format_bytes(size_bytes),
            )

            return BackupResult(
                success=True,
                backup_path=backup_path,
                size_bytes=size_bytes,
                checksum=checksum,
                duration_seconds=duration,
                tables_backed_up=tables,
            )

        except sqlite3.Error as e:
            logger.error("Hot backup failed: %s", e)
            # Clean up partial backup
            if backup_path.exists():
                backup_path.unlink()
            return BackupResult(
                success=False,
                error_message=f"SQLite error: {e}",
            )
        except OSError as e:
            logger.error("Hot backup failed: %s", e)
            return BackupResult(
                success=False,
                error_message=f"IO error: {e}",
            )
        finally:
            with self._lock:
                self._backup_in_progress = False

    def create_migration_backup(self) -> BackupResult:
        """Create a pre-migration backup with special naming.

        Returns:
            BackupResult with success status.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"pre_migration_{timestamp}.db"
        backup_path = DEFAULT_MIGRATION_BACKUP_DIR / name
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        result = self.create_hot_backup(name=str(backup_path))

        if result.success:
            logger.info("Migration backup created at %s", result.backup_path)

        return result

    def create_sql_export(
        self,
        tables: list[str] | None = None,
        compress: bool = True,
    ) -> BackupResult:
        """Create a SQL dump export for data portability.

        Args:
            tables: List of tables to export. All if None.
            compress: Whether to gzip the output.

        Returns:
            BackupResult with success status.
        """
        start_time = time.time()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"jarvis_export_{timestamp}.sql"
        export_path = self.export_dir / export_name

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Build dump command
                if tables:
                    # Export specific tables
                    table_list = " ".join(tables)
                    dump_cmd = f".dump {table_list}"
                else:
                    # Export all
                    dump_cmd = ".dump"

                # Execute dump
                result = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'")

                with open(export_path, "w") as f:
                    f.write("-- JARVIS Database Export\n")
                    f.write(f"-- Generated: {datetime.now().isoformat()}\n")
                    f.write(f"-- Source: {self.db_path}\n\n")
                    f.write("BEGIN TRANSACTION;\n\n")

                    # Get tables to export
                    tables_to_export = tables or self._get_all_tables(conn)

                    for table in tables_to_export:
                        # Export table schema
                        cursor = conn.execute(
                            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                            (table,),
                        )
                        row = cursor.fetchone()
                        if row and row[0]:
                            f.write(f"{row[0]};\n\n")

                        # Export data
                        cursor = conn.execute(f"SELECT * FROM {table}")
                        columns = [desc[0] for desc in cursor.description]

                        for row in cursor:
                            values = []
                            for val in row:
                                if val is None:
                                    values.append("NULL")
                                elif isinstance(val, (int, float)):
                                    values.append(str(val))
                                else:
                                    escaped = str(val).replace("'", "''")
                                    values.append(f"'{escaped}'")

                            f.write(
                                f"INSERT INTO {table} ({','.join(columns)}) "
                                f"VALUES ({','.join(values)});\n"
                            )

                        f.write("\n")

                    f.write("COMMIT;\n")

            # Compress if requested
            if compress:
                gz_path = export_path.with_suffix(".sql.gz")
                with open(export_path, "rb") as f_in:
                    with gzip.open(gz_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                export_path.unlink()
                export_path = gz_path

            duration = time.time() - start_time
            size_bytes = export_path.stat().st_size
            checksum = self._calculate_checksum(export_path)

            logger.info(
                "SQL export created at %s (size: %s, time: %.2fs)",
                export_path,
                self._format_bytes(size_bytes),
                duration,
            )

            return BackupResult(
                success=True,
                backup_path=export_path,
                size_bytes=size_bytes,
                checksum=checksum,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error("SQL export failed: %s", e)
            if export_path.exists():
                export_path.unlink()
            return BackupResult(
                success=False,
                error_message=f"Export failed: {e}",
            )

    def restore_from_backup(
        self,
        backup_path: Path,
        verify_integrity: bool = True,
        create_safety_copy: bool = True,
    ) -> RestoreResult:
        """Restore database from a backup.

        Args:
            backup_path: Path to the backup file.
            verify_integrity: Whether to run integrity check after restore.
            create_safety_copy: Whether to keep a copy of current DB.

        Returns:
            RestoreResult with success status and metadata.
        """
        start_time = time.time()

        if not backup_path.exists():
            return RestoreResult(
                success=False,
                error_message=f"Backup file not found: {backup_path}",
            )

        try:
            # Create safety copy if requested
            safety_path = None
            if create_safety_copy and self.db_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safety_name = f"jarvis_pre_restore_{timestamp}.db"
                safety_path = self.backup_dir / safety_name
                shutil.copy2(self.db_path, safety_path)
                logger.info("Created safety copy at %s", safety_path)

            # Verify backup integrity before restore
            if verify_integrity:
                with sqlite3.connect(str(backup_path)) as conn:
                    cursor = conn.execute("PRAGMA integrity_check")
                    result = cursor.fetchone()
                    if result[0] != "ok":
                        return RestoreResult(
                            success=False,
                            error_message=f"Backup integrity check failed: {result[0]}",
                        )

            # Perform restore
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_path, self.db_path)

            duration = time.time() - start_time

            # Post-restore verification
            integrity_passed = True
            schema_version = None

            if verify_integrity:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.execute("PRAGMA integrity_check")
                    result = cursor.fetchone()
                    integrity_passed = result[0] == "ok"

                    cursor = conn.execute(
                        "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
                    )
                    row = cursor.fetchone()
                    if row:
                        schema_version = row[0]

            logger.info(
                "Database restored from %s in %.2f seconds",
                backup_path,
                duration,
            )

            return RestoreResult(
                success=True,
                restored_path=self.db_path,
                backup_path=backup_path,
                duration_seconds=duration,
                integrity_check_passed=integrity_passed,
                schema_version=schema_version,
            )

        except Exception as e:
            logger.error("Restore failed: %s", e)
            return RestoreResult(
                success=False,
                error_message=f"Restore failed: {e}",
            )

    def verify_backup(self, backup_path: Path) -> HealthReport:
        """Verify a backup file is valid and healthy.

        Args:
            backup_path: Path to the backup file.

        Returns:
            HealthReport with detailed status.
        """
        warnings: list[str] = []

        try:
            with sqlite3.connect(str(backup_path)) as conn:
                # Integrity check
                cursor = conn.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()
                integrity_passed = integrity_result[0] == "ok"

                if not integrity_passed:
                    warnings.append(f"Integrity check failed: {integrity_result[0]}")

                # Page info
                cursor = conn.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]

                cursor = conn.execute("PRAGMA freelist_count")
                freelist_count = cursor.fetchone()[0]

                # Journal mode
                cursor = conn.execute("PRAGMA journal_mode")
                journal_mode = cursor.fetchone()[0]

                # Foreign keys
                cursor = conn.execute("PRAGMA foreign_key_check")
                fk_violations = cursor.fetchall()
                foreign_keys_ok = len(fk_violations) == 0

                if not foreign_keys_ok:
                    warnings.append(f"Foreign key violations: {len(fk_violations)}")

                # File size
                size_bytes = backup_path.stat().st_size

                # WAL size
                wal_path = backup_path.with_suffix(".db-wal")
                wal_size_bytes = wal_path.stat().st_size if wal_path.exists() else 0

                return HealthReport(
                    timestamp=datetime.now(),
                    integrity_check_passed=integrity_passed,
                    page_count=page_count,
                    freelist_count=freelist_count,
                    journal_mode=journal_mode,
                    foreign_keys_ok=foreign_keys_ok,
                    size_bytes=size_bytes,
                    wal_size_bytes=wal_size_bytes,
                    warnings=warnings,
                )

        except sqlite3.Error as e:
            return HealthReport(
                timestamp=datetime.now(),
                integrity_check_passed=False,
                page_count=0,
                freelist_count=0,
                journal_mode="unknown",
                foreign_keys_ok=False,
                size_bytes=0,
                wal_size_bytes=0,
                warnings=[f"Failed to open backup: {e}"],
            )

    def list_backups(self) -> list[Path]:
        """List all available backup files.

        Returns:
            List of backup file paths, sorted by modification time (newest first).
        """
        if not self.backup_dir.exists():
            return []

        backups = [
            f
            for f in self.backup_dir.iterdir()
            if f.suffix == ".db" and f.name.startswith("jarvis_")
        ]
        backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return backups

    def cleanup_old_backups(
        self,
        max_age_days: int = HOT_BACKUP_RETENTION_DAYS,
        dry_run: bool = False,
    ) -> list[Path]:
        """Remove backups older than specified age.

        Args:
            max_age_days: Maximum age in days.
            dry_run: If True, only report what would be deleted.

        Returns:
            List of removed (or would-remove) backup paths.
        """
        cutoff = datetime.now() - timedelta(days=max_age_days)
        removed: list[Path] = []

        for backup in self.list_backups():
            mtime = datetime.fromtimestamp(backup.stat().st_mtime)
            if mtime < cutoff:
                if not dry_run:
                    backup.unlink()
                    logger.info("Removed old backup: %s", backup)
                removed.append(backup)

        return removed

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _format_bytes(self, size: int) -> str:
        """Format byte size for human readability."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def _get_table_counts(self, db_path: Path) -> dict[str, int]:
        """Get row counts for all tables in a database."""
        counts: dict[str, int] = {}
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor]

                for table in tables:
                    try:
                        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                        counts[table] = cursor.fetchone()[0]
                    except sqlite3.Error:
                        counts[table] = -1
        except sqlite3.Error:
            pass

        return counts

    def _get_all_tables(self, conn: sqlite3.Connection) -> list[str]:
        """Get list of all user tables in database."""
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return [row[0] for row in cursor]


class WalArchive:
    """WAL file archiving for point-in-time recovery."""

    def __init__(self, archive_dir: Path | None = None) -> None:
        """Initialize WAL archive manager.

        Args:
            archive_dir: Directory for WAL archives. Uses default if None.
        """
        self.archive_dir = archive_dir or DEFAULT_WAL_ARCHIVE_DIR
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def archive_current_wal(self, db_path: Path) -> Path | None:
        """Archive the current WAL file if it exists.

        Args:
            db_path: Path to the database file.

        Returns:
            Path to archived file, or None if no WAL exists.
        """
        wal_path = db_path.with_suffix(".db-wal")
        if not wal_path.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"jarvis_{timestamp}.wal"
        archive_path = self.archive_dir / archive_name

        shutil.copy2(wal_path, archive_path)
        logger.debug("Archived WAL to %s", archive_path)

        return archive_path

    def cleanup_old_archives(self, max_age_hours: int = WAL_ARCHIVE_RETENTION_HOURS) -> int:
        """Remove archived WAL files older than specified age.

        Args:
            max_age_hours: Maximum age in hours.

        Returns:
            Number of files removed.
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        removed = 0

        for wal_file in self.archive_dir.glob("*.wal"):
            mtime = datetime.fromtimestamp(wal_file.stat().st_mtime)
            if mtime < cutoff:
                wal_file.unlink()
                removed += 1

        if removed > 0:
            logger.info("Cleaned up %d old WAL archives", removed)

        return removed


def get_latest_backup(backup_dir: Path | None = None) -> Path | None:
    """Get the most recent backup file.

    Args:
        backup_dir: Directory to search. Uses default if None.

    Returns:
        Path to most recent backup, or None if none found.
    """
    manager = BackupManager(backup_dir=backup_dir)
    backups = manager.list_backups()
    return backups[0] if backups else None


def quick_backup() -> BackupResult:
    """Create a quick hot backup with default settings.

    Returns:
        BackupResult from the operation.
    """
    manager = BackupManager()
    return manager.create_hot_backup()


def quick_restore(backup_path: Path | None = None) -> RestoreResult:
    """Restore from the most recent backup or specified path.

    Args:
        backup_path: Specific backup to restore from. Uses latest if None.

    Returns:
        RestoreResult from the operation.
    """
    manager = BackupManager()

    if backup_path is None:
        backup_path = get_latest_backup()
        if backup_path is None:
            return RestoreResult(
                success=False,
                error_message="No backups found",
            )

    return manager.restore_from_backup(backup_path)
