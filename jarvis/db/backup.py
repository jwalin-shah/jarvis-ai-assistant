"""Database backup and restore utilities for JARVIS SQLite database.

Provides automated backup creation, rotation, and restore capabilities
for the ~/.jarvis/jarvis.db database.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import time
from pathlib import Path

from jarvis.db.models import JARVIS_DB_PATH

logger = logging.getLogger(__name__)

# Backup configuration
BACKUP_DIR = JARVIS_DB_PATH.parent / "backups"
MAX_BACKUPS = 5
BACKUP_SUFFIX = ".backup"


class BackupManager:
    """Manages database backups with rotation.

    Creates timestamped backup copies and maintains a maximum
    number of recent backups. Uses SQLite's backup API for
    consistency (no partial writes).
    """

    def __init__(
        self,
        db_path: Path | None = None,
        backup_dir: Path | None = None,
        max_backups: int = MAX_BACKUPS,
    ) -> None:
        self.db_path = db_path or JARVIS_DB_PATH
        self.backup_dir = backup_dir or BACKUP_DIR
        self.max_backups = max_backups

    def create_backup(self) -> Path | None:
        """Create a backup of the database using SQLite backup API.

        Returns:
            Path to the backup file, or None if backup failed.
        """
        if not self.db_path.exists():
            logger.warning("Database does not exist at %s, skipping backup", self.db_path)
            return None

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"jarvis_{timestamp}{BACKUP_SUFFIX}"

        try:
            source = sqlite3.connect(str(self.db_path))
            dest = sqlite3.connect(str(backup_path))
            try:
                source.backup(dest)
                logger.info("Database backup created: %s", backup_path)
            finally:
                dest.close()
                source.close()

            self._rotate_backups()
            return backup_path

        except Exception:
            logger.exception("Failed to create database backup")
            if backup_path.exists():
                backup_path.unlink()
            return None

    def restore_backup(self, backup_path: Path) -> bool:
        """Restore database from a backup file.

        Args:
            backup_path: Path to the backup file to restore from.

        Returns:
            True if restore succeeded.
        """
        if not backup_path.exists():
            logger.error("Backup file not found: %s", backup_path)
            return False

        try:
            # Validate the backup is a valid SQLite database
            conn = sqlite3.connect(str(backup_path))
            conn.execute("SELECT count(*) FROM sqlite_master")
            conn.close()

            shutil.copy2(str(backup_path), str(self.db_path))
            logger.info("Database restored from %s", backup_path)
            return True

        except Exception:
            logger.exception("Failed to restore database from %s", backup_path)
            return False

    def list_backups(self) -> list[Path]:
        """List available backups, newest first."""
        if not self.backup_dir.exists():
            return []
        backups = sorted(
            self.backup_dir.glob(f"*{BACKUP_SUFFIX}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return backups

    def _rotate_backups(self) -> None:
        """Remove old backups exceeding max_backups."""
        backups = self.list_backups()
        for old_backup in backups[self.max_backups :]:
            try:
                old_backup.unlink()
                logger.debug("Removed old backup: %s", old_backup)
            except OSError:
                logger.warning("Failed to remove old backup: %s", old_backup)


def get_backup_manager() -> BackupManager:
    """Get a BackupManager instance with default settings."""
    return BackupManager()
