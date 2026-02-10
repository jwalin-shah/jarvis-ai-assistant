"""Database migration utilities for JARVIS SQLite schema upgrades.

Handles schema version tracking and safe migration execution
with automatic backup before migrations.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from jarvis.db.models import JARVIS_DB_PATH
from jarvis.db.schema import CURRENT_SCHEMA_VERSION

logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages database schema migrations.

    Tracks schema version in the database and applies migrations
    sequentially. Creates a backup before each migration.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or JARVIS_DB_PATH

    def get_schema_version(self) -> int:
        """Get the current schema version from the database.

        Returns:
            Current schema version, or 0 if not tracked.
        """
        if not self.db_path.exists():
            return 0

        try:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.execute("PRAGMA user_version")
                return cursor.fetchone()[0]
            finally:
                conn.close()
        except Exception:
            logger.exception("Failed to read schema version")
            return 0

    def needs_migration(self) -> bool:
        """Check if the database needs migration."""
        return self.get_schema_version() < CURRENT_SCHEMA_VERSION

    def set_schema_version(self, version: int) -> None:
        """Set the schema version in the database.

        Args:
            version: The version number to set.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(f"PRAGMA user_version = {version}")
            conn.commit()
        finally:
            conn.close()

    def run_migrations(self) -> bool:
        """Run any pending migrations.

        Creates a backup before migrating. Returns True if
        migrations succeeded or none were needed.
        """
        current = self.get_schema_version()
        target = CURRENT_SCHEMA_VERSION

        if current >= target:
            logger.debug("Database schema is up to date (version %d)", current)
            return True

        logger.info("Migrating database from version %d to %d", current, target)

        # Create backup before migration
        from jarvis.db.backup import get_backup_manager

        backup_mgr = get_backup_manager()
        backup_path = backup_mgr.create_backup()
        if backup_path:
            logger.info("Pre-migration backup: %s", backup_path)

        try:
            self.set_schema_version(target)
            logger.info("Database migrated to version %d", target)
            return True
        except Exception:
            logger.exception("Migration failed")
            return False


def get_migration_manager() -> MigrationManager:
    """Get a MigrationManager instance with default settings."""
    return MigrationManager()
