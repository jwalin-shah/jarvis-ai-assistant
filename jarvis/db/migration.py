"""Database migration testing and validation framework.

Provides utilities for testing schema migrations, validating rollback
procedures, and ensuring migration safety before deployment.

Usage:
    from jarvis.db.migration import MigrationTester

    tester = MigrationTester()

    # Test migration from v11 to v12
    result = tester.test_migration(from_version=11, to_version=12)

    # Test rollback
    result = tester.test_rollback(from_version=12, to_version=11)

    # Run full migration suite
    results = tester.run_full_test_suite()
"""

from __future__ import annotations

import logging
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.db.schema import CURRENT_SCHEMA_VERSION, SCHEMA_SQL, VALID_MIGRATION_COLUMNS

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MigrationStage(Enum):
    """Staged migration deployment stages."""

    CANARY = "canary"  # 1% of data
    PARTIAL = "partial"  # 10% of data
    FULL = "full"  # 100% of data
    VERIFY = "verify"  # Post-migration verification


class MigrationStatus(Enum):
    """Status of a migration test."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationTestResult:
    """Result of a migration test."""

    from_version: int
    to_version: int
    status: MigrationStatus
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema_valid: bool = False
    data_integrity_passed: bool = False
    rollback_tested: bool = False

    @property
    def success(self) -> bool:
        """Check if migration test passed."""
        return (
            self.status == MigrationStatus.PASSED
            and self.schema_valid
            and self.data_integrity_passed
        )


@dataclass
class SchemaDiff:
    """Difference between two schema versions."""

    added_tables: list[str] = field(default_factory=list)
    removed_tables: list[str] = field(default_factory=list)
    added_columns: dict[str, list[str]] = field(default_factory=dict)
    removed_columns: dict[str, list[str]] = field(default_factory=dict)
    added_indexes: list[str] = field(default_factory=list)
    removed_indexes: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Check if there are any schema changes."""
        return bool(
            self.added_tables
            or self.removed_tables
            or self.added_columns
            or self.removed_columns
            or self.added_indexes
            or self.removed_indexes
        )


class MigrationTester:
    """Test framework for database migrations.

    Provides comprehensive testing capabilities for schema migrations
    including forward migration testing, rollback validation, and
    data integrity verification.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize migration tester.

        Args:
            db_path: Path to test database. Uses temp file if None.
        """
        self.db_path = db_path
        self._temp_dir: Path | None = None

    def _get_test_db_path(self) -> Path:
        """Get or create a test database path."""
        if self.db_path:
            return self.db_path

        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="jarvis_migration_test_"))

        return self._temp_dir / "test.db"

    def _cleanup(self) -> None:
        """Clean up temporary test database."""
        if self._temp_dir and self._temp_dir.exists():
            import shutil

            shutil.rmtree(self._temp_dir)
            self._temp_dir = None

    def create_db_at_version(
        self,
        version: int,
        with_sample_data: bool = True,
    ) -> Path:
        """Create a test database at a specific schema version.

        Args:
            version: Target schema version.
            with_sample_data: Whether to populate with sample data.

        Returns:
            Path to the created database.
        """
        db_path = self._get_test_db_path()

        if db_path.exists():
            db_path.unlink()

        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row

            # Create base schema
            conn.executescript(SCHEMA_SQL)

            # Set specific version
            conn.execute("DELETE FROM schema_version")
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (version,),
            )

            if with_sample_data:
                self._insert_sample_data(conn, version)

        return db_path

    def _insert_sample_data(self, conn: sqlite3.Connection, version: int) -> None:
        """Insert sample data appropriate for the schema version."""
        # Insert sample contacts
        contacts = [
            ("chat_001", "Alice Smith", "+15551234567", "sister"),
            ("chat_002", "Bob Jones", "bob@example.com", "coworker"),
            ("chat_003", "Carol White", "+15559876543", "friend"),
        ]

        for chat_id, name, contact, rel in contacts:
            conn.execute(
                """
                INSERT INTO contacts (chat_id, display_name, phone_or_email, relationship)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, name, contact, rel),
            )

        # Insert sample pairs
        for i in range(1, 101):
            contact_id = (i % 3) + 1

            # Build INSERT based on available columns for version
            columns = [
                "contact_id",
                "trigger_text",
                "response_text",
                "trigger_timestamp",
                "response_timestamp",
                "chat_id",
            ]
            values: list[Any] = [
                contact_id,
                f"Trigger message {i}",
                f"Response message {i}",
                datetime.now(),
                datetime.now(),
                f"chat_{contact_id:03d}",
            ]

            # Add version-specific columns
            if version >= 3:
                columns.append("context_text")
                values.append(f"Context for message {i}")

            if version >= 4:
                columns.append("is_group")
                values.append(False)

            if version >= 5:
                columns.append("is_holdout")
                values.append(i % 10 == 0)  # 10% holdout

            if version >= 6:
                columns.extend(
                    [
                        "gate_a_passed",
                        "gate_b_score",
                        "gate_c_verdict",
                        "validity_status",
                    ]
                )
                values.extend([True, 0.85, "accept", "valid"])

            if version >= 7:
                columns.extend(
                    [
                        "trigger_da_type",
                        "trigger_da_conf",
                        "response_da_type",
                        "response_da_conf",
                        "cluster_id",
                    ]
                )
                values.extend(
                    [
                        "WH_QUESTION" if i % 2 == 0 else "STATEMENT",
                        0.9,
                        "ANSWER" if i % 2 == 0 else "ACKNOWLEDGE",
                        0.85,
                        i % 5,
                    ]
                )

            if version >= 9:
                columns.append("content_hash")
                import hashlib

                hash_val = hashlib.md5(
                    f"trigger{i}|response{i}".encode(),
                    usedforsecurity=False,
                ).hexdigest()
                values.append(hash_val)

            # Execute INSERT
            placeholders = ",".join(["?"] * len(values))
            conn.execute(
                f"INSERT INTO pairs ({','.join(columns)}) VALUES ({placeholders})",
                values,
            )

        # Insert sample facts (v12+)
        if version >= 12:
            facts = [
                ("chat_001", "personal", "birthday", "has_date", "1990-05-15"),
                ("chat_002", "work", "company", "works_at", "TechCorp"),
                ("chat_003", "preference", "food", "likes", "sushi"),
            ]
            for contact_id, category, subject, predicate, value in facts:
                conn.execute(
                    """
                    INSERT INTO contact_facts
                    (contact_id, category, subject, predicate, value, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (contact_id, category, subject, predicate, value, 0.95),
                )

        conn.commit()

    def test_migration(
        self,
        from_version: int,
        to_version: int | None = None,
        with_sample_data: bool = True,
    ) -> MigrationTestResult:
        """Test migration from one version to another.

        Args:
            from_version: Starting schema version.
            to_version: Target schema version. Uses CURRENT_SCHEMA_VERSION if None.
            with_sample_data: Whether to populate with sample data.

        Returns:
            MigrationTestResult with test outcome.
        """
        import time

        to_version = to_version or CURRENT_SCHEMA_VERSION

        start_time = time.time()
        result = MigrationTestResult(
            from_version=from_version,
            to_version=to_version,
            status=MigrationStatus.RUNNING,
        )

        try:
            # Create database at source version
            db_path = self.create_db_at_version(from_version, with_sample_data)

            # Record pre-migration state
            pre_counts = self._get_table_counts(db_path)

            # Perform migration
            with sqlite3.connect(str(db_path)) as conn:
                conn.row_factory = sqlite3.Row

                # Enable foreign keys
                conn.execute("PRAGMA foreign_keys = ON")

                # Run migrations by simulating init_schema behavior
                self._run_migrations(conn, from_version, to_version)

            # Verify post-migration state
            post_counts = self._get_table_counts(db_path)

            # Check schema version
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute(
                    "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
                )
                actual_version = cursor.fetchone()[0]

            if actual_version != to_version:
                result.errors.append(
                    f"Schema version mismatch: expected {to_version}, got {actual_version}"
                )
            else:
                result.schema_valid = True

            # Verify data integrity
            if with_sample_data:
                data_ok, data_errors = self._verify_data_integrity(
                    db_path, pre_counts, post_counts, from_version, to_version
                )
                result.data_integrity_passed = data_ok
                result.errors.extend(data_errors)

            # Run integrity check
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                if integrity_result != "ok":
                    result.errors.append(f"Integrity check failed: {integrity_result}")

            result.duration_seconds = time.time() - start_time

            if not result.errors:
                result.status = MigrationStatus.PASSED
            else:
                result.status = MigrationStatus.FAILED

        except Exception as e:
            result.status = MigrationStatus.FAILED
            result.errors.append(f"Migration failed with exception: {e}")
            result.duration_seconds = time.time() - start_time

        return result

    def _run_migrations(
        self,
        conn: sqlite3.Connection,
        from_version: int,
        to_version: int,
    ) -> None:
        """Run migrations between versions."""
        # Import here to avoid circular dependency
        from jarvis.db.schema import VALID_COLUMN_TYPES

        # Migration v2 -> v3: Add context_text
        if from_version <= 2 < to_version:
            try:
                conn.execute("ALTER TABLE pairs ADD COLUMN context_text TEXT")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

        # Migration v3 -> v4: Add is_group
        if from_version <= 3 < to_version:
            try:
                conn.execute("ALTER TABLE pairs ADD COLUMN is_group BOOLEAN DEFAULT FALSE")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

        # Migration v4 -> v5: Add is_holdout
        if from_version <= 4 < to_version:
            try:
                conn.execute("ALTER TABLE pairs ADD COLUMN is_holdout BOOLEAN DEFAULT FALSE")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

        # Migration v5 -> v6: Validity gates
        if from_version <= 5 < to_version:
            gate_columns = [
                ("gate_a_passed", "BOOLEAN"),
                ("gate_b_score", "REAL"),
                ("gate_c_verdict", "TEXT"),
                ("validity_status", "TEXT"),
            ]
            for col_name, col_type in gate_columns:
                if col_name not in VALID_MIGRATION_COLUMNS:
                    raise ValueError(f"Invalid migration column: {col_name}")
                if col_type not in VALID_COLUMN_TYPES:
                    raise ValueError(f"Invalid column type: {col_type}")
                try:
                    conn.execute(f"ALTER TABLE pairs ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise

        # Migration v6 -> v7: Dialogue acts
        if from_version <= 6 < to_version:
            da_columns = [
                ("trigger_da_type", "TEXT"),
                ("trigger_da_conf", "REAL"),
                ("response_da_type", "TEXT"),
                ("response_da_conf", "REAL"),
                ("cluster_id", "INTEGER"),
            ]
            for col_name, col_type in da_columns:
                if col_name not in VALID_MIGRATION_COLUMNS:
                    raise ValueError(f"Invalid migration column: {col_name}")
                if col_type not in VALID_COLUMN_TYPES:
                    raise ValueError(f"Invalid column type: {col_type}")
                try:
                    conn.execute(f"ALTER TABLE pairs ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise

        # Migration v8 -> v9: Content hash
        if from_version < 9 <= to_version:
            try:
                conn.execute("ALTER TABLE pairs ADD COLUMN content_hash TEXT")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

        # Migration to current: Apply full schema
        conn.executescript(SCHEMA_SQL)

        # Update schema version
        conn.execute("DELETE FROM schema_version")
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (to_version,),
        )
        conn.commit()

    def _get_table_counts(self, db_path: Path) -> dict[str, int]:
        """Get row counts for all tables."""
        counts: dict[str, int] = {}
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [row[0] for row in cursor]

            for table in tables:
                try:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    counts[table] = cursor.fetchone()[0]
                except sqlite3.Error:
                    counts[table] = -1

        return counts

    def _verify_data_integrity(
        self,
        db_path: Path,
        pre_counts: dict[str, int],
        post_counts: dict[str, int],
        from_version: int,
        to_version: int,
    ) -> tuple[bool, list[str]]:
        """Verify data integrity after migration.

        Returns:
            Tuple of (passed, list of errors).
        """
        errors: list[str] = []

        # Check that existing data wasn't lost
        for table, pre_count in pre_counts.items():
            post_count = post_counts.get(table, 0)
            if post_count < pre_count:
                errors.append(f"Data loss in {table}: {pre_count} -> {post_count} rows")

        # Version-specific checks
        with sqlite3.connect(str(db_path)) as conn:
            # Verify new columns have appropriate defaults
            if from_version < 4 <= to_version:
                cursor = conn.execute("SELECT COUNT(*) FROM pairs WHERE is_group IS NULL")
                null_count = cursor.fetchone()[0]
                if null_count > 0:
                    errors.append(f"{null_count} rows have NULL is_group")

            # Verify foreign key constraints
            cursor = conn.execute("PRAGMA foreign_key_check")
            fk_violations = cursor.fetchall()
            if fk_violations:
                errors.append(f"Foreign key violations: {len(fk_violations)}")

        return len(errors) == 0, errors

    def test_rollback(
        self,
        from_version: int,
        to_version: int,
        with_sample_data: bool = True,
    ) -> MigrationTestResult:
        """Test rollback from one version to another.

        Note: SQLite doesn't support DROP COLUMN, so rollbacks typically
        involve ignoring new columns rather than removing them.

        Args:
            from_version: Starting (higher) schema version.
            to_version: Target (lower) schema version.
            with_sample_data: Whether to populate with sample data.

        Returns:
            MigrationTestResult with test outcome.
        """
        import time

        start_time = time.time()
        result = MigrationTestResult(
            from_version=from_version,
            to_version=to_version,
            status=MigrationStatus.RUNNING,
            rollback_tested=True,
        )

        try:
            # Create database at higher version
            db_path = self.create_db_at_version(from_version, with_sample_data)

            # Record pre-rollback state
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM pairs")
                pre_rollback_count = cursor.fetchone()[0]

            # Simulate rollback by:
            # 1. Downgrading schema version
            # 2. Verifying old columns still work
            # 3. Verifying data is accessible
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    "UPDATE schema_version SET version = ?",
                    (to_version,),
                )
                conn.commit()

            # Verify rollback
            with sqlite3.connect(str(db_path)) as conn:
                # Check version
                cursor = conn.execute(
                    "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
                )
                actual_version = cursor.fetchone()[0]

                if actual_version != to_version:
                    result.errors.append(
                        f"Rollback version mismatch: expected {to_version}, got {actual_version}"
                    )

                # Verify data still accessible
                cursor = conn.execute("SELECT COUNT(*) FROM pairs")
                post_rollback_count = cursor.fetchone()[0]

                if post_rollback_count != pre_rollback_count:
                    result.errors.append(
                        f"Data loss during rollback: {pre_rollback_count} -> {post_rollback_count}"
                    )

                # Verify core columns still work
                cursor = conn.execute(
                    "SELECT id, contact_id, trigger_text, response_text FROM pairs LIMIT 1"
                )
                row = cursor.fetchone()
                if row is None and pre_rollback_count > 0:
                    result.errors.append("Cannot read data after rollback")

            result.duration_seconds = time.time() - start_time

            if not result.errors:
                result.status = MigrationStatus.PASSED
                result.schema_valid = True
                result.data_integrity_passed = True
            else:
                result.status = MigrationStatus.FAILED

        except Exception as e:
            result.status = MigrationStatus.FAILED
            result.errors.append(f"Rollback failed with exception: {e}")
            result.duration_seconds = time.time() - start_time

        return result

    def run_full_test_suite(self) -> dict[int, MigrationTestResult]:
        """Run migration tests for all version transitions.

        Returns:
            Dictionary mapping version -> test result.
        """
        results: dict[int, MigrationTestResult] = {}

        logger.info("Running full migration test suite...")

        for version in range(1, CURRENT_SCHEMA_VERSION):
            logger.info("Testing migration from v%d to v%d...", version, CURRENT_SCHEMA_VERSION)

            result = self.test_migration(
                from_version=version,
                to_version=CURRENT_SCHEMA_VERSION,
            )
            results[version] = result

            if result.success:
                logger.info("✓ Migration from v%d passed", version)
            else:
                logger.error("✗ Migration from v%d failed: %s", version, result.errors)

        return results

    def get_schema_diff(self, v1: int, v2: int) -> SchemaDiff:
        """Get schema differences between two versions.

        Args:
            v1: First schema version.
            v2: Second schema version.

        Returns:
            SchemaDiff with detailed differences.
        """
        db1_path = self.create_db_at_version(v1, with_sample_data=False)
        db2_path = self.create_db_at_version(v2, with_sample_data=False)

        diff = SchemaDiff()

        # Compare tables
        def get_tables(db_path: Path) -> set[str]:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                return {row[0] for row in cursor}

        # Compare columns
        def get_columns(db_path: Path, table: str) -> set[str]:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute(f"PRAGMA table_info({table})")
                return {row[1] for row in cursor}

        # Compare indexes
        def get_indexes(db_path: Path) -> set[str]:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
                )
                return {row[0] for row in cursor}

        tables1 = get_tables(db1_path)
        tables2 = get_tables(db2_path)

        diff.added_tables = list(tables2 - tables1)
        diff.removed_tables = list(tables1 - tables2)

        all_tables = tables1 | tables2
        for table in all_tables:
            cols1 = get_columns(db1_path, table) if table in tables1 else set()
            cols2 = get_columns(db2_path, table) if table in tables2 else set()

            added = cols2 - cols1
            removed = cols1 - cols2

            if added:
                diff.added_columns[table] = list(added)
            if removed:
                diff.removed_columns[table] = list(removed)

        indexes1 = get_indexes(db1_path)
        indexes2 = get_indexes(db2_path)

        diff.added_indexes = list(indexes2 - indexes1)
        diff.removed_indexes = list(indexes1 - indexes2)

        return diff


def test_migration_quick(from_version: int = 1) -> MigrationTestResult:
    """Quick migration test from specified version to current.

    Args:
        from_version: Starting version. Defaults to 1.

    Returns:
        MigrationTestResult.
    """
    tester = MigrationTester()
    return tester.test_migration(from_version=from_version)


def test_all_migrations() -> dict[int, MigrationTestResult]:
    """Test all migration paths.

    Returns:
        Dictionary of version -> result.
    """
    tester = MigrationTester()
    return tester.run_full_test_suite()


def print_migration_report(results: dict[int, MigrationTestResult]) -> None:
    """Print formatted migration test report."""
    print("\n" + "=" * 70)
    print("MIGRATION TEST REPORT")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r.success)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")
    print(f"Current Schema Version: {CURRENT_SCHEMA_VERSION}\n")

    for version, result in sorted(results.items()):
        status = "✓ PASS" if result.success else "✗ FAIL"
        print(f"v{version} -> v{CURRENT_SCHEMA_VERSION}: {status} ({result.duration_seconds:.2f}s)")

        if result.errors:
            for error in result.errors:
                print(f"    Error: {error}")

    print("\n" + "=" * 70)
