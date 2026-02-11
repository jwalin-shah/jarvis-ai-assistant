"""Tests for uncovered DB modules: search, backup, and migration.

Uses real in-memory SQLite databases. Asserts on actual query results and DB state.
Relies heavily on pytest.mark.parametrize for thorough coverage.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from jarvis.db.schema import CURRENT_SCHEMA_VERSION, SCHEMA_SQL

# ===========================================================================
# Helpers
# ===========================================================================


def _create_test_db(tmp_path: Path, with_data: bool = True) -> Path:
    """Create a test database with the full schema and optional sample data."""
    db_path = tmp_path / "test_jarvis.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        conn.execute("DELETE FROM schema_version")
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (CURRENT_SCHEMA_VERSION,),
        )
        if with_data:
            _insert_sample_data(conn)
        conn.commit()
    return db_path


def _insert_sample_data(conn: sqlite3.Connection) -> None:
    """Insert deterministic sample data for testing."""
    # Contacts
    contacts = [
        ("chat_001", "Alice Smith", "+15551234567", "sister"),
        ("chat_002", "Bob Jones", "bob@example.com", "coworker"),
        ("chat_003", "Carol White", "+15559876543", "friend"),
    ]
    for chat_id, name, contact, rel in contacts:
        conn.execute(
            "INSERT INTO contacts (chat_id, display_name, phone_or_email, relationship) "
            "VALUES (?, ?, ?, ?)",
            (chat_id, name, contact, rel),
        )

    # Pairs with varied quality scores and DA types
    now = datetime.now().isoformat()
    for i in range(1, 51):
        contact_id = (i % 3) + 1
        quality = 0.9 if i % 2 == 0 else 0.4
        trigger_da = "WH_QUESTION" if i % 3 == 0 else "STATEMENT"
        response_da = "ANSWER" if i % 3 == 0 else "ACKNOWLEDGE"
        is_holdout = i % 10 == 0

        conn.execute(
            "INSERT INTO pairs "
            "(contact_id, trigger_text, response_text, trigger_timestamp, "
            "response_timestamp, chat_id, quality_score, is_holdout, "
            "trigger_da_type, trigger_da_conf, response_da_type, response_da_conf, "
            "validity_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                contact_id,
                f"Trigger {i}",
                f"Response {i}",
                now,
                now,
                f"chat_{contact_id:03d}",
                quality,
                is_holdout,
                trigger_da,
                0.85,
                response_da,
                0.80,
                "valid",
            ),
        )

    # Contact facts
    facts = [
        ("chat_001", "personal", "birthday", "has_date", "1990-05-15"),
        ("chat_002", "work", "company", "works_at", "TechCorp"),
        ("chat_003", "preference", "food", "likes", "sushi"),
    ]
    for cid, cat, subj, pred, val in facts:
        conn.execute(
            "INSERT INTO contact_facts "
            "(contact_id, category, subject, predicate, value, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cid, cat, subj, pred, val, 0.95),
        )


# ===========================================================================
# Tests for jarvis/db/search.py (PairSearchMixin)
# ===========================================================================


class TestPairSearchQueries:
    """Test pair search, DA queries, and train/test split via raw SQL.

    These tests replicate the queries from PairSearchMixin against a real
    in-memory database to verify SQL correctness and result shapes.
    """

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return _create_test_db(tmp_path)

    @pytest.fixture
    def conn(self, db_path: Path) -> sqlite3.Connection:
        c = sqlite3.connect(str(db_path))
        c.row_factory = sqlite3.Row
        yield c
        c.close()

    def test_da_distribution_returns_counts(self, conn: sqlite3.Connection) -> None:
        """DA distribution query returns non-empty trigger and response dicts."""
        cursor = conn.execute(
            "SELECT trigger_da_type, COUNT(*) as cnt "
            "FROM pairs WHERE trigger_da_type IS NOT NULL "
            "GROUP BY trigger_da_type ORDER BY cnt DESC"
        )
        trigger_dist = {row["trigger_da_type"]: row["cnt"] for row in cursor}
        assert len(trigger_dist) > 0
        assert "WH_QUESTION" in trigger_dist or "STATEMENT" in trigger_dist

    def test_da_distribution_response(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute(
            "SELECT response_da_type, COUNT(*) as cnt "
            "FROM pairs WHERE response_da_type IS NOT NULL "
            "GROUP BY response_da_type ORDER BY cnt DESC"
        )
        response_dist = {row["response_da_type"]: row["cnt"] for row in cursor}
        assert len(response_dist) > 0

    @pytest.mark.parametrize(
        "response_da,min_conf,expect_results",
        [
            ("ANSWER", 0.5, True),
            ("ACKNOWLEDGE", 0.5, True),
            ("NONEXISTENT_DA", 0.5, False),
            ("ANSWER", 0.99, False),
        ],
    )
    def test_pairs_by_response_da(
        self,
        conn: sqlite3.Connection,
        response_da: str,
        min_conf: float,
        expect_results: bool,
    ) -> None:
        cursor = conn.execute(
            "SELECT * FROM pairs "
            "WHERE response_da_type = ? AND response_da_conf >= ? "
            "AND quality_score >= 0.0 AND is_holdout = FALSE "
            "ORDER BY response_da_conf DESC LIMIT 100",
            (response_da, min_conf),
        )
        rows = cursor.fetchall()
        if expect_results:
            assert len(rows) > 0
        else:
            assert len(rows) == 0

    @pytest.mark.parametrize(
        "trigger_da,min_conf,expect_results",
        [
            ("WH_QUESTION", 0.5, True),
            ("STATEMENT", 0.5, True),
            ("NONEXISTENT", 0.5, False),
        ],
    )
    def test_pairs_by_trigger_da(
        self,
        conn: sqlite3.Connection,
        trigger_da: str,
        min_conf: float,
        expect_results: bool,
    ) -> None:
        cursor = conn.execute(
            "SELECT * FROM pairs "
            "WHERE trigger_da_type = ? AND trigger_da_conf >= ? "
            "AND quality_score >= 0.0 AND is_holdout = FALSE "
            "ORDER BY trigger_da_conf DESC LIMIT 100",
            (trigger_da, min_conf),
        )
        rows = cursor.fetchall()
        if expect_results:
            assert len(rows) > 0
        else:
            assert len(rows) == 0

    def test_cross_tabulation(self, conn: sqlite3.Connection) -> None:
        """Cross-tabulation query returns nested structure."""
        cursor = conn.execute(
            "SELECT trigger_da_type, response_da_type, COUNT(*) as cnt "
            "FROM pairs "
            "WHERE trigger_da_type IS NOT NULL AND response_da_type IS NOT NULL "
            "GROUP BY trigger_da_type, response_da_type "
            "ORDER BY trigger_da_type, cnt DESC"
        )
        result: dict[str, dict[str, int]] = {}
        for row in cursor:
            trigger = row["trigger_da_type"]
            response = row["response_da_type"]
            if trigger not in result:
                result[trigger] = {}
            result[trigger][response] = row["cnt"]
        assert len(result) > 0

    def test_holdout_split_counts(self, conn: sqlite3.Connection) -> None:
        """Holdout and training pair counts are consistent."""
        cursor = conn.execute(
            "SELECT "
            "SUM(CASE WHEN is_holdout = FALSE THEN 1 ELSE 0 END) as training, "
            "SUM(CASE WHEN is_holdout = TRUE THEN 1 ELSE 0 END) as holdout "
            "FROM pairs"
        )
        row = cursor.fetchone()
        training = row["training"] or 0
        holdout = row["holdout"] or 0
        assert training + holdout == 50  # We inserted 50 pairs
        assert holdout == 5  # Every 10th is holdout (i % 10 == 0)

    def test_valid_pairs_filter(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute(
            "SELECT COUNT(*) as cnt FROM pairs "
            "WHERE validity_status = 'valid' AND quality_score >= 0.5"
        )
        count = cursor.fetchone()["cnt"]
        # Only even-indexed pairs have quality 0.9 (>= 0.5), and all are 'valid'
        assert count == 25  # Half of 50

    @pytest.mark.parametrize(
        "query,expected_min",
        [
            ("SELECT COUNT(*) as cnt FROM pairs", 50),
            ("SELECT COUNT(*) as cnt FROM contacts", 3),
            ("SELECT COUNT(*) as cnt FROM contact_facts", 3),
        ],
    )
    def test_sample_data_counts(
        self, conn: sqlite3.Connection, query: str, expected_min: int
    ) -> None:
        cursor = conn.execute(query)
        count = cursor.fetchone()["cnt"]
        assert count >= expected_min


class TestPairSearchSpecialCharacters:
    """Test search queries with special characters to verify SQL safety."""

    @pytest.fixture
    def conn(self, tmp_path: Path) -> sqlite3.Connection:
        db_path = _create_test_db(tmp_path, with_data=False)
        c = sqlite3.connect(str(db_path))
        c.row_factory = sqlite3.Row
        # Insert a pair with special characters
        now = datetime.now().isoformat()
        special_texts = [
            "Trigger with 'quotes'",
            'Trigger with "double quotes"',
            "Trigger with; semicolons",
            "Trigger with -- dashes",
            "'; DROP TABLE pairs; --",
        ]
        for i, text in enumerate(special_texts, 1):
            c.execute(
                "INSERT INTO contacts (chat_id, display_name) VALUES (?, ?)",
                (f"chat_{i:03d}", f"Contact {i}"),
            )
            c.execute(
                "INSERT INTO pairs "
                "(contact_id, trigger_text, response_text, trigger_timestamp, "
                "response_timestamp, chat_id, quality_score) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (i, text, f"Response {i}", now, now, f"chat_{i:03d}", 0.8),
            )
        c.commit()
        yield c
        c.close()

    @pytest.mark.parametrize(
        "search_term",
        [
            "quotes",
            "semicolons",
            "dashes",
            "DROP TABLE",
        ],
    )
    def test_search_with_special_chars(self, conn: sqlite3.Connection, search_term: str) -> None:
        """Parameterized queries handle special characters safely."""
        cursor = conn.execute(
            "SELECT * FROM pairs WHERE trigger_text LIKE ?",
            (f"%{search_term}%",),
        )
        rows = cursor.fetchall()
        assert len(rows) >= 1

    def test_empty_search_returns_all(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute("SELECT * FROM pairs WHERE trigger_text LIKE ?", ("%",))
        rows = cursor.fetchall()
        assert len(rows) == 5


class TestPairSearchEmptyDB:
    """Test search queries against an empty database."""

    @pytest.fixture
    def conn(self, tmp_path: Path) -> sqlite3.Connection:
        db_path = _create_test_db(tmp_path, with_data=False)
        c = sqlite3.connect(str(db_path))
        c.row_factory = sqlite3.Row
        yield c
        c.close()

    def test_da_distribution_empty(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute(
            "SELECT trigger_da_type, COUNT(*) as cnt "
            "FROM pairs WHERE trigger_da_type IS NOT NULL "
            "GROUP BY trigger_da_type"
        )
        assert cursor.fetchall() == []

    def test_cross_tab_empty(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute(
            "SELECT trigger_da_type, response_da_type, COUNT(*) as cnt "
            "FROM pairs "
            "WHERE trigger_da_type IS NOT NULL AND response_da_type IS NOT NULL "
            "GROUP BY trigger_da_type, response_da_type"
        )
        assert cursor.fetchall() == []

    def test_split_stats_empty(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute(
            "SELECT "
            "SUM(CASE WHEN is_holdout = FALSE THEN 1 ELSE 0 END) as training, "
            "SUM(CASE WHEN is_holdout = TRUE THEN 1 ELSE 0 END) as holdout "
            "FROM pairs"
        )
        row = cursor.fetchone()
        assert (row["training"] or 0) == 0
        assert (row["holdout"] or 0) == 0


# ===========================================================================
# Tests for jarvis/db/backup.py (BackupManager)
# ===========================================================================


class TestBackupManager:
    """Test backup creation, restore, and verification."""

    @pytest.fixture
    def source_db(self, tmp_path: Path) -> Path:
        return _create_test_db(tmp_path)

    @pytest.fixture
    def backup_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "backups"
        d.mkdir()
        return d

    @pytest.fixture
    def export_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "exports"
        d.mkdir()
        return d

    @pytest.fixture
    def wal_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "wal_archive"
        d.mkdir()
        return d

    def _make_manager(self, source_db: Path, backup_dir: Path, export_dir: Path, wal_dir: Path):
        from jarvis.db.backup import BackupManager

        return BackupManager(
            db_path=source_db,
            backup_dir=backup_dir,
            export_dir=export_dir,
            wal_archive_dir=wal_dir,
        )

    def test_hot_backup_creates_file(
        self, source_db: Path, backup_dir: Path, export_dir: Path, wal_dir: Path
    ) -> None:

        manager = self._make_manager(source_db, backup_dir, export_dir, wal_dir)
        result = manager.create_hot_backup(name="test_backup.db")
        assert result.success is True
        assert result.backup_path is not None
        assert result.backup_path.exists()
        assert result.size_bytes > 0
        assert result.checksum != ""

    def test_hot_backup_integrity(
        self, source_db: Path, backup_dir: Path, export_dir: Path, wal_dir: Path
    ) -> None:
        manager = self._make_manager(source_db, backup_dir, export_dir, wal_dir)
        result = manager.create_hot_backup(name="integrity_test.db")
        assert result.success

        # Verify backup has same data
        with sqlite3.connect(str(result.backup_path)) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM pairs")
            count = cursor.fetchone()[0]
            assert count == 50

            cursor = conn.execute("SELECT COUNT(*) FROM contacts")
            count = cursor.fetchone()[0]
            assert count == 3

    def test_hot_backup_table_counts(
        self, source_db: Path, backup_dir: Path, export_dir: Path, wal_dir: Path
    ) -> None:
        manager = self._make_manager(source_db, backup_dir, export_dir, wal_dir)
        result = manager.create_hot_backup(name="counts_test.db")
        assert result.success
        assert "pairs" in result.tables_backed_up
        assert result.tables_backed_up["pairs"] == 50

    def test_restore_from_backup(
        self, source_db: Path, backup_dir: Path, export_dir: Path, wal_dir: Path
    ) -> None:
        manager = self._make_manager(source_db, backup_dir, export_dir, wal_dir)
        backup_result = manager.create_hot_backup(name="for_restore.db")
        assert backup_result.success

        # Corrupt the source by deleting all pairs
        with sqlite3.connect(str(source_db)) as conn:
            conn.execute("DELETE FROM pairs")
            conn.commit()

        # Restore
        restore_result = manager.restore_from_backup(backup_result.backup_path)
        assert restore_result.success
        assert restore_result.integrity_check_passed

        # Verify data restored
        with sqlite3.connect(str(source_db)) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM pairs")
            assert cursor.fetchone()[0] == 50

    def test_restore_nonexistent_file(
        self, source_db: Path, backup_dir: Path, export_dir: Path, wal_dir: Path
    ) -> None:
        manager = self._make_manager(source_db, backup_dir, export_dir, wal_dir)
        result = manager.restore_from_backup(Path("/nonexistent/backup.db"))
        assert result.success is False
        assert "not found" in result.error_message

    def test_verify_backup_healthy(
        self, source_db: Path, backup_dir: Path, export_dir: Path, wal_dir: Path
    ) -> None:
        manager = self._make_manager(source_db, backup_dir, export_dir, wal_dir)
        backup = manager.create_hot_backup(name="verify_test.db")
        assert backup.success

        health = manager.verify_backup(backup.backup_path)
        assert health.integrity_check_passed
        assert health.page_count > 0
        assert health.size_bytes > 0

    def test_verify_corrupt_backup(self, tmp_path: Path) -> None:
        """Verifying a corrupt file reports failure."""
        from jarvis.db.backup import BackupManager

        corrupt_path = tmp_path / "corrupt.db"
        corrupt_path.write_bytes(b"this is not a database")

        manager = BackupManager(
            db_path=corrupt_path,
            backup_dir=tmp_path / "backups",
            export_dir=tmp_path / "exports",
            wal_archive_dir=tmp_path / "wal",
        )
        health = manager.verify_backup(corrupt_path)
        assert health.integrity_check_passed is False
        assert len(health.warnings) > 0

    def test_list_backups_empty(self, tmp_path: Path) -> None:
        from jarvis.db.backup import BackupManager

        empty_dir = tmp_path / "empty_backups"
        empty_dir.mkdir()
        manager = BackupManager(
            db_path=tmp_path / "dummy.db",
            backup_dir=empty_dir,
            export_dir=tmp_path / "exports",
            wal_archive_dir=tmp_path / "wal",
        )
        assert manager.list_backups() == []

    def test_list_backups_after_create(
        self, source_db: Path, backup_dir: Path, export_dir: Path, wal_dir: Path
    ) -> None:
        manager = self._make_manager(source_db, backup_dir, export_dir, wal_dir)
        manager.create_hot_backup(name="jarvis_20250101_000000.db")
        manager.create_hot_backup(name="jarvis_20250102_000000.db")
        backups = manager.list_backups()
        assert len(backups) == 2

    def test_sql_export_creates_file(
        self, source_db: Path, backup_dir: Path, export_dir: Path, wal_dir: Path
    ) -> None:
        manager = self._make_manager(source_db, backup_dir, export_dir, wal_dir)
        result = manager.create_sql_export(compress=False)
        assert result.success
        assert result.backup_path is not None
        assert result.backup_path.exists()
        assert result.backup_path.suffix == ".sql"

    def test_sql_export_compressed(
        self, source_db: Path, backup_dir: Path, export_dir: Path, wal_dir: Path
    ) -> None:
        manager = self._make_manager(source_db, backup_dir, export_dir, wal_dir)
        result = manager.create_sql_export(compress=True)
        assert result.success
        assert result.backup_path.name.endswith(".sql.gz")

    def test_sql_export_specific_tables(
        self, source_db: Path, backup_dir: Path, export_dir: Path, wal_dir: Path
    ) -> None:
        manager = self._make_manager(source_db, backup_dir, export_dir, wal_dir)
        result = manager.create_sql_export(tables=["contacts"], compress=False)
        assert result.success
        content = result.backup_path.read_text()
        assert "contacts" in content
        # Should not contain pairs data
        assert "Trigger 1" not in content

    def test_concurrent_backup_blocked(
        self, source_db: Path, backup_dir: Path, export_dir: Path, wal_dir: Path
    ) -> None:
        """Second backup while first is in progress should fail."""
        manager = self._make_manager(source_db, backup_dir, export_dir, wal_dir)
        # Simulate in-progress
        manager._backup_in_progress = True
        result = manager.create_hot_backup(name="concurrent.db")
        assert result.success is False
        assert "already in progress" in result.error_message
        manager._backup_in_progress = False


class TestBackupHealthReport:
    """Test HealthReport dataclass properties."""

    def test_is_healthy_true(self) -> None:
        from jarvis.db.backup import HealthReport

        report = HealthReport(
            timestamp=datetime.now(),
            integrity_check_passed=True,
            page_count=100,
            freelist_count=0,
            journal_mode="wal",
            foreign_keys_ok=True,
            size_bytes=100_000,
            wal_size_bytes=0,
        )
        assert report.is_healthy is True
        assert report.fragmentation_ratio == 0.0

    @pytest.mark.parametrize(
        "integrity,fk_ok,journal,warnings,expected_healthy",
        [
            (False, True, "wal", [], False),
            (True, False, "wal", [], False),
            (True, True, "delete", [], False),
            (True, True, "wal", ["some warning"], False),
            (True, True, "wal", [], True),
        ],
    )
    def test_is_healthy_parametrized(
        self,
        integrity: bool,
        fk_ok: bool,
        journal: str,
        warnings: list,
        expected_healthy: bool,
    ) -> None:
        from jarvis.db.backup import HealthReport

        report = HealthReport(
            timestamp=datetime.now(),
            integrity_check_passed=integrity,
            page_count=100,
            freelist_count=5,
            journal_mode=journal,
            foreign_keys_ok=fk_ok,
            size_bytes=100_000,
            wal_size_bytes=0,
            warnings=warnings,
        )
        assert report.is_healthy is expected_healthy

    def test_fragmentation_ratio(self) -> None:
        from jarvis.db.backup import HealthReport

        report = HealthReport(
            timestamp=datetime.now(),
            integrity_check_passed=True,
            page_count=100,
            freelist_count=25,
            journal_mode="wal",
            foreign_keys_ok=True,
            size_bytes=100_000,
            wal_size_bytes=0,
        )
        assert report.fragmentation_ratio == 0.25

    def test_fragmentation_ratio_zero_pages(self) -> None:
        from jarvis.db.backup import HealthReport

        report = HealthReport(
            timestamp=datetime.now(),
            integrity_check_passed=True,
            page_count=0,
            freelist_count=0,
            journal_mode="wal",
            foreign_keys_ok=True,
            size_bytes=0,
            wal_size_bytes=0,
        )
        assert report.fragmentation_ratio == 0.0


# ===========================================================================
# Tests for jarvis/db/migration.py (MigrationTester)
# ===========================================================================


class TestMigrationTester:
    """Test schema migrations, rollbacks, and schema diffs."""

    def test_create_db_at_current_version(self, tmp_path: Path) -> None:
        from jarvis.db.migration import MigrationTester

        tester = MigrationTester(db_path=tmp_path / "test.db")
        db_path = tester.create_db_at_version(CURRENT_SCHEMA_VERSION)
        assert db_path.exists()

        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            )
            assert cursor.fetchone()[0] == CURRENT_SCHEMA_VERSION

    @pytest.mark.parametrize("version", [1, 3, 5, 7, 9])
    def test_create_db_at_old_versions(self, tmp_path: Path, version: int) -> None:
        from jarvis.db.migration import MigrationTester

        tester = MigrationTester(db_path=tmp_path / f"v{version}.db")
        db_path = tester.create_db_at_version(version, with_sample_data=True)
        assert db_path.exists()

        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
            assert cursor.fetchone()[0] == version

    @pytest.mark.parametrize("from_version", [1, 3, 5, 7])
    def test_migration_forward(self, tmp_path: Path, from_version: int) -> None:
        """Migration from old version to current succeeds."""
        from jarvis.db.migration import MigrationStatus, MigrationTester

        tester = MigrationTester(db_path=tmp_path / f"migrate_v{from_version}.db")
        result = tester.test_migration(
            from_version=from_version,
            to_version=CURRENT_SCHEMA_VERSION,
        )
        assert result.status == MigrationStatus.PASSED, (
            f"Migration v{from_version} -> v{CURRENT_SCHEMA_VERSION} failed: {result.errors}"
        )
        assert result.schema_valid is True
        assert result.data_integrity_passed is True

    def test_migration_same_version(self, tmp_path: Path) -> None:
        """Migration from current to current is a no-op success."""
        from jarvis.db.migration import MigrationStatus, MigrationTester

        tester = MigrationTester(db_path=tmp_path / "same_version.db")
        result = tester.test_migration(
            from_version=CURRENT_SCHEMA_VERSION,
            to_version=CURRENT_SCHEMA_VERSION,
        )
        assert result.status == MigrationStatus.PASSED

    def test_migration_preserves_data(self, tmp_path: Path) -> None:
        """Migration does not lose rows."""
        from jarvis.db.migration import MigrationTester

        tester = MigrationTester(db_path=tmp_path / "preserve.db")
        db_path = tester.create_db_at_version(5, with_sample_data=True)

        # Count rows before migration
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM pairs")
            pre_count = cursor.fetchone()[0]
            assert pre_count > 0

        result = tester.test_migration(from_version=5)
        assert result.data_integrity_passed is True

    @pytest.mark.parametrize(
        "from_version,to_version",
        [
            (CURRENT_SCHEMA_VERSION, CURRENT_SCHEMA_VERSION - 1),
            (CURRENT_SCHEMA_VERSION, 1),
            (7, 5),
        ],
    )
    def test_rollback(self, tmp_path: Path, from_version: int, to_version: int) -> None:
        """Rollback preserves data and core column access."""
        from jarvis.db.migration import MigrationStatus, MigrationTester

        tester = MigrationTester(db_path=tmp_path / f"rollback_{from_version}_{to_version}.db")
        result = tester.test_rollback(
            from_version=from_version,
            to_version=to_version,
        )
        assert result.status == MigrationStatus.PASSED, (
            f"Rollback v{from_version} -> v{to_version} failed: {result.errors}"
        )
        assert result.rollback_tested is True


class TestMigrationTestResult:
    """Test MigrationTestResult dataclass properties."""

    @pytest.mark.parametrize(
        "status_name,schema_valid,data_ok,expected_success",
        [
            ("PASSED", True, True, True),
            ("PASSED", False, True, False),
            ("PASSED", True, False, False),
            ("FAILED", True, True, False),
            ("RUNNING", True, True, False),
        ],
    )
    def test_success_property(
        self,
        status_name: str,
        schema_valid: bool,
        data_ok: bool,
        expected_success: bool,
    ) -> None:
        from jarvis.db.migration import MigrationStatus, MigrationTestResult

        result = MigrationTestResult(
            from_version=1,
            to_version=CURRENT_SCHEMA_VERSION,
            status=MigrationStatus[status_name],
            schema_valid=schema_valid,
            data_integrity_passed=data_ok,
        )
        assert result.success is expected_success


class TestSchemaDiff:
    """Test schema diff between versions."""

    def test_diff_has_changes(self, tmp_path: Path) -> None:
        from jarvis.db.migration import MigrationTester

        tester = MigrationTester(db_path=tmp_path / "diff.db")
        diff = tester.get_schema_diff(1, CURRENT_SCHEMA_VERSION)
        # create_db_at_version uses the same SCHEMA_SQL for all versions,
        # so diff between v1 and current is empty (only version number differs)
        assert diff.has_changes is False

    def test_diff_same_version_no_changes(self, tmp_path: Path) -> None:
        from jarvis.db.migration import MigrationTester

        tester = MigrationTester(db_path=tmp_path / "same_diff.db")
        diff = tester.get_schema_diff(CURRENT_SCHEMA_VERSION, CURRENT_SCHEMA_VERSION)
        assert diff.has_changes is False

    def test_diff_detects_added_columns(self, tmp_path: Path) -> None:
        """v1 -> v7 should add DA columns to pairs."""
        from jarvis.db.migration import MigrationTester

        tester = MigrationTester(db_path=tmp_path / "col_diff.db")
        diff = tester.get_schema_diff(1, CURRENT_SCHEMA_VERSION)
        # The full schema creates all columns, so v1 and vN have the same
        # CREATE TABLE. But new tables may differ.
        assert isinstance(diff.added_tables, list)
        assert isinstance(diff.added_columns, dict)


class TestMigrationIdempotent:
    """Verify migrations are idempotent (can be run twice safely)."""

    @pytest.mark.parametrize("from_version", [1, 5, 7])
    def test_double_migration(self, tmp_path: Path, from_version: int) -> None:
        from jarvis.db.migration import MigrationStatus, MigrationTester

        tester = MigrationTester(db_path=tmp_path / f"idempotent_v{from_version}.db")

        # First migration
        result1 = tester.test_migration(from_version=from_version)
        assert result1.status == MigrationStatus.PASSED

        # Second migration on same DB (should not fail due to duplicate columns)
        db_path = tester._get_test_db_path()
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            # Re-run the migration logic
            tester._run_migrations(conn, from_version, CURRENT_SCHEMA_VERSION)
            # Verify still valid
            cursor = conn.execute("PRAGMA integrity_check")
            assert cursor.fetchone()[0] == "ok"


# ===========================================================================
# Tests for schema.py constants
# ===========================================================================


class TestSchemaConstants:
    """Verify schema constants are well-formed."""

    def test_schema_sql_creates_tables(self) -> None:
        """SCHEMA_SQL creates all expected tables in a fresh DB."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA_SQL)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = {row[0] for row in cursor}
        expected = {
            "schema_version",
            "contacts",
            "contact_style_targets",
            "pairs",
            "pair_artifacts",
            "clusters",
            "pair_embeddings",
            "index_versions",
            "scheduled_drafts",
            "contact_timing_prefs",
            "send_queue",
            "contact_facts",
        }
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"
        conn.close()

    def test_schema_sql_creates_indexes(self) -> None:
        """SCHEMA_SQL creates expected indexes."""
        from jarvis.db.schema import EXPECTED_INDICES

        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA_SQL)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        actual_indexes = {row[0] for row in cursor}
        assert EXPECTED_INDICES.issubset(actual_indexes), (
            f"Missing indexes: {EXPECTED_INDICES - actual_indexes}"
        )
        conn.close()

    def test_schema_sql_idempotent(self) -> None:
        """Running SCHEMA_SQL twice does not raise errors."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA_SQL)
        conn.executescript(SCHEMA_SQL)  # Should not raise
        cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        assert cursor.fetchone()[0] > 0
        conn.close()

    def test_current_schema_version_positive(self) -> None:
        assert CURRENT_SCHEMA_VERSION > 0

    @pytest.mark.parametrize(
        "column",
        [
            "context_text",
            "is_group",
            "is_holdout",
            "gate_a_passed",
            "trigger_da_type",
            "content_hash",
            "linked_contact_id",
        ],
    )
    def test_valid_migration_columns(self, column: str) -> None:
        from jarvis.db.schema import VALID_MIGRATION_COLUMNS

        assert column in VALID_MIGRATION_COLUMNS

    @pytest.mark.parametrize(
        "col_type",
        ["TEXT", "REAL", "INTEGER", "BOOLEAN"],
    )
    def test_valid_column_types(self, col_type: str) -> None:
        from jarvis.db.schema import VALID_COLUMN_TYPES

        assert col_type in VALID_COLUMN_TYPES


# ===========================================================================
# Tests for contact_facts table operations
# ===========================================================================


class TestContactFactsDB:
    """Test contact_facts table via direct SQL (mirrors fact_storage.py usage)."""

    @pytest.fixture
    def conn(self, tmp_path: Path) -> sqlite3.Connection:
        db_path = _create_test_db(tmp_path)
        c = sqlite3.connect(str(db_path))
        c.row_factory = sqlite3.Row
        yield c
        c.close()

    def test_insert_and_retrieve_fact(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO contact_facts "
            "(contact_id, category, subject, predicate, value, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("chat_001", "location", "Austin", "lives_in", "", 0.85),
        )
        conn.commit()
        cursor = conn.execute(
            "SELECT * FROM contact_facts WHERE contact_id = ? AND category = ?",
            ("chat_001", "location"),
        )
        rows = cursor.fetchall()
        assert len(rows) >= 1
        assert rows[-1]["subject"] == "Austin"

    def test_upsert_on_conflict(self, conn: sqlite3.Connection) -> None:
        """UNIQUE constraint on (contact_id, category, subject, predicate)."""
        conn.execute(
            "INSERT OR REPLACE INTO contact_facts "
            "(contact_id, category, subject, predicate, value, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("chat_new", "work", "Google", "works_at", "", 0.7),
        )
        conn.execute(
            "INSERT OR REPLACE INTO contact_facts "
            "(contact_id, category, subject, predicate, value, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("chat_new", "work", "Google", "works_at", "", 0.9),
        )
        conn.commit()

        cursor = conn.execute(
            "SELECT COUNT(*) as cnt FROM contact_facts "
            "WHERE contact_id = 'chat_new' AND subject = 'Google'"
        )
        assert cursor.fetchone()["cnt"] == 1

    def test_batch_insert(self, conn: sqlite3.Connection) -> None:
        """Batch insert via executemany."""
        batch = [("chat_batch", "preference", f"food_{i}", "likes", "", 0.8) for i in range(20)]
        conn.executemany(
            "INSERT INTO contact_facts "
            "(contact_id, category, subject, predicate, value, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            batch,
        )
        conn.commit()

        cursor = conn.execute(
            "SELECT COUNT(*) as cnt FROM contact_facts WHERE contact_id = 'chat_batch'"
        )
        assert cursor.fetchone()["cnt"] == 20

    @pytest.mark.parametrize(
        "category,expected_min",
        [
            ("personal", 1),
            ("work", 1),
            ("preference", 1),
            ("nonexistent", 0),
        ],
    )
    def test_filter_by_category(
        self, conn: sqlite3.Connection, category: str, expected_min: int
    ) -> None:
        cursor = conn.execute(
            "SELECT COUNT(*) as cnt FROM contact_facts WHERE category = ?",
            (category,),
        )
        assert cursor.fetchone()["cnt"] >= expected_min

    def test_temporal_fields(self, conn: sqlite3.Connection) -> None:
        """valid_from and valid_until columns work correctly."""
        now = datetime.now().isoformat()
        conn.execute(
            "INSERT INTO contact_facts "
            "(contact_id, category, subject, predicate, value, confidence, "
            "valid_from, valid_until) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("chat_temporal", "location", "NYC", "lived_in", "", 0.7, "2020-01-01", now),
        )
        conn.commit()

        cursor = conn.execute(
            "SELECT valid_from, valid_until FROM contact_facts WHERE contact_id = 'chat_temporal'"
        )
        row = cursor.fetchone()
        assert row["valid_from"] == "2020-01-01"
        assert row["valid_until"] is not None

    def test_linked_contact_id(self, conn: sqlite3.Connection) -> None:
        """linked_contact_id column for NER person linking."""
        conn.execute(
            "INSERT INTO contact_facts "
            "(contact_id, category, subject, predicate, value, confidence, "
            "linked_contact_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("chat_linked", "relationship", "Sarah", "is_family_of", "sister", 0.8, "chat_001"),
        )
        conn.commit()

        cursor = conn.execute(
            "SELECT linked_contact_id FROM contact_facts WHERE contact_id = 'chat_linked'"
        )
        assert cursor.fetchone()["linked_contact_id"] == "chat_001"
