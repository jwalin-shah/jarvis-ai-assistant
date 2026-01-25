"""Unit tests for Workstream 7: Permission Monitor and Schema Detector.

Tests TCC permission monitoring and chat.db schema detection.
"""

import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from contracts.health import Permission, PermissionStatus, SchemaInfo
from core.health.permissions import (
    FIX_INSTRUCTIONS,
    TCCPermissionMonitor,
    get_permission_monitor,
    reset_permission_monitor,
)
from core.health.schema import (
    KNOWN_SCHEMAS,
    ChatDBSchemaDetector,
    get_schema_detector,
    reset_schema_detector,
)

# =============================================================================
# TCCPermissionMonitor Tests
# =============================================================================


class TestTCCPermissionMonitorInitialization:
    """Tests for TCCPermissionMonitor initialization."""

    def test_default_initialization(self):
        """Monitor initializes with default paths."""
        monitor = TCCPermissionMonitor()
        assert monitor._messages_db_path == Path.home() / "Library" / "Messages" / "chat.db"

    def test_custom_paths(self):
        """Monitor accepts custom paths."""
        custom_path = Path("/custom/path/chat.db")
        monitor = TCCPermissionMonitor(messages_db_path=custom_path)
        assert monitor._messages_db_path == custom_path

    def test_singleton_pattern(self):
        """Get singleton returns same instance."""
        reset_permission_monitor()
        m1 = get_permission_monitor()
        m2 = get_permission_monitor()
        assert m1 is m2
        reset_permission_monitor()

    def test_reset_singleton(self):
        """Reset creates new singleton instance."""
        reset_permission_monitor()
        m1 = get_permission_monitor()
        reset_permission_monitor()
        m2 = get_permission_monitor()
        assert m1 is not m2
        reset_permission_monitor()


class TestPermissionChecks:
    """Tests for individual permission checks."""

    def test_check_full_disk_access_granted(self, tmp_path):
        """Full disk access returns True when file is readable."""
        # Create a mock chat.db file
        db_path = tmp_path / "chat.db"
        db_path.write_bytes(b"SQLite format 3\0" + b"\0" * 100)

        monitor = TCCPermissionMonitor(messages_db_path=db_path)
        status = monitor.check_permission(Permission.FULL_DISK_ACCESS)

        assert status.granted is True
        assert status.permission == Permission.FULL_DISK_ACCESS
        assert status.fix_instructions == ""

    def test_check_full_disk_access_denied_no_file(self, tmp_path):
        """Full disk access returns False when file doesn't exist."""
        db_path = tmp_path / "nonexistent.db"
        monitor = TCCPermissionMonitor(messages_db_path=db_path)
        status = monitor.check_permission(Permission.FULL_DISK_ACCESS)

        assert status.granted is False
        assert status.fix_instructions != ""

    def test_check_full_disk_access_denied_permission_error(self, tmp_path):
        """Full disk access returns False on PermissionError."""
        db_path = tmp_path / "chat.db"
        monitor = TCCPermissionMonitor(messages_db_path=db_path)

        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            with patch.object(Path, "exists", return_value=True):
                status = monitor.check_permission(Permission.FULL_DISK_ACCESS)

        assert status.granted is False

    def test_check_contacts_access_granted(self, tmp_path):
        """Contacts access returns True when directory is readable."""
        contacts_path = tmp_path / "AddressBook"
        contacts_path.mkdir()
        (contacts_path / "test.db").touch()

        monitor = TCCPermissionMonitor(contacts_path=contacts_path)
        status = monitor.check_permission(Permission.CONTACTS)

        assert status.granted is True

    def test_check_contacts_access_denied(self, tmp_path):
        """Contacts access returns False when directory doesn't exist."""
        contacts_path = tmp_path / "nonexistent"
        monitor = TCCPermissionMonitor(contacts_path=contacts_path)
        status = monitor.check_permission(Permission.CONTACTS)

        assert status.granted is False

    def test_check_calendar_access_granted(self, tmp_path):
        """Calendar access returns True when directory is readable."""
        calendar_path = tmp_path / "Calendars"
        calendar_path.mkdir()

        monitor = TCCPermissionMonitor(calendar_path=calendar_path)
        status = monitor.check_permission(Permission.CALENDAR)

        assert status.granted is True

    def test_check_calendar_access_denied(self, tmp_path):
        """Calendar access returns False when directory doesn't exist."""
        calendar_path = tmp_path / "nonexistent"
        monitor = TCCPermissionMonitor(calendar_path=calendar_path)
        status = monitor.check_permission(Permission.CALENDAR)

        assert status.granted is False

    def test_check_automation_always_true(self):
        """Automation permission returns True by default."""
        monitor = TCCPermissionMonitor()
        status = monitor.check_permission(Permission.AUTOMATION)

        assert status.granted is True


class TestPermissionStatus:
    """Tests for PermissionStatus dataclass."""

    def test_status_contains_timestamp(self, tmp_path):
        """Status includes ISO timestamp."""
        db_path = tmp_path / "chat.db"
        monitor = TCCPermissionMonitor(messages_db_path=db_path)
        status = monitor.check_permission(Permission.FULL_DISK_ACCESS)

        assert status.last_checked is not None
        # Should be ISO 8601 format
        assert "T" in status.last_checked

    def test_status_contains_fix_instructions_when_denied(self, tmp_path):
        """Fix instructions provided when permission is denied."""
        db_path = tmp_path / "nonexistent.db"
        monitor = TCCPermissionMonitor(messages_db_path=db_path)
        status = monitor.check_permission(Permission.FULL_DISK_ACCESS)

        assert status.granted is False
        assert "System Preferences" in status.fix_instructions
        assert "Full Disk Access" in status.fix_instructions

    def test_fix_instructions_empty_when_granted(self, tmp_path):
        """Fix instructions empty when permission is granted."""
        db_path = tmp_path / "chat.db"
        db_path.write_bytes(b"SQLite format 3\0" + b"\0" * 100)

        monitor = TCCPermissionMonitor(messages_db_path=db_path)
        status = monitor.check_permission(Permission.FULL_DISK_ACCESS)

        assert status.granted is True
        assert status.fix_instructions == ""


class TestCheckAll:
    """Tests for check_all method."""

    def test_check_all_returns_all_permissions(self, tmp_path):
        """check_all returns status for all permission types."""
        monitor = TCCPermissionMonitor(
            messages_db_path=tmp_path / "chat.db",
            contacts_path=tmp_path / "contacts",
            calendar_path=tmp_path / "calendar",
        )
        statuses = monitor.check_all()

        assert len(statuses) == len(Permission)
        permissions_checked = {s.permission for s in statuses}
        assert permissions_checked == set(Permission)

    def test_check_all_includes_granted_and_denied(self, tmp_path):
        """check_all can include both granted and denied permissions."""
        # Set up: one granted (automation), others denied (paths don't exist)
        monitor = TCCPermissionMonitor(
            messages_db_path=tmp_path / "chat.db",
            contacts_path=tmp_path / "contacts",
            calendar_path=tmp_path / "calendar",
        )
        statuses = monitor.check_all()

        # Automation should be granted (always True)
        automation_status = next(s for s in statuses if s.permission == Permission.AUTOMATION)
        assert automation_status.granted is True

        # Full disk access should be denied (file doesn't exist)
        fda_status = next(s for s in statuses if s.permission == Permission.FULL_DISK_ACCESS)
        assert fda_status.granted is False


class TestWaitForPermission:
    """Tests for wait_for_permission method."""

    def test_wait_returns_true_when_already_granted(self, tmp_path):
        """wait_for_permission returns immediately when permission granted."""
        db_path = tmp_path / "chat.db"
        db_path.write_bytes(b"SQLite format 3\0" + b"\0" * 100)

        monitor = TCCPermissionMonitor(messages_db_path=db_path)
        start = time.time()
        result = monitor.wait_for_permission(Permission.FULL_DISK_ACCESS, timeout_seconds=5)
        elapsed = time.time() - start

        assert result is True
        assert elapsed < 1  # Should return almost immediately

    def test_wait_returns_false_on_timeout(self, tmp_path):
        """wait_for_permission returns False after timeout."""
        db_path = tmp_path / "nonexistent.db"
        monitor = TCCPermissionMonitor(messages_db_path=db_path)

        start = time.time()
        result = monitor.wait_for_permission(Permission.FULL_DISK_ACCESS, timeout_seconds=2)
        elapsed = time.time() - start

        assert result is False
        assert elapsed >= 2  # Should wait for full timeout

    def test_wait_returns_true_when_permission_appears(self, tmp_path):
        """wait_for_permission returns True when permission becomes available."""
        db_path = tmp_path / "chat.db"
        monitor = TCCPermissionMonitor(messages_db_path=db_path)

        # Create file after a delay in another thread
        def create_file():
            time.sleep(0.5)
            db_path.write_bytes(b"SQLite format 3\0" + b"\0" * 100)

        thread = threading.Thread(target=create_file)
        thread.start()

        result = monitor.wait_for_permission(Permission.FULL_DISK_ACCESS, timeout_seconds=5)
        thread.join()

        assert result is True


class TestGetSummary:
    """Tests for get_summary method."""

    def test_get_summary_returns_dict(self, tmp_path):
        """get_summary returns dictionary mapping names to booleans."""
        monitor = TCCPermissionMonitor(
            messages_db_path=tmp_path / "chat.db",
        )
        summary = monitor.get_summary()

        assert isinstance(summary, dict)
        assert all(isinstance(k, str) for k in summary.keys())
        assert all(isinstance(v, bool) for v in summary.values())

    def test_get_summary_uses_permission_values(self, tmp_path):
        """get_summary keys match Permission enum values."""
        monitor = TCCPermissionMonitor(messages_db_path=tmp_path / "chat.db")
        summary = monitor.get_summary()

        expected_keys = {p.value for p in Permission}
        assert set(summary.keys()) == expected_keys


class TestFixInstructions:
    """Tests for fix instruction constants."""

    def test_all_permissions_have_instructions(self):
        """All permission types have fix instructions."""
        for permission in Permission:
            assert permission in FIX_INSTRUCTIONS
            assert len(FIX_INSTRUCTIONS[permission]) > 0

    def test_instructions_mention_system_preferences(self):
        """Instructions include System Preferences path."""
        for permission in Permission:
            instructions = FIX_INSTRUCTIONS[permission]
            assert "System Preferences" in instructions or "System Settings" in instructions


class TestPermissionMonitorThreadSafety:
    """Thread safety tests for TCCPermissionMonitor."""

    def test_concurrent_checks(self, tmp_path):
        """Concurrent permission checks are thread-safe."""
        db_path = tmp_path / "chat.db"
        db_path.write_bytes(b"SQLite format 3\0" + b"\0" * 100)
        monitor = TCCPermissionMonitor(messages_db_path=db_path)

        results = []
        lock = threading.Lock()

        def check_permissions():
            for _ in range(20):
                status = monitor.check_permission(Permission.FULL_DISK_ACCESS)
                with lock:
                    results.append(status.granted)

        threads = [threading.Thread(target=check_permissions) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 100
        assert all(r is True for r in results)


class TestProtocolCompliance:
    """Verify TCCPermissionMonitor implements PermissionMonitor protocol."""

    def test_has_check_permission(self):
        """Monitor has check_permission method."""
        monitor = TCCPermissionMonitor()
        assert hasattr(monitor, "check_permission")
        assert callable(monitor.check_permission)

    def test_has_check_all(self):
        """Monitor has check_all method."""
        monitor = TCCPermissionMonitor()
        assert hasattr(monitor, "check_all")
        assert callable(monitor.check_all)

    def test_has_wait_for_permission(self):
        """Monitor has wait_for_permission method."""
        monitor = TCCPermissionMonitor()
        assert hasattr(monitor, "wait_for_permission")
        assert callable(monitor.wait_for_permission)

    def test_check_permission_returns_permission_status(self, tmp_path):
        """check_permission returns PermissionStatus."""
        monitor = TCCPermissionMonitor(messages_db_path=tmp_path / "chat.db")
        result = monitor.check_permission(Permission.FULL_DISK_ACCESS)
        assert isinstance(result, PermissionStatus)

    def test_check_all_returns_list_of_permission_status(self, tmp_path):
        """check_all returns list of PermissionStatus."""
        monitor = TCCPermissionMonitor(messages_db_path=tmp_path / "chat.db")
        result = monitor.check_all()
        assert isinstance(result, list)
        assert all(isinstance(s, PermissionStatus) for s in result)

    def test_wait_for_permission_returns_bool(self, tmp_path):
        """wait_for_permission returns boolean."""
        monitor = TCCPermissionMonitor(messages_db_path=tmp_path / "chat.db")
        result = monitor.wait_for_permission(Permission.AUTOMATION, timeout_seconds=1)
        assert isinstance(result, bool)


# =============================================================================
# ChatDBSchemaDetector Tests
# =============================================================================


class TestChatDBSchemaDetectorInitialization:
    """Tests for ChatDBSchemaDetector initialization."""

    def test_initialization(self):
        """Detector initializes correctly."""
        detector = ChatDBSchemaDetector()
        assert detector._schema_cache == {}

    def test_singleton_pattern(self):
        """Get singleton returns same instance."""
        reset_schema_detector()
        d1 = get_schema_detector()
        d2 = get_schema_detector()
        assert d1 is d2
        reset_schema_detector()

    def test_reset_singleton(self):
        """Reset creates new singleton instance."""
        reset_schema_detector()
        d1 = get_schema_detector()
        reset_schema_detector()
        d2 = get_schema_detector()
        assert d1 is not d2
        reset_schema_detector()


class TestSchemaDetection:
    """Tests for schema version detection."""

    def _create_v14_database(self, db_path: Path) -> None:
        """Create a minimal v14-like chat.db for testing."""
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create core tables
        cursor.execute("""
            CREATE TABLE message (
                ROWID INTEGER PRIMARY KEY,
                text TEXT,
                handle_id INTEGER,
                date INTEGER,
                is_from_me INTEGER,
                attributedBody BLOB,
                thread_originator_guid TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE chat (
                ROWID INTEGER PRIMARY KEY,
                guid TEXT,
                display_name TEXT,
                chat_identifier TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE handle (
                ROWID INTEGER PRIMARY KEY,
                id TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE chat_message_join (
                chat_id INTEGER,
                message_id INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE chat_handle_join (
                chat_id INTEGER,
                handle_id INTEGER
            )
        """)

        conn.commit()
        conn.close()

    def _create_v15_database(self, db_path: Path) -> None:
        """Create a minimal v15-like chat.db for testing."""
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create core tables with v15-specific columns
        cursor.execute("""
            CREATE TABLE message (
                ROWID INTEGER PRIMARY KEY,
                text TEXT,
                handle_id INTEGER,
                date INTEGER,
                is_from_me INTEGER,
                attributedBody BLOB,
                thread_originator_guid TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE chat (
                ROWID INTEGER PRIMARY KEY,
                guid TEXT,
                display_name TEXT,
                chat_identifier TEXT,
                service_name TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE handle (
                ROWID INTEGER PRIMARY KEY,
                id TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE chat_message_join (
                chat_id INTEGER,
                message_id INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE chat_handle_join (
                chat_id INTEGER,
                handle_id INTEGER
            )
        """)

        conn.commit()
        conn.close()

    def test_detect_v14_schema(self, tmp_path):
        """Detects v14 schema correctly."""
        db_path = tmp_path / "chat.db"
        self._create_v14_database(db_path)

        detector = ChatDBSchemaDetector()
        info = detector.detect(str(db_path))

        assert info.version == "v14"
        assert info.known_schema is True
        assert info.compatible is True
        assert "message" in info.tables
        assert "chat" in info.tables

    def test_detect_v15_schema(self, tmp_path):
        """Detects v15 schema correctly."""
        db_path = tmp_path / "chat.db"
        self._create_v15_database(db_path)

        detector = ChatDBSchemaDetector()
        info = detector.detect(str(db_path))

        assert info.version == "v15"
        assert info.known_schema is True
        assert info.compatible is True

    def test_detect_nonexistent_file(self, tmp_path):
        """Returns unknown for nonexistent file."""
        db_path = tmp_path / "nonexistent.db"
        detector = ChatDBSchemaDetector()
        info = detector.detect(str(db_path))

        assert info.version == "unknown"
        assert info.compatible is False
        assert info.known_schema is False
        assert info.tables == []

    def test_caches_result(self, tmp_path):
        """Schema detection result is cached."""
        db_path = tmp_path / "chat.db"
        self._create_v14_database(db_path)

        detector = ChatDBSchemaDetector()
        info1 = detector.detect(str(db_path))
        info2 = detector.detect(str(db_path))

        # Should be same object from cache
        assert info1 is info2

    def test_clear_cache(self, tmp_path):
        """clear_cache removes cached results."""
        db_path = tmp_path / "chat.db"
        self._create_v14_database(db_path)

        detector = ChatDBSchemaDetector()
        info1 = detector.detect(str(db_path))
        detector.clear_cache()
        info2 = detector.detect(str(db_path))

        # Should be different objects after cache clear
        assert info1 is not info2


class TestSchemaInfo:
    """Tests for SchemaInfo dataclass."""

    def test_schema_info_has_all_fields(self, tmp_path):
        """SchemaInfo contains all required fields."""
        db_path = tmp_path / "chat.db"

        # Create minimal v14 db
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE message (ROWID INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE chat (ROWID INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE handle (ROWID INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.close()

        detector = ChatDBSchemaDetector()
        info = detector.detect(str(db_path))

        assert hasattr(info, "version")
        assert hasattr(info, "tables")
        assert hasattr(info, "compatible")
        assert hasattr(info, "migration_needed")
        assert hasattr(info, "known_schema")


class TestGetQuery:
    """Tests for get_query method."""

    def test_get_conversations_query(self):
        """Returns conversations query."""
        detector = ChatDBSchemaDetector()
        query = detector.get_query("conversations", "v14")

        assert "SELECT" in query
        assert "chat" in query
        assert "LIMIT" in query

    def test_get_messages_query(self):
        """Returns messages query."""
        detector = ChatDBSchemaDetector()
        query = detector.get_query("messages", "v14")

        assert "SELECT" in query
        assert "message" in query
        assert "chat.guid = ?" in query

    def test_get_search_query(self):
        """Returns search query."""
        detector = ChatDBSchemaDetector()
        query = detector.get_query("search", "v14")

        assert "SELECT" in query
        assert "LIKE" in query

    def test_get_context_query(self):
        """Returns context query."""
        detector = ChatDBSchemaDetector()
        query = detector.get_query("context", "v14")

        assert "SELECT" in query
        assert "ABS" in query

    def test_unknown_query_raises_key_error(self):
        """Raises KeyError for unknown query name."""
        detector = ChatDBSchemaDetector()

        with pytest.raises(KeyError):
            detector.get_query("nonexistent", "v14")

    def test_unknown_version_falls_back_to_v14(self):
        """Unknown schema version falls back to v14 queries."""
        detector = ChatDBSchemaDetector()
        query_v14 = detector.get_query("messages", "v14")
        query_unknown = detector.get_query("messages", "v99")

        assert query_v14 == query_unknown

    def test_query_with_since_filter(self):
        """Query includes since filter when requested."""
        detector = ChatDBSchemaDetector()
        query = detector.get_query("conversations", "v14", with_since_filter=True)

        assert "last_message_date > ?" in query

    def test_query_without_since_filter(self):
        """Query excludes since filter when not requested."""
        detector = ChatDBSchemaDetector()
        query = detector.get_query("conversations", "v14", with_since_filter=False)

        assert "last_message_date > ?" not in query

    def test_query_with_before_filter(self):
        """Query includes before filter when requested."""
        detector = ChatDBSchemaDetector()
        query = detector.get_query("messages", "v14", with_before_filter=True)

        assert "message.date < ?" in query

    def test_query_without_before_filter(self):
        """Query excludes before filter when not requested."""
        detector = ChatDBSchemaDetector()
        query = detector.get_query("messages", "v14", with_before_filter=False)

        assert "message.date < ?" not in query


class TestGetAvailableQueries:
    """Tests for get_available_queries method."""

    def test_returns_list_of_query_names(self):
        """Returns list of available query names."""
        detector = ChatDBSchemaDetector()
        queries = detector.get_available_queries("v14")

        assert isinstance(queries, list)
        assert "conversations" in queries
        assert "messages" in queries
        assert "search" in queries
        assert "context" in queries


class TestIsVersionSupported:
    """Tests for is_version_supported method."""

    def test_v14_is_supported(self):
        """v14 is a supported version."""
        detector = ChatDBSchemaDetector()
        assert detector.is_version_supported("v14") is True

    def test_v15_is_supported(self):
        """v15 is a supported version."""
        detector = ChatDBSchemaDetector()
        assert detector.is_version_supported("v15") is True

    def test_unknown_not_supported(self):
        """Unknown version is not supported."""
        detector = ChatDBSchemaDetector()
        assert detector.is_version_supported("v99") is False
        assert detector.is_version_supported("unknown") is False


class TestKnownSchemas:
    """Tests for KNOWN_SCHEMAS constant."""

    def test_known_schemas_has_v14_and_v15(self):
        """Known schemas includes v14 and v15."""
        assert "v14" in KNOWN_SCHEMAS
        assert "v15" in KNOWN_SCHEMAS

    def test_schemas_have_required_tables(self):
        """Schema definitions include required tables."""
        for version, config in KNOWN_SCHEMAS.items():
            assert "required_tables" in config
            required = config["required_tables"]
            assert "message" in required
            assert "chat" in required
            assert "handle" in required


class TestSchemaDetectorThreadSafety:
    """Thread safety tests for ChatDBSchemaDetector."""

    def test_concurrent_detections(self, tmp_path):
        """Concurrent schema detections are thread-safe."""
        # Create test databases
        db_paths = []
        for i in range(5):
            db_path = tmp_path / f"chat_{i}.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("""
                CREATE TABLE message (
                    ROWID INTEGER PRIMARY KEY,
                    thread_originator_guid TEXT
                )
            """)
            conn.execute("CREATE TABLE chat (ROWID INTEGER PRIMARY KEY)")
            conn.execute("CREATE TABLE handle (ROWID INTEGER PRIMARY KEY)")
            conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
            conn.close()
            db_paths.append(str(db_path))

        detector = ChatDBSchemaDetector()
        results = []
        lock = threading.Lock()

        def detect_schema(path):
            for _ in range(10):
                info = detector.detect(path)
                with lock:
                    results.append(info)

        threads = [threading.Thread(target=detect_schema, args=(path,)) for path in db_paths]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 50  # 5 paths * 10 iterations


class TestSchemaDetectorProtocolCompliance:
    """Verify ChatDBSchemaDetector implements SchemaDetector protocol."""

    def test_has_detect(self):
        """Detector has detect method."""
        detector = ChatDBSchemaDetector()
        assert hasattr(detector, "detect")
        assert callable(detector.detect)

    def test_has_get_query(self):
        """Detector has get_query method."""
        detector = ChatDBSchemaDetector()
        assert hasattr(detector, "get_query")
        assert callable(detector.get_query)

    def test_detect_returns_schema_info(self, tmp_path):
        """detect returns SchemaInfo."""
        detector = ChatDBSchemaDetector()
        result = detector.detect(str(tmp_path / "nonexistent.db"))
        assert isinstance(result, SchemaInfo)

    def test_get_query_returns_string(self):
        """get_query returns string."""
        detector = ChatDBSchemaDetector()
        result = detector.get_query("messages", "v14")
        assert isinstance(result, str)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPermissionSchemaIntegration:
    """Integration tests combining permission monitoring and schema detection."""

    def test_full_disk_access_then_schema_detect(self, tmp_path):
        """Schema detection works after permission check passes."""
        # Create mock chat.db
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE message (
                ROWID INTEGER PRIMARY KEY,
                thread_originator_guid TEXT
            )
        """)
        conn.execute("CREATE TABLE chat (ROWID INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE handle (ROWID INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        conn.close()

        # Check permission
        perm_monitor = TCCPermissionMonitor(messages_db_path=db_path)
        perm_status = perm_monitor.check_permission(Permission.FULL_DISK_ACCESS)

        # If permission granted, detect schema
        if perm_status.granted:
            schema_detector = ChatDBSchemaDetector()
            schema_info = schema_detector.detect(str(db_path))

            assert schema_info.version == "v14"
            assert schema_info.compatible is True

    def test_workflow_with_missing_permissions(self, tmp_path):
        """Graceful handling when permissions are denied."""
        # Path doesn't exist - simulates denied permission
        db_path = tmp_path / "nonexistent.db"

        perm_monitor = TCCPermissionMonitor(messages_db_path=db_path)
        perm_status = perm_monitor.check_permission(Permission.FULL_DISK_ACCESS)

        assert perm_status.granted is False
        assert len(perm_status.fix_instructions) > 0

        # Even without permission, schema detector handles gracefully
        schema_detector = ChatDBSchemaDetector()
        schema_info = schema_detector.detect(str(db_path))

        assert schema_info.version == "unknown"
        assert schema_info.compatible is False
