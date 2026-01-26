"""Unit tests for JARVIS Setup Wizard.

Tests cover permission checking, database validation, config initialization,
model checking, and the overall setup flow. Filesystem and permissions are mocked.
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from contracts.health import Permission, PermissionStatus, SchemaInfo
from contracts.memory import MemoryMode
from jarvis.setup import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TEMPLATE_THRESHOLD,
    CheckResult,
    CheckStatus,
    JarvisConfig,
    PermissionMonitorImpl,
    SchemaDetectorImpl,
    SetupResult,
    SetupWizard,
    open_system_preferences_fda,
    run_setup,
)


class TestJarvisConfigInSetup:
    """Tests for JarvisConfig Pydantic model as used in setup.

    Note: Full JarvisConfig tests are in test_config.py. These tests
    verify basic functionality as used by the setup wizard.
    """

    def test_default_values(self):
        """Test default configuration values."""
        config = JarvisConfig()
        assert config.model_path == DEFAULT_MODEL_PATH
        assert config.template_similarity_threshold == DEFAULT_TEMPLATE_THRESHOLD

    def test_model_dump(self):
        """Test conversion to dictionary via model_dump."""
        config = JarvisConfig(
            model_path="custom/model",
            template_similarity_threshold=0.8,
        )
        data = config.model_dump()
        assert data["model_path"] == "custom/model"
        assert data["template_similarity_threshold"] == 0.8
        assert "ui" in data  # Nested configs should be included
        assert "search" in data
        assert "chat" in data

    def test_model_validate(self):
        """Test creation from dictionary via model_validate."""
        data = {
            "model_path": "custom/model",
            "template_similarity_threshold": 0.6,
        }
        config = JarvisConfig.model_validate(data)
        assert config.model_path == "custom/model"
        assert config.template_similarity_threshold == 0.6

    def test_model_validate_with_defaults(self):
        """Test creation from partial dictionary uses defaults."""
        config = JarvisConfig.model_validate({})
        assert config.model_path == DEFAULT_MODEL_PATH
        assert config.template_similarity_threshold == DEFAULT_TEMPLATE_THRESHOLD


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_basic_check_result(self):
        """Test basic CheckResult creation."""
        result = CheckResult(
            name="Test Check",
            status=CheckStatus.PASS,
            message="All good",
        )
        assert result.name == "Test Check"
        assert result.status == CheckStatus.PASS
        assert result.message == "All good"
        assert result.details is None
        assert result.fix_instructions is None

    def test_check_result_with_all_fields(self):
        """Test CheckResult with all fields."""
        result = CheckResult(
            name="Test Check",
            status=CheckStatus.FAIL,
            message="Something wrong",
            details="More info here",
            fix_instructions="Do this to fix",
        )
        assert result.details == "More info here"
        assert result.fix_instructions == "Do this to fix"


class TestPermissionMonitorImpl:
    """Tests for PermissionMonitorImpl."""

    def test_check_full_disk_access_granted(self, tmp_path, monkeypatch):
        """Test FDA check when permission is granted."""
        # Create a fake chat.db
        fake_db = tmp_path / "chat.db"
        fake_db.write_bytes(b"fake data")

        # Patch CHAT_DB_PATH to point to our fake db
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", fake_db)

        monitor = PermissionMonitorImpl()
        status = monitor.check_permission(Permission.FULL_DISK_ACCESS)

        assert status.granted is True
        assert status.permission == Permission.FULL_DISK_ACCESS
        assert status.last_checked  # Should have a timestamp

    def test_check_full_disk_access_denied(self, tmp_path, monkeypatch):
        """Test FDA check when permission is denied."""
        # Create a fake path that doesn't exist
        fake_db = tmp_path / "nonexistent" / "chat.db"

        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", fake_db)

        # Mock the home directory TCC path check to also fail
        def mock_exists(self):
            return False

        with patch.object(Path, "exists", return_value=False):
            monitor = PermissionMonitorImpl()
            # Since both paths don't exist, it assumes access is granted
            status = monitor.check_permission(Permission.FULL_DISK_ACCESS)
            assert status.granted is True  # Default when files don't exist

    def test_check_full_disk_access_permission_error(self, tmp_path, monkeypatch):
        """Test FDA check when reading raises PermissionError."""
        fake_db = tmp_path / "chat.db"
        fake_db.write_bytes(b"data")

        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", fake_db)

        # Mock open to raise PermissionError
        original_open = Path.open

        def mock_open(self, *args, **kwargs):
            if str(self) == str(fake_db):
                raise PermissionError("Access denied")
            return original_open(self, *args, **kwargs)

        with patch.object(Path, "open", mock_open):
            monitor = PermissionMonitorImpl()
            status = monitor.check_permission(Permission.FULL_DISK_ACCESS)
            assert status.granted is False

    def test_check_all_permissions(self, tmp_path, monkeypatch):
        """Test checking all permissions."""
        fake_db = tmp_path / "chat.db"
        fake_db.write_bytes(b"data")
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", fake_db)

        monitor = PermissionMonitorImpl()
        statuses = monitor.check_all()

        assert len(statuses) == 1
        assert statuses[0].permission == Permission.FULL_DISK_ACCESS

    def test_fix_instructions_present(self):
        """Test that fix instructions are provided."""
        monitor = PermissionMonitorImpl()
        status = monitor.check_permission(Permission.FULL_DISK_ACCESS)

        assert status.fix_instructions
        has_sys_settings = "System Settings" in status.fix_instructions
        has_fda = "Full Disk Access" in status.fix_instructions
        assert has_sys_settings or has_fda


class TestSchemaDetectorImpl:
    """Tests for SchemaDetectorImpl."""

    def test_detect_nonexistent_db(self, tmp_path):
        """Test detection on non-existent database."""
        detector = SchemaDetectorImpl()
        info = detector.detect(str(tmp_path / "nonexistent.db"))

        assert info.version == "unknown"
        assert info.tables == []
        assert info.compatible is False
        assert info.known_schema is False

    def test_detect_valid_v14_schema(self, tmp_path):
        """Test detection of v14 schema."""
        import sqlite3

        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create required tables for v14
        cursor.execute("""
            CREATE TABLE message (
                ROWID INTEGER PRIMARY KEY,
                text TEXT,
                date INTEGER,
                is_from_me INTEGER,
                handle_id INTEGER,
                thread_originator_guid TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE chat (
                ROWID INTEGER PRIMARY KEY,
                guid TEXT,
                display_name TEXT
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

        detector = SchemaDetectorImpl()
        info = detector.detect(str(db_path))

        assert info.version == "v14"
        assert info.compatible is True
        assert info.known_schema is True
        assert "message" in info.tables
        assert "chat" in info.tables

    def test_detect_v15_schema(self, tmp_path):
        """Test detection of v15 schema."""
        import sqlite3

        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create v15 schema with service_name in chat table
        cursor.execute("""
            CREATE TABLE message (
                ROWID INTEGER PRIMARY KEY,
                text TEXT,
                date INTEGER,
                is_from_me INTEGER,
                handle_id INTEGER,
                thread_originator_guid TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE chat (
                ROWID INTEGER PRIMARY KEY,
                guid TEXT,
                display_name TEXT,
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

        detector = SchemaDetectorImpl()
        info = detector.detect(str(db_path))

        assert info.version == "v15"
        assert info.compatible is True
        assert info.known_schema is True


class TestSetupWizard:
    """Tests for SetupWizard."""

    @pytest.fixture
    def mock_console(self):
        """Create a mock console that captures output."""
        console = MagicMock()
        return console

    @pytest.fixture
    def mock_permission_monitor(self):
        """Create a mock permission monitor."""
        monitor = MagicMock(spec=PermissionMonitorImpl)
        monitor.check_permission.return_value = PermissionStatus(
            permission=Permission.FULL_DISK_ACCESS,
            granted=True,
            last_checked=datetime.now().isoformat(),
            fix_instructions="",
        )
        return monitor

    @pytest.fixture
    def mock_schema_detector(self, tmp_path):
        """Create a mock schema detector."""
        detector = MagicMock(spec=SchemaDetectorImpl)
        detector.detect.return_value = SchemaInfo(
            version="v14",
            tables=["message", "chat", "handle", "chat_message_join", "chat_handle_join"],
            compatible=True,
            migration_needed=False,
            known_schema=True,
        )
        return detector

    def test_run_check_only_no_config_created(
        self, tmp_path, mock_console, mock_permission_monitor, mock_schema_detector, monkeypatch
    ):
        """Test check-only mode doesn't create config."""
        # Setup mock paths
        config_dir = tmp_path / ".jarvis"
        config_file = config_dir / "config.json"
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_DIR", config_dir)
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_FILE", config_file)
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", tmp_path / "chat.db")

        # Mock memory controller
        mock_controller = MagicMock()
        mock_controller.get_state.return_value = MagicMock(available_mb=8000)
        mock_controller.get_mode.return_value = MemoryMode.FULL

        # Mock platform
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("platform.mac_ver", lambda: ("14.0", ("", "", ""), "arm64"))

        with patch("core.memory.controller.get_memory_controller", return_value=mock_controller):
            wizard = SetupWizard(
                console=mock_console,
                permission_monitor=mock_permission_monitor,
                schema_detector=mock_schema_detector,
            )
            result = wizard.run(check_only=True)

        assert result.config_created is False
        assert not config_file.exists()

    def test_run_creates_config(
        self, tmp_path, mock_console, mock_permission_monitor, mock_schema_detector, monkeypatch
    ):
        """Test full run creates config file."""
        config_dir = tmp_path / ".jarvis"
        config_file = config_dir / "config.json"
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_DIR", config_dir)
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_FILE", config_file)
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", tmp_path / "chat.db")

        mock_controller = MagicMock()
        mock_controller.get_state.return_value = MagicMock(available_mb=8000)
        mock_controller.get_mode.return_value = MemoryMode.FULL

        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("platform.mac_ver", lambda: ("14.0", ("", "", ""), "arm64"))

        with patch("core.memory.controller.get_memory_controller", return_value=mock_controller):
            wizard = SetupWizard(
                console=mock_console,
                permission_monitor=mock_permission_monitor,
                schema_detector=mock_schema_detector,
            )
            result = wizard.run(check_only=False)

        assert result.config_created is True
        assert config_file.exists()

        # Verify config contents (using new Pydantic schema)
        with config_file.open() as f:
            config = json.load(f)
        assert config["model_path"] == DEFAULT_MODEL_PATH
        assert config["template_similarity_threshold"] == DEFAULT_TEMPLATE_THRESHOLD
        # New config should have nested sections
        assert "ui" in config
        assert "search" in config
        assert "chat" in config

    def test_preserves_existing_config(
        self, tmp_path, mock_console, mock_permission_monitor, mock_schema_detector, monkeypatch
    ):
        """Test that existing config is preserved and migrated."""
        config_dir = tmp_path / ".jarvis"
        config_file = config_dir / "config.json"
        config_dir.mkdir()

        # Create existing v1 config (old format without version)
        existing_config = {
            "model_path": "custom/model",
            "template_similarity_threshold": 0.9,
            "imessage_default_limit": 100,
        }
        with config_file.open("w") as f:
            json.dump(existing_config, f)

        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_DIR", config_dir)
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_FILE", config_file)
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", tmp_path / "chat.db")

        mock_controller = MagicMock()
        mock_controller.get_state.return_value = MagicMock(available_mb=8000)
        mock_controller.get_mode.return_value = MemoryMode.FULL

        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("platform.mac_ver", lambda: ("14.0", ("", "", ""), "arm64"))

        with patch("core.memory.controller.get_memory_controller", return_value=mock_controller):
            wizard = SetupWizard(
                console=mock_console,
                permission_monitor=mock_permission_monitor,
                schema_detector=mock_schema_detector,
            )
            result = wizard.run(check_only=False)

        # Config should not be recreated (just updated via migration)
        assert result.config_created is False

        # Original values should be preserved, and migration applied
        with config_file.open() as f:
            config = json.load(f)
        assert config["model_path"] == "custom/model"
        assert config["template_similarity_threshold"] == 0.9
        # Migration should have added new sections
        assert "ui" in config
        assert "search" in config
        # imessage_default_limit should be migrated to search.default_limit
        assert config["search"]["default_limit"] == 100

    def test_platform_check_macos(self, mock_console, monkeypatch):
        """Test platform check on macOS."""
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("platform.mac_ver", lambda: ("14.2.1", ("", "", ""), "arm64"))

        wizard = SetupWizard(console=mock_console)
        wizard._check_platform()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.PASS
        assert "macOS 14.2.1" in wizard._checks[0].message

    def test_platform_check_linux(self, mock_console, monkeypatch):
        """Test platform check on Linux (warning)."""
        monkeypatch.setattr("platform.system", lambda: "Linux")

        wizard = SetupWizard(console=mock_console)
        wizard._check_platform()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.WARN
        assert "Linux" in wizard._checks[0].message

    def test_permission_check_granted(self, mock_console, mock_permission_monitor):
        """Test permission check when FDA is granted."""
        wizard = SetupWizard(
            console=mock_console,
            permission_monitor=mock_permission_monitor,
        )
        wizard._check_permissions()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.PASS
        assert wizard._checks[0].name == "Full Disk Access"

    def test_permission_check_denied(self, mock_console):
        """Test permission check when FDA is denied."""
        monitor = MagicMock(spec=PermissionMonitorImpl)
        monitor.check_permission.return_value = PermissionStatus(
            permission=Permission.FULL_DISK_ACCESS,
            granted=False,
            last_checked=datetime.now().isoformat(),
            fix_instructions="Grant FDA in System Settings",
        )

        wizard = SetupWizard(console=mock_console, permission_monitor=monitor)
        wizard._check_permissions()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.FAIL
        assert wizard._checks[0].fix_instructions is not None

    def test_database_check_not_found(
        self, mock_console, mock_schema_detector, tmp_path, monkeypatch
    ):
        """Test database check when chat.db doesn't exist."""
        fake_path = tmp_path / "nonexistent" / "chat.db"
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", fake_path)

        wizard = SetupWizard(
            console=mock_console,
            schema_detector=mock_schema_detector,
        )
        wizard._check_database()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.WARN
        assert "not found" in wizard._checks[0].message

    def test_database_check_compatible(
        self, mock_console, mock_schema_detector, tmp_path, monkeypatch
    ):
        """Test database check with compatible schema."""
        db_path = tmp_path / "chat.db"
        db_path.write_bytes(b"fake")
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", db_path)

        wizard = SetupWizard(
            console=mock_console,
            schema_detector=mock_schema_detector,
        )
        wizard._check_database()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.PASS
        assert "v14" in wizard._checks[0].message

    def test_memory_check_with_controller(self, mock_console, monkeypatch):
        """Test memory check using MemoryController."""
        mock_controller = MagicMock()
        mock_controller.get_state.return_value = MagicMock(available_mb=10000)
        mock_controller.get_mode.return_value = MemoryMode.FULL

        with patch("core.memory.controller.get_memory_controller", return_value=mock_controller):
            wizard = SetupWizard(console=mock_console)
            mode = wizard._check_memory()

        assert mode == MemoryMode.FULL
        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.PASS
        assert "FULL" in wizard._checks[0].message

    def test_memory_check_fallback_psutil(self, mock_console, monkeypatch):
        """Test memory check fallback to psutil when controller unavailable."""

        # Make get_memory_controller raise ImportError
        def mock_import_error():
            raise ImportError("No controller")

        mock_mem = MagicMock()
        mock_mem.available = 6 * 1024**3  # 6GB

        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        with patch("core.memory.controller.get_memory_controller", side_effect=ImportError):
            wizard = SetupWizard(console=mock_console)
            mode = wizard._check_memory()

        assert mode == MemoryMode.LITE
        assert len(wizard._checks) == 1
        assert "LITE" in wizard._checks[0].message

    def test_model_check_downloaded(self, mock_console, monkeypatch):
        """Test model check when model is in cache."""
        with patch("huggingface_hub.try_to_load_from_cache", return_value="/path/to/config.json"):
            wizard = SetupWizard(console=mock_console)
            wizard._check_model()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.PASS
        assert DEFAULT_MODEL_PATH in wizard._checks[0].message

    def test_model_check_not_downloaded(self, mock_console):
        """Test model check when model is not in cache."""
        with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
            wizard = SetupWizard(console=mock_console)
            wizard._check_model()

        assert len(wizard._checks) == 1
        assert wizard._checks[0].status == CheckStatus.WARN
        assert wizard._checks[0].fix_instructions is not None
        assert "huggingface-cli download" in wizard._checks[0].fix_instructions

    def test_model_check_import_error(self, mock_console, tmp_path, monkeypatch):
        """Test model check when huggingface_hub not installed."""
        # Mock the home directory for cache check
        cache_path = tmp_path / ".cache" / "huggingface" / "hub"
        cache_path.mkdir(parents=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with patch("huggingface_hub.try_to_load_from_cache", side_effect=ImportError):
            wizard = SetupWizard(console=mock_console)
            wizard._check_model()

        assert len(wizard._checks) == 1
        # Should be WARN since model dir not found
        assert wizard._checks[0].status == CheckStatus.WARN


class TestSetupResult:
    """Tests for SetupResult."""

    def test_success_with_all_passing(self):
        """Test result is success when all checks pass."""
        checks = [
            CheckResult(name="Check1", status=CheckStatus.PASS, message="OK"),
            CheckResult(name="Check2", status=CheckStatus.PASS, message="OK"),
            CheckResult(name="Check3", status=CheckStatus.WARN, message="Warning"),
        ]
        result = SetupResult(success=True, checks=checks)
        assert result.success is True

    def test_failure_with_failing_check(self):
        """Test result is failure when any check fails."""
        checks = [
            CheckResult(name="Check1", status=CheckStatus.PASS, message="OK"),
            CheckResult(name="Check2", status=CheckStatus.FAIL, message="Failed"),
        ]
        result = SetupResult(success=False, checks=checks)
        assert result.success is False


class TestOpenSystemPreferences:
    """Tests for open_system_preferences_fda."""

    def test_non_darwin_returns_false(self, monkeypatch):
        """Test returns False on non-Darwin platforms."""
        monkeypatch.setattr("platform.system", lambda: "Linux")
        assert open_system_preferences_fda() is False

    def test_darwin_opens_preferences(self, monkeypatch):
        """Test attempts to open System Preferences on Darwin."""
        monkeypatch.setattr("platform.system", lambda: "Darwin")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = open_system_preferences_fda()

        assert result is True
        mock_run.assert_called_once()
        # Check it tried to open the privacy pane
        args = mock_run.call_args[0][0]
        assert "open" in args


class TestRunSetup:
    """Tests for run_setup convenience function."""

    def test_run_setup_creates_wizard(self, tmp_path, monkeypatch):
        """Test run_setup creates and runs a wizard."""
        config_dir = tmp_path / ".jarvis"
        config_file = config_dir / "config.json"
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_DIR", config_dir)
        monkeypatch.setattr("jarvis.setup.JARVIS_CONFIG_FILE", config_file)
        monkeypatch.setattr("jarvis.setup.CHAT_DB_PATH", tmp_path / "chat.db")
        monkeypatch.setattr("platform.system", lambda: "Linux")

        mock_mem = MagicMock()
        mock_mem.available = 16 * 1024**3
        monkeypatch.setattr("psutil.virtual_memory", lambda: mock_mem)

        with patch("core.memory.controller.get_memory_controller", side_effect=ImportError):
            with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
                result = run_setup(check_only=True)

        assert isinstance(result, SetupResult)
        assert len(result.checks) > 0


class TestCheckStatusEnum:
    """Tests for CheckStatus enum."""

    def test_all_status_values(self):
        """Test all check status values exist."""
        assert CheckStatus.PASS.value == "pass"
        assert CheckStatus.WARN.value == "warn"
        assert CheckStatus.FAIL.value == "fail"
        assert CheckStatus.SKIP.value == "skip"
