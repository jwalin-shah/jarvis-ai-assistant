"""Integration tests for the JARVIS CLI.

Tests the command-line interface for developer tools (serve, health, benchmark, db).
User-facing features (chat, reply, search) are in the desktop app.
"""

from unittest.mock import MagicMock, patch

import pytest

from core.health import reset_degradation_controller
from core.memory import reset_memory_controller
from jarvis.cli import (
    FEATURE_CHAT,
    FEATURE_IMESSAGE,
    cmd_health,
    cmd_version,
    create_parser,
    initialize_system,
    main,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before and after each test."""
    reset_memory_controller()
    reset_degradation_controller()
    yield
    reset_memory_controller()
    reset_degradation_controller()


class TestCreateParser:
    """Tests for argument parser creation."""

    def test_parser_creates_successfully(self):
        """Parser is created without errors."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "jarvis"

    def test_parser_has_verbose_flag(self):
        """Parser has verbose flag."""
        parser = create_parser()
        args = parser.parse_args(["--verbose", "health"])
        assert args.verbose is True

    def test_parser_has_version_flag(self):
        """Parser has version flag."""
        parser = create_parser()
        args = parser.parse_args(["--version"])
        assert args.version is True

    def test_parser_health_command(self):
        """Parser parses health command."""
        parser = create_parser()
        args = parser.parse_args(["health"])
        assert args.command == "health"

    def test_parser_benchmark_command(self):
        """Parser parses benchmark command with type."""
        parser = create_parser()
        args = parser.parse_args(["benchmark", "memory"])
        assert args.command == "benchmark"
        assert args.type == "memory"

    def test_parser_benchmark_with_output(self):
        """Parser parses benchmark command with output file."""
        parser = create_parser()
        args = parser.parse_args(["benchmark", "latency", "-o", "results.json"])
        assert args.type == "latency"
        assert args.output == "results.json"

    def test_parser_benchmark_validates_type(self):
        """Parser validates benchmark type."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["benchmark", "invalid"])

    def test_parser_serve_command(self):
        """Parser parses serve command."""
        parser = create_parser()
        args = parser.parse_args(["serve"])
        assert args.command == "serve"
        assert args.host == "127.0.0.1"
        assert args.port == 8000

    def test_parser_serve_with_options(self):
        """Parser parses serve command with options."""
        parser = create_parser()
        args = parser.parse_args(["serve", "--host", "0.0.0.0", "-p", "3000", "--reload"])
        assert args.host == "0.0.0.0"
        assert args.port == 3000
        assert args.reload is True

    def test_parser_db_command(self):
        """Parser parses db command."""
        parser = create_parser()
        args = parser.parse_args(["db", "stats"])
        assert args.command == "db"
        assert args.db_command == "stats"


class TestInitializeSystem:
    """Tests for system initialization."""

    def test_initialize_returns_success(self):
        """Initialize system returns success flag."""
        success, warnings = initialize_system()
        assert success is True

    def test_initialize_registers_features(self):
        """Initialize system registers degradation features."""
        from core.health import get_degradation_controller

        initialize_system()
        controller = get_degradation_controller()
        health = controller.get_health()

        assert FEATURE_CHAT in health
        assert FEATURE_IMESSAGE in health


class TestCmdVersion:
    """Tests for version command."""

    def test_version_returns_zero(self):
        """Version command returns success exit code."""
        parser = create_parser()
        args = parser.parse_args(["version"])
        exit_code = cmd_version(args)
        assert exit_code == 0


class TestCmdHealth:
    """Tests for health command."""

    def test_health_returns_zero(self):
        """Health command returns success exit code."""
        initialize_system()
        parser = create_parser()
        args = parser.parse_args(["health"])
        exit_code = cmd_health(args)
        assert exit_code == 0


class TestMain:
    """Tests for main entry point."""

    def test_main_with_no_args_shows_help(self):
        """Main with no args shows help and returns zero."""
        exit_code = main([])
        assert exit_code == 0

    def test_main_with_version(self):
        """Main with --version returns zero."""
        exit_code = main(["--version"])
        assert exit_code == 0

    def test_main_with_health(self):
        """Main with health command returns zero."""
        exit_code = main(["health"])
        assert exit_code == 0


class TestBenchmark:
    """Tests for benchmark command."""

    @patch("subprocess.run")
    def test_benchmark_calls_subprocess(self, mock_run):
        """Benchmark command calls subprocess with correct module."""
        mock_run.return_value = MagicMock(returncode=0)

        exit_code = main(["benchmark", "memory"])

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "-m" in call_args
        assert "evals.benchmarks.memory.run" in call_args
        assert exit_code == 0

    @patch("subprocess.run")
    def test_benchmark_with_output(self, mock_run):
        """Benchmark command passes output file to subprocess."""
        mock_run.return_value = MagicMock(returncode=0)

        exit_code = main(["benchmark", "latency", "-o", "results.json"])

        call_args = mock_run.call_args[0][0]
        assert "--output" in call_args
        assert "results.json" in call_args
        assert exit_code == 0

    @patch("subprocess.run")
    def test_benchmark_propagates_exit_code(self, mock_run):
        """Benchmark command propagates subprocess exit code."""
        mock_run.return_value = MagicMock(returncode=42)

        exit_code = main(["benchmark", "hhem"])

        assert exit_code == 42

    @patch("subprocess.run")
    def test_benchmark_hhem_calls_correct_module(self, mock_run):
        """Benchmark hhem calls correct module."""
        mock_run.return_value = MagicMock(returncode=0)

        exit_code = main(["benchmark", "hhem"])

        call_args = mock_run.call_args[0][0]
        assert "evals.benchmarks.hallucination.run" in call_args
        assert exit_code == 0


class TestFallbackBehaviors:
    """Tests for fallback and degraded behaviors."""

    def test_template_only_response(self):
        """Template-only response works without model."""
        from jarvis.system import _template_only_response

        result = _template_only_response("hello")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fallback_response(self):
        """Fallback response returns static message."""
        from jarvis.system import _fallback_response

        result = _fallback_response()
        assert isinstance(result, str)
        assert "unable" in result.lower() or "check" in result.lower()

    def test_imessage_degraded(self):
        """iMessage degraded mode returns empty list."""
        from jarvis.system import _imessage_degraded

        result = _imessage_degraded("test")
        assert result == []

    def test_imessage_fallback(self):
        """iMessage fallback returns empty list."""
        from jarvis.system import _imessage_fallback

        result = _imessage_fallback()
        assert result == []


class TestCleanup:
    """Tests for cleanup function."""

    def test_cleanup_resets_controllers(self):
        """Cleanup resets memory and degradation controllers."""
        from jarvis.cli import cleanup

        initialize_system()
        cleanup()

    def test_cleanup_handles_errors(self):
        """Cleanup handles errors gracefully."""
        from jarvis.cli import cleanup

        cleanup()
        cleanup()  # Second call should also be safe


class TestModuleExecution:
    """Tests for module execution via python -m jarvis."""

    def test_module_entry_point_exists(self):
        """jarvis.__main__ exists and is importable."""
        import jarvis.__main__

        assert hasattr(jarvis.__main__, "run")


class TestVerboseMode:
    """Tests for verbose logging mode."""

    def test_verbose_flag_sets_logging(self):
        """Verbose flag enables debug logging."""
        from jarvis.cli import setup_logging

        setup_logging(verbose=False)
        setup_logging(verbose=True)


class TestAccessChecks:
    """Tests for access check functions."""

    def test_check_imessage_access_returns_bool(self):
        """Check iMessage access returns boolean."""
        from jarvis.system import _check_imessage_access

        result = _check_imessage_access()
        assert isinstance(result, bool)

    def test_check_imessage_access_exception_returns_false(self):
        """Check iMessage access returns False on exception."""
        from jarvis.system import _check_imessage_access

        with patch("integrations.imessage.ChatDBReader") as mock_reader:
            mock_reader.side_effect = Exception("DB error")
            result = _check_imessage_access()
        assert result is False


class TestDbCommand:
    """Tests for db command."""

    def test_parser_db_init(self):
        """Parser parses db init command."""
        parser = create_parser()
        args = parser.parse_args(["db", "init"])
        assert args.db_command == "init"

    def test_parser_db_init_with_force(self):
        """Parser parses db init with force flag."""
        parser = create_parser()
        args = parser.parse_args(["db", "init", "--force"])
        assert args.force is True

    def test_parser_db_add_contact(self):
        """Parser parses db add-contact command."""
        parser = create_parser()
        args = parser.parse_args(["db", "add-contact", "--name", "John"])
        assert args.db_command == "add-contact"
        assert args.name == "John"

    def test_parser_db_list_contacts(self):
        """Parser parses db list-contacts command."""
        parser = create_parser()
        args = parser.parse_args(["db", "list-contacts"])
        assert args.db_command == "list-contacts"

    def test_parser_db_extract(self):
        """Parser parses db extract command."""
        parser = create_parser()
        args = parser.parse_args(["db", "extract"])
        assert args.db_command == "extract"

    def test_parser_db_stats(self):
        """Parser parses db stats command."""
        parser = create_parser()
        args = parser.parse_args(["db", "stats"])
        assert args.db_command == "stats"

    def test_parser_db_stats_with_gate_breakdown(self):
        """Parser parses db stats with gate-breakdown flag."""
        parser = create_parser()
        args = parser.parse_args(["db", "stats", "--gate-breakdown"])
        assert args.gate_breakdown is True

