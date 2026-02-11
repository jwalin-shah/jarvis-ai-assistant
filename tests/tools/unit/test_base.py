"""Unit tests for Tool base class."""

from pathlib import Path

import pytest

from jarvis.tools.base import Tool, ToolResult


class DummyTool(Tool):
    """Concrete implementation for testing."""

    name = "dummy"
    description = "A dummy tool for testing"

    def run(self, **kwargs):
        return ToolResult(success=True, message="Done")


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_boolean_conversion_success(self):
        """Successful result is truthy."""
        result = ToolResult(success=True, message="OK")
        assert bool(result) is True
        assert result  # Works in if statement

    def test_boolean_conversion_failure(self):
        """Failed result is falsy."""
        result = ToolResult(success=False, message="Failed")
        assert bool(result) is False
        assert not result  # Works in if statement

    def test_to_dict(self):
        """Conversion to dictionary for JSON."""
        result = ToolResult(
            success=True,
            message="Test",
            data={"key": "value"},
            artifacts=[Path("/tmp/file.txt")],
            metrics={"accuracy": 0.95},
        )

        d = result.to_dict()
        assert d["success"] is True
        assert d["message"] == "Test"
        assert d["data"]["key"] == "value"
        assert d["artifacts"] == ["/tmp/file.txt"]
        assert d["metrics"]["accuracy"] == 0.95


class TestToolBase:
    """Tests for Tool base class."""

    def test_initialization(self):
        """Tool can be initialized with config."""
        tool = DummyTool({"key": "value"})
        assert tool.config["key"] == "value"

    def test_default_config(self):
        """Tool works with empty config."""
        tool = DummyTool()
        assert tool.config == {}

    def test_logger_lazy_load(self):
        """Logger is created on first access."""
        tool = DummyTool()
        assert tool._logger is None
        logger = tool.logger
        assert logger is not None
        assert tool._logger is logger  # Same instance

    def test_get_config_with_default(self):
        """get_config returns default for missing keys."""
        tool = DummyTool({"exists": "yes"})
        assert tool.get_config("exists") == "yes"
        assert tool.get_config("missing") is None
        assert tool.get_config("missing", "default") == "default"

    def test_require_config_present(self):
        """require_config returns value if present."""
        tool = DummyTool({"required": "value"})
        assert tool.require_config("required") == "value"

    def test_require_config_missing(self):
        """require_config raises if missing."""
        tool = DummyTool()
        with pytest.raises(ValueError, match="Required config missing: missing"):
            tool.require_config("missing")

    def test_validation_empty(self):
        """Base validation returns empty list for valid config."""
        tool = DummyTool()
        errors = tool.validate()
        assert errors == []

    def test_dry_run_success(self):
        """dry_run returns success when validation passes."""
        tool = DummyTool()
        result = tool.dry_run()
        assert result.success is True
        assert "Validation passed" in result.message

    def test_dry_run_failure(self):
        """dry_run returns failure when validation fails."""

        class InvalidTool(Tool):
            name = "invalid"
            description = "Always invalid"
            required_config = ["missing_key"]

            def run(self, **kwargs):
                return ToolResult(success=True, message="Done")

        tool = InvalidTool()
        result = tool.dry_run()
        assert result.success is False
        assert "Validation failed" in result.message
        assert len(result.data["errors"]) == 1


class TestToolValidation:
    """Tests for Tool validation logic."""

    def test_required_config_check(self):
        """Validation checks required config keys."""

        class RequiredConfigTool(Tool):
            name = "required_test"
            description = "Test"
            required_config = ["data_dir", "output"]

            def run(self, **kwargs):
                return ToolResult(success=True, message="Done")

        tool = RequiredConfigTool({"data_dir": "/tmp"})  # Missing "output"
        errors = tool.validate()
        assert any("output" in e for e in errors)

    def test_path_validation(self, tmp_path: Path):
        """Validation checks path existence."""

        class PathTool(Tool):
            name = "path_test"
            description = "Test"
            path_config_keys = ["data_dir"]

            def run(self, **kwargs):
                return ToolResult(success=True, message="Done")

        # Non-existent path
        tool = PathTool({"data_dir": tmp_path / "nonexistent"})
        errors = tool.validate()
        assert any("does not exist" in e for e in errors)

        # Existing path
        existing = tmp_path / "exists"
        existing.mkdir()
        tool = PathTool({"data_dir": existing})
        errors = tool.validate()
        assert not any("does not exist" in e for e in errors)
