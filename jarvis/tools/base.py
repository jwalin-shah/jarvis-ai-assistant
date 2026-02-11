"""Base classes for JARVIS tools.

All tools must inherit from Tool and implement the run() method.
This provides consistent behavior, logging, and result handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolResult:
    """Standard result format for all tools.

    Attributes:
        success: Whether the tool completed successfully
        message: Human-readable status message
        data: Optional structured data from the tool
        artifacts: Paths to generated files/artifacts
        metrics: Numerical metrics (accuracy, duration, etc.)
        warnings: Non-fatal issues that occurred
    """

    success: bool
    message: str
    data: dict[str, Any] | None = None
    artifacts: list[Path] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Allow boolean check: if result: ..."""
        return self.success

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "artifacts": [str(p) for p in self.artifacts],
            "metrics": self.metrics,
            "warnings": self.warnings,
        }


class Tool(ABC):
    """Base class for all JARVIS tools.

    Subclasses must define:
    - name: Tool identifier (snake_case)
    - description: Short human-readable description
    - version: Semver string

    Example:
        @tool("train_category", "Train category classifier", "1.0.0")
        class CategoryTrainer(Tool):
            def run(self, **kwargs) -> ToolResult:
                # Implementation
                return ToolResult(success=True, message="Training complete")
    """

    name: str = ""
    description: str = ""
    version: str = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize tool with configuration.

        Args:
            config: Tool-specific configuration dictionary
        """
        self.config = config or {}
        self._logger = None

    @property
    def logger(self):
        """Lazy-loaded logger for the tool."""
        if self._logger is None:
            from jarvis.tools.logging import get_tool_logger

            self._logger = get_tool_logger(self.name)
        return self._logger

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Execute the tool.

        Args:
            **kwargs: Runtime parameters (override config)

        Returns:
            ToolResult with success status and outputs
        """
        pass

    def validate(self) -> list[str]:
        """Validate tool configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required config keys
        required = getattr(self, "required_config", [])
        for key in required:
            if key not in self.config:
                errors.append(f"Missing required config: {key}")

        # Check path existence
        path_keys = getattr(self, "path_config_keys", [])
        for key in path_keys:
            if key in self.config:
                path = Path(self.config[key])
                if not path.exists():
                    errors.append(f"Path does not exist: {key}={path}")

        return errors

    def dry_run(self) -> ToolResult:
        """Validate without executing.

        Returns:
            ToolResult indicating validation status
        """
        errors = self.validate()
        if errors:
            return ToolResult(
                success=False,
                message=f"Validation failed with {len(errors)} error(s)",
                data={"errors": errors},
            )
        return ToolResult(
            success=True,
            message="Validation passed",
        )

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        return self.config.get(key, default)

    def require_config(self, key: str) -> Any:
        """Get required config value, raise if missing."""
        if key not in self.config:
            raise ValueError(f"Required config missing: {key}")
        return self.config[key]
