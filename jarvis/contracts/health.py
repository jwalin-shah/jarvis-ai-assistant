"""System health monitoring interfaces.

Workstreams 6 and 7 implement against these contracts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    pass


class FeatureState(Enum):
    """Health state of a feature."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


class Permission(Enum):
    """macOS permissions required by JARVIS."""

    FULL_DISK_ACCESS = "full_disk_access"
    CONTACTS = "contacts"
    CALENDAR = "calendar"
    AUTOMATION = "automation"


@dataclass
class PermissionStatus:
    """Status of a single permission.

    Attributes:
        permission: The permission being checked.
        granted: Whether the permission is currently granted.
        last_checked: ISO format timestamp of last check.
        fix_instructions: User-friendly instructions if permission not granted.
    """

    permission: Permission
    granted: bool
    last_checked: str
    fix_instructions: str


@dataclass
class SchemaInfo:
    """Information about iMessage chat.db schema.

    Attributes:
        version: Detected schema version string.
        tables: List of table names in the database.
        compatible: Whether this schema is compatible with JARVIS.
        migration_needed: Whether migration is needed for compatibility.
        known_schema: Whether this schema version is recognized.
    """

    version: str
    tables: list[str]
    compatible: bool
    migration_needed: bool
    known_schema: bool


@dataclass
class DegradationPolicy:
    """Policy for degrading a feature gracefully.

    Attributes:
        feature_name: Unique name identifying the feature.
        health_check: Function that returns True if feature is healthy.
        degraded_behavior: Function to call when feature is degraded.
        fallback_behavior: Function to call when feature has failed.
        recovery_check: Function that returns True if feature can recover.
        max_failures: Maximum failures before switching to fallback.
    """

    feature_name: str
    health_check: Callable[[], bool]
    degraded_behavior: Callable[..., Any]
    fallback_behavior: Callable[..., Any]
    recovery_check: Callable[[], bool]
    max_failures: int = 3

    def __post_init__(self) -> None:
        """Validate field constraints."""
        if not self.feature_name.strip():
            msg = "feature_name cannot be empty"
            raise ValueError(msg)
        if self.max_failures < 1:
            msg = f"max_failures must be >= 1, got {self.max_failures}"
            raise ValueError(msg)


class DegradationController(Protocol):
    """Interface for graceful degradation (Workstream 6)."""

    def register_feature(self, policy: DegradationPolicy) -> None:
        """Register a feature with its degradation policy.

        Args:
            policy: Degradation policy defining health checks and fallbacks.
        """
        ...

    def execute(self, feature_name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute feature with automatic fallback on failure.

        Args:
            feature_name: Name of the registered feature to execute.
            *args: Positional arguments to pass to the feature.
            **kwargs: Keyword arguments to pass to the feature.

        Returns:
            Result from the feature execution (healthy, degraded, or fallback).
        """
        ...

    def get_health(self) -> dict[str, FeatureState]:
        """Return health status of all features.

        Returns:
            Dictionary mapping feature names to their current health state.
        """
        ...

    def reset_feature(self, feature_name: str) -> None:
        """Reset failure count and try healthy mode again.

        Args:
            feature_name: Name of the feature to reset.
        """
        ...


class PermissionMonitor(Protocol):
    """Interface for TCC permission monitoring (Workstream 7)."""

    def check_permission(self, permission: Permission) -> PermissionStatus:
        """Check if a specific permission is granted.

        Args:
            permission: Permission to check.

        Returns:
            Status with granted flag and fix instructions if not granted.
        """
        ...

    def check_all(self) -> list[PermissionStatus]:
        """Check all required permissions.

        Returns:
            List of status objects for all required permissions.
        """
        ...

    def wait_for_permission(self, permission: Permission, timeout_seconds: int) -> bool:
        """Block until permission granted or timeout.

        Args:
            permission: Permission to wait for.
            timeout_seconds: Maximum time to wait in seconds.

        Returns:
            True if permission was granted within timeout, False otherwise.
        """
        ...


class SchemaDetector(Protocol):
    """Interface for chat.db schema detection (Workstream 7)."""

    def detect(self, db_path: str) -> SchemaInfo:
        """Detect schema version and compatibility.

        Args:
            db_path: Path to the chat.db database file.

        Returns:
            Schema information including version and compatibility status.
        """
        ...

    def get_query(self, query_name: str, schema_version: str) -> str:
        """Get appropriate SQL query for the detected schema.

        Args:
            query_name: Name of the query to retrieve.
            schema_version: Detected schema version.

        Returns:
            SQL query string appropriate for the schema version.
        """
        ...
