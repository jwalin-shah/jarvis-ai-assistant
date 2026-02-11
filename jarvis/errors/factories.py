"""Convenience factory functions for common error scenarios.

Provides shorthand functions for creating commonly-used error instances
with appropriate error codes and details pre-filled.
"""

from __future__ import annotations

from typing import Any

from jarvis.errors.base import ErrorCode
from jarvis.errors.domain import (
    CalendarAccessError,
    TaskError,
    TaskNotFoundError,
    ValidationError,
)
from jarvis.errors.model import (
    ModelGenerationError,
    ModelLoadError,
    iMessageAccessError,
)


def task_not_found(task_id: str) -> TaskNotFoundError:
    """Create a TaskNotFoundError for a missing task."""
    return TaskNotFoundError(
        f"Task not found: {task_id}",
        task_id=task_id,
    )


def task_invalid_status(task_id: str, current_status: str, required_status: str) -> TaskError:
    """Create a TaskError for invalid task status."""
    return TaskError(
        f"Task {task_id} has status '{current_status}', but requires '{required_status}'",
        task_id=task_id,
        code=ErrorCode.TSK_INVALID_STATUS,
        details={"current_status": current_status, "required_status": required_status},
    )


def model_not_found(model_path: str) -> ModelLoadError:
    """Create a ModelLoadError for a missing model."""
    return ModelLoadError(
        f"Model not found at: {model_path}",
        model_path=model_path,
        code=ErrorCode.MDL_NOT_FOUND,
    )


def model_out_of_memory(
    model_name: str, available_mb: int | None = None, required_mb: int | None = None
) -> ModelLoadError:
    """Create a ModelLoadError for out of memory during loading."""
    details: dict[str, Any] = {}
    if available_mb is not None:
        details["available_mb"] = available_mb
    if required_mb is not None:
        details["required_mb"] = required_mb

    return ModelLoadError(
        f"Insufficient memory to load model: {model_name}",
        model_name=model_name,
        code=ErrorCode.RES_MEMORY_EXHAUSTED,
        details=details,
    )


def imessage_permission_denied(db_path: str | None = None) -> iMessageAccessError:
    """Create an iMessageAccessError for Full Disk Access requirement."""
    return iMessageAccessError(
        "Full Disk Access is required to read iMessages",
        db_path=db_path,
        requires_permission=True,
    )


def imessage_db_not_found(db_path: str) -> iMessageAccessError:
    """Create an iMessageAccessError for missing database."""
    return iMessageAccessError(
        f"iMessage database not found at: {db_path}",
        db_path=db_path,
        code=ErrorCode.MSG_DB_NOT_FOUND,
    )


def validation_required(field: str) -> ValidationError:
    """Create a ValidationError for a missing required field."""
    return ValidationError(
        f"Missing required field: {field}",
        field=field,
        code=ErrorCode.VAL_MISSING_REQUIRED,
    )


def validation_type_error(field: str, value: Any, expected: str) -> ValidationError:
    """Create a ValidationError for an incorrect type."""
    return ValidationError(
        f"Invalid type for '{field}': expected {expected}, got {type(value).__name__}",
        field=field,
        value=value,
        expected=expected,
        code=ErrorCode.VAL_TYPE_ERROR,
    )


def model_generation_timeout(
    model_name: str, timeout_seconds: float, prompt: str | None = None
) -> ModelGenerationError:
    """Create a ModelGenerationError for generation timeout."""
    return ModelGenerationError(
        f"Model generation timed out after {timeout_seconds} seconds",
        model_name=model_name,
        timeout_seconds=timeout_seconds,
        prompt=prompt,
        code=ErrorCode.MDL_TIMEOUT,
    )


def calendar_permission_denied() -> CalendarAccessError:
    """Create a CalendarAccessError for Calendar permission requirement."""
    return CalendarAccessError(
        "Calendar access is required to manage events",
        requires_permission=True,
    )
