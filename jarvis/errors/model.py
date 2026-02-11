"""Model and iMessage error classes.

Contains errors for model loading/generation and iMessage database access.
"""

from __future__ import annotations

from typing import Any

from jarvis.errors.base import ErrorCode, JarvisError

# Model Errors


class ModelError(JarvisError):
    """Base class for model-related errors."""

    default_message = "Model error"
    default_code = ErrorCode.MDL_GENERATION_FAILED

    def __init__(
        self,
        message: str | None = None,
        *,
        model_name: str | None = None,
        model_path: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if model_name:
            details["model_name"] = model_name
        if model_path:
            details["model_path"] = model_path
        super().__init__(message, code=code, details=details, cause=cause)


class ModelLoadError(ModelError):
    """Raised when model loading fails."""

    default_message = "Failed to load model"
    default_code = ErrorCode.MDL_LOAD_FAILED


class ModelGenerationError(ModelError):
    """Raised when text generation fails."""

    default_message = "Text generation failed"
    default_code = ErrorCode.MDL_GENERATION_FAILED

    def __init__(
        self,
        message: str | None = None,
        *,
        prompt: str | None = None,
        timeout_seconds: float | None = None,
        model_name: str | None = None,
        model_path: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds
            code = code or ErrorCode.MDL_TIMEOUT
        if prompt is not None:
            details["prompt_preview"] = prompt[:200] + "..." if len(prompt) > 200 else prompt
        super().__init__(
            message,
            model_name=model_name,
            model_path=model_path,
            code=code,
            details=details,
            cause=cause,
        )


# iMessage Errors


class iMessageError(JarvisError):  # noqa: N801 - iMessage is a brand name
    """Base class for iMessage-related errors."""

    default_message = "iMessage error"
    default_code = ErrorCode.MSG_QUERY_FAILED

    def __init__(
        self,
        message: str | None = None,
        *,
        db_path: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if db_path:
            details["db_path"] = db_path
        super().__init__(message, code=code, details=details, cause=cause)


class iMessageAccessError(iMessageError):  # noqa: N801 - iMessage is a brand name
    """Raised when iMessage database access is denied."""

    default_message = "Cannot access iMessage database"
    default_code = ErrorCode.MSG_ACCESS_DENIED

    def __init__(
        self,
        message: str | None = None,
        *,
        db_path: str | None = None,
        requires_permission: bool = False,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if requires_permission:
            details["requires_permission"] = True
            details["permission_instructions"] = [
                "Open System Settings",
                "Go to Privacy & Security > Full Disk Access",
                "Add and enable your terminal application",
                "Restart JARVIS",
            ]
        super().__init__(message, db_path=db_path, code=code, details=details, cause=cause)


class iMessageQueryError(iMessageError):  # noqa: N801 - iMessage is a brand name
    """Raised when iMessage database queries fail."""

    default_message = "iMessage query failed"
    default_code = ErrorCode.MSG_QUERY_FAILED

    def __init__(
        self,
        message: str | None = None,
        *,
        query: str | None = None,
        db_path: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if query is not None:
            details["query_preview"] = query[:200] + "..." if len(query) > 200 else query
        super().__init__(message, db_path=db_path, code=code, details=details, cause=cause)
