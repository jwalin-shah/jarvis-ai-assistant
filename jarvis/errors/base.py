"""Base error classes and error codes for JARVIS.

Contains ErrorCode enum, JarvisError base class, and ConfigurationError.
All JARVIS-specific exceptions inherit from JarvisError.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """Standard error codes for JARVIS errors.

    These codes can be used to programmatically identify error types
    and are included in API error responses.
    """

    # Configuration errors (CFG_*)
    CFG_INVALID = "CFG_INVALID"
    CFG_MISSING = "CFG_MISSING"
    CFG_MIGRATION_FAILED = "CFG_MIGRATION_FAILED"

    # Model errors (MDL_*)
    MDL_LOAD_FAILED = "MDL_LOAD_FAILED"
    MDL_NOT_FOUND = "MDL_NOT_FOUND"
    MDL_GENERATION_FAILED = "MDL_GENERATION_FAILED"
    MDL_TIMEOUT = "MDL_TIMEOUT"
    MDL_INVALID_REQUEST = "MDL_INVALID_REQUEST"

    # iMessage errors (MSG_*)
    MSG_ACCESS_DENIED = "MSG_ACCESS_DENIED"
    MSG_DB_NOT_FOUND = "MSG_DB_NOT_FOUND"
    MSG_QUERY_FAILED = "MSG_QUERY_FAILED"
    MSG_SCHEMA_UNSUPPORTED = "MSG_SCHEMA_UNSUPPORTED"
    MSG_SEND_FAILED = "MSG_SEND_FAILED"

    # Validation errors (VAL_*)
    VAL_INVALID_INPUT = "VAL_INVALID_INPUT"
    VAL_MISSING_REQUIRED = "VAL_MISSING_REQUIRED"
    VAL_TYPE_ERROR = "VAL_TYPE_ERROR"

    # Resource errors (RES_*)
    RES_MEMORY_LOW = "RES_MEMORY_LOW"
    RES_MEMORY_EXHAUSTED = "RES_MEMORY_EXHAUSTED"
    RES_DISK_FULL = "RES_DISK_FULL"
    RES_DISK_ACCESS = "RES_DISK_ACCESS"

    # Task errors (TSK_*)
    TSK_NOT_FOUND = "TSK_NOT_FOUND"
    TSK_INVALID_STATUS = "TSK_INVALID_STATUS"
    TSK_EXECUTION_FAILED = "TSK_EXECUTION_FAILED"
    TSK_CANCELLED = "TSK_CANCELLED"
    TSK_QUEUE_FULL = "TSK_QUEUE_FULL"

    # Calendar errors (CAL_*)
    CAL_ACCESS_DENIED = "CAL_ACCESS_DENIED"
    CAL_NOT_AVAILABLE = "CAL_NOT_AVAILABLE"
    CAL_CREATE_FAILED = "CAL_CREATE_FAILED"
    CAL_PARSE_FAILED = "CAL_PARSE_FAILED"

    # Experiment errors (EXP_*)
    EXP_NOT_FOUND = "EXP_NOT_FOUND"
    EXP_INVALID_CONFIG = "EXP_INVALID_CONFIG"
    EXP_VARIANT_NOT_FOUND = "EXP_VARIANT_NOT_FOUND"
    EXP_ALREADY_EXISTS = "EXP_ALREADY_EXISTS"

    # Feedback errors (FBK_*)
    FBK_NOT_FOUND = "FBK_NOT_FOUND"
    FBK_INVALID_ACTION = "FBK_INVALID_ACTION"
    FBK_STORE_ERROR = "FBK_STORE_ERROR"

    # Graph errors (GRF_*)
    GRF_BUILD_FAILED = "GRF_BUILD_FAILED"
    GRF_CONTACT_NOT_FOUND = "GRF_CONTACT_NOT_FOUND"

    # Database errors (DB_*)
    DB_CONNECTION_FAILED = "DB_CONNECTION_FAILED"
    DB_QUERY_FAILED = "DB_QUERY_FAILED"
    DB_INTEGRITY_ERROR = "DB_INTEGRITY_ERROR"

    # Export errors (EXPORT_*)
    EXPORT_GENERATION_FAILED = "EXPORT_GENERATION_FAILED"
    EXPORT_INVALID_FORMAT = "EXPORT_INVALID_FORMAT"

    # Embedding errors (EMB_*)
    EMB_ENCODING_FAILED = "EMB_ENCODING_FAILED"
    EMB_INDEX_NOT_READY = "EMB_INDEX_NOT_READY"

    # Generic errors
    UNKNOWN = "UNKNOWN"


class JarvisError(Exception):
    """Base exception for all JARVIS errors.

    All JARVIS-specific exceptions inherit from this class, enabling
    consistent error handling patterns across the codebase.

    Attributes:
        message: Human-readable error message.
        code: Machine-readable error code from ErrorCode enum.
        details: Optional additional context about the error.
        cause: Optional original exception that caused this error.
    """

    default_message: str = "An error occurred"
    default_code: ErrorCode = ErrorCode.UNKNOWN

    def __init__(
        self,
        message: str | None = None,
        *,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.code = code or self.default_code
        self.details = details or {}
        self.cause = cause

        super().__init__(self.message)

        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        parts = [f"{self.__class__.__name__}({self.message!r}"]
        if self.code != self.default_code:
            parts.append(f", code={self.code.value!r}")
        if self.details:
            parts.append(f", details={self.details!r}")
        if self.cause:
            parts.append(f", cause={self.cause!r}")
        parts.append(")")
        return "".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for API responses."""
        result: dict[str, Any] = {
            "error": self.__class__.__name__,
            "code": self.code.value,
            "detail": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result


# Configuration Errors


class ConfigurationError(JarvisError):
    """Raised for configuration and settings issues."""

    default_message = "Configuration error"
    default_code = ErrorCode.CFG_INVALID

    def __init__(
        self,
        message: str | None = None,
        *,
        config_key: str | None = None,
        config_path: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if config_key:
            details["config_key"] = config_key
        if config_path:
            details["config_path"] = config_path
        super().__init__(message, code=code, details=details, cause=cause)
