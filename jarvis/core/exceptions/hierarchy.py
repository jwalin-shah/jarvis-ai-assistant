from __future__ import annotations

from typing import Any

from jarvis.core.exceptions.codes import ErrorCode


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
        """Initialize a JARVIS error.

        Args:
            message: Human-readable error message.
            code: Error code for programmatic handling.
            details: Additional context as key-value pairs.
            cause: Original exception that caused this error.
        """
        self.message = message or self.default_message
        self.code = code or self.default_code
        self.details = details or {}
        self.cause = cause

        # Build full message for Exception base class
        super().__init__(self.message)

        # Chain the cause if provided
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        """Return human-readable representation."""
        return self.message

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
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
        """Convert error to dictionary for API responses.

        Returns:
            Dictionary with error, code, and detail fields.
        """
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


class iMessageError(JarvisError):  # noqa: N801
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


class iMessageAccessError(iMessageError):  # noqa: N801
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


class iMessageQueryError(iMessageError):  # noqa: N801
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


class ConversationNotFoundError(iMessageError):  # noqa: N801
    """Raised when a specific conversation is not found."""

    default_message = "Conversation not found"
    default_code = ErrorCode.MSG_NOT_FOUND


# Validation Errors


class ValidationError(JarvisError):
    """Raised for input validation failures."""

    default_message = "Validation error"
    default_code = ErrorCode.VAL_INVALID_INPUT

    def __init__(
        self,
        message: str | None = None,
        *,
        field: str | None = None,
        value: Any = None,
        expected: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        if expected:
            details["expected"] = expected
        super().__init__(message, code=code, details=details, cause=cause)


# Resource Errors


class ResourceError(JarvisError):
    """Base class for system resource errors."""

    default_message = "Resource error"
    default_code = ErrorCode.RES_MEMORY_LOW

    def __init__(
        self,
        message: str | None = None,
        *,
        resource_type: str | None = None,
        available: int | float | None = None,
        required: int | float | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if resource_type:
            details["resource_type"] = resource_type
        if available is not None:
            details["available"] = available
        if required is not None:
            details["required"] = required
        super().__init__(message, code=code, details=details, cause=cause)


class MemoryResourceError(ResourceError):
    """Raised when memory resources are insufficient."""

    default_message = "Insufficient memory"
    default_code = ErrorCode.RES_MEMORY_LOW

    def __init__(
        self,
        message: str | None = None,
        *,
        available_mb: int | None = None,
        required_mb: int | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(
            message,
            resource_type="memory",
            available=available_mb,
            required=required_mb,
            code=code,
            details=details,
            cause=cause,
        )


class DiskResourceError(ResourceError):
    """Raised when disk resources are insufficient or inaccessible."""

    default_message = "Disk resource error"
    default_code = ErrorCode.RES_DISK_ACCESS

    def __init__(
        self,
        message: str | None = None,
        *,
        path: str | None = None,
        available_mb: int | None = None,
        required_mb: int | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if path:
            details["path"] = path
        super().__init__(
            message,
            resource_type="disk",
            available=available_mb,
            required=required_mb,
            code=code,
            details=details,
            cause=cause,
        )


# Task Errors


class TaskError(JarvisError):
    """Base class for task-related errors."""

    default_message = "Task error"
    default_code = ErrorCode.TSK_EXECUTION_FAILED

    def __init__(
        self,
        message: str | None = None,
        *,
        task_id: str | None = None,
        task_type: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if task_id:
            details["task_id"] = task_id
        if task_type:
            details["task_type"] = task_type
        super().__init__(message, code=code, details=details, cause=cause)


class TaskNotFoundError(TaskError):
    """Raised when a task is not found."""

    default_message = "Task not found"
    default_code = ErrorCode.TSK_NOT_FOUND


class TaskExecutionError(TaskError):
    """Raised when task execution fails."""

    default_message = "Task execution failed"
    default_code = ErrorCode.TSK_EXECUTION_FAILED


# Calendar Errors


class CalendarError(JarvisError):
    """Base class for calendar-related errors."""

    default_message = "Calendar error"
    default_code = ErrorCode.CAL_NOT_AVAILABLE


class CalendarAccessError(CalendarError):
    """Raised when calendar access is denied."""

    default_message = "Cannot access Calendar"
    default_code = ErrorCode.CAL_ACCESS_DENIED

    def __init__(
        self,
        message: str | None = None,
        *,
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
                "Go to Privacy & Security > Calendars",
                "Enable access for your terminal application",
                "Restart JARVIS",
            ]
        super().__init__(message, code=code, details=details, cause=cause)


class CalendarCreateError(CalendarError):
    """Raised when creating a calendar event fails."""

    default_message = "Failed to create calendar event"
    default_code = ErrorCode.CAL_CREATE_FAILED

    def __init__(
        self,
        message: str | None = None,
        *,
        calendar_id: str | None = None,
        event_title: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if calendar_id:
            details["calendar_id"] = calendar_id
        if event_title:
            details["event_title"] = event_title
        super().__init__(message, code=code, details=details, cause=cause)


class EventParseError(CalendarError):
    """Raised when parsing event data from text fails."""

    default_message = "Failed to parse event from text"
    default_code = ErrorCode.CAL_PARSE_FAILED

    def __init__(
        self,
        message: str | None = None,
        *,
        source_text: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if source_text is not None:
            details["source_text"] = (
                source_text[:200] + "..." if len(source_text) > 200 else source_text
            )
        super().__init__(message, code=code, details=details, cause=cause)


# Experiment Errors


class ExperimentError(JarvisError):
    """Base class for experiment-related errors."""

    default_message = "Experiment error"
    default_code = ErrorCode.EXP_NOT_FOUND

    def __init__(
        self,
        message: str | None = None,
        *,
        experiment_name: str | None = None,
        variant_id: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if experiment_name:
            details["experiment_name"] = experiment_name
        if variant_id:
            details["variant_id"] = variant_id
        super().__init__(message, code=code, details=details, cause=cause)


class ExperimentNotFoundError(ExperimentError):
    """Raised when an experiment is not found."""

    default_message = "Experiment not found"
    default_code = ErrorCode.EXP_NOT_FOUND


class ExperimentConfigError(ExperimentError):
    """Raised when experiment configuration is invalid."""

    default_message = "Invalid experiment configuration"
    default_code = ErrorCode.EXP_INVALID_CONFIG


# Feedback Errors


class FeedbackError(JarvisError):
    """Base class for feedback-related errors."""

    default_message = "Feedback error"
    default_code = ErrorCode.FBK_STORE_ERROR

    def __init__(
        self,
        message: str | None = None,
        *,
        feedback_id: int | None = None,
        suggestion_id: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if feedback_id is not None:
            details["feedback_id"] = feedback_id
        if suggestion_id:
            details["suggestion_id"] = suggestion_id
        super().__init__(message, code=code, details=details, cause=cause)


class FeedbackNotFoundError(FeedbackError):
    """Raised when a feedback record is not found."""

    default_message = "Feedback not found"
    default_code = ErrorCode.FBK_NOT_FOUND


class FeedbackInvalidActionError(FeedbackError):
    """Raised when an invalid feedback action is provided."""

    default_message = "Invalid feedback action"
    default_code = ErrorCode.FBK_INVALID_ACTION


# Graph Errors


class GraphError(JarvisError):
    """Base class for knowledge graph-related errors."""

    default_message = "Graph operation failed"
    default_code = ErrorCode.GRF_BUILD_FAILED

    def __init__(
        self,
        message: str | None = None,
        *,
        contact_id: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if contact_id:
            details["contact_id"] = contact_id
        super().__init__(message, code=code, details=details, cause=cause)


class GraphContactNotFoundError(GraphError):
    """Raised when a contact is not found in the graph."""

    default_message = "Contact not found"
    default_code = ErrorCode.GRF_CONTACT_NOT_FOUND


# Database Errors


class DatabaseError(JarvisError):
    """Base class for database-related errors."""

    default_message = "Database operation failed"
    default_code = ErrorCode.DB_QUERY_FAILED

    def __init__(
        self,
        message: str | None = None,
        *,
        query: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if query is not None:
            details["query_preview"] = query[:200] + "..." if len(query) > 200 else query
        super().__init__(message, code=code, details=details, cause=cause)


# Export Errors


class ExportError(JarvisError):
    """Base class for export-related errors."""

    default_message = "Export operation failed"
    default_code = ErrorCode.EXPORT_GENERATION_FAILED

    def __init__(
        self,
        message: str | None = None,
        *,
        export_format: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if export_format:
            details["export_format"] = export_format
        super().__init__(message, code=code, details=details, cause=cause)


# Embedding Errors


class EmbeddingError(JarvisError):
    """Base class for embedding-related errors."""

    default_message = "Embedding operation failed"
    default_code = ErrorCode.EMB_ENCODING_FAILED

    def __init__(
        self,
        message: str | None = None,
        *,
        model_name: str | None = None,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if model_name:
            details["model_name"] = model_name
        super().__init__(message, code=code, details=details, cause=cause)
