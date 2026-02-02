"""Health and system models.

Contains schemas for health checks, model info, and error responses.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelInfo(BaseModel):
    """Current model information.

    Provides details about the currently loaded MLX language model.

    Example:
        ```json
        {
            "id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "display_name": "Qwen 0.5B (Fast)",
            "loaded": true,
            "memory_usage_mb": 450.5,
            "quality_tier": "basic"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "display_name": "Qwen 0.5B (Fast)",
                "loaded": True,
                "memory_usage_mb": 450.5,
                "quality_tier": "basic",
            }
        }
    )

    id: str | None = Field(
        default=None,
        description="Model identifier (HuggingFace path)",
        examples=["mlx-community/Qwen2.5-0.5B-Instruct-4bit"],
    )
    display_name: str = Field(
        ...,
        description="Human-readable model name",
        examples=["Qwen 0.5B (Fast)", "Qwen 1.5B (Balanced)"],
    )
    loaded: bool = Field(
        ...,
        description="True if the model is currently loaded in memory",
    )
    memory_usage_mb: float = Field(
        ...,
        description="Current memory usage of the model in megabytes",
        examples=[450.5, 1024.0],
        ge=0,
    )
    quality_tier: str | None = Field(
        default=None,
        description="Quality tier: 'basic', 'good', or 'best'",
        examples=["basic", "good", "best"],
    )


class HealthResponse(BaseModel):
    """System health status response.

    Comprehensive health check including memory, permissions, model state,
    and overall system status.

    Example:
        ```json
        {
            "status": "healthy",
            "imessage_access": true,
            "memory_available_gb": 12.5,
            "memory_used_gb": 3.5,
            "memory_mode": "FULL",
            "model_loaded": true,
            "permissions_ok": true,
            "jarvis_rss_mb": 256.5,
            "jarvis_vms_mb": 1024.0
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "imessage_access": True,
                "memory_available_gb": 12.5,
                "memory_used_gb": 3.5,
                "memory_mode": "FULL",
                "model_loaded": True,
                "permissions_ok": True,
                "details": None,
                "jarvis_rss_mb": 256.5,
                "jarvis_vms_mb": 1024.0,
                "model": None,
                "recommended_model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "system_ram_gb": 16.0,
            }
        }
    )

    status: str = Field(
        ...,
        description="Overall health status: 'healthy', 'degraded', or 'unhealthy'",
        examples=["healthy", "degraded", "unhealthy"],
    )
    imessage_access: bool = Field(
        ...,
        description="True if Full Disk Access is granted for iMessage database",
    )
    memory_available_gb: float = Field(
        ...,
        description="Available system memory in gigabytes",
        examples=[12.5, 4.0],
        ge=0,
    )
    memory_used_gb: float = Field(
        ...,
        description="Used system memory in gigabytes",
        examples=[3.5, 8.0],
        ge=0,
    )
    memory_mode: str = Field(
        ...,
        description="Memory controller mode: 'FULL', 'LITE', or 'MINIMAL'",
        examples=["FULL", "LITE", "MINIMAL"],
    )
    model_loaded: bool = Field(
        ...,
        description="True if the MLX language model is loaded",
    )
    permissions_ok: bool = Field(
        ...,
        description="True if all required permissions are granted",
    )
    details: dict[str, str] | None = Field(
        default=None,
        description="Additional details about any issues detected",
        examples=[{"imessage": "Full Disk Access required"}],
    )
    jarvis_rss_mb: float = Field(
        default=0.0,
        description="JARVIS process Resident Set Size (actual RAM usage) in MB",
        examples=[256.5, 512.0],
        ge=0,
    )
    jarvis_vms_mb: float = Field(
        default=0.0,
        description="JARVIS process Virtual Memory Size in MB",
        examples=[1024.0, 2048.0],
        ge=0,
    )
    model: ModelInfo | None = Field(
        default=None,
        description="Information about the currently loaded model",
    )
    recommended_model: str | None = Field(
        default=None,
        description="Recommended model ID based on system RAM",
        examples=["mlx-community/Qwen2.5-1.5B-Instruct-4bit"],
    )
    system_ram_gb: float | None = Field(
        default=None,
        description="Total system RAM in gigabytes",
        examples=[16.0, 32.0],
        ge=0,
    )


class ErrorResponse(BaseModel):
    """Standardized error response model.

    All API errors return this format for consistent client handling.

    Attributes:
        error: The exception class name (e.g., "ValidationError", "ModelLoadError").
        code: Machine-readable error code (e.g., "VAL_INVALID_INPUT", "MDL_LOAD_FAILED").
        detail: Human-readable error message describing what went wrong.
        details: Optional additional context about the error (field names, paths, etc.).

    Example Response (400 Bad Request):
        {
            "error": "ValidationError",
            "code": "VAL_MISSING_REQUIRED",
            "detail": "Missing required field: email",
            "details": {"field": "email"}
        }

    Example Response (403 Forbidden):
        {
            "error": "iMessageAccessError",
            "code": "MSG_ACCESS_DENIED",
            "detail": "Full Disk Access is required to read iMessages",
            "details": {
                "requires_permission": true,
                "permission_instructions": [
                    "Open System Settings",
                    "Go to Privacy & Security > Full Disk Access",
                    "Add and enable your terminal application",
                    "Restart JARVIS"
                ]
            }
        }

    Example Response (503 Service Unavailable):
        {
            "error": "ModelLoadError",
            "code": "RES_MEMORY_EXHAUSTED",
            "detail": "Insufficient memory to load model: qwen-3b",
            "details": {"available_mb": 1024, "required_mb": 2048}
        }
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "code": "VAL_INVALID_INPUT",
                "detail": "Missing required field: email",
                "details": {"field": "email"},
            }
        }
    )

    error: str = Field(
        ...,
        description="Exception class name (e.g., 'ValidationError')",
        examples=["ValidationError", "ModelLoadError", "iMessageAccessError"],
    )
    code: str = Field(
        ...,
        description="Machine-readable error code for programmatic handling",
        examples=["VAL_INVALID_INPUT", "MDL_LOAD_FAILED", "MSG_ACCESS_DENIED"],
    )
    detail: str = Field(
        ...,
        description="Human-readable error message",
        examples=["Missing required field: email", "Failed to load model"],
    )
    details: dict[str, str | int | float | bool | list[str]] | None = Field(
        default=None,
        description="Additional context about the error (optional)",
        examples=[{"field": "email"}, {"available_mb": 1024, "required_mb": 2048}],
    )
