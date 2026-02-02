"""Settings models.

Contains schemas for application settings, model selection, and configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AvailableModelInfo(BaseModel):
    """Information about an available model.

    Details about a model that can be selected for use.

    Example:
        ```json
        {
            "model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "name": "Qwen 0.5B (Fast)",
            "size_gb": 0.4,
            "quality_tier": "basic",
            "ram_requirement_gb": 4.0,
            "is_downloaded": true,
            "is_loaded": false,
            "is_recommended": false,
            "description": "Fastest responses, good for simple tasks"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "name": "Qwen 0.5B (Fast)",
                "size_gb": 0.4,
                "quality_tier": "basic",
                "ram_requirement_gb": 4.0,
                "is_downloaded": True,
                "is_loaded": False,
                "is_recommended": False,
                "description": "Fastest responses, good for simple tasks",
            }
        }
    )

    model_id: str = Field(
        ...,
        description="Unique model identifier (HuggingFace path)",
        examples=["mlx-community/Qwen2.5-0.5B-Instruct-4bit"],
    )
    name: str = Field(
        ...,
        description="Human-readable model name",
        examples=["Qwen 0.5B (Fast)", "Qwen 1.5B (Balanced)"],
    )
    size_gb: float = Field(
        ...,
        description="Model size on disk in gigabytes",
        examples=[0.4, 1.0, 2.0],
        ge=0,
    )
    quality_tier: str = Field(
        ...,
        description="Quality tier: 'basic', 'good', or 'best'",
        examples=["basic", "good", "best"],
    )
    ram_requirement_gb: float = Field(
        ...,
        description="Minimum RAM required to run this model",
        examples=[4.0, 8.0, 16.0],
        ge=0,
    )
    is_downloaded: bool = Field(
        ...,
        description="True if the model is downloaded locally",
    )
    is_loaded: bool = Field(
        ...,
        description="True if the model is currently loaded in memory",
    )
    is_recommended: bool = Field(
        ...,
        description="True if this is the recommended model for the system",
    )
    description: str | None = Field(
        default=None,
        description="Brief description of the model's characteristics",
        examples=["Fastest responses, good for simple tasks"],
    )


class GenerationSettings(BaseModel):
    """Generation parameter settings.

    Controls how the AI model generates text.

    Example:
        ```json
        {
            "temperature": 0.7,
            "max_tokens_reply": 150,
            "max_tokens_summary": 500
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "temperature": 0.7,
                "max_tokens_reply": 150,
                "max_tokens_summary": 500,
            }
        }
    )

    temperature: float = Field(
        default=0.7,
        ge=0.1,
        le=1.0,
        description="Sampling temperature (0.1=focused, 1.0=creative)",
        examples=[0.7, 0.5, 0.9],
    )
    max_tokens_reply: int = Field(
        default=150,
        ge=50,
        le=300,
        description="Maximum tokens for reply generation",
        examples=[150, 200],
    )
    max_tokens_summary: int = Field(
        default=500,
        ge=200,
        le=1000,
        description="Maximum tokens for summary generation",
        examples=[500, 750],
    )


class BehaviorSettings(BaseModel):
    """Behavior preference settings.

    Controls JARVIS behavior and defaults.

    Example:
        ```json
        {
            "auto_suggest_replies": true,
            "suggestion_count": 3,
            "context_messages_reply": 20,
            "context_messages_summary": 50
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "auto_suggest_replies": True,
                "suggestion_count": 3,
                "context_messages_reply": 20,
                "context_messages_summary": 50,
            }
        }
    )

    auto_suggest_replies: bool = Field(
        default=True,
        description="Automatically suggest replies when viewing conversations",
    )
    suggestion_count: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Default number of reply suggestions to generate",
    )
    context_messages_reply: int = Field(
        default=20,
        ge=10,
        le=50,
        description="Default number of messages to use for reply context",
    )
    context_messages_summary: int = Field(
        default=50,
        ge=20,
        le=100,
        description="Default number of messages to use for summaries",
    )


class SystemInfo(BaseModel):
    """Read-only system information.

    Current system state including memory and model status.

    Example:
        ```json
        {
            "system_ram_gb": 16.0,
            "current_memory_usage_gb": 8.5,
            "model_loaded": true,
            "model_memory_usage_gb": 0.5,
            "imessage_access": true
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "system_ram_gb": 16.0,
                "current_memory_usage_gb": 8.5,
                "model_loaded": True,
                "model_memory_usage_gb": 0.5,
                "imessage_access": True,
            }
        }
    )

    system_ram_gb: float = Field(
        ...,
        description="Total system RAM in gigabytes",
        examples=[16.0, 32.0],
        ge=0,
    )
    current_memory_usage_gb: float = Field(
        ...,
        description="Current system memory usage in gigabytes",
        examples=[8.5, 12.0],
        ge=0,
    )
    model_loaded: bool = Field(
        ...,
        description="True if an AI model is currently loaded",
    )
    model_memory_usage_gb: float = Field(
        ...,
        description="Memory used by the loaded model in gigabytes",
        examples=[0.5, 1.5],
        ge=0,
    )
    imessage_access: bool = Field(
        ...,
        description="True if iMessage database access is available",
    )


class SettingsResponse(BaseModel):
    """Complete settings response.

    Full settings state including model, generation, behavior, and system info.

    Example:
        ```json
        {
            "model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "generation": {"temperature": 0.7, "max_tokens_reply": 150},
            "behavior": {"auto_suggest_replies": true, "suggestion_count": 3},
            "system": {"system_ram_gb": 16.0, "model_loaded": true}
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "generation": {
                    "temperature": 0.7,
                    "max_tokens_reply": 150,
                    "max_tokens_summary": 500,
                },
                "behavior": {
                    "auto_suggest_replies": True,
                    "suggestion_count": 3,
                    "context_messages_reply": 20,
                    "context_messages_summary": 50,
                },
                "system": {
                    "system_ram_gb": 16.0,
                    "current_memory_usage_gb": 8.5,
                    "model_loaded": True,
                    "model_memory_usage_gb": 0.5,
                    "imessage_access": True,
                },
            }
        }
    )

    model_id: str = Field(
        ...,
        description="Currently selected model ID",
        examples=["mlx-community/Qwen2.5-0.5B-Instruct-4bit"],
    )
    generation: GenerationSettings = Field(
        ...,
        description="Generation parameter settings",
    )
    behavior: BehaviorSettings = Field(
        ...,
        description="Behavior preference settings",
    )
    system: SystemInfo = Field(
        ...,
        description="Read-only system information",
    )


class SettingsUpdateRequest(BaseModel):
    """Request to update settings.

    Partial update - only provided fields are changed.

    Example:
        ```json
        {
            "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "generation": {"temperature": 0.8}
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "generation": {"temperature": 0.8},
            }
        }
    )

    model_id: str | None = Field(
        default=None,
        description="New model ID to switch to",
        examples=["mlx-community/Qwen2.5-1.5B-Instruct-4bit"],
    )
    generation: GenerationSettings | None = Field(
        default=None,
        description="Updated generation settings",
    )
    behavior: BehaviorSettings | None = Field(
        default=None,
        description="Updated behavior settings",
    )


class DownloadStatus(BaseModel):
    """Model download status.

    Current status of a model download operation.

    Example:
        ```json
        {
            "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "status": "downloading",
            "progress": 45.5,
            "error": null
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "status": "downloading",
                "progress": 45.5,
                "error": None,
            }
        }
    )

    model_id: str = Field(
        ...,
        description="Model ID being downloaded",
        examples=["mlx-community/Qwen2.5-1.5B-Instruct-4bit"],
    )
    status: str = Field(
        ...,
        description="Download status: 'downloading', 'completed', or 'failed'",
        examples=["downloading", "completed", "failed"],
    )
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Download progress percentage (0-100)",
        examples=[45.5, 100.0],
    )
    error: str | None = Field(
        default=None,
        description="Error message if download failed",
        examples=[None, "Network error: connection timed out"],
    )


class ActivateResponse(BaseModel):
    """Response after activating a model.

    Result of switching to a different model.

    Example:
        ```json
        {
            "success": true,
            "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "error": null
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "error": None,
            }
        }
    )

    success: bool = Field(
        ...,
        description="True if the model was activated successfully",
    )
    model_id: str = Field(
        ...,
        description="The model ID that was activated",
        examples=["mlx-community/Qwen2.5-1.5B-Instruct-4bit"],
    )
    error: str | None = Field(
        default=None,
        description="Error message if activation failed",
        examples=[None, "Model not downloaded. Please download first."],
    )
