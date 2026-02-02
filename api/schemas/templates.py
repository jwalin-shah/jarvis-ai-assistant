"""Custom template models.

Contains schemas for user-defined response templates.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CustomTemplateResponse(BaseModel):
    """Response model for a custom template.

    Represents a user-defined template for custom response patterns.

    Example:
        ```json
        {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "name": "Work Acknowledgment",
            "template_text": "Thanks for the update! I'll review and get back to you.",
            "trigger_phrases": ["got your update", "thanks for sending", "received the file"],
            "category": "work",
            "tags": ["professional", "acknowledgment"],
            "min_group_size": null,
            "max_group_size": null,
            "enabled": true,
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": "2024-01-15T10:30:00Z",
            "usage_count": 42
        }
        ```
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Work Acknowledgment",
                "template_text": "Thanks for the update! I'll review and get back to you.",
                "trigger_phrases": [
                    "got your update",
                    "thanks for sending",
                    "received the file",
                ],
                "category": "work",
                "tags": ["professional", "acknowledgment"],
                "min_group_size": None,
                "max_group_size": None,
                "enabled": True,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "usage_count": 42,
            }
        },
    )

    id: str = Field(
        ...,
        description="Unique template identifier (UUID)",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    name: str = Field(
        ...,
        description="Human-readable template name",
        examples=["Work Acknowledgment", "Casual Greeting"],
    )
    template_text: str = Field(
        ...,
        description="The response text to return when matched",
        examples=["Thanks for the update! I'll review and get back to you."],
    )
    trigger_phrases: list[str] = Field(
        ...,
        description="List of phrases that should trigger this template",
        examples=[["got your update", "thanks for sending", "received the file"]],
    )
    category: str = Field(
        default="general",
        description="Category for organization",
        examples=["work", "personal", "casual", "general"],
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Additional tags for filtering and organization",
        examples=[["professional", "acknowledgment"]],
    )
    min_group_size: int | None = Field(
        default=None,
        description="Minimum group size to apply this template (null = any)",
        examples=[None, 2, 5],
    )
    max_group_size: int | None = Field(
        default=None,
        description="Maximum group size to apply this template (null = any)",
        examples=[None, 5, 10],
    )
    enabled: bool = Field(
        default=True,
        description="Whether this template is active",
    )
    created_at: str = Field(
        ...,
        description="When the template was created (ISO format)",
        examples=["2024-01-15T10:30:00Z"],
    )
    updated_at: str = Field(
        ...,
        description="When the template was last modified (ISO format)",
        examples=["2024-01-15T10:30:00Z"],
    )
    usage_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this template has been matched",
        examples=[42, 0],
    )


class CustomTemplateCreateRequest(BaseModel):
    """Request to create a new custom template.

    Example:
        ```json
        {
            "name": "Work Acknowledgment",
            "template_text": "Thanks for the update! I'll review and get back to you.",
            "trigger_phrases": ["got your update", "thanks for sending"],
            "category": "work",
            "tags": ["professional"]
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Work Acknowledgment",
                "template_text": "Thanks for the update! I'll review and get back to you.",
                "trigger_phrases": ["got your update", "thanks for sending"],
                "category": "work",
                "tags": ["professional"],
            }
        }
    )

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Human-readable template name",
        examples=["Work Acknowledgment"],
    )
    template_text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The response text to return when matched",
        examples=["Thanks for the update! I'll review and get back to you."],
    )
    trigger_phrases: list[str] = Field(
        ...,
        min_length=1,
        description="List of phrases that should trigger this template (at least one required)",
        examples=[["got your update", "thanks for sending"]],
    )
    category: str = Field(
        default="general",
        max_length=50,
        description="Category for organization",
        examples=["work", "personal", "casual"],
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Additional tags for filtering",
        examples=[["professional", "acknowledgment"]],
    )
    min_group_size: int | None = Field(
        default=None,
        ge=1,
        description="Minimum group size to apply this template",
    )
    max_group_size: int | None = Field(
        default=None,
        ge=1,
        description="Maximum group size to apply this template",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this template is active",
    )


class CustomTemplateUpdateRequest(BaseModel):
    """Request to update an existing custom template.

    Partial update - only provided fields are changed.

    Example:
        ```json
        {
            "name": "Updated Name",
            "enabled": false
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Updated Name",
                "enabled": False,
            }
        }
    )

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Human-readable template name",
    )
    template_text: str | None = Field(
        default=None,
        min_length=1,
        max_length=1000,
        description="The response text to return when matched",
    )
    trigger_phrases: list[str] | None = Field(
        default=None,
        description="List of phrases that should trigger this template",
    )
    category: str | None = Field(
        default=None,
        max_length=50,
        description="Category for organization",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Additional tags for filtering",
    )
    min_group_size: int | None = Field(
        default=None,
        ge=1,
        description="Minimum group size to apply this template",
    )
    max_group_size: int | None = Field(
        default=None,
        ge=1,
        description="Maximum group size to apply this template",
    )
    enabled: bool | None = Field(
        default=None,
        description="Whether this template is active",
    )


class CustomTemplateListResponse(BaseModel):
    """Response containing a list of custom templates.

    Example:
        ```json
        {
            "templates": [...],
            "total": 10,
            "categories": ["work", "personal"],
            "tags": ["professional", "casual"]
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "templates": [],
                "total": 10,
                "categories": ["work", "personal"],
                "tags": ["professional", "casual"],
            }
        }
    )

    templates: list[CustomTemplateResponse] = Field(
        ...,
        description="List of custom templates",
    )
    total: int = Field(
        ...,
        ge=0,
        description="Total number of templates",
    )
    categories: list[str] = Field(
        default_factory=list,
        description="All unique categories",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="All unique tags",
    )


class CustomTemplateUsageStats(BaseModel):
    """Usage statistics for custom templates.

    Example:
        ```json
        {
            "total_templates": 15,
            "enabled_templates": 12,
            "total_usage": 250,
            "usage_by_category": {"work": 150, "personal": 100},
            "top_templates": [{"id": "...", "name": "Greeting", "usage_count": 50}]
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_templates": 15,
                "enabled_templates": 12,
                "total_usage": 250,
                "usage_by_category": {"work": 150, "personal": 100},
                "top_templates": [{"id": "abc123", "name": "Greeting", "usage_count": 50}],
            }
        }
    )

    total_templates: int = Field(
        ...,
        ge=0,
        description="Total number of custom templates",
    )
    enabled_templates: int = Field(
        ...,
        ge=0,
        description="Number of enabled templates",
    )
    total_usage: int = Field(
        ...,
        ge=0,
        description="Total usage count across all templates",
    )
    usage_by_category: dict[str, int] = Field(
        default_factory=dict,
        description="Usage counts grouped by category",
    )
    top_templates: list[dict[str, str | int]] = Field(
        default_factory=list,
        description="Top templates by usage count",
    )


class CustomTemplateTestRequest(BaseModel):
    """Request to test a template against sample inputs.

    Example:
        ```json
        {
            "trigger_phrases": ["got your update", "thanks for sending"],
            "test_inputs": ["got your email update", "thanks for the info"]
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trigger_phrases": ["got your update", "thanks for sending"],
                "test_inputs": ["got your email update", "thanks for the info"],
            }
        }
    )

    trigger_phrases: list[str] = Field(
        ...,
        min_length=1,
        description="Template trigger phrases to test",
    )
    test_inputs: list[str] = Field(
        ...,
        min_length=1,
        description="Sample inputs to test against",
    )


class CustomTemplateTestResult(BaseModel):
    """Result of testing a single input against template triggers.

    Example:
        ```json
        {
            "input": "got your email update",
            "matched": true,
            "best_match": "got your update",
            "similarity": 0.85
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "input": "got your email update",
                "matched": True,
                "best_match": "got your update",
                "similarity": 0.85,
            }
        }
    )

    input: str = Field(
        ...,
        description="The test input that was evaluated",
    )
    matched: bool = Field(
        ...,
        description="Whether the input matched any trigger phrase",
    )
    best_match: str | None = Field(
        default=None,
        description="The trigger phrase with highest similarity",
    )
    similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score (0.0 to 1.0)",
    )


class CustomTemplateTestResponse(BaseModel):
    """Response containing test results for template matching.

    Example:
        ```json
        {
            "results": [...],
            "match_rate": 0.75,
            "threshold": 0.7
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [],
                "match_rate": 0.75,
                "threshold": 0.7,
            }
        }
    )

    results: list[CustomTemplateTestResult] = Field(
        ...,
        description="Test results for each input",
    )
    match_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Percentage of inputs that matched",
    )
    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity threshold used for matching",
    )


class CustomTemplateExportRequest(BaseModel):
    """Request to export templates.

    Example:
        ```json
        {
            "template_ids": ["id1", "id2"]
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "template_ids": None,
            }
        }
    )

    template_ids: list[str] | None = Field(
        default=None,
        description="Specific template IDs to export, or null for all",
    )


class CustomTemplateExportResponse(BaseModel):
    """Response containing exported templates.

    Example:
        ```json
        {
            "version": 1,
            "export_date": "2024-01-15T10:30:00Z",
            "template_count": 5,
            "templates": [...]
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "version": 1,
                "export_date": "2024-01-15T10:30:00Z",
                "template_count": 5,
                "templates": [],
            }
        }
    )

    version: int = Field(
        ...,
        description="Export format version",
    )
    export_date: str = Field(
        ...,
        description="When the export was created",
    )
    template_count: int = Field(
        ...,
        ge=0,
        description="Number of templates in export",
    )
    templates: list[dict[str, Any]] = Field(
        ...,
        description="Template data for import",
    )


class CustomTemplateImportRequest(BaseModel):
    """Request to import templates.

    Example:
        ```json
        {
            "data": {"version": 1, "templates": [...]},
            "overwrite": false
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data": {"version": 1, "templates": []},
                "overwrite": False,
            }
        }
    )

    data: dict[str, Any] = Field(
        ...,
        description="Export data to import",
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing templates with same ID",
    )


class CustomTemplateImportResponse(BaseModel):
    """Response after importing templates.

    Example:
        ```json
        {
            "imported": 5,
            "skipped": 0,
            "errors": 0,
            "total_templates": 15
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "imported": 5,
                "skipped": 0,
                "errors": 0,
                "total_templates": 15,
            }
        }
    )

    imported: int = Field(
        ...,
        ge=0,
        description="Number of templates successfully imported",
    )
    skipped: int = Field(
        ...,
        ge=0,
        description="Number of templates skipped",
    )
    errors: int = Field(
        ...,
        ge=0,
        description="Number of templates that failed to import",
    )
    total_templates: int = Field(
        ...,
        ge=0,
        description="Total templates after import",
    )
