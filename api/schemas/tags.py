"""Tag and SmartFolder API schemas.

Contains Pydantic models for tag management, smart folders, and auto-tagging.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# -----------------------------------------------------------------------------
# Tag Schemas
# -----------------------------------------------------------------------------


class TagBase(BaseModel):
    """Base fields for tag models."""

    name: str = Field(
        ...,
        description="Display name for the tag",
        examples=["Work", "Family", "Urgent"],
        min_length=1,
        max_length=100,
    )
    color: str = Field(
        default="#3b82f6",
        description="Hex color code for the tag",
        examples=["#ef4444", "#22c55e", "#3b82f6"],
        pattern=r"^#[0-9a-fA-F]{6}$",
    )
    icon: str = Field(
        default="tag",
        description="Icon name for the tag",
        examples=["tag", "star", "briefcase", "home"],
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the tag",
        examples=["Work-related conversations", "Family members"],
        max_length=500,
    )
    parent_id: int | None = Field(
        default=None,
        description="Parent tag ID for hierarchical tags",
        examples=[1, None],
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names for the tag",
        examples=[["trabajo", "travail"]],
    )


class TagCreate(TagBase):
    """Request to create a new tag."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Work",
                "color": "#3b82f6",
                "icon": "briefcase",
                "description": "Work-related conversations",
                "aliases": ["job", "office"],
            }
        }
    )


class TagUpdate(BaseModel):
    """Request to update an existing tag."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Work Projects",
                "color": "#0066cc",
            }
        }
    )

    name: str | None = Field(default=None, min_length=1, max_length=100)
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    icon: str | None = None
    description: str | None = None
    parent_id: int | None = None
    aliases: list[str] | None = None
    sort_order: int | None = None


class TagResponse(TagBase):
    """Tag response model."""

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "name": "Work",
                "color": "#3b82f6",
                "icon": "briefcase",
                "description": "Work-related conversations",
                "parent_id": None,
                "aliases": ["job", "office"],
                "sort_order": 0,
                "is_system": False,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            }
        },
    )

    id: int = Field(..., description="Unique tag identifier")
    sort_order: int = Field(default=0, description="Order within siblings")
    is_system: bool = Field(default=False, description="True if system-generated tag")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")
    updated_at: datetime | None = Field(default=None, description="Last update timestamp")


class TagWithPath(TagResponse):
    """Tag response with hierarchical path."""

    path: str = Field(
        ...,
        description="Full hierarchical path (e.g., 'Work/Projects/Alpha')",
        examples=["Work/Projects/Alpha", "Personal"],
    )


class TagListResponse(BaseModel):
    """Response containing a list of tags."""

    tags: list[TagResponse] = Field(..., description="List of tags")
    total: int = Field(..., description="Total number of tags", ge=0)


# -----------------------------------------------------------------------------
# Conversation Tag Schemas
# -----------------------------------------------------------------------------


class ConversationTagRequest(BaseModel):
    """Request to add a tag to a conversation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tag_id": 1,
            }
        }
    )

    tag_id: int = Field(..., description="Tag ID to add")


class ConversationTagResponse(BaseModel):
    """Response with tag assignment details."""

    model_config = ConfigDict(from_attributes=True)

    chat_id: str = Field(..., description="Conversation identifier")
    tag: TagResponse = Field(..., description="The assigned tag")
    added_at: datetime | None = Field(default=None, description="When the tag was added")
    added_by: str = Field(default="user", description="Who/what added the tag")
    confidence: float = Field(
        default=1.0, description="Confidence score for auto-assigned tags", ge=0, le=1
    )


class ConversationTagsResponse(BaseModel):
    """Response with all tags for a conversation."""

    chat_id: str = Field(..., description="Conversation identifier")
    tags: list[ConversationTagResponse] = Field(..., description="List of tag assignments")


class BulkTagRequest(BaseModel):
    """Request for bulk tag operations."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_ids": ["chat123", "chat456", "chat789"],
                "tag_ids": [1, 2],
            }
        }
    )

    chat_ids: list[str] = Field(
        ..., description="List of conversation IDs", min_length=1, max_length=1000
    )
    tag_ids: list[int] = Field(..., description="List of tag IDs", min_length=1, max_length=50)


class BulkTagResponse(BaseModel):
    """Response for bulk tag operations."""

    affected_count: int = Field(..., description="Number of tag assignments affected", ge=0)
    chat_ids: list[str] = Field(..., description="Conversation IDs that were modified")


# -----------------------------------------------------------------------------
# Smart Folder Schemas
# -----------------------------------------------------------------------------


class RuleConditionSchema(BaseModel):
    """A single condition in a smart folder rule."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "field": "unread_count",
                "operator": "greater_than",
                "value": 0,
            }
        }
    )

    field: str = Field(
        ...,
        description="Field to evaluate",
        examples=["unread_count", "last_message_date", "display_name", "tags"],
    )
    operator: str = Field(
        ...,
        description="Comparison operator",
        examples=["equals", "contains", "greater_than", "in_last_days", "has_tag"],
    )
    value: str | int | float | bool | list | None = Field(
        default=None,
        description="Value to compare against",
    )


class SmartFolderRulesSchema(BaseModel):
    """Rules configuration for a smart folder."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "match": "all",
                "conditions": [
                    {"field": "unread_count", "operator": "greater_than", "value": 0},
                    {"field": "is_group", "operator": "equals", "value": False},
                ],
                "sort_by": "last_message_date",
                "sort_order": "desc",
                "limit": 50,
            }
        }
    )

    match: str = Field(
        default="all",
        description="Match type: 'all' requires all conditions, 'any' requires at least one",
        pattern="^(all|any)$",
    )
    conditions: list[RuleConditionSchema] = Field(
        default_factory=list, description="List of rule conditions"
    )
    sort_by: str = Field(default="last_message_date", description="Field to sort results by")
    sort_order: str = Field(default="desc", description="Sort order: 'asc' or 'desc'")
    limit: int = Field(default=0, description="Maximum results (0 for unlimited)", ge=0)


class SmartFolderBase(BaseModel):
    """Base fields for smart folder models."""

    name: str = Field(
        ...,
        description="Display name for the folder",
        examples=["Unread", "Work Urgent", "Recent"],
        min_length=1,
        max_length=100,
    )
    icon: str = Field(
        default="folder",
        description="Icon name for the folder",
        examples=["folder", "inbox", "mail", "flag"],
    )
    color: str = Field(
        default="#64748b",
        description="Hex color code for the folder",
        pattern=r"^#[0-9a-fA-F]{6}$",
    )
    rules: SmartFolderRulesSchema = Field(
        default_factory=SmartFolderRulesSchema, description="Folder rules configuration"
    )


class SmartFolderCreate(SmartFolderBase):
    """Request to create a new smart folder."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Needs Response",
                "icon": "mail",
                "color": "#ef4444",
                "rules": {
                    "match": "all",
                    "conditions": [
                        {"field": "needs_response", "operator": "equals", "value": True}
                    ],
                },
            }
        }
    )


class SmartFolderUpdate(BaseModel):
    """Request to update an existing smart folder."""

    name: str | None = Field(default=None, min_length=1, max_length=100)
    icon: str | None = None
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    rules: SmartFolderRulesSchema | None = None
    sort_order: int | None = None


class SmartFolderResponse(SmartFolderBase):
    """Smart folder response model."""

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "name": "Unread",
                "icon": "mail",
                "color": "#ef4444",
                "rules": {"match": "all", "conditions": [], "sort_by": "last_message_date"},
                "sort_order": 0,
                "is_default": True,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            }
        },
    )

    id: int = Field(..., description="Unique folder identifier")
    sort_order: int = Field(default=0, description="Order in folder list")
    is_default: bool = Field(default=False, description="True if this is a default folder")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")
    updated_at: datetime | None = Field(default=None, description="Last update timestamp")


class SmartFolderListResponse(BaseModel):
    """Response containing a list of smart folders."""

    folders: list[SmartFolderResponse] = Field(..., description="List of smart folders")
    total: int = Field(..., description="Total number of folders", ge=0)


class SmartFolderPreviewRequest(BaseModel):
    """Request to preview a smart folder's rules."""

    rules: SmartFolderRulesSchema = Field(..., description="Rules to preview")
    limit: int = Field(default=10, description="Maximum preview results", ge=1, le=50)


class SmartFolderPreviewResponse(BaseModel):
    """Preview response for smart folder rules."""

    total_matches: int = Field(..., description="Total conversations that match", ge=0)
    preview: list[dict] = Field(..., description="Sample of matching conversations")
    has_more: bool = Field(..., description="True if there are more matches beyond preview")


# -----------------------------------------------------------------------------
# Tag Rule Schemas
# -----------------------------------------------------------------------------


class TagRuleBase(BaseModel):
    """Base fields for auto-tagging rule models."""

    name: str = Field(
        ...,
        description="Display name for the rule",
        examples=["Auto-tag work emails", "Mark urgent"],
        min_length=1,
        max_length=100,
    )
    trigger: str = Field(
        default="on_new_message",
        description="When the rule should be evaluated",
        examples=["on_new_message", "on_keyword_match", "on_contact_match"],
    )
    conditions: list[RuleConditionSchema] = Field(
        default_factory=list, description="Conditions that must be met"
    )
    tag_ids: list[int] = Field(..., description="Tag IDs to apply when rule matches")
    priority: int = Field(default=0, description="Higher priority rules are evaluated first", ge=0)
    is_enabled: bool = Field(default=True, description="Whether the rule is active")


class TagRuleCreate(TagRuleBase):
    """Request to create a new tag rule."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Tag work conversations",
                "trigger": "on_keyword_match",
                "conditions": [
                    {"field": "last_message_text", "operator": "contains", "value": "meeting"}
                ],
                "tag_ids": [1],
                "priority": 10,
            }
        }
    )


class TagRuleUpdate(BaseModel):
    """Request to update an existing tag rule."""

    name: str | None = Field(default=None, min_length=1, max_length=100)
    trigger: str | None = None
    conditions: list[RuleConditionSchema] | None = None
    tag_ids: list[int] | None = None
    priority: int | None = None
    is_enabled: bool | None = None


class TagRuleResponse(TagRuleBase):
    """Tag rule response model."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Unique rule identifier")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")
    last_triggered_at: datetime | None = Field(default=None, description="Last trigger timestamp")
    trigger_count: int = Field(default=0, description="Number of times rule has triggered")


class TagRuleListResponse(BaseModel):
    """Response containing a list of tag rules."""

    rules: list[TagRuleResponse] = Field(..., description="List of tag rules")
    total: int = Field(..., description="Total number of rules", ge=0)


# -----------------------------------------------------------------------------
# Tag Suggestion Schemas
# -----------------------------------------------------------------------------


class TagSuggestionResponse(BaseModel):
    """A suggested tag for a conversation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tag_id": 1,
                "tag_name": "Work",
                "confidence": 0.85,
                "reason": "Content matches 'Work' patterns",
                "source": "content",
            }
        }
    )

    tag_id: int | None = Field(
        default=None, description="Existing tag ID (None for new suggestions)"
    )
    tag_name: str = Field(..., description="Suggested tag name")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)
    reason: str | None = Field(default=None, description="Explanation for the suggestion")
    source: str = Field(
        default="content",
        description="What generated the suggestion",
        examples=["content", "sentiment", "time", "contact", "history"],
    )


class TagSuggestionsRequest(BaseModel):
    """Request for tag suggestions."""

    chat_id: str = Field(..., description="Conversation to get suggestions for")
    limit: int = Field(default=5, description="Maximum number of suggestions", ge=1, le=20)


class TagSuggestionsResponse(BaseModel):
    """Response with tag suggestions."""

    chat_id: str = Field(..., description="Conversation identifier")
    suggestions: list[TagSuggestionResponse] = Field(..., description="List of suggestions")


class SuggestionFeedbackRequest(BaseModel):
    """Request to record feedback on a suggestion."""

    chat_id: str = Field(..., description="Conversation identifier")
    tag_id: int = Field(..., description="Tag ID that was suggested")
    accepted: bool = Field(..., description="Whether the suggestion was accepted")


# -----------------------------------------------------------------------------
# Tag Statistics Schemas
# -----------------------------------------------------------------------------


class TagUsageStats(BaseModel):
    """Usage statistics for a tag."""

    id: int = Field(..., description="Tag identifier")
    name: str = Field(..., description="Tag name")
    count: int = Field(..., description="Number of conversations with this tag", ge=0)


class TagStatisticsResponse(BaseModel):
    """Overall tag statistics."""

    total_tags: int = Field(..., description="Total number of tags", ge=0)
    total_tagged_conversations: int = Field(
        ..., description="Number of conversations with at least one tag", ge=0
    )
    average_tags_per_conversation: float = Field(
        ..., description="Average tags per tagged conversation", ge=0
    )
    most_used_tags: list[TagUsageStats] = Field(..., description="Most frequently used tags")
