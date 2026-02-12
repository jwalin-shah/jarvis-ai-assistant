"""Export and digest models.

Contains schemas for data export and daily/weekly digest functionality.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ExportFormatEnum(str, Enum):
    """Supported export formats."""

    JSON = "json"
    CSV = "csv"
    TXT = "txt"


class ExportDateRange(BaseModel):
    """Date range filter for exports."""

    start: datetime | None = Field(default=None, description="Start date (inclusive)")
    end: datetime | None = Field(default=None, description="End date (inclusive)")


class ExportConversationRequest(BaseModel):
    """Request to export a single conversation."""

    format: ExportFormatEnum = Field(
        default=ExportFormatEnum.JSON,
        description="Export format (json, csv, txt)",
    )
    date_range: ExportDateRange | None = Field(
        default=None,
        description="Optional date range filter",
    )
    include_attachments: bool = Field(
        default=False,
        description="Include attachment info in export (CSV only)",
    )
    limit: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum messages to export",
    )


class ExportSearchRequest(BaseModel):
    """Request to export search results."""

    query: str = Field(..., min_length=1, description="Search query")
    format: ExportFormatEnum = Field(
        default=ExportFormatEnum.JSON,
        description="Export format (json, csv, txt)",
    )
    limit: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Maximum results to export",
    )
    sender: str | None = Field(default=None, description="Filter by sender")
    date_range: ExportDateRange | None = Field(
        default=None,
        description="Optional date range filter",
    )


class ExportBackupRequest(BaseModel):
    """Request for full conversation backup."""

    conversation_limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum conversations to include per page",
    )
    messages_per_conversation: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Maximum messages per conversation",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of conversations to skip (for paginated backup)",
    )
    date_range: ExportDateRange | None = Field(
        default=None,
        description="Optional date range filter for messages",
    )


class ExportResponse(BaseModel):
    """Response containing exported data."""

    success: bool
    format: str
    filename: str
    data: str = Field(..., description="Exported data as string")
    message_count: int
    export_type: str = Field(
        default="conversation",
        description="Type of export: conversation, search, or backup",
    )


class DigestPeriodEnum(str, Enum):
    """Time period for digest generation."""

    DAILY = "daily"
    WEEKLY = "weekly"


class DigestFormatEnum(str, Enum):
    """Supported digest export formats."""

    MARKDOWN = "markdown"
    HTML = "html"


class UnansweredConversationResponse(BaseModel):
    """A conversation with unanswered messages.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "display_name": "John Doe",
            "participants": ["+15551234567"],
            "unanswered_count": 3,
            "last_message_date": "2024-01-15T10:30:00Z",
            "last_message_preview": "Hey, are you free tomorrow?",
            "is_group": false
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "display_name": "John Doe",
                "participants": ["+15551234567"],
                "unanswered_count": 3,
                "last_message_date": "2024-01-15T10:30:00Z",
                "last_message_preview": "Hey, are you free tomorrow?",
                "is_group": False,
            }
        }
    )

    chat_id: str = Field(..., description="Conversation identifier")
    display_name: str = Field(..., description="Display name for the conversation")
    participants: list[str] = Field(..., description="List of participants")
    unanswered_count: int = Field(..., ge=0, description="Number of unanswered messages")
    last_message_date: datetime | None = Field(
        default=None, description="Date of last unanswered message"
    )
    last_message_preview: str | None = Field(
        default=None, description="Preview of last unanswered message"
    )
    is_group: bool = Field(..., description="Whether this is a group conversation")


class GroupHighlightResponse(BaseModel):
    """Highlight from an active group chat.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "display_name": "Family Group",
            "participants": ["+15551234567", "+15559876543"],
            "message_count": 45,
            "active_participants": ["John", "Jane"],
            "top_topics": ["meeting", "plans"],
            "last_activity": "2024-01-15T18:30:00Z"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "display_name": "Family Group",
                "participants": ["+15551234567", "+15559876543"],
                "message_count": 45,
                "active_participants": ["John", "Jane"],
                "top_topics": ["meeting", "plans"],
                "last_activity": "2024-01-15T18:30:00Z",
            }
        }
    )

    chat_id: str = Field(..., description="Conversation identifier")
    display_name: str = Field(..., description="Group display name")
    participants: list[str] = Field(..., description="List of participants")
    message_count: int = Field(..., ge=0, description="Message count in period")
    active_participants: list[str] = Field(..., description="Most active participants")
    top_topics: list[str] = Field(..., description="Detected topics")
    last_activity: datetime | None = Field(default=None, description="Last activity timestamp")


class ActionItemResponse(BaseModel):
    """Detected action item from messages.

    Example:
        ```json
        {
            "text": "Can you send me the report?",
            "chat_id": "chat123456789",
            "conversation_name": "John Doe",
            "sender": "John Doe",
            "date": "2024-01-15T10:30:00Z",
            "message_id": 12345,
            "item_type": "task"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Can you send me the report?",
                "chat_id": "chat123456789",
                "conversation_name": "John Doe",
                "sender": "John Doe",
                "date": "2024-01-15T10:30:00Z",
                "message_id": 12345,
                "item_type": "task",
            }
        }
    )

    text: str = Field(..., description="Action item text")
    chat_id: str = Field(..., description="Conversation identifier")
    conversation_name: str = Field(..., description="Conversation display name")
    sender: str = Field(..., description="Sender of the message")
    date: datetime = Field(..., description="Message date")
    message_id: int = Field(..., description="Message identifier")
    item_type: str = Field(
        ...,
        description="Type of action item (task, question, event, reminder)",
        examples=["task", "question", "event", "reminder"],
    )


class MessageStatsResponse(BaseModel):
    """Message volume statistics.

    Example:
        ```json
        {
            "total_sent": 150,
            "total_received": 200,
            "total_messages": 350,
            "active_conversations": 12,
            "most_active_conversation": "Family Group",
            "most_active_count": 45,
            "avg_messages_per_day": 50.0,
            "busiest_hour": 14,
            "hourly_distribution": {"9": 20, "14": 35}
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_sent": 150,
                "total_received": 200,
                "total_messages": 350,
                "active_conversations": 12,
                "most_active_conversation": "Family Group",
                "most_active_count": 45,
                "avg_messages_per_day": 50.0,
                "busiest_hour": 14,
                "hourly_distribution": {"9": 20, "14": 35},
            }
        }
    )

    total_sent: int = Field(..., ge=0, description="Total messages sent")
    total_received: int = Field(..., ge=0, description="Total messages received")
    total_messages: int = Field(..., ge=0, description="Total messages")
    active_conversations: int = Field(..., ge=0, description="Active conversations count")
    most_active_conversation: str | None = Field(
        default=None, description="Name of most active conversation"
    )
    most_active_count: int = Field(default=0, ge=0, description="Message count in most active")
    avg_messages_per_day: float = Field(..., ge=0, description="Average messages per day")
    busiest_hour: int | None = Field(
        default=None, ge=0, le=23, description="Hour with most messages (0-23)"
    )
    hourly_distribution: dict[str, int] = Field(
        default_factory=dict, description="Message count by hour"
    )


class DigestResponse(BaseModel):
    """Complete digest response.

    Example:
        ```json
        {
            "period": "daily",
            "generated_at": "2024-01-15T08:00:00Z",
            "start_date": "2024-01-14T08:00:00Z",
            "end_date": "2024-01-15T08:00:00Z",
            "needs_attention": [],
            "highlights": [],
            "action_items": [],
            "stats": {...}
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "period": "daily",
                "generated_at": "2024-01-15T08:00:00Z",
                "start_date": "2024-01-14T08:00:00Z",
                "end_date": "2024-01-15T08:00:00Z",
                "needs_attention": [],
                "highlights": [],
                "action_items": [],
                "stats": {
                    "total_sent": 50,
                    "total_received": 75,
                    "total_messages": 125,
                    "active_conversations": 5,
                    "most_active_conversation": "Family Group",
                    "most_active_count": 30,
                    "avg_messages_per_day": 125.0,
                    "busiest_hour": 14,
                    "hourly_distribution": {},
                },
            }
        }
    )

    period: DigestPeriodEnum = Field(..., description="Digest period")
    generated_at: datetime = Field(..., description="When the digest was generated")
    start_date: datetime = Field(..., description="Start of digest period")
    end_date: datetime = Field(..., description="End of digest period")
    needs_attention: list[UnansweredConversationResponse] = Field(
        default_factory=list, description="Conversations needing attention"
    )
    highlights: list[GroupHighlightResponse] = Field(
        default_factory=list, description="Group chat highlights"
    )
    action_items: list[ActionItemResponse] = Field(
        default_factory=list, description="Detected action items"
    )
    stats: MessageStatsResponse = Field(..., description="Message statistics")


class DigestGenerateRequest(BaseModel):
    """Request to generate a digest.

    Example:
        ```json
        {
            "period": "daily",
            "end_date": "2024-01-15T08:00:00Z"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "period": "daily",
                "end_date": "2024-01-15T08:00:00Z",
            }
        }
    )

    period: DigestPeriodEnum = Field(
        default=DigestPeriodEnum.DAILY, description="Digest period (daily or weekly)"
    )
    end_date: datetime | None = Field(
        default=None, description="End date for digest (defaults to now)"
    )


class DigestExportRequest(BaseModel):
    """Request to export a digest.

    Example:
        ```json
        {
            "period": "daily",
            "format": "markdown",
            "end_date": "2024-01-15T08:00:00Z"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "period": "daily",
                "format": "markdown",
                "end_date": "2024-01-15T08:00:00Z",
            }
        }
    )

    period: DigestPeriodEnum = Field(default=DigestPeriodEnum.DAILY, description="Digest period")
    format: DigestFormatEnum = Field(default=DigestFormatEnum.MARKDOWN, description="Export format")
    end_date: datetime | None = Field(
        default=None, description="End date for digest (defaults to now)"
    )


class DigestExportResponse(BaseModel):
    """Response containing exported digest.

    Example:
        ```json
        {
            "success": true,
            "format": "markdown",
            "filename": "jarvis_digest_daily_20240115.md",
            "data": "# JARVIS Daily Digest..."
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "format": "markdown",
                "filename": "jarvis_digest_daily_20240115.md",
                "data": "# JARVIS Daily Digest...",
            }
        }
    )

    success: bool = Field(..., description="Whether export succeeded")
    format: str = Field(..., description="Export format used")
    filename: str = Field(..., description="Suggested filename")
    data: str = Field(..., description="Exported content")


class DigestPreferencesResponse(BaseModel):
    """Digest preferences response.

    Example:
        ```json
        {
            "enabled": true,
            "schedule": "daily",
            "preferred_time": "08:00",
            "include_action_items": true,
            "include_stats": true,
            "max_conversations": 50,
            "export_format": "markdown"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": True,
                "schedule": "daily",
                "preferred_time": "08:00",
                "include_action_items": True,
                "include_stats": True,
                "max_conversations": 50,
                "export_format": "markdown",
            }
        }
    )

    enabled: bool = Field(..., description="Whether digest generation is enabled")
    schedule: str = Field(..., description="Digest schedule (daily or weekly)")
    preferred_time: str = Field(..., description="Preferred generation time (HH:MM)")
    include_action_items: bool = Field(..., description="Include action items")
    include_stats: bool = Field(..., description="Include statistics")
    max_conversations: int = Field(..., description="Max conversations to analyze")
    export_format: str = Field(..., description="Default export format")


class DigestPreferencesUpdateRequest(BaseModel):
    """Request to update digest preferences.

    Example:
        ```json
        {
            "enabled": true,
            "schedule": "weekly"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": True,
                "schedule": "weekly",
            }
        }
    )

    enabled: bool | None = Field(default=None, description="Enable/disable digest")
    schedule: str | None = Field(default=None, description="Schedule (daily/weekly)")
    preferred_time: str | None = Field(default=None, description="Preferred time (HH:MM)")
    include_action_items: bool | None = Field(default=None, description="Include action items")
    include_stats: bool | None = Field(default=None, description="Include statistics")
    max_conversations: int | None = Field(
        default=None, ge=10, le=200, description="Max conversations"
    )
    export_format: str | None = Field(default=None, description="Export format")
