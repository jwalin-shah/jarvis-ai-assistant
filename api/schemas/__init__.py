"""Pydantic schemas for API responses.

Converts dataclasses from contracts/ to Pydantic models for FastAPI serialization.
All schemas include OpenAPI metadata for automatic documentation generation.

This package re-exports all schemas for backward compatibility:
    from api.schemas import MessageResponse  # Still works
"""

from __future__ import annotations

# Analytics and profile models
from api.schemas.analytics import (
    ConversationInsightsResponse,
    FrequencyTrendsResponse,
    RefreshProfileRequest,
    RefreshProfileResponse,
    RelationshipHealthResponse,
    RelationshipProfileResponse,
    ResponsePatternsResponse,
    SentimentResponse,
    SentimentTrendResponse,
    StyleGuideResponse,
    ToneProfileResponse,
    TopicDistributionResponse,
)

# Attachment models
from api.schemas.attachments import (
    AttachmentStatsResponse,
    AttachmentTypeEnum,
    AttachmentWithContextResponse,
    ExtendedAttachmentResponse,
    StorageByConversationResponse,
    StorageSummaryResponse,
)

# Calendar models
from api.schemas.calendar import (
    CalendarEventResponse,
    CalendarResponse,
    CreateEventFromDetectedRequest,
    CreateEventRequest,
    CreateEventResponse,
    DetectedEventResponse,
    DetectEventsFromMessagesRequest,
    DetectEventsRequest,
)

# Draft and messaging models
from api.schemas.drafts import (
    ContextInfo,
    DateRange,
    DraftReplyRequest,
    DraftReplyResponse,
    DraftSuggestion,
    DraftSummaryRequest,
    DraftSummaryResponse,
    SendAttachmentRequest,
    SendMessageRequest,
    SendMessageResponse,
)

# Export and digest models
from api.schemas.export import (
    ActionItemResponse,
    DigestExportRequest,
    DigestExportResponse,
    DigestFormatEnum,
    DigestGenerateRequest,
    DigestPeriodEnum,
    DigestPreferencesResponse,
    DigestPreferencesUpdateRequest,
    DigestResponse,
    ExportBackupRequest,
    ExportConversationRequest,
    ExportDateRange,
    ExportFormatEnum,
    ExportResponse,
    ExportSearchRequest,
    GroupHighlightResponse,
    MessageStatsResponse,
    UnansweredConversationResponse,
)

# Core iMessage models
from api.schemas.messages import (
    AttachmentResponse,
    ConversationResponse,
    ConversationsListResponse,
    MessageResponse,
    MessagesListResponse,
    ReactionResponse,
)

# Settings models
from api.schemas.settings import (
    ActivateResponse,
    AvailableModelInfo,
    BehaviorSettings,
    DownloadStatus,
    GenerationSettings,
    SettingsResponse,
    SettingsUpdateRequest,
    SystemInfo,
)

# Statistics models
from api.schemas.stats import (
    ConversationStatsResponse,
    HourlyActivity,
    TimeRangeEnum,
    WordFrequency,
)

# Health and system models
from api.schemas.system import (
    ErrorResponse,
    HealthResponse,
    ModelInfo,
)

# Template models
from api.schemas.templates import (
    CustomTemplateCreateRequest,
    CustomTemplateExportRequest,
    CustomTemplateExportResponse,
    CustomTemplateImportRequest,
    CustomTemplateImportResponse,
    CustomTemplateListResponse,
    CustomTemplateResponse,
    CustomTemplateTestRequest,
    CustomTemplateTestResponse,
    CustomTemplateTestResult,
    CustomTemplateUpdateRequest,
    CustomTemplateUsageStats,
)

# Thread models
from api.schemas.threads import (
    ThreadedMessageResponse,
    ThreadedViewResponse,
    ThreadingConfigRequest,
    ThreadResponse,
)

__all__ = [
    # Messages
    "AttachmentResponse",
    "ConversationResponse",
    "ConversationsListResponse",
    "MessageResponse",
    "MessagesListResponse",
    "ReactionResponse",
    # System
    "ErrorResponse",
    "HealthResponse",
    "ModelInfo",
    # Drafts
    "ContextInfo",
    "DateRange",
    "DraftReplyRequest",
    "DraftReplyResponse",
    "DraftSuggestion",
    "DraftSummaryRequest",
    "DraftSummaryResponse",
    "SendAttachmentRequest",
    "SendMessageRequest",
    "SendMessageResponse",
    # Settings
    "ActivateResponse",
    "AvailableModelInfo",
    "BehaviorSettings",
    "DownloadStatus",
    "GenerationSettings",
    "SettingsResponse",
    "SettingsUpdateRequest",
    "SystemInfo",
    # Stats
    "ConversationStatsResponse",
    "HourlyActivity",
    "TimeRangeEnum",
    "WordFrequency",
    # Attachments
    "AttachmentStatsResponse",
    "AttachmentTypeEnum",
    "AttachmentWithContextResponse",
    "ExtendedAttachmentResponse",
    "StorageByConversationResponse",
    "StorageSummaryResponse",
    # Threads
    "ThreadedMessageResponse",
    "ThreadedViewResponse",
    "ThreadingConfigRequest",
    "ThreadResponse",
    # Templates
    "CustomTemplateCreateRequest",
    "CustomTemplateExportRequest",
    "CustomTemplateExportResponse",
    "CustomTemplateImportRequest",
    "CustomTemplateImportResponse",
    "CustomTemplateListResponse",
    "CustomTemplateResponse",
    "CustomTemplateTestRequest",
    "CustomTemplateTestResponse",
    "CustomTemplateTestResult",
    "CustomTemplateUpdateRequest",
    "CustomTemplateUsageStats",
    # Analytics
    "ConversationInsightsResponse",
    "FrequencyTrendsResponse",
    "RefreshProfileRequest",
    "RefreshProfileResponse",
    "RelationshipHealthResponse",
    "RelationshipProfileResponse",
    "ResponsePatternsResponse",
    "SentimentResponse",
    "SentimentTrendResponse",
    "StyleGuideResponse",
    "ToneProfileResponse",
    "TopicDistributionResponse",
    # Calendar
    "CalendarEventResponse",
    "CalendarResponse",
    "CreateEventFromDetectedRequest",
    "CreateEventRequest",
    "CreateEventResponse",
    "DetectedEventResponse",
    "DetectEventsFromMessagesRequest",
    "DetectEventsRequest",
    # Export
    "ActionItemResponse",
    "DigestExportRequest",
    "DigestExportResponse",
    "DigestFormatEnum",
    "DigestGenerateRequest",
    "DigestPeriodEnum",
    "DigestPreferencesResponse",
    "DigestPreferencesUpdateRequest",
    "DigestResponse",
    "ExportBackupRequest",
    "ExportConversationRequest",
    "ExportDateRange",
    "ExportFormatEnum",
    "ExportResponse",
    "ExportSearchRequest",
    "GroupHighlightResponse",
    "MessageStatsResponse",
    "UnansweredConversationResponse",
]
