"""Pydantic schemas for JARVIS v2 API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# Health
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "2.0.0"
    model_loaded: bool = False
    imessage_accessible: bool = False


class EmbeddingCacheStats(BaseModel):
    total_entries: int = 0
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    error: str | None = None


# Conversations
class ConversationResponse(BaseModel):
    chat_id: str
    display_name: str | None
    participants: list[str]
    last_message_date: datetime | None
    last_message_text: str | None
    last_message_is_from_me: bool
    message_count: int
    is_group: bool


class ConversationListResponse(BaseModel):
    conversations: list[ConversationResponse]
    total: int


# Messages
class MessageResponse(BaseModel):
    id: int
    text: str
    sender: str
    sender_name: str | None  # Resolved contact name
    is_from_me: bool
    timestamp: datetime | None
    chat_id: str


class MessageListResponse(BaseModel):
    messages: list[MessageResponse]
    chat_id: str
    total: int


# Reply Generation
class GenerateRepliesRequest(BaseModel):
    chat_id: str
    num_replies: int = Field(default=3, ge=1, le=5)


class GeneratedReplyResponse(BaseModel):
    text: str
    reply_type: str
    confidence: float


class GenerateRepliesResponse(BaseModel):
    replies: list[GeneratedReplyResponse]
    chat_id: str
    model_used: str
    generation_time_ms: float
    context_summary: str


# Send Message
class SendMessageRequest(BaseModel):
    chat_id: str
    text: str


class SendMessageResponse(BaseModel):
    success: bool
    error: str | None = None


# Settings
class SettingsResponse(BaseModel):
    model_id: str
    auto_suggest: bool = True
    max_replies: int = 3


class SettingsUpdateRequest(BaseModel):
    model_id: str | None = None
    auto_suggest: bool | None = None
    max_replies: int | None = None
