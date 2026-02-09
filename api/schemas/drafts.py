"""Draft, reply, and messaging models.

Contains schemas for sending messages, generating draft replies, and summaries.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SendMessageRequest(BaseModel):
    """Request to send an iMessage.

    Send a text message to an individual or group conversation.

    Example:
        ```json
        {
            "text": "Hey, are you free for lunch?",
            "recipient": "+15551234567",
            "is_group": false
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Hey, are you free for lunch?",
                "recipient": "+15551234567",
                "is_group": False,
            }
        }
    )

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Message text content to send",
        examples=["Hey, are you free for lunch?", "Thanks!"],
    )
    recipient: str | None = Field(
        default=None,
        description="Recipient phone number or email (required for individual chats)",
        examples=["+15551234567", "john@example.com"],
    )
    is_group: bool = Field(
        default=False,
        description="Set to true for group chats (uses chat_id from path)",
    )


class SendAttachmentRequest(BaseModel):
    """Request to send a file attachment.

    Send a file (image, document, etc.) to an individual or group conversation.

    Example:
        ```json
        {
            "file_path": "/Users/john/Documents/photo.jpg",
            "recipient": "+15551234567",
            "is_group": false
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_path": "/Users/john/Documents/photo.jpg",
                "recipient": "+15551234567",
                "is_group": False,
            }
        }
    )

    file_path: str = Field(
        ...,
        description="Absolute path to the file to send",
        examples=["/Users/john/Documents/photo.jpg"],
    )
    recipient: str | None = Field(
        default=None,
        description="Recipient phone number or email (required for individual chats)",
        examples=["+15551234567", "john@example.com"],
    )
    is_group: bool = Field(
        default=False,
        description="Set to true for group chats (uses chat_id from path)",
    )

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate file path is within user home directory."""
        from pathlib import Path

        resolved = Path(v).resolve()
        home = Path.home()
        if not str(resolved).startswith(str(home)):
            raise ValueError(f"file_path must be within user home directory ({home})")
        return str(resolved)


class SendMessageResponse(BaseModel):
    """Response after sending a message.

    Indicates whether the send operation succeeded or failed.

    Example:
        ```json
        {
            "success": true,
            "error": null
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "error": None,
            }
        }
    )

    success: bool = Field(
        ...,
        description="True if the message was sent successfully",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the send failed",
        examples=[None, "Automation permission denied"],
    )


class DraftSuggestion(BaseModel):
    """A suggested reply for a conversation.

    AI-generated reply suggestion with confidence score.

    Example:
        ```json
        {
            "text": "Sounds great! What time works for you?",
            "confidence": 0.85
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Sounds great! What time works for you?",
                "confidence": 0.85,
            }
        }
    )

    text: str = Field(
        ...,
        description="Suggested reply text",
        examples=["Sounds great! What time works for you?", "Sure, I'll take a look!"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0) indicating relevance",
        examples=[0.85, 0.7],
    )


class ContextInfo(BaseModel):
    """Information about the context used to generate a reply.

    Metadata about the conversation context used for AI generation.

    Example:
        ```json
        {
            "num_messages": 20,
            "participants": ["John Doe", "Jane Smith"],
            "last_message": "Are you free for dinner?"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "num_messages": 20,
                "participants": ["John Doe", "Jane Smith"],
                "last_message": "Are you free for dinner?",
            }
        }
    )

    num_messages: int = Field(
        ...,
        description="Number of messages used for context",
        examples=[20, 50],
        ge=0,
    )
    participants: list[str] = Field(
        ...,
        description="Names of conversation participants",
        examples=[["John Doe", "Jane Smith"]],
    )
    last_message: str | None = Field(
        default=None,
        description="The most recent message in the conversation",
        examples=["Are you free for dinner?"],
    )


class DraftReplyRequest(BaseModel):
    """Request for draft reply generation.

    Request AI-generated reply suggestions for a conversation.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "instruction": "accept enthusiastically",
            "num_suggestions": 3,
            "context_messages": 20
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "instruction": "accept enthusiastically",
                "num_suggestions": 3,
                "context_messages": 20,
            }
        }
    )

    chat_id: str = Field(
        ...,
        description="Conversation ID to generate replies for",
        examples=["chat123456789"],
    )
    instruction: str | None = Field(
        default=None,
        description="Optional instruction for reply tone/content",
        examples=["accept enthusiastically", "politely decline", "ask for more details"],
    )
    num_suggestions: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of reply suggestions to generate (1-5)",
    )
    context_messages: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Number of previous messages to use for context (5-50)",
    )


class DraftReplyResponse(BaseModel):
    """Response with generated reply suggestions.

    Contains AI-generated reply suggestions with context metadata.

    Example:
        ```json
        {
            "suggestions": [
                {"text": "Yes, I'd love to! What time?", "confidence": 0.9},
                {"text": "Sure! Let me know the details.", "confidence": 0.8}
            ],
            "context_used": {
                "num_messages": 20,
                "participants": ["John"],
                "last_message": "Dinner tonight?"
            }
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "suggestions": [
                    {"text": "Yes, I'd love to! What time?", "confidence": 0.9},
                    {"text": "Sure! Let me know the details.", "confidence": 0.8},
                ],
                "context_used": {
                    "num_messages": 20,
                    "participants": ["John"],
                    "last_message": "Dinner tonight?",
                },
            }
        }
    )

    suggestions: list[DraftSuggestion] = Field(
        ...,
        description="List of generated reply suggestions",
    )
    context_used: ContextInfo = Field(
        ...,
        description="Information about the context used for generation",
    )


class DraftSummaryRequest(BaseModel):
    """Request for conversation summarization.

    Request an AI-generated summary of a conversation.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "num_messages": 50
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "num_messages": 50,
            }
        }
    )

    chat_id: str = Field(
        ...,
        description="Conversation ID to summarize",
        examples=["chat123456789"],
    )
    num_messages: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of messages to include in summary (10-200)",
    )


class DateRange(BaseModel):
    """Date range for a conversation summary.

    Represents the time span covered by a summary.

    Example:
        ```json
        {
            "start": "2024-01-01",
            "end": "2024-01-15"
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start": "2024-01-01",
                "end": "2024-01-15",
            }
        }
    )

    start: str = Field(
        ...,
        description="Start date of the summarized period (YYYY-MM-DD)",
        examples=["2024-01-01"],
    )
    end: str = Field(
        ...,
        description="End date of the summarized period (YYYY-MM-DD)",
        examples=["2024-01-15"],
    )


class DraftSummaryResponse(BaseModel):
    """Response with conversation summary.

    Contains AI-generated summary with key points and date range.

    Example:
        ```json
        {
            "summary": "Discussion about planning a weekend trip to the beach.",
            "key_points": [
                "Decided on Saturday departure",
                "Meeting at John's place at 9am",
                "Bringing snacks and sunscreen"
            ],
            "date_range": {"start": "2024-01-10", "end": "2024-01-15"}
        }
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "summary": "Discussion about planning a weekend trip to the beach.",
                "key_points": [
                    "Decided on Saturday departure",
                    "Meeting at John's place at 9am",
                    "Bringing snacks and sunscreen",
                ],
                "date_range": {"start": "2024-01-10", "end": "2024-01-15"},
            }
        }
    )

    summary: str = Field(
        ...,
        description="Brief summary of the conversation",
        examples=["Discussion about planning a weekend trip to the beach."],
    )
    key_points: list[str] = Field(
        ...,
        description="List of key points extracted from the conversation",
    )
    date_range: DateRange = Field(
        ...,
        description="Date range of messages included in the summary",
    )
