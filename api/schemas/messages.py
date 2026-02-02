"""Core iMessage data models.

Contains schemas for messages, conversations, attachments, and reactions.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AttachmentResponse(BaseModel):
    """iMessage attachment metadata.

    Represents a file attachment (image, video, document, etc.) sent or
    received in an iMessage conversation.

    Example:
        ```json
        {
            "filename": "IMG_1234.jpg",
            "file_path": "~/Library/Messages/Attachments/.../IMG_1234.jpg",
            "mime_type": "image/jpeg",
            "file_size": 245760
        }
        ```
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "filename": "IMG_1234.jpg",
                "file_path": "~/Library/Messages/Attachments/.../IMG_1234.jpg",
                "mime_type": "image/jpeg",
                "file_size": 245760,
            }
        },
    )

    filename: str = Field(
        ...,
        description="Original filename of the attachment",
        examples=["IMG_1234.jpg", "Document.pdf"],
    )
    file_path: str | None = Field(
        default=None,
        description="Absolute path to the attachment file on disk",
        examples=["~/Library/Messages/Attachments/.../IMG_1234.jpg"],
    )
    mime_type: str | None = Field(
        default=None,
        description="MIME type of the attachment",
        examples=["image/jpeg", "application/pdf", "video/mp4"],
    )
    file_size: int | None = Field(
        default=None,
        description="File size in bytes",
        examples=[245760, 1048576],
        ge=0,
    )


class ReactionResponse(BaseModel):
    """iMessage tapback reaction.

    Represents a tapback reaction (love, like, dislike, laugh, emphasis,
    question) added to a message by a participant.

    Example:
        ```json
        {
            "type": "love",
            "sender": "+15551234567",
            "sender_name": "John Doe",
            "date": "2024-01-15T10:30:00Z"
        }
        ```
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "type": "love",
                "sender": "+15551234567",
                "sender_name": "John Doe",
                "date": "2024-01-15T10:30:00Z",
            }
        },
    )

    type: str = Field(
        ...,
        description="Tapback reaction type",
        examples=["love", "like", "dislike", "laugh", "emphasis", "question"],
    )
    sender: str = Field(
        ...,
        description="Phone number or email of the person who added the reaction",
        examples=["+15551234567", "john@example.com"],
    )
    sender_name: str | None = Field(
        default=None,
        description="Display name of the sender from Contacts",
        examples=["John Doe", "Mom"],
    )
    date: datetime = Field(
        ...,
        description="Timestamp when the reaction was added",
    )


class MessageResponse(BaseModel):
    """iMessage response model.

    Represents a single message in a conversation, including metadata,
    attachments, and reactions.

    Example:
        ```json
        {
            "id": 12345,
            "chat_id": "chat123456789",
            "sender": "+15551234567",
            "sender_name": "John Doe",
            "text": "Hey, are you free for lunch tomorrow?",
            "date": "2024-01-15T10:30:00Z",
            "is_from_me": false,
            "attachments": [],
            "reply_to_id": null,
            "reactions": [{"type": "love", "sender": "+15559876543"}],
            "is_system_message": false
        }
        ```
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 12345,
                "chat_id": "chat123456789",
                "sender": "+15551234567",
                "sender_name": "John Doe",
                "text": "Hey, are you free for lunch tomorrow?",
                "date": "2024-01-15T10:30:00Z",
                "is_from_me": False,
                "attachments": [],
                "reply_to_id": None,
                "reactions": [],
                "date_delivered": None,
                "date_read": None,
                "is_system_message": False,
            }
        },
    )

    id: int = Field(
        ...,
        description="Unique message identifier from the iMessage database",
        examples=[12345, 67890],
    )
    chat_id: str = Field(
        ...,
        description="Conversation identifier this message belongs to",
        examples=["chat123456789", "iMessage;-;+15551234567"],
    )
    sender: str = Field(
        ...,
        description="Phone number or email of the sender",
        examples=["+15551234567", "john@example.com"],
    )
    sender_name: str | None = Field(
        default=None,
        description="Display name of the sender from Contacts",
        examples=["John Doe", "Mom"],
    )
    text: str = Field(
        ...,
        description="Message text content",
        examples=["Hey, are you free for lunch tomorrow?", "Sounds great!"],
    )
    date: datetime = Field(
        ...,
        description="Timestamp when the message was sent/received",
    )
    is_from_me: bool = Field(
        ...,
        description="True if the message was sent by the current user",
    )
    attachments: list[AttachmentResponse] = Field(
        default_factory=list,
        description="List of file attachments in this message",
    )
    reply_to_id: int | None = Field(
        default=None,
        description="Message ID this is a reply to (threaded conversation)",
        examples=[12340, None],
    )
    reactions: list[ReactionResponse] = Field(
        default_factory=list,
        description="List of tapback reactions on this message",
    )
    date_delivered: datetime | None = Field(
        default=None,
        description="Timestamp when the message was delivered",
    )
    date_read: datetime | None = Field(
        default=None,
        description="Timestamp when the message was read",
    )
    is_system_message: bool = Field(
        default=False,
        description="True for system messages (e.g., 'John left the group')",
    )


class MessagesListResponse(BaseModel):
    """Response containing a list of messages with metadata."""

    messages: list[MessageResponse] = Field(
        ...,
        description="List of messages",
    )
    chat_id: str = Field(
        ...,
        description="Conversation ID these messages belong to",
    )
    total: int = Field(
        ...,
        description="Total number of messages returned",
        ge=0,
    )


class ConversationResponse(BaseModel):
    """Conversation summary response model.

    Provides a summary of a conversation including participants, message count,
    and preview of the last message.

    Example:
        ```json
        {
            "chat_id": "chat123456789",
            "participants": ["+15551234567", "+15559876543"],
            "display_name": "Family Group",
            "last_message_date": "2024-01-15T10:30:00Z",
            "message_count": 150,
            "is_group": true,
            "last_message_text": "See you at dinner!"
        }
        ```
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "chat_id": "chat123456789",
                "participants": ["+15551234567", "+15559876543"],
                "display_name": "Family Group",
                "last_message_date": "2024-01-15T10:30:00Z",
                "message_count": 150,
                "is_group": True,
                "last_message_text": "See you at dinner!",
            }
        },
    )

    chat_id: str = Field(
        ...,
        description="Unique conversation identifier",
        examples=["chat123456789", "iMessage;-;+15551234567"],
    )
    participants: list[str] = Field(
        ...,
        description="List of participant phone numbers or emails",
        examples=[["+15551234567", "+15559876543"]],
    )
    display_name: str | None = Field(
        default=None,
        description="Group name or contact display name",
        examples=["Family Group", "John Doe"],
    )
    last_message_date: datetime = Field(
        ...,
        description="Timestamp of the most recent message",
    )
    message_count: int = Field(
        ...,
        description="Total number of messages in the conversation",
        examples=[150, 42],
        ge=0,
    )
    is_group: bool = Field(
        ...,
        description="True if this is a group conversation",
    )
    last_message_text: str | None = Field(
        default=None,
        description="Preview of the most recent message text",
        examples=["See you at dinner!", "Thanks!"],
    )


class ConversationsListResponse(BaseModel):
    """Response containing a list of conversations with pagination metadata."""

    conversations: list[ConversationResponse] = Field(
        ...,
        description="List of conversations",
    )
    total: int = Field(
        ...,
        description="Total number of conversations returned",
        ge=0,
    )
