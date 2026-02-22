"""iMessage integration interfaces.

Workstream 10 implements against these contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass
class Attachment:
    """iMessage attachment metadata.

    Attributes:
        filename: Name of the attachment file.
        file_path: Full filesystem path to attachment file.
        mime_type: MIME type (e.g., "image/jpeg").
        file_size: Size in bytes.
        width: Image/video width in pixels.
        height: Image/video height in pixels.
        duration_seconds: Audio/video duration in seconds.
        created_date: When attachment was created/received.
        is_sticker: Whether attachment is a sticker.
        uti: Uniform Type Identifier (e.g., "public.jpeg").
    """

    filename: str
    file_path: str | None
    mime_type: str | None
    file_size: int | None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    created_date: datetime | None = None
    is_sticker: bool = False
    uti: str | None = None

    def __post_init__(self) -> None:
        """Validate field constraints."""
        if self.file_size is not None and self.file_size < 0:
            msg = f"file_size must be >= 0, got {self.file_size}"
            raise ValueError(msg)
        if self.width is not None and self.width < 0:
            msg = f"width must be >= 0, got {self.width}"
            raise ValueError(msg)
        if self.height is not None and self.height < 0:
            msg = f"height must be >= 0, got {self.height}"
            raise ValueError(msg)
        if self.duration_seconds is not None and self.duration_seconds < 0:
            msg = f"duration_seconds must be >= 0, got {self.duration_seconds}"
            raise ValueError(msg)


@dataclass
class AttachmentSummary:
    """Summary of attachments for a conversation.

    Attributes:
        total_count: Total number of attachments.
        total_size_bytes: Total size of all attachments in bytes.
        by_type: Count by type (images/videos/audio/documents/other).
        size_by_type: Size in bytes by type.
    """

    total_count: int
    total_size_bytes: int
    by_type: dict[str, int]
    size_by_type: dict[str, int]

    def __post_init__(self) -> None:
        """Validate field constraints."""
        if self.total_count < 0:
            msg = f"total_count must be >= 0, got {self.total_count}"
            raise ValueError(msg)
        if self.total_size_bytes < 0:
            msg = f"total_size_bytes must be >= 0, got {self.total_size_bytes}"
            raise ValueError(msg)


@dataclass
class Reaction:
    """iMessage tapback reaction.

    Tapback types: love, like, dislike, laugh, emphasize, question
    Also supports "removed_" prefix for removed reactions.

    Attributes:
        type: Reaction type (e.g., "love", "like", "removed_love").
        sender: Phone number or email of sender.
        sender_name: Resolved contact name if available.
        date: When the reaction was added.
    """

    type: str
    sender: str
    sender_name: str | None
    date: datetime

    def __post_init__(self) -> None:
        """Validate field constraints."""
        valid_reactions = {"love", "like", "dislike", "laugh", "emphasize", "question"}
        reaction_type = self.type.removeprefix("removed_")
        if reaction_type not in valid_reactions:
            msg = (
                f"type must be one of {valid_reactions} "
                f"(optionally prefixed with 'removed_'), got {self.type}"
            )
            raise ValueError(msg)


@dataclass
class Message:
    """Normalized iMessage representation.

    Attributes:
        id: Unique message identifier.
        chat_id: ID of conversation containing this message.
        sender: Phone number or email of sender.
        sender_name: Resolved contact name if available.
        text: Message text content.
        date: When message was sent/received.
        is_from_me: Whether this message was sent by the user.
        attachments: List of attachments.
        reply_to_id: ID of message this is replying to.
        reactions: List of tapback reactions.
        date_delivered: When message was delivered (only for sent messages).
        date_read: When message was read (only for sent messages).
        is_system_message: Whether this is a system message (e.g., group events).
    """

    id: int
    chat_id: str
    sender: str
    sender_name: str | None
    text: str
    date: datetime
    is_from_me: bool
    attachments: list[Attachment] = field(default_factory=list)
    reply_to_id: int | None = None
    reactions: list[Reaction] = field(default_factory=list)
    date_delivered: datetime | None = None
    date_read: datetime | None = None
    is_system_message: bool = False

    def __post_init__(self) -> None:
        """Validate field constraints."""
        if self.id < 0:
            msg = f"id must be >= 0, got {self.id}"
            raise ValueError(msg)
        if self.reply_to_id is not None and self.reply_to_id < 0:
            msg = f"reply_to_id must be >= 0, got {self.reply_to_id}"
            raise ValueError(msg)


@dataclass
class Conversation:
    """iMessage conversation summary.

    Attributes:
        chat_id: Unique conversation identifier.
        participants: List of participant phone numbers or emails.
        display_name: Display name for the conversation.
        last_message_date: Date of the most recent message.
        message_count: Total number of messages in conversation.
        is_group: Whether this is a group conversation.
        last_message_text: Preview text of the most recent message.
    """

    chat_id: str
    participants: list[str]
    display_name: str | None
    last_message_date: datetime
    message_count: int
    is_group: bool
    last_message_text: str | None = None

    def __post_init__(self) -> None:
        """Validate field constraints."""
        if self.message_count < 0:
            msg = f"message_count must be >= 0, got {self.message_count}"
            raise ValueError(msg)
        if not self.participants:
            msg = "Conversation must have at least one participant"
            raise ValueError(msg)
        if self.is_group and len(self.participants) < 2:
            msg = f"Group conversation must have >= 2 participants, got {len(self.participants)}"
            raise ValueError(msg)


class iMessageReader(Protocol):  # noqa: N801 - iMessage is a proper noun
    """Interface for iMessage integration (Workstream 10)."""

    def check_access(self) -> bool:
        """Check if we have permission to read chat.db."""
        ...

    def get_conversation(self, chat_id: str) -> Conversation | None:
        """Get a single conversation by chat_id."""
        ...

    def get_conversations(
        self, limit: int = 50, since: datetime | None = None
    ) -> list[Conversation]:
        """Get recent conversations."""
        ...

    def get_messages(
        self, chat_id: str, limit: int = 100, before: datetime | None = None
    ) -> list[Message]:
        """Get messages from a conversation."""
        ...

    def get_messages_batch(
        self, chat_ids: list[str], limit_per_chat: int = 100
    ) -> dict[str, list[Message]]:
        """Get messages for multiple conversations in a single SQL query."""
        ...

    def search(
        self,
        query: str,
        limit: int = 50,
        sender: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        chat_id: str | None = None,
        has_attachments: bool | None = None,
    ) -> list[Message]:
        """Full-text search across messages with optional filters.

        Args:
            query: Search query string
            limit: Maximum number of results
            sender: Filter by sender phone number or email
            after: Filter for messages after this datetime
            before: Filter for messages before this datetime
            chat_id: Filter by conversation ID
            has_attachments: Filter for messages with/without attachments
        """
        ...

    def get_conversation_context(
        self, chat_id: str, around_message_id: int, context_messages: int = 5
    ) -> list[Message]:
        """Get messages around a specific message for context."""
        ...
