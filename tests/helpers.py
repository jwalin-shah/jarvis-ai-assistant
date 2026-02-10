"""Shared test helpers and mocks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from contracts.imessage import Attachment, Conversation, Message, Reaction

# --- Context Fetching Stubs ---


@dataclass
class ReplyContext:
    """Context for generating a reply."""

    messages: list[Message]
    formatted_context: str
    last_received_message: Message
    conversation_summary: str | None = None


class MockContextFetcher:
    """Mock context fetcher for testing.

    In production, this retrieves and formats conversation context.
    """

    def __init__(self, reader: Any):
        self.reader = reader

    def get_reply_context(self, chat_id: str, num_messages: int = 10) -> ReplyContext:
        """Get context for generating a reply."""
        messages = self.reader.get_messages(chat_id, limit=num_messages)

        # Format messages as context string
        formatted_lines = []
        for msg in messages:
            sender = "You" if msg.is_from_me else (msg.sender_name or msg.sender)
            formatted_lines.append(f"{sender}: {msg.text}")
        formatted_context = "\n".join(formatted_lines)

        # Find last message not from user
        last_received = None
        for msg in reversed(messages):
            if not msg.is_from_me:
                last_received = msg
                break

        if last_received is None and messages:
            last_received = messages[-1]

        return ReplyContext(
            messages=messages,
            formatted_context=formatted_context,
            last_received_message=last_received,
        )

    def get_summary_context(
        self, chat_id: str, num_messages: int = 50
    ) -> tuple[list[Message], str]:
        """Get context for summarizing a conversation."""
        messages = self.reader.get_messages(chat_id, limit=num_messages)
        formatted_lines = []
        for msg in messages:
            sender = "You" if msg.is_from_me else (msg.sender_name or msg.sender)
            formatted_lines.append(f"[{msg.date.strftime('%Y-%m-%d %H:%M')}] {sender}: {msg.text}")
        return messages, "\n".join(formatted_lines)


# --- Prompt Building Stubs ---


def build_reply_prompt(context: str, last_message: str) -> str:
    """Build a prompt for generating a reply.

    Args:
        context: Formatted conversation history
        last_message: The message to reply to

    Returns:
        Formatted prompt for the generator
    """
    return f"""You are helping compose a friendly, natural reply to a message.

Conversation history:
{context}

The last message you need to reply to is:
"{last_message}"

Generate a brief, natural reply (1-2 sentences). Be conversational and friendly.
Reply:"""


def build_summary_prompt(context: str, num_messages: int) -> str:
    """Build a prompt for summarizing a conversation.

    Args:
        context: Formatted conversation history
        num_messages: Number of messages being summarized

    Returns:
        Formatted prompt for the generator
    """
    return f"""Summarize the following conversation ({num_messages} messages) in 2-3 sentences.
Focus on the main topics discussed and any action items.

Conversation:
{context}

Summary:"""


# --- Test Data Helpers ---


def create_mock_message(
    text: str,
    is_from_me: bool,
    sender: str = "+1234567890",
    sender_name: str | None = "John",
    msg_id: int = 1,
    chat_id: str = "chat123",
    date: datetime | None = None,
    attachments: list[Attachment] | None = None,
    reactions: list[Reaction] | None = None,
) -> Message:
    """Create a mock Message for testing."""
    return Message(
        id=msg_id,
        chat_id=chat_id,
        sender="" if is_from_me else sender,
        sender_name=None if is_from_me else sender_name,
        text=text,
        date=date or datetime(2024, 1, 15, 10, 30),
        is_from_me=is_from_me,
        attachments=attachments or [],
        reactions=reactions or [],
    )


def create_mock_conversation(
    chat_id: str = "chat123",
    participants: list[str] | None = None,
    display_name: str | None = "John",
    message_count: int = 50,
    is_group: bool = False,
) -> Conversation:
    """Create a mock Conversation for testing."""
    return Conversation(
        chat_id=chat_id,
        participants=participants or ["+1234567890"],
        display_name=display_name,
        last_message_date=datetime(2024, 1, 15, 10, 30),
        message_count=message_count,
        is_group=is_group,
    )
