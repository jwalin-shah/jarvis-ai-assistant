"""Context fetcher service for RAG-based iMessage retrieval.

Retrieves and formats iMessage data for use in LLM prompts.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from contracts.imessage import Conversation, Message, iMessageReader

logger = logging.getLogger(__name__)


@dataclass
class ReplyContext:
    """Context for generating a reply to a conversation."""

    chat_id: str
    participant_names: list[str]
    messages: list[Message]
    formatted_context: str  # Ready for LLM prompt
    last_received_message: Message | None
    current_topic: str | None = None  # Topic label of current conversation thread
    contact_facts: str = ""  # Formatted facts for prompt
    relationship_graph: str = ""  # Formatted graph context
    contact_profile: dict[str, Any] | None = None  # Profile for style matching


@dataclass
class SummaryContext:
    """Context for summarizing a conversation."""

    chat_id: str
    participant_names: list[str]
    messages: list[Message]
    formatted_context: str
    date_range: tuple[datetime, datetime]


@dataclass
class SearchContext:
    """Context for search results."""

    query: str
    results: list[Message]
    formatted_context: str


class ContextFetcher:
    """Fetches and formats iMessage context for RAG.

    Uses dependency injection for the iMessage reader to enable
    testing with mock implementations.

    Example:
        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            fetcher = ContextFetcher(reader)
            context = fetcher.get_reply_context("iMessage;-;+15551234567")
            # Example: print(context.formatted_context)
    """

    def __init__(self, reader: iMessageReader, max_cached_conversations: int = 500) -> None:
        """Initialize the context fetcher.

        Args:
            reader: An iMessageReader implementation (e.g., ChatDBReader)
            max_cached_conversations: Maximum number of conversations to cache
                for name resolution (default 500)
        """
        self._reader = reader
        self._max_cached_conversations = max_cached_conversations
        # Cache conversations for name resolution
        self._conversations_cache: list[Conversation] | None = None

    def clear_cache(self) -> None:
        """Clear the conversation cache.

        Call this method if you need to refresh conversation data,
        for example after new messages arrive or conversations change.
        """
        self._conversations_cache = None

    def _get_conversations(self) -> list[Conversation]:
        """Get cached conversations, loading if necessary."""
        if self._conversations_cache is None:
            self._conversations_cache = self._reader.get_conversations(
                limit=self._max_cached_conversations
            )
        return self._conversations_cache

    def _get_participant_names(self, chat_id: str) -> list[str]:
        """Get participant names for a conversation.

        Args:
            chat_id: The conversation ID

        Returns:
            List of participant display names
        """
        conversations = self._get_conversations()
        for conv in conversations:
            if conv.chat_id == chat_id:
                # Use display_name if available, otherwise use participants
                if conv.display_name:
                    return [conv.display_name]
                return conv.participants
        return []

    def _format_sender_name(self, message: Message) -> str:
        """Format the sender name for display.

        Args:
            message: The message to format sender for

        Returns:
            Display name for the sender
        """
        if message.is_from_me:
            return "Me"
        return message.sender_name or message.sender

    def _format_messages(self, messages: list[Message], participant_names: list[str]) -> str:
        """Format messages as turn-based context for LLM.

        Args:
            messages: List of messages to format
            participant_names: List of participant names for header

        Returns:
            Formatted string ready for LLM prompt
        """
        if not messages:
            return ""

        lines = []

        # Header with participants
        if participant_names:
            names_str = ", ".join(participant_names)
            lines.append(f"Conversation with: {names_str}")
        lines.append("---")

        # Turn-based grouping: merge consecutive messages from same sender
        if messages:
            current_sender = self._format_sender_name(messages[0])
            current_msgs: list[str] = []

            for msg in messages:
                if msg.is_system_message:
                    # Flush current turn before system message
                    if current_msgs:
                        lines.append(f"{current_sender}: {' '.join(current_msgs)}")
                        current_msgs = []

                    timestamp = msg.date.strftime("%H:%M")
                    lines.append(f"[{timestamp}] [System] {msg.text}")
                    continue

                sender = self._format_sender_name(msg)
                text = msg.text or ""

                # Add attachment info
                if msg.attachments:
                    attachment_info = []
                    for att in msg.attachments:
                        if att.mime_type and att.mime_type.startswith("image/"):
                            attachment_info.append("[Image]")
                        elif att.mime_type and att.mime_type.startswith("video/"):
                            attachment_info.append("[Video]")
                        elif att.mime_type and att.mime_type.startswith("audio/"):
                            attachment_info.append("[Audio]")
                        else:
                            attachment_info.append(f"[Attachment: {att.filename}]")

                    text = f"{text} {' '.join(attachment_info)}".strip()

                if not text:
                    continue

                if sender == current_sender:
                    current_msgs.append(text)
                else:
                    # Flush previous turn
                    if current_msgs:
                        lines.append(f"{current_sender}: {' '.join(current_msgs)}")
                    current_sender = sender
                    current_msgs = [text]

            # Final flush
            if current_msgs:
                lines.append(f"{current_sender}: {' '.join(current_msgs)}")

        lines.append("---")

        return "\n".join(lines)

    def _format_search_results(self, query: str, messages: list[Message]) -> str:
        """Format search results as context for LLM.

        Args:
            query: The search query
            messages: List of matching messages

        Returns:
            Formatted string ready for LLM prompt
        """
        if not messages:
            return f'No messages found matching "{query}"'

        lines = [f'Search results for "{query}":']
        lines.append("---")

        for msg in messages:
            timestamp = msg.date.strftime("%Y-%m-%d %H:%M")
            sender = self._format_sender_name(msg)
            text = msg.text

            # Include attachment info if present
            if msg.attachments and not text:
                text = f"[{len(msg.attachments)} attachment(s)]"

            lines.append(f"[{timestamp}] {sender}: {text}")

        lines.append("---")

        return "\n".join(lines)

    def get_reply_context(self, chat_id: str, num_messages: int = 20) -> ReplyContext:
        """Get context for replying to a conversation.

        Retrieves recent messages from a conversation, formatted for LLM use.

        Args:
            chat_id: The conversation ID
            num_messages: Number of recent messages to retrieve (default 20)

        Returns:
            ReplyContext with formatted messages and metadata
        """
        messages = self._reader.get_messages(chat_id, limit=num_messages)

        # Messages come newest-first, reverse for chronological order
        messages = messages[::-1]

        # Find last received message (not from me) - iterate backwards
        last_received: Message | None = None
        last_received_idx = -1
        for i, msg in enumerate(reversed(messages)):
            if not msg.is_from_me:
                last_received = msg
                last_received_idx = len(messages) - 1 - i
                break

        # TRUNCATE: Only include messages up to the last received one
        if last_received_idx != -1:
            context_messages = messages[: last_received_idx + 1]
        else:
            context_messages = messages

        participant_names = self._get_participant_names(chat_id)
        formatted = self._format_messages(context_messages, participant_names)

        # Find current topic from segments
        current_topic = self._get_current_topic(chat_id, context_messages)

        # V4 ENHANCEMENT: Fetch facts and profile
        contact_facts = ""
        relationship_graph = ""
        contact_profile = None

        try:
            from jarvis.contacts.fact_storage import get_facts_for_contact
            from jarvis.prompts.contact import format_facts_for_prompt

            facts = get_facts_for_contact(chat_id)
            contact_facts = format_facts_for_prompt(facts)
        except Exception:  # nosec B110
            pass

        try:
            from jarvis.contacts.contact_profile import get_contact_profile

            profile = get_contact_profile(chat_id)
            if profile:
                contact_profile = profile.to_dict()
        except Exception:  # nosec B110
            pass

        return ReplyContext(
            chat_id=chat_id,
            participant_names=participant_names,
            messages=context_messages,
            formatted_context=formatted,
            last_received_message=last_received,
            current_topic=current_topic,
            contact_facts=contact_facts,
            relationship_graph=relationship_graph,
            contact_profile=contact_profile,
        )

    def _get_current_topic(self, chat_id: str, messages: list[Message]) -> str | None:
        """Get the topic label of the most recent segment."""
        if not messages:
            return None

        # Get the newest message timestamp
        newest_time = max(m.date for m in messages)

        try:
            from jarvis.db import get_db
            from jarvis.topics.segment_storage import get_segments_for_chat

            db = get_db()
            with db.connection() as conn:
                # Get most recent segment that overlaps with our messages
                segments = get_segments_for_chat(conn, chat_id, limit=5)
                for seg in segments:
                    # Segments are ordered by start_time DESC
                    if seg.get("start_time") and seg["start_time"] <= newest_time:
                        return seg.get("topic_label")
        except Exception:  # nosec B110
            pass

        return None

    def get_summary_context(self, chat_id: str, num_messages: int = 50) -> SummaryContext:
        """Get context for summarizing a conversation.

        Retrieves messages from a conversation for summarization.

        Args:
            chat_id: The conversation ID
            num_messages: Number of messages to retrieve (default 50)

        Returns:
            SummaryContext with formatted messages and date range
        """
        messages = self._reader.get_messages(chat_id, limit=num_messages)

        # Messages come newest-first, reverse for chronological order
        messages = messages[::-1]

        participant_names = self._get_participant_names(chat_id)
        formatted = self._format_messages(messages, participant_names)

        # Calculate date range
        if messages:
            date_range = (messages[0].date, messages[-1].date)
        else:
            # Fallback: no messages found, use current time
            # This preserves the original behavior while documenting the edge case
            logger.debug(
                "No messages found for summary context (chat_id=%s), "
                "using current time as fallback",
                chat_id,
            )
            now = datetime.now()
            date_range = (now, now)

        return SummaryContext(
            chat_id=chat_id,
            participant_names=participant_names,
            messages=messages,
            formatted_context=formatted,
            date_range=date_range,
        )

    def get_search_context(self, query: str, limit: int = 20) -> SearchContext:
        """Get context from search results.

        Searches messages and formats results for LLM use.

        Args:
            query: Search query string
            limit: Maximum number of results (default 20)

        Returns:
            SearchContext with formatted search results
        """
        results = self._reader.search(query, limit=limit)
        formatted = self._format_search_results(query, results)

        return SearchContext(
            query=query,
            results=results,
            formatted_context=formatted,
        )

    def find_conversation_by_name(self, name: str) -> str | None:
        """Find a conversation by participant name.

        Performs fuzzy matching against conversation display names
        and participant identifiers.

        Args:
            name: Name to search for (case insensitive)

        Returns:
            chat_id if found, None otherwise
        """
        if not name:
            return None

        name_lower = name.lower().strip()
        conversations = self._get_conversations()

        # First pass: exact match on display name
        for conv in conversations:
            if conv.display_name and conv.display_name.lower() == name_lower:
                return conv.chat_id

        # Second pass: partial match on display name
        for conv in conversations:
            if conv.display_name and name_lower in conv.display_name.lower():
                return conv.chat_id

        # Third pass: match on participant identifiers (phone/email)
        for conv in conversations:
            for participant in conv.participants:
                # Check if name matches participant (case insensitive)
                if name_lower in participant.lower():
                    return conv.chat_id

        # Fourth pass: check if it's a phone number pattern
        # Remove common formatting and check against normalized participants
        name_digits = "".join(c for c in name if c.isdigit())
        if len(name_digits) >= 7:  # Minimum for a partial phone match
            for conv in conversations:
                for participant in conv.participants:
                    participant_digits = "".join(c for c in participant if c.isdigit())
                    if name_digits in participant_digits or participant_digits.endswith(
                        name_digits
                    ):
                        return conv.chat_id

        return None
