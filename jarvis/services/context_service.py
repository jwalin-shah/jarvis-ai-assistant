"""Context Service - Handles data fetching and context preparation for the router."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from jarvis.contacts.contact_profile import MIN_MESSAGES_FOR_PROFILE, get_contact_profile
from jarvis.contacts.contact_profile_context import (
    ContactProfileContext,
    is_contact_profile_context_enabled,
)
from jarvis.db import Contact, JarvisDB, get_db
from jarvis.search.semantic_search import SearchFilters

if TYPE_CHECKING:
    from integrations.imessage.reader import ChatDBReader
    from jarvis.search.semantic_search import SemanticSearcher

logger = logging.getLogger(__name__)


class ContextService:
    """Service for fetching conversation and contact context."""

    def __init__(
        self,
        db: JarvisDB | None = None,
        imessage_reader: ChatDBReader | None = None,
        semantic_searcher: SemanticSearcher | None = None,
    ) -> None:
        """Initialize the context service.

        Args:
            db: Database instance.
            imessage_reader: iMessage reader instance.
            semantic_searcher: Semantic searcher instance.
        """
        self._db = db or get_db()
        self._imessage_reader = imessage_reader
        self._semantic_searcher = semantic_searcher

    @property
    def db(self) -> JarvisDB:
        return self._db

    def get_contact(self, contact_id: int | None, chat_id: str | None) -> Contact | None:
        """Look up contact by contact_id or chat_id."""
        if contact_id:
            return self.db.get_contact(contact_id)
        if chat_id:
            return self.db.get_contact_by_chat_id(chat_id)
        return None

    def fetch_conversation_context(self, chat_id: str, limit: int = 10) -> list[str]:
        """Fetch recent conversation history from iMessage."""
        if not self._imessage_reader:
            return []

        try:
            messages = self._imessage_reader.get_messages(chat_id, limit=limit)
            if not messages:
                return []

            # Format messages for context (newest first, so reverse for chronological)
            context_messages = []
            for msg in reversed(messages):
                sender = "You" if msg.is_from_me else (msg.sender_name or msg.sender or "Them")
                text = msg.text or ""
                if text:
                    context_messages.append(f"[{sender}]: {text}")

            return context_messages

        except Exception as e:
            logger.warning("Failed to fetch conversation context: %s", e)
            return []

    def search_examples(
        self,
        incoming: str,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search messages for similar conversations."""
        try:
            if not self._semantic_searcher:
                return []

            filters = SearchFilters(chat_id=chat_id)
            results = self._semantic_searcher.search(
                query=incoming,
                filters=filters,
                limit=5,
            )

            exchanges = []
            for r in results:
                if not r.message.is_from_me:
                    # Found a similar incoming message, find my response to it
                    if self._imessage_reader:
                        # Get messages immediately after this one
                        after_messages = self._imessage_reader.get_messages_after(
                            r.message.id, chat_id=r.message.chat_id, limit=1
                        )
                        if after_messages and after_messages[0].is_from_me:
                            exchanges.append(
                                {
                                    "trigger_text": r.message.text,
                                    "response_text": after_messages[0].text,
                                    "similarity": r.similarity,
                                }
                            )

            return exchanges
        except Exception as e:
            logger.warning("Semantic search failed: %s", e)
            return []

    def get_relationship_profile(
        self, contact: Contact | None, chat_id: str | None
    ) -> tuple[dict[str, Any] | None, ContactProfileContext | None]:
        """Get relationship profile and context for a contact."""
        relationship_profile = None
        contact_context: ContactProfileContext | None = None

        if contact:
            profile = None
            if is_contact_profile_context_enabled():
                profile = get_contact_profile(chat_id) if chat_id else None
                if profile is None and contact.id:
                    profile = get_contact_profile(str(contact.id))

                if profile and profile.message_count >= MIN_MESSAGES_FOR_PROFILE:
                    contact_context = ContactProfileContext.from_contact_profile(profile)
                    relationship_profile = contact_context.to_prompt_payload()
                    logger.debug(
                        "Using contact profile context for %s (formality=%.2f, %d messages)",
                        contact.display_name,
                        profile.formality_score,
                        profile.message_count,
                    )

            if contact_context is None:
                relationship_profile = {
                    "tone": contact.style_notes or "casual",
                    "relationship": contact.relationship or "friend",
                }

        return relationship_profile, contact_context
