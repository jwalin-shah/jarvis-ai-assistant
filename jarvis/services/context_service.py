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

if TYPE_CHECKING:
    from integrations.imessage.reader import ChatDBReader
    from jarvis.search.vec_search import VecSearcher
    from models.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class ContextService:
    """Service for fetching conversation and contact context."""

    def __init__(
        self,
        db: JarvisDB | None = None,
        imessage_reader: ChatDBReader | None = None,
        semantic_searcher: Any | None = None,
        vec_searcher: VecSearcher | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        """Initialize the context service.

        Args:
            db: Database instance.
            imessage_reader: iMessage reader instance.
            semantic_searcher: Legacy semantic searcher (deprecated, ignored).
            vec_searcher: VecSearcher for chunk-based retrieval.
            reranker: Optional cross-encoder reranker for improving retrieval quality.
        """
        self._db = db or get_db()
        self._imessage_reader = imessage_reader
        self._vec_searcher = vec_searcher
        self._reranker = reranker

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

    def _get_vec_searcher(self) -> VecSearcher | None:
        """Get or lazily initialize the VecSearcher."""
        if self._vec_searcher is None:
            try:
                from jarvis.search.vec_search import get_vec_searcher

                self._vec_searcher = get_vec_searcher(self._db)
            except Exception:
                pass
        return self._vec_searcher

    def search_examples(
        self,
        incoming: str,
        chat_id: str | None = None,
        contact_id: int | None = None,
        embedder: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Search vec_chunks for similar conversation segments.

        Returns chunks with trigger/response text for few-shot prompting.
        When segment data is available, enriches results with full segment
        context (all message ROWIDs for coreference-preserving RAG).

        Args:
            embedder: Optional embedder override (e.g. CachedEmbedder) to avoid
                re-encoding the query if the caller already computed the embedding.
        """
        try:
            searcher = self._get_vec_searcher()
            if not searcher:
                return []

            # Try segment-aware search first for richer context
            try:
                segment_results = searcher.search_with_full_segments(
                    query=incoming,
                    limit=5,
                    contact_id=contact_id,
                    embedder=embedder,
                )
                if segment_results:
                    exchanges = []
                    for r in segment_results:
                        if r.get("trigger_text") and r.get("response_text"):
                            entry = {
                                "trigger_text": r["trigger_text"],
                                "response_text": r["response_text"],
                                "similarity": r["score"],
                                "topic": r.get("topic"),
                            }
                            # Include segment metadata if available
                            if "segment" in r:
                                entry["segment"] = r["segment"]
                            exchanges.append(entry)

                    if exchanges:
                        # Cross-encoder reranking
                        if self._reranker is not None and len(exchanges) > 1:
                            exchanges = self._reranker.rerank(
                                query=incoming,
                                candidates=exchanges,
                                text_key="trigger_text",
                                top_k=3,
                            )
                        return exchanges
            except Exception:
                pass  # Fall through to standard search

            # Standard search path (no segment data)
            if contact_id is not None:
                results = searcher.search_with_pairs(
                    query=incoming,
                    limit=5,
                    contact_id=contact_id,
                    embedder=embedder,
                )
            else:
                results = searcher.search_with_pairs_global(
                    query=incoming,
                    limit=5,
                    embedder=embedder,
                )

            exchanges = []
            for r in results:
                if r.trigger_text and r.response_text:
                    exchanges.append(
                        {
                            "trigger_text": r.trigger_text,
                            "response_text": r.response_text,
                            "similarity": r.score,
                            "topic": r.topic,
                        }
                    )

            # Cross-encoder reranking (if enabled and useful)
            if self._reranker is not None and len(exchanges) > 1:
                exchanges = self._reranker.rerank(
                    query=incoming,
                    candidates=exchanges,
                    text_key="trigger_text",
                    top_k=3,
                )

            return exchanges
        except Exception as e:
            logger.warning("Chunk search failed: %s", e)
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
