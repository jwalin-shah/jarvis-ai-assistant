"""Context Service - Handles data fetching and context preparation for the router."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from jarvis.contacts.contact_profile import MIN_MESSAGES_FOR_PROFILE, get_contact_profile
from jarvis.contacts.contact_profile_context import (
    ContactProfileContext,
    is_contact_profile_context_enabled,
)
from jarvis.contacts.junk_filters import is_bot_message
from jarvis.db import Contact, JarvisDB, get_db
from jarvis.search.vec_search import VecSearchResult

if TYPE_CHECKING:
    from integrations.imessage.reader import ChatDBReader
    from jarvis.search.vec_search import VecSearcher
    from models.cross_encoder import CrossEncoderReranker

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

    def is_bot_chat(self, chat_id: str | None, contact_name: str | None = None) -> bool:
        """Determine if a chat is with an automated service or bot.

        Uses heuristics like shortcodes (<= 6 digits) and keyword matching
        on the contact name, plus specialized junk filters.
        """
        if not chat_id:
            return False

        # 1. Content-based filter (from junk_filters)
        # We don't have the text here, but we can check the chat_id
        if is_bot_message("", chat_id):
            return True

        # 2. Check for shortcodes or service-like identifiers
        # chat_id often looks like "iMessage;-;+1234567890" or "iMessage;-;12345"
        identifier = chat_id.rsplit(";", 1)[-1] if ";" in chat_id else chat_id
        if identifier.isdigit() and len(identifier) <= 6:
            return True

        # 3. Keyword matching on contact name or text pattern
        if contact_name:
            cname_lower = contact_name.lower()
            bot_keywords = [
                "alert",
                "notice",
                "verification",
                "system",
                "no-reply",
                "automated",
                "reminder",
                "support",
            ]
            if any(k in cname_lower for k in bot_keywords):
                return True
        
        # Check if the chat ID or identifier contains bot markers
        if identifier and any(k in identifier.lower() for k in ["info", "united", "reply"]):
            return True

        # 4. Check for alphanumeric sender IDs (Business handles like "INFO", "UNITED")
        if identifier and not identifier.isdigit() and "+" not in identifier and "@" not in identifier:
            # If it's a short alphanumeric string without typical phone/email markers
            if len(identifier) < 15:
                return True

        return False

    def fetch_conversation_context(
        self, chat_id: str, limit: int = 10
    ) -> tuple[list[str], set[str]]:
        """Fetch recent conversation history from iMessage.

        Returns:
            Tuple of (list of formatted message strings, set of participant names).
        """
        if not self._imessage_reader:
            return [], set()

        try:
            messages = self._imessage_reader.get_messages(chat_id, limit=limit)
            if not messages:
                return [], set()

            # Format messages for context (newest first, so reverse for chronological)
            chronological = list(reversed(messages))
            context_turns = []
            participants: set[str] = set()

            current_sender: str | None = None
            current_text_parts: list[str] = []
            last_msg_time: datetime | None = None

            # Filter messages: Keep all within last 24 hours, or the most recent cluster
            # Handle timezone-aware vs naive datetimes
            first_msg_tz = chronological[0].date.tzinfo if chronological else None
            now = datetime.now(first_msg_tz)
            cutoff = now - timedelta(hours=24)

            # We already have chronological list. Let's find the start index.
            start_idx = 0
            for i, msg in enumerate(chronological):
                if msg.date > cutoff:
                    start_idx = i
                    break
                # If message is older than 24h but very close to the next one (< 1h),
                # we might want to keep it, but user asked to skip if far apart.
                # Simplest is to just use 24h cutoff for "recent" context.

            recent_chronological = chronological[start_idx:]
            if not recent_chronological and chronological:
                # If nothing in last 24h, just take the last 3 messages anyway
                recent_chronological = chronological[-3:]
            
            # Respect limit: only take the most recent messages up to 'limit'
            if len(recent_chronological) > limit:
                recent_chronological = recent_chronological[-limit:]

            last_date: date | None = None

            for msg in recent_chronological:
                sender = "You" if msg.is_from_me else (msg.sender_name or msg.sender or "Contact")
                participants.add(sender)
                text = (msg.text or "").strip()
                if not text:
                    continue

                # Add date header if date changed
                msg_date = msg.date.date()
                if msg_date != last_date:
                    if current_sender is not None:
                        context_turns.append(f"{current_sender}: {' '.join(current_text_parts)}")
                        current_sender = None
                        current_text_parts = []
                    
                    date_label = "Today" if msg_date == now.date() else "Yesterday" if msg_date == (now.date() - timedelta(days=1)) else msg_date.strftime("%A, %b %d")
                    context_turns.append(f"--- {date_label} ---")
                    last_date = msg_date

                # Add time gap marker if > 6 hours
                if last_msg_time and (msg.date - last_msg_time).total_seconds() > 21600:
                    if current_sender is not None:
                        context_turns.append(f"{current_sender}: {' '.join(current_text_parts)}")
                        current_sender = None
                        current_text_parts = []

                    gap_hours = int((msg.date - last_msg_time).total_seconds() / 3600)
                    context_turns.append(f"({gap_hours} hours later)")

                timestamp_str = msg.date.strftime("%H:%M")

                if sender == current_sender:
                    current_text_parts.append(text)
                else:
                    if current_sender is not None:
                        context_turns.append(f"{current_sender}: {' '.join(current_text_parts)}")
                    current_sender = sender
                    current_text_parts = [f"[{timestamp_str}] {text}"]

                last_msg_time = msg.date

            if current_sender is not None:
                context_turns.append(f"{current_sender}: {' '.join(current_text_parts)}")

            return context_turns, participants

        except Exception as e:
            logger.warning("Failed to fetch conversation context: %s", e)
            return [], set()

    def _get_vec_searcher(self) -> VecSearcher | None:
        """Get or lazily initialize the VecSearcher."""
        if self._vec_searcher is None:
            try:
                from jarvis.search.vec_search import get_vec_searcher

                self._vec_searcher = get_vec_searcher(self._db)
            except Exception:  # nosec B110
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
                        if r.get("context_text") and r.get("reply_text"):
                            entry = {
                                "context_text": r["context_text"],
                                "reply_text": r["reply_text"],
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
                                text_key="context_text",
                                top_k=3,
                            )
                        return exchanges
            except Exception:  # nosec B110
                pass  # Fall through to standard search

            # Standard search path (no segment data)
            if contact_id is not None:
                results = searcher.search_with_chunks(
                    query=incoming,
                    limit=5,
                    contact_id=contact_id,
                    embedder=embedder,
                )
            else:
                results = searcher.search_with_chunks_global(
                    query=incoming,
                    limit=5,
                    embedder=embedder,
                )

            exchanges = []
            result: VecSearchResult
            for result in results:
                if result.context_text and result.reply_text:
                    exchanges.append(
                        {
                            "context_text": result.context_text,
                            "reply_text": result.reply_text,
                            "similarity": result.score,
                            "topic": result.topic,
                        }
                    )

            # Cross-encoder reranking (if enabled and useful)
            if self._reranker is not None and len(exchanges) > 1:
                exchanges = self._reranker.rerank(
                    query=incoming,
                    candidates=exchanges,
                    text_key="context_text",
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
