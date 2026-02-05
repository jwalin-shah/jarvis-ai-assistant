"""Typed contact profile context used by prompts and routing."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.contacts.contact_profile import ContactProfile
    from jarvis.embeddings import EmbeddingStoreProfile


_ENABLE_CONTACT_PROFILE_CONTEXT = os.environ.get(
    "JARVIS_ENABLE_CONTACT_PROFILE_CONTEXT", "1"
).lower() in {"1", "true", "yes"}


def is_contact_profile_context_enabled() -> bool:
    """Whether contact-profile-driven context should be used."""
    return _ENABLE_CONTACT_PROFILE_CONTEXT


@dataclass
class ContactProfileContext:
    tone: str = "casual"
    avg_message_length: float = 50.0
    emoji_frequency: float | None = None
    response_patterns: dict[str, float | int] | None = None
    relationship: str | None = None
    relationship_confidence: float | None = None
    style_guide: str | None = None
    greeting_style: list[str] = field(default_factory=list)
    signoff_style: list[str] = field(default_factory=list)
    top_topics: list[str] = field(default_factory=list)

    def to_prompt_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tone": self.tone,
            "avg_message_length": self.avg_message_length,
        }
        if self.response_patterns:
            payload["response_patterns"] = self.response_patterns
        if self.relationship:
            payload["relationship"] = self.relationship
        if self.style_guide:
            payload["style_guide"] = self.style_guide
        if self.emoji_frequency is not None:
            payload["emoji_frequency"] = self.emoji_frequency
        if self.greeting_style:
            payload["greeting_style"] = self.greeting_style
        if self.signoff_style:
            payload["signoff_style"] = self.signoff_style
        if self.top_topics:
            payload["top_topics"] = self.top_topics
        return payload

    @classmethod
    def from_contact_profile(cls, profile: ContactProfile) -> ContactProfileContext:
        """Build context from a full contact profile."""
        tone = "professional" if profile.formality_score >= 0.7 else "casual"
        if profile.formality == "very_casual":
            tone = "casual"
        style_guide = cls._format_style_guide(profile)
        return cls(
            tone=tone,
            avg_message_length=profile.avg_message_length,
            emoji_frequency=profile.emoji_frequency,
            relationship=profile.relationship or "friend",
            relationship_confidence=profile.relationship_confidence,
            style_guide=style_guide,
            greeting_style=profile.greeting_style,
            signoff_style=profile.signoff_style,
            top_topics=profile.top_topics,
        )

    @classmethod
    def from_relationship_profile(
        cls, profile: EmbeddingStoreProfile
    ) -> ContactProfileContext:
        """Build context from the embedding store's relationship profile."""
        return cls(
            tone=profile.typical_tone,
            avg_message_length=profile.avg_message_length or 50.0,
            response_patterns=profile.response_patterns or None,
        )

    @staticmethod
    def _format_style_guide(profile: ContactProfile) -> str | None:
        if profile.message_count < 1:
            return None
        try:
            from jarvis.contacts.contact_profile import format_style_guide

            return format_style_guide(profile)
        except ImportError:
            return None
