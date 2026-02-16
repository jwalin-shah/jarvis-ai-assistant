"""Stub module for backward compatibility.

This module provides minimal exports to support legacy imports.
The functionality has been migrated to jarvis.contacts.contact_profile.

NOTE: This is a temporary stub. Code should be updated to use
jarvis.contacts.contact_profile.ContactProfile instead.
"""

from dataclasses import dataclass
from typing import Any

MIN_MESSAGES_FOR_PROFILE = 5


@dataclass
class ToneProfile:
    """Stub for backward compatibility."""

    formality_score: float = 0.5


@dataclass
class RelationshipProfile:
    """Stub for backward compatibility.

    Use jarvis.contacts.contact_profile.ContactProfile instead.
    """

    contact_id: str
    message_count: int = 0
    tone_profile: ToneProfile | None = None

    def __post_init__(self) -> None:
        if self.tone_profile is None:
            self.tone_profile = ToneProfile()


def build_relationship_profile(contact_id: str, messages: list[Any]) -> RelationshipProfile:
    """Stub for backward compatibility."""
    return RelationshipProfile(contact_id=contact_id, message_count=len(messages))


def profile_needs_refresh(profile: RelationshipProfile) -> bool:
    """Stub for backward compatibility."""
    return True


def load_profile(contact_id: str) -> RelationshipProfile | None:
    """Stub for backward compatibility."""
    return None


def delete_profile(contact_id: str) -> None:
    """Stub for backward compatibility."""
    pass


def select_matching_examples(
    relationship_profile: RelationshipProfile,
    casual_examples: list[tuple[str, str]],
    professional_examples: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Stub for backward compatibility."""
    return []


def get_voice_guidance(contact_id: str) -> dict[str, str]:
    """Stub for backward compatibility."""
    return {
        "tone": "casual",
        "length": "medium",
    }


def generate_style_guide(relationship_profile: RelationshipProfile) -> str:
    """Stub for backward compatibility."""
    return "casual"
