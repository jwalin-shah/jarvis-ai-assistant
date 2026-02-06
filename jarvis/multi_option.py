"""Multi-Option Generation - Generate diverse response options for commitment questions.

For commitment questions (invitations, requests, yes/no questions), generates
3 diverse options representing different response types:
- AGREE: Positive acceptance
- DECLINE: Polite rejection
- DEFER: Non-committal, need to check

This follows the Smart Reply pattern from Google's research, ensuring users
have meaningful choices rather than variations of the same response.

Usage:
    from jarvis.multi_option import generate_response_options, get_multi_option_generator

    generator = get_multi_option_generator()

    result = generator.generate_options(
        trigger="Want to grab lunch tomorrow?",
        contact_name="Sarah",
    )

    for option in result.options:
        print(f"{option.response_type}: {option.text}")
    # AGREE: Yeah I'm down!
    # DECLINE: Can't tomorrow, sorry
    # DEFER: Let me check my schedule
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jarvis.classifiers.response_classifier import (
    ResponseType,
)
from jarvis.search.retrieval import TypedRetriever

if TYPE_CHECKING:
    from jarvis.db import Contact

logger = logging.getLogger(__name__)

# Confidence threshold for retrieval - below this, use LLM generation
MIN_RETRIEVAL_CONFIDENCE = 0.5


# Response type priority for multi-option generation
# AGREE first (most common response), then alternatives
OPTION_PRIORITY = [
    ResponseType.AGREE,
    ResponseType.DECLINE,
    ResponseType.DEFER,
]

# Trigger types that should use multi-option generation
# New trigger classifier uses coarser labels: "commitment" covers invitations/requests
# Keep old labels for backwards compatibility with any remaining old classifier usage
COMMITMENT_TRIGGER_TYPES = frozenset(
    {
        # New hybrid classifier labels (TriggerType enum values)
        "commitment",
        # Legacy DA classifier labels (for backwards compatibility)
        "INVITATION",
        "REQUEST",
        "YN_QUESTION",
    }
)

# Patterns that look like REQUEST but are actually INFO_STATEMENT (status updates)
# These should NOT trigger commitment options even if classifier says REQUEST
INFO_STATEMENT_PATTERNS = [
    # Location/transit status
    re.compile(r"^(i'?m |i am )?(on my way|omw|otw|heading|coming|leaving|almost)", re.IGNORECASE),
    re.compile(r"^(just |almost )?(left|arrived|got here|parked|here)", re.IGNORECASE),
    re.compile(r"^(be there|gonna be|will be|should be) (in |soon|shortly)", re.IGNORECASE),
    re.compile(r"^\d+ min(ute)?s?( away| out)?$", re.IGNORECASE),
    re.compile(r"^(eta|here in) \d+", re.IGNORECASE),
    # Running late
    re.compile(r"^(i'?m |i am )?(running|gonna be) late", re.IGNORECASE),
    re.compile(r"^(sorry.*)?(running behind|stuck in traffic)", re.IGNORECASE),
    re.compile(r"^(got |hit )?(stuck|held up|delayed)", re.IGNORECASE),
    # Simple status
    re.compile(r"^(i'?m |i am )?(here|home|at |in the)", re.IGNORECASE),
    re.compile(r"^(just |already )?(woke up|got up|finished|done)", re.IGNORECASE),
]

# WH_QUESTION patterns - asking for info, not invitations
# "Who's coming?" asks for info, "Want to come?" is an invitation
WH_QUESTION_PATTERNS = [
    # Who questions (asking about people, not inviting)
    re.compile(r"^who'?s (coming|going|there|all |gonna)", re.IGNORECASE),
    re.compile(r"^who (else |all )?(is |are )?(coming|going|there)", re.IGNORECASE),
    re.compile(r"^who (did|will|should|can)", re.IGNORECASE),
    # What/when/where questions
    re.compile(r"^what time", re.IGNORECASE),
    re.compile(r"^what'?s the (time|plan|address|spot)", re.IGNORECASE),
    re.compile(r"^when (is|are|do|does|should|will)", re.IGNORECASE),
    re.compile(r"^where (is|are|do|should|at)", re.IGNORECASE),
    re.compile(r"^how (long|much|many|far)", re.IGNORECASE),
]


def _is_info_statement(text: str) -> bool:
    """Check if text matches INFO_STATEMENT patterns (not a real commitment trigger)."""
    text = text.strip()
    for pattern in INFO_STATEMENT_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _is_wh_question(text: str) -> bool:
    """Check if text is a WH_QUESTION (asking for info, not a commitment trigger)."""
    text = text.strip()
    for pattern in WH_QUESTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


@dataclass
class ResponseOption:
    """A single response option with metadata."""

    text: str
    response_type: ResponseType
    confidence: float
    source: str  # 'template', 'generated', 'fallback'

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "type": self.response_type.value,
            "response": self.text,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass
class MultiOptionResult:
    """Result from multi-option generation."""

    trigger: str
    trigger_da: str | None
    is_commitment: bool
    options: list[ResponseOption] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "trigger": self.trigger,
            "trigger_da": self.trigger_da,
            "is_commitment": self.is_commitment,
            "options": [opt.to_dict() for opt in self.options],
            "suggestions": [opt.text for opt in self.options],  # Backward compatible
        }

    @property
    def has_options(self) -> bool:
        """Check if we have any options."""
        return len(self.options) > 0

    def get_option(self, response_type: ResponseType) -> ResponseOption | None:
        """Get option by response type."""
        for opt in self.options:
            if opt.response_type == response_type:
                return opt
        return None


# =============================================================================
# Fallback Templates
# =============================================================================

# Simple fallback templates when we don't have personalized examples
FALLBACK_TEMPLATES: dict[ResponseType, list[str]] = {
    ResponseType.AGREE: [
        "Yeah I'm down!",
        "Sure, sounds good!",
        "Yes!",
        "Definitely!",
        "Count me in!",
    ],
    ResponseType.DECLINE: [
        "Can't make it, sorry",
        "Not today, unfortunately",
        "I'll have to pass",
        "Sorry, I'm busy",
        "Rain check?",
    ],
    ResponseType.DEFER: [
        "Let me check and get back to you",
        "Maybe, I'll let you know",
        "Not sure yet, I'll see",
        "Let me think about it",
        "Possibly, need to check my schedule",
    ],
    ResponseType.QUESTION: [
        "What time?",
        "Where at?",
        "Who else is coming?",
    ],
    ResponseType.ACKNOWLEDGE: [
        "Got it!",
        "Okay",
        "Sounds good",
    ],
    ResponseType.REACT_POSITIVE: [
        "That's awesome!",
        "Congrats!",
        "Amazing!",
    ],
    ResponseType.REACT_SYMPATHY: [
        "I'm sorry to hear that",
        "That sucks",
        "Here for you",
    ],
}


class MultiOptionGenerator:
    """Generates diverse response options for commitment questions.

    Delegates to the unified ReplyService.
    """

    def __init__(
        self,
        retriever: TypedRetriever | None = None,
        max_options: int = 3,
    ) -> None:
        """Initialize the generator.

        Args:
            retriever: TypedRetriever.
            max_options: Maximum number of options.
        """
        from jarvis.reply_service import get_reply_service

        self._service = get_reply_service()
        if retriever:
            from jarvis.reply_service import ReplyService

            self._service = ReplyService(retriever=retriever)
        self._max_options = max_options

    @property
    def retriever(self) -> TypedRetriever:
        return self._service.retriever

    def is_commitment_trigger(self, trigger: str) -> tuple[bool, str | None]:
        """Check if trigger is a commitment question."""
        if _is_info_statement(trigger):
            return False, "statement"
        if _is_wh_question(trigger):
            return False, "question"

        trigger_da, _ = self.retriever.classify_trigger(trigger)
        is_commitment = trigger_da in COMMITMENT_TRIGGER_TYPES
        return is_commitment, trigger_da

    def generate_options(
        self,
        trigger: str,
        contact_name: str | None = None,
        contact: Contact | None = None,
        chat_id: str | None = None,
        force_commitment: bool = False,
    ) -> MultiOptionResult:
        """Generate diverse response options for a trigger."""
        return self._service.generate_options(
            incoming=trigger,
            chat_id=chat_id,
            force_commitment=force_commitment,
        )


# =============================================================================
# Singleton Access
# =============================================================================

_generator: MultiOptionGenerator | None = None
_generator_lock = threading.Lock()


def get_multi_option_generator() -> MultiOptionGenerator:
    """Get or create the singleton MultiOptionGenerator instance."""
    global _generator

    if _generator is None:
        with _generator_lock:
            if _generator is None:
                _generator = MultiOptionGenerator()

    return _generator


def reset_multi_option_generator() -> None:
    """Reset the singleton generator."""
    global _generator

    with _generator_lock:
        _generator = None


# =============================================================================
# Convenience Function
# =============================================================================


def generate_response_options(
    trigger: str,
    contact_name: str | None = None,
    chat_id: str | None = None,
) -> MultiOptionResult:
    """Generate response options for a trigger.

    Convenience function that uses the singleton generator.

    Args:
        trigger: Trigger message text.
        contact_name: Optional contact name.
        chat_id: Optional chat_id for loading relationship profile.

    Returns:
        MultiOptionResult with diverse options.
    """
    return get_multi_option_generator().generate_options(
        trigger=trigger,
        contact_name=contact_name,
        chat_id=chat_id,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ResponseOption",
    "MultiOptionResult",
    "MultiOptionGenerator",
    "get_multi_option_generator",
    "reset_multi_option_generator",
    "generate_response_options",
    "COMMITMENT_TRIGGER_TYPES",
]
