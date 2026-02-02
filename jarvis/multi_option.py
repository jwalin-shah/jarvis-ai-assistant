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

from jarvis.response_classifier import (
    ResponseType,
)
from jarvis.retrieval import TypedRetriever, get_typed_retriever

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

    For triggers like invitations and requests, generates multiple options
    representing different response intents (AGREE, DECLINE, DEFER).

    Strategy (simple and fast):
    1. FAISS retrieval: Get personalized examples from user's message history
    2. Static fallback: Use high-quality template responses if no history found

    No LLM needed - commitment responses are simple enough that templates work great,
    and this keeps latency under 200ms instead of 1-2 seconds with LLM.

    Thread Safety:
        This class is thread-safe. Dependencies loaded lazily with locking.
    """

    def __init__(
        self,
        retriever: TypedRetriever | None = None,
        max_options: int = 3,
    ) -> None:
        """Initialize the generator.

        Args:
            retriever: TypedRetriever for getting examples. Created lazily if None.
            max_options: Maximum number of options to generate.
        """
        self._retriever = retriever
        self._max_options = max_options

    @property
    def retriever(self) -> TypedRetriever:
        """Get or create the retriever."""
        if self._retriever is None:
            self._retriever = get_typed_retriever()
        return self._retriever

    def is_commitment_trigger(self, trigger: str) -> tuple[bool, str | None]:
        """Check if trigger is a commitment question.

        Args:
            trigger: Trigger text.

        Returns:
            Tuple of (is_commitment, trigger_da_type).
        """
        # First check for INFO_STATEMENT patterns (status updates that shouldn't trigger)
        # This catches false positives where classifier says REQUEST but it's really status
        if _is_info_statement(trigger):
            return False, "statement"  # New-style label

        # Check for WH_QUESTION patterns (asking for info, not invitations)
        # "Who's coming?" asks for info, "Want to come?" is an invitation
        if _is_wh_question(trigger):
            return False, "question"  # New-style label

        trigger_da, conf = self.retriever.classify_trigger(trigger)
        is_commitment = trigger_da in COMMITMENT_TRIGGER_TYPES
        return is_commitment, trigger_da

    def _get_template_option(
        self,
        trigger: str,
        response_type: ResponseType,
        examples: list,
    ) -> ResponseOption | None:
        """Get a template option from retrieved examples.

        Args:
            trigger: Trigger text.
            response_type: Target response type.
            examples: Retrieved examples of this type.

        Returns:
            ResponseOption or None if no good example found.
        """
        if not examples:
            return None

        # Use the best example (highest similarity)
        best = examples[0]
        return ResponseOption(
            text=best.response_text,
            response_type=response_type,
            confidence=best.similarity,
            source="template",
        )

    def _get_fallback_option(self, response_type: ResponseType) -> ResponseOption:
        """Get a fallback template option.

        Args:
            response_type: Target response type.

        Returns:
            ResponseOption with a generic template.
        """
        import random

        templates = FALLBACK_TEMPLATES.get(response_type, ["Okay"])
        text = random.choice(templates)

        return ResponseOption(
            text=text,
            response_type=response_type,
            confidence=0.5,
            source="fallback",
        )

    def _generate_llm_option(
        self,
        trigger: str,
        response_type: ResponseType,
        style_guide: str,
        examples: list | None = None,
    ) -> ResponseOption | None:
        """Generate a response using the LLM.

        Args:
            trigger: The message to respond to.
            response_type: Target response type (AGREE, DECLINE, DEFER).
            style_guide: Style guidance from the relationship profile.
            examples: Optional few-shot examples.

        Returns:
            ResponseOption or None if generation fails or LLM unavailable.
        """
        from contracts.models import GenerationRequest
        from jarvis.generation import can_use_llm, generate_with_fallback
        from jarvis.prompts import COMMITMENT_PROMPT

        # Check if LLM is available before attempting generation
        can_generate, reason = can_use_llm()
        if not can_generate:
            logger.debug("LLM unavailable for %s: %s", response_type.value, reason)
            return None  # Fall back to static templates

        # Format examples section if we have any
        examples_section = ""
        if examples:
            example_lines = []
            for ex in examples[:2]:  # Max 2 examples
                example_lines.append(f"Message: {ex.trigger_text}")
                example_lines.append(f"Response: {ex.response_text}")
                example_lines.append("")
            if example_lines:
                examples_section = "\n### Examples:\n" + "\n".join(example_lines)

        # Format the prompt
        response_type_name = response_type.value.upper()
        response_type_lower = response_type.value.lower()

        prompt = COMMITMENT_PROMPT.template.format(
            response_type=response_type_name,
            response_type_lower=response_type_lower,
            style_guide=style_guide,
            trigger=trigger,
            examples_section=examples_section,
        )

        try:
            request = GenerationRequest(
                prompt=prompt,
                context_documents=[],
                few_shot_examples=[],
                max_tokens=30,
                temperature=0.7,  # Slightly creative for variety
            )

            response = generate_with_fallback(request)

            if response.text and response.finish_reason != "error":
                # Clean up the response
                text = response.text.strip()
                # Remove any trailing punctuation repetition
                text = text.rstrip(".")
                # Take only first line if multiple
                text = text.split("\n")[0].strip()

                if text and len(text) < 100:  # Sanity check
                    return ResponseOption(
                        text=text,
                        response_type=response_type,
                        confidence=0.7,  # Medium confidence for LLM
                        source="generated",
                    )

        except Exception as e:
            logger.warning("LLM generation failed for %s: %s", response_type.value, e)

        return None

    def generate_options(
        self,
        trigger: str,
        contact_name: str | None = None,
        contact: Contact | None = None,
        chat_id: str | None = None,
        force_commitment: bool = False,
    ) -> MultiOptionResult:
        """Generate diverse response options for a trigger.

        Args:
            trigger: Trigger message text.
            contact_name: Optional contact name for personalization.
            contact: Optional Contact object with style info.
            chat_id: Optional chat_id for loading relationship profile.
            force_commitment: If True, treat as commitment even if not detected.

        Returns:
            MultiOptionResult with options for different response types.
        """
        # Check if this is a commitment trigger
        is_commitment, trigger_da = self.is_commitment_trigger(trigger)

        if force_commitment:
            is_commitment = True

        if not is_commitment:
            # Not a commitment question - return single option
            return MultiOptionResult(
                trigger=trigger,
                trigger_da=trigger_da,
                is_commitment=False,
                options=[],
            )

        # Load relationship profile for style guidance
        style_guide = "Use a casual, friendly tone."  # Default
        if chat_id:
            try:
                from jarvis.embedding_profile import (
                    generate_embedding_style_guide,
                    load_embedding_profile,
                )

                profile = load_embedding_profile(chat_id)
                if profile:
                    style_guide = generate_embedding_style_guide(profile)
                    logger.debug("Loaded profile for %s: %s", chat_id[:15], style_guide[:50])
            except Exception as e:
                logger.debug("Failed to load profile for %s: %s", chat_id, e)

        # OPTIMIZATION: Create cached embedder for reuse across all FAISS searches
        from jarvis.embedding_adapter import CachedEmbedder, get_embedder

        cached_embedder = CachedEmbedder(get_embedder())

        # Get examples for commitment types
        # Pass trigger_da to avoid re-classification, and cached_embedder for efficiency
        multi_examples = self.retriever.get_examples_for_commitment(
            trigger=trigger,
            k_per_type=3,
            embedder=cached_embedder,
            trigger_da=trigger_da,  # Reuse classification from above
        )

        # Generate options for each type
        # Strategy: FAISS retrieval (high confidence) → LLM (low confidence) → static fallback
        options: list[ResponseOption] = []

        for response_type in OPTION_PRIORITY:
            if len(options) >= self._max_options:
                break

            examples = multi_examples.get_examples(response_type)

            # Strategy 1: Use retrieved examples if high confidence
            if examples and examples[0].similarity >= MIN_RETRIEVAL_CONFIDENCE:
                option = self._get_template_option(trigger, response_type, examples)
                if option:
                    logger.debug(
                        "%s: using template (conf=%.2f)",
                        response_type.value,
                        examples[0].similarity,
                    )
                    options.append(option)
                    continue

            # Strategy 2: LLM generation for low confidence (personalized via style guide)
            llm_option = self._generate_llm_option(
                trigger=trigger,
                response_type=response_type,
                style_guide=style_guide,
                examples=examples,  # Pass examples for few-shot even if low confidence
            )
            if llm_option:
                logger.debug("%s: using LLM generation", response_type.value)
                options.append(llm_option)
                continue

            # Strategy 3: Static fallback templates (if LLM fails)
            logger.debug("%s: using static fallback", response_type.value)
            options.append(self._get_fallback_option(response_type))

        # Ensure diversity - no duplicate texts
        seen_texts = set()
        unique_options = []
        for opt in options:
            text_lower = opt.text.lower().strip()
            if text_lower not in seen_texts:
                seen_texts.add(text_lower)
                unique_options.append(opt)

        return MultiOptionResult(
            trigger=trigger,
            trigger_da=trigger_da,
            is_commitment=True,
            options=unique_options,
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
