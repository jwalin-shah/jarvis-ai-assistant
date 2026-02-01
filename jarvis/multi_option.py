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
COMMITMENT_TRIGGER_TYPES = frozenset({
    # New hybrid classifier labels (TriggerType enum values)
    "commitment",
    # Legacy DA classifier labels (for backwards compatibility)
    "INVITATION",
    "REQUEST",
    "YN_QUESTION",
})

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

    Two strategies:
    1. Template mode: Use retrieved examples directly (fast, personalized)
    2. Generate mode: Use LLM with typed examples as few-shot (slower, flexible)

    Thread Safety:
        This class is thread-safe. Dependencies loaded lazily with locking.
    """

    def __init__(
        self,
        retriever: TypedRetriever | None = None,
        use_llm: bool = True,
        max_options: int = 3,
    ) -> None:
        """Initialize the generator.

        Args:
            retriever: TypedRetriever for getting examples. Created lazily if None.
            use_llm: If True, use LLM for generation. If False, use templates only.
            max_options: Maximum number of options to generate.
        """
        self._retriever = retriever
        self._generator = None
        self._lock = threading.Lock()
        self._use_llm = use_llm
        self._max_options = max_options

    @property
    def retriever(self) -> TypedRetriever:
        """Get or create the retriever."""
        if self._retriever is None:
            self._retriever = get_typed_retriever()
        return self._retriever

    @property
    def generator(self):
        """Get or create the LLM generator."""
        if self._generator is None and self._use_llm:
            with self._lock:
                if self._generator is None:
                    try:
                        from models import get_generator
                        self._generator = get_generator(skip_templates=True)
                    except Exception as e:
                        logger.warning("Failed to load generator: %s", e)
        return self._generator

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

    def _generate_all_options_batched(
        self,
        trigger: str,
        types_needed: list[ResponseType],
        examples_by_type: dict[ResponseType, list],
        contact_name: str | None = None,
    ) -> dict[ResponseType, ResponseOption]:
        """Generate all options in a single LLM call (batched).

        This is more efficient than calling the LLM 3 times separately.
        Reduces latency from ~3x to ~1x for the generation phase.

        Args:
            trigger: Trigger text.
            types_needed: List of response types to generate.
            examples_by_type: Examples for each type.
            contact_name: Optional contact name for personalization.

        Returns:
            Dict mapping response type to generated option.
        """
        if not self.generator or not types_needed:
            return {}

        try:
            from contracts.models import GenerationRequest

            # Build the batched prompt
            type_instructions = {
                ResponseType.AGREE: ("YES/accept", "Yeah I'm down, Sure, Definitely"),
                ResponseType.DECLINE: ("NO/decline politely", "Can't sorry, Nah I'm busy"),
                ResponseType.DEFER: ("UNSURE/need to check", "Let me check, Maybe, I'll see"),
            }

            # Format: ask for all needed types in one prompt
            type_lines = []
            for rt in types_needed:
                instr, examples = type_instructions.get(rt, (rt.value, ""))
                type_lines.append(f"{rt.value}: ({instr}) e.g. {examples}")

            prompt = f"""Generate short iMessage replies for this message. One reply for each type.

Message: "{trigger}"

Rules for ALL replies:
- Maximum 5 words each
- Casual texting style
- NO emojis, NO "Hey!"
- Just the reply text

Generate exactly these replies:
{chr(10).join(type_lines)}

Replies:"""

            request = GenerationRequest(
                prompt=prompt,
                context_documents=[],
                few_shot_examples=[],
                max_tokens=80,  # Enough for 3 short responses
                temperature=0.7,
            )

            response = self.generator.generate(request)
            text = response.text.strip()

            # Parse the response - look for TYPE: reply patterns
            results: dict[ResponseType, ResponseOption] = {}
            lines = text.split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                for rt in types_needed:
                    # Look for patterns like "AGREE: yes" or "AGREE - yes"
                    prefixes = [f"{rt.value}:", f"{rt.value} -", f"{rt.value.lower()}:"]
                    for prefix in prefixes:
                        if line.lower().startswith(prefix.lower()):
                            reply_text = line[len(prefix):].strip()
                            reply_text = self._cleanup_generated_text(reply_text)

                            if reply_text and len(reply_text) >= 2 and rt not in results:
                                results[rt] = ResponseOption(
                                    text=reply_text,
                                    response_type=rt,
                                    confidence=0.7,
                                    source="generated_batch",
                                )
                            break

            logger.debug(
                "Batched generation: needed %d types, got %d",
                len(types_needed), len(results)
            )
            return results

        except Exception as e:
            logger.warning("Batched LLM generation failed: %s", e)
            return {}

    def _generate_option(
        self,
        trigger: str,
        response_type: ResponseType,
        examples: list,
        contact_name: str | None = None,
    ) -> ResponseOption | None:
        """Generate a single option using LLM (fallback for when batching fails).

        Args:
            trigger: Trigger text.
            response_type: Target response type.
            examples: Few-shot examples.
            contact_name: Optional contact name for personalization.

        Returns:
            ResponseOption or None if generation fails.
        """
        if not self.generator:
            return None

        try:
            from contracts.models import GenerationRequest

            # Build few-shot examples from retrieved data
            few_shot = []
            for ex in examples[:3]:
                if ex.similarity > 0.3:  # Only use relevant examples
                    few_shot.append((ex.trigger_text, ex.response_text))

            # Type-specific instructions and inline examples
            type_config = {
                ResponseType.AGREE: {
                    "desc": "saying YES, accepting the invitation",
                    "examples": ["Yeah I'm down", "Sure", "Yes", "Definitely", "Let's do it"],
                    "tone": "enthusiastic but brief",
                },
                ResponseType.DECLINE: {
                    "desc": "saying NO, politely declining",
                    "examples": ["Can't today sorry", "Nah I'm busy", "Not gonna work", "I'll pass"],
                    "tone": "apologetic but brief",
                },
                ResponseType.DEFER: {
                    "desc": "being UNSURE, saying you need to check",
                    "examples": ["Let me check", "Maybe", "I'll see", "Not sure yet"],
                    "tone": "non-committal",
                },
            }

            config = type_config.get(response_type, {
                "desc": response_type.value.lower(),
                "examples": [],
                "tone": "casual",
            })

            # Build concise prompt optimized for short responses
            prompt = f"""Generate a short iMessage reply that is {config["desc"]}.

Message: "{trigger}"

Rules:
- Maximum 5-7 words
- Casual texting style (lowercase ok)
- NO emojis
- NO greetings like "Hey!"
- Just the response, nothing else

Examples of {response_type.value} responses: {", ".join(config["examples"])}

Reply:"""

            request = GenerationRequest(
                prompt=prompt,
                context_documents=[],
                few_shot_examples=few_shot if few_shot else [],
                max_tokens=30,  # Shorter to avoid rambling
                temperature=0.8,  # Some variety
            )

            response = self.generator.generate(request)
            text = response.text.strip()

            # Cleanup: remove quotes, "Hey!", emojis, etc.
            text = self._cleanup_generated_text(text)

            if not text or len(text) < 2:
                return None

            return ResponseOption(
                text=text,
                response_type=response_type,
                confidence=0.7,
                source="generated",
            )

        except Exception as e:
            logger.warning("LLM generation failed for %s: %s", response_type.value, e)
            return None

    def _cleanup_generated_text(self, text: str) -> str:
        """Clean up LLM-generated text to match iMessage style."""
        import re

        # Remove surrounding quotes
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]

        # Remove common LLM prefixes
        prefixes_to_remove = [
            r"^(Hey!?\s*)",
            r"^(Hi!?\s*)",
            r"^(Sure!?\s*,?\s*)",  # Keep "Sure" alone, remove "Sure, ..."
            r"^(Here'?s? (a |my )?(casual |short )?response:?\s*)",
            r"^(Response:?\s*)",
            r"^(Reply:?\s*)",
        ]
        for pattern in prefixes_to_remove:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove emojis (common unicode ranges)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)

        # Remove trailing punctuation cleanup artifacts
        text = text.strip()
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace

        # Truncate if too long (keep first sentence)
        if len(text) > 50:
            # Find first sentence end
            for end in [".", "!", "?"]:
                idx = text.find(end)
                if 0 < idx < 50:
                    text = text[: idx + 1]
                    break
            else:
                text = text[:50].rsplit(" ", 1)[0]  # Cut at word boundary

        return text.strip()

    def generate_options(
        self,
        trigger: str,
        contact_name: str | None = None,
        contact: Contact | None = None,
        force_commitment: bool = False,
    ) -> MultiOptionResult:
        """Generate diverse response options for a trigger.

        Args:
            trigger: Trigger message text.
            contact_name: Optional contact name for personalization.
            contact: Optional Contact object with style info.
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
        options: list[ResponseOption] = []
        types_needing_generation: list[ResponseType] = []

        # Phase 1: Try templates first (fast path)
        for response_type in OPTION_PRIORITY:
            if len(options) >= self._max_options:
                break

            examples = multi_examples.get_examples(response_type)

            # Strategy 1: Try template from examples
            if examples:
                option = self._get_template_option(trigger, response_type, examples)
                if option and option.confidence >= 0.5:
                    options.append(option)
                    continue

            # Mark for LLM generation
            if self._use_llm:
                types_needing_generation.append(response_type)

        # Phase 2: Batched LLM generation (1 call instead of N)
        if types_needing_generation and self._use_llm:
            # Collect examples for batched generation
            examples_for_batch = {
                rt: multi_examples.get_examples(rt)
                for rt in types_needing_generation
            }

            # Single batched LLM call for all needed types
            batched_results = self._generate_all_options_batched(
                trigger=trigger,
                types_needed=types_needing_generation,
                examples_by_type=examples_for_batch,
                contact_name=contact_name,
            )

            # Add successful batched results
            for response_type in types_needing_generation:
                if response_type in batched_results:
                    options.append(batched_results[response_type])
                    types_needing_generation.remove(response_type)

            # Phase 3: Individual generation for any that failed batching
            for response_type in types_needing_generation:
                examples = multi_examples.get_examples(response_type)
                option = self._generate_option(
                    trigger, response_type, examples, contact_name
                )
                if option:
                    options.append(option)
                else:
                    # Strategy 4: Fallback template
                    options.append(self._get_fallback_option(response_type))

        # Phase 4: Fallback for any remaining types (if LLM disabled)
        types_covered = {opt.response_type for opt in options}
        for response_type in OPTION_PRIORITY:
            if len(options) >= self._max_options:
                break
            if response_type not in types_covered:
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
) -> MultiOptionResult:
    """Generate response options for a trigger.

    Convenience function that uses the singleton generator.

    Args:
        trigger: Trigger message text.
        contact_name: Optional contact name.

    Returns:
        MultiOptionResult with diverse options.
    """
    return get_multi_option_generator().generate_options(
        trigger=trigger,
        contact_name=contact_name,
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
