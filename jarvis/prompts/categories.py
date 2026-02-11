"""Category configuration and mapping for reply routing.

Contains category-to-behavior mapping, template responses for
acknowledge/closing categories, and utility functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jarvis.prompts.examples import (
    CASUAL_REPLY_EXAMPLES,
    PROMPT_LAST_UPDATED,
    PROMPT_VERSION,
    SUMMARIZATION_EXAMPLES,
    PromptMetadata,
)
from jarvis.prompts.templates import MAX_PROMPT_TOKENS

if TYPE_CHECKING:
    from jarvis.classifiers.response_mobilization import MobilizationResult


# System prompt for chat-based reply generation
CHAT_SYSTEM_PROMPT = (
    "Generate a casual text message reply. One short sentence max. "
    'No AI phrases like "I understand" or "Let me know". '
    "No emojis unless they used them first."
)


# Category mapping: map old/runtime signals to new 6-category schema
CATEGORY_MAP: dict[str, str] = {
    # Old 5-category schema -> new 6-category
    "ack": "acknowledge",
    "info": "request",
    "emotional": "emotion",
    "social": "statement",
    # Old 4-category schema -> new 6-category
    "brief": "request",
    "warm": "emotion",
    # Runtime signals -> new categories
    "quick_exchange": "acknowledge",
    "logistics": "request",
    "emotional_support": "emotion",
    "catching_up": "statement",
    "planning": "request",
    # Tone -> category defaults
    "casual": "statement",
    "professional": "request",
    "mixed": "statement",
    "celebration": "social",
    # Ambiguous / low context
    "clarify": "clarify",
    "edge_case": "clarify",
    "unknown": "clarify",
    "information": "clarify",
}


def resolve_category(
    last_message: str,
    context: list[str] | None = None,
    tone: str = "casual",
    mobilization: MobilizationResult | None = None,
) -> str:
    """Classify a message into an optimization category.

    Uses the SVM-based category classifier when available, falling back
    to the static CATEGORY_MAP for graceful degradation.

    Args:
        last_message: The incoming message text.
        context: Recent conversation messages.
        tone: Detected tone (casual/professional/mixed).
        mobilization: Pre-computed mobilization result.

    Returns:
        Optimization category name.
    """
    try:
        from jarvis.classifiers.category_classifier import classify_category

        result = classify_category(last_message, context=context, mobilization=mobilization)
        return result.category
    except Exception:
        # Graceful fallback to static mapping
        return CATEGORY_MAP.get(tone, "statement")


@dataclass
class CategoryConfig:
    """Configuration for a message category.

    Attributes:
        skip_slm: If True, skip SLM generation (use template response).
        prompt: Prompt template key to use (if not skipping SLM).
        context_depth: Number of messages to include in context.
        system_prompt: Optional category-specific system prompt.
    """

    skip_slm: bool
    prompt: str | None
    context_depth: int
    system_prompt: str | None = None


# Category configurations (maps category -> routing behavior)
CATEGORY_CONFIGS: dict[str, CategoryConfig] = {
    "closing": CategoryConfig(
        skip_slm=True,
        prompt=None,
        context_depth=0,
        system_prompt=None,
    ),
    "acknowledge": CategoryConfig(
        skip_slm=True,
        prompt=None,
        context_depth=0,
        system_prompt=None,
    ),
    "question": CategoryConfig(
        skip_slm=False,
        prompt="reply_generation",
        context_depth=5,
        system_prompt="They asked a question. Just answer it, keep it short.",
    ),
    "request": CategoryConfig(
        skip_slm=False,
        prompt="reply_generation",
        context_depth=5,
        system_prompt="They're asking you to do something. Say yes, no, or ask a follow-up.",
    ),
    "emotion": CategoryConfig(
        skip_slm=False,
        prompt="reply_generation",
        context_depth=3,
        system_prompt="They're sharing something emotional. Be a good friend, not a therapist.",
    ),
    "statement": CategoryConfig(
        skip_slm=False,
        prompt="reply_generation",
        context_depth=3,
        system_prompt="They're sharing or chatting. React naturally like a friend would.",
    ),
}


def get_category_config(category: str) -> CategoryConfig:
    """Get routing configuration for a category.

    Args:
        category: Category name (closing, acknowledge, question, request, emotion, statement).

    Returns:
        CategoryConfig for the category, or default (statement) if unknown.
    """
    return CATEGORY_CONFIGS.get(category, CATEGORY_CONFIGS["statement"])


# Template responses for acknowledge/closing categories (when skip_slm=True)
ACKNOWLEDGE_TEMPLATES: list[str] = [
    "ok",
    "sounds good",
    "got it",
    "thanks",
    "np",
    "\U0001f44d",
    "for sure",
    "alright",
    "bet",
    "cool",
]

CLOSING_TEMPLATES: list[str] = [
    "bye!",
    "see ya",
    "later!",
    "talk soon",
    "ttyl",
    "peace",
    "catch you later",
    "gn",
]


# =============================================================================
# Utility Functions
# =============================================================================


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text.

    Uses a simple heuristic of ~4 characters per token.
    This is an approximation; actual token counts vary by model.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


def is_within_token_limit(prompt: str, limit: int = MAX_PROMPT_TOKENS) -> bool:
    """Check if a prompt is within the token limit.

    Args:
        prompt: The prompt to check
        limit: Maximum allowed tokens (default: MAX_PROMPT_TOKENS)

    Returns:
        True if prompt is within limit, False otherwise
    """
    return estimate_tokens(prompt) <= limit


# =============================================================================
# Compatibility Exports
# =============================================================================

# Convert FewShotExample lists to tuple format for GenerationRequest compatibility
REPLY_EXAMPLES: list[tuple[str, str]] = [(ex.context, ex.output) for ex in CASUAL_REPLY_EXAMPLES]

SUMMARY_EXAMPLES: list[tuple[str, str]] = SUMMARIZATION_EXAMPLES


# =============================================================================
# API-Style Prompt Examples
# =============================================================================

# These examples use the instruction-based format for API endpoints
# (e.g., the drafts router). They include explicit instructions.

API_REPLY_EXAMPLES_METADATA = PromptMetadata(
    name="api_reply_examples",
    version=PROMPT_VERSION,
    last_updated=PROMPT_LAST_UPDATED,
    description="Few-shot examples for API reply generation with explicit instructions",
)

API_REPLY_EXAMPLES: list[tuple[str, str]] = [
    (
        "Last message: 'Hey, are you free for dinner tomorrow?'\n"
        "Instruction: accept enthusiastically",
        "Yes, absolutely! I'd love to! What time works for you?",
    ),
    (
        "Last message: 'Can you review this document by EOD?'\n"
        "Instruction: confirm and ask for details",
        "Sure, I can take a look. Which sections should I focus on?",
    ),
    (
        "Last message: 'Thanks for your help yesterday!'\nInstruction: None",
        "You're welcome! Happy I could help.",
    ),
]

API_SUMMARY_EXAMPLES_METADATA = PromptMetadata(
    name="api_summary_examples",
    version=PROMPT_VERSION,
    last_updated=PROMPT_LAST_UPDATED,
    description="Few-shot examples for API conversation summarization",
)

API_SUMMARY_EXAMPLES: list[tuple[str, str]] = [
    (
        "Conversation about planning a birthday party with 5 messages "
        "discussing date, venue, and guest list.",
        "Summary: Planning discussion for a birthday party.\n"
        "Key points:\n- Deciding on date and venue\n- Creating guest list",
    ),
]
