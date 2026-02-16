"""Centralized generation configuration defaults.

This module contains the single source of truth for all generation parameters.
All other modules should import from here instead of hardcoding values.

Usage:
    from jarvis.prompts.generation_config import (
        DEFAULT_REPETITION_PENALTY,
        DEFAULT_MAX_TOKENS,
        get_generation_config,
    )

    config = get_generation_config(pressure="low")
    result = loader.generate_sync(prompt, **config)
"""

from __future__ import annotations

from typing import Literal

# =============================================================================
# Optimized Generation Defaults (Single Source of Truth)
# =============================================================================
# These values are the result of extensive testing on real iMessage data.
# See: results/OPTIMIZED_SETTINGS.md for methodology and results.

DEFAULT_TEMPERATURE: float = 0.15
"""Slightly higher than 0.1 for naturalness without chaos."""

DEFAULT_TOP_P: float = 0.9
"""Nucleus sampling threshold - good balance of diversity."""

DEFAULT_TOP_K: int = 50
"""Top-k sampling limit."""

DEFAULT_REPETITION_PENALTY: float = 1.15
"""Key parameter: 1.15 prevents echoing while keeping responses natural.

Values tested:
- 1.05: Too low, model echoes input (e.g., "Lmk when u coming" -> "Lmk when u coming!")
- 1.10: Better, occasional echoing
- 1.15: Optimal - no echoing, natural responses
- 1.20: No echoing but can sound stilted
- 1.30+: Unnatural, forces too much variety
"""

DEFAULT_MAX_TOKENS: int = 25
"""Hard constraint on response length.

Eval dataset "ideal" responses are longer, but real texting prefers brevity.
- 50 tokens: Too verbose, model writes paragraphs
- 25 tokens: Good for most replies (1-2 sentences)
- 20 tokens: Better for brief acks
- 15 tokens: Too short for complex answers
"""

DEFAULT_CONTEXT_DEPTH: int = 15
"""Number of conversation turns to include.

- 10: Sometimes misses important context
- 15: Optimal for texting conversations
- 20: More context but increases prompt size ~30%
"""

# =============================================================================
# Pressure-Based Configuration
# =============================================================================


def get_max_tokens_for_pressure(pressure: Literal["none", "low", "medium", "high"]) -> int:
    """Get max tokens based on conversation pressure.

    Args:
        pressure: Conversation pressure level from response mobilization.

    Returns:
        Maximum tokens appropriate for this pressure level.
    """
    return {
        "none": 12,  # Brief acknowledgment
        "low": 15,  # Casual response
        "medium": 20,  # Standard reply
        "high": 25,  # Complex answer (still brief!)
    }.get(pressure, 20)


def get_generation_config(
    pressure: Literal["none", "low", "medium", "high"] = "medium",
    temperature: float | None = None,
    repetition_penalty: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
) -> dict[str, float | int]:
    """Get complete generation config with optimized defaults.

    Args:
        pressure: Conversation pressure level (affects max_tokens)
        temperature: Override default temperature
        repetition_penalty: Override default repetition penalty
        max_tokens: Override pressure-based max tokens
        top_p: Override default top_p

    Returns:
        Dictionary ready to pass to generate_sync()
    """
    return {
        "temperature": temperature if temperature is not None else DEFAULT_TEMPERATURE,
        "repetition_penalty": (
            repetition_penalty if repetition_penalty is not None else DEFAULT_REPETITION_PENALTY
        ),
        "max_tokens": (
            max_tokens if max_tokens is not None else get_max_tokens_for_pressure(pressure)
        ),
        "top_p": top_p if top_p is not None else DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
    }


# =============================================================================
# Category-Specific Context Depths
# =============================================================================

CATEGORY_CONTEXT_DEPTHS: dict[str, int] = {
    "closing": 0,  # No context needed
    "acknowledge": 0,  # No context needed
    "question": 15,  # Need context to answer
    "request": 15,  # Need context to respond
    "emotion": 15,  # Need context to empathize
    "statement": 15,  # Need context to react
}
"""Context depth per message category."""
