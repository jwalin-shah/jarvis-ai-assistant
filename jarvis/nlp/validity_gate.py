"""Validity gate implementation for candidate exchanges.

This module provides a three-layer validation system for candidate exchanges:
    Gate A: Pure rule-based rejection (no embeddings)
    Gate B: Embedding similarity with topic-shift penalty
    Gate C: NLI cross-encoder for borderline cases (optional)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jarvis.text_normalizer import (
    is_acknowledgment_only,
    is_emoji_only,
    is_garbage_message,
    is_reaction,
    is_spam_message,
)


@dataclass
class GateConfig:
    """Configuration for validity gates."""

    # Gate A: Rule-based
    reject_reactions: bool = True
    reject_acknowledgments: bool = True
    reject_garbage: bool = True
    reject_spam: bool = True
    reject_emojis: bool = True

    # Gate B: Embeddings (placeholders)
    embedding_threshold: float = 0.8

    # Gate C: NLI (placeholders)
    nli_threshold: float = 0.6


class ValidityGate:
    """Three-layer validation for candidate exchanges.

    Gate A: Pure rule-based rejection (no embeddings)
    Gate B: Embedding similarity with topic-shift penalty
    Gate C: NLI cross-encoder for borderline cases (optional)
    """

    def __init__(
        self,
        embedder: Any = None,
        nli_model: Any = None,
        config: GateConfig | None = None,
    ) -> None:
        """Initialize the validity gate.

        Args:
            embedder: UnifiedEmbedder instance (or None).
            nli_model: NLI model instance (or None).
            config: GateConfig instance (or None).
        """
        self.embedder = embedder
        self.nli_model = nli_model
        self.config = config or GateConfig()

    def _gate_a_rules(self, text: str) -> tuple[bool, str]:
        """Apply Gate A rules (pure rule-based rejection).

        Args:
            text: Candidate text to validate.

        Returns:
            Tuple of (passed, reason). passed is True if valid, False if rejected.
            reason provides a string explanation for rejection or "passed".
        """
        if not text or not text.strip():
            return False, "empty_text"

        if self.config.reject_reactions and is_reaction(text):
            return False, "reaction"

        if self.config.reject_garbage and is_garbage_message(text):
            return False, "garbage"

        if self.config.reject_spam and is_spam_message(text):
            return False, "spam"

        if self.config.reject_emojis and is_emoji_only(text):
            return False, "emoji_only"

        if self.config.reject_acknowledgments and is_acknowledgment_only(text):
            return False, "acknowledgment"

        return True, "passed"

    def validate(self, text: str) -> tuple[bool, str]:
        """Validate a candidate exchange using all gates.

        Args:
            text: Candidate text to validate.

        Returns:
            Tuple of (passed, reason). passed is True if valid, False if rejected.
            reason provides a string explanation for rejection or "passed".
        """
        # Gate A: Rule-based
        passed_a, reason_a = self._gate_a_rules(text)
        if not passed_a:
            return False, reason_a

        # Gate B & C (stubs for now as dependencies are missing)
        # TODO: Implement Gate B (embeddings) and Gate C (NLI)
        return True, "passed"
