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

    # Gate B: Embeddings
    embedding_threshold: float = 0.75  # Lowered from 0.8 for better recall

    # Gate C: NLI
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
            nli_model: NLICrossEncoder instance (or None).
            config: GateConfig instance (or None).
        """
        self._embedder = embedder
        self._nli_model = nli_model
        self.config = config or GateConfig()
        self._context_embeddings: list | None = None

    def set_context(self, context_texts: list[str]) -> None:
        """Set conversation context for embedding-based validation.

        Args:
            context_texts: List of previous messages in conversation.
        """
        if self._embedder is None:
            return

        try:
            self._context_embeddings = self._embedder.encode(context_texts)
        except Exception:
            self._context_embeddings = None

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

    def _gate_b_embeddings(self, text: str) -> tuple[bool, str]:
        """Apply Gate B - embedding similarity check.

        Checks if the candidate text is too similar to recent context (duplicate)
        or too different (topic shift).

        Args:
            text: Candidate text to validate.

        Returns:
            Tuple of (passed, reason).
        """
        if self._embedder is None or self._context_embeddings is None:
            # Skip Gate B if no embedder available
            return True, "passed"

        try:
            candidate_emb = self._embedder.encode([text])
            if candidate_emb is None or len(candidate_emb) == 0:
                return True, "passed"

            candidate_vec = candidate_emb[0]

            # Check similarity with each context message
            for ctx_emb in self._context_embeddings:
                # Cosine similarity (both normalized)
                similarity = float(candidate_vec @ ctx_emb)

                # Reject if too similar (duplicate/near-duplicate)
                if similarity > 0.95:
                    return False, "too_similar_to_context"

            # Passed embedding check
            return True, "passed"

        except Exception:
            # Fail open on errors
            return True, "passed"

    def _gate_c_nli(self, text: str) -> tuple[bool, str]:
        """Apply Gate C - NLI cross-encoder for entailment.

        Uses NLI to detect if candidate contradicts recent messages
        (topic shift detection).

        Args:
            text: Candidate text to validate.

        Returns:
            Tuple of (passed, reason).
        """
        if self._nli_model is None:
            # Skip Gate C if no NLI model available
            return True, "passed"

        # Get recent context for contradiction check
        context_texts = []
        if hasattr(self, '_context_texts'):
            context_texts = self._context_texts[-3:] if len(self._context_texts) > 3 else self._context_texts

        if not context_texts:
            return True, "passed"

        try:
            # Check entailment with each context message
            for ctx_text in context_texts:
                result = self._nli_model.predict(ctx_text, text)
                entailment_score = result.get("entailment", 0.0)

                # If candidate contradicts context (low entailment, high contradiction)
                if entailment_score < self.config.nli_threshold:
                    # Check for contradiction
                    contradiction_score = result.get("contradiction", 0.0)
                    if contradiction_score > 0.7:
                        return False, "contradicts_context"

            return True, "passed"

        except Exception:
            # Fail open on errors
            return True, "passed"

    def validate(self, text: str, context_texts: list[str] | None = None) -> tuple[bool, str]:
        """Validate a candidate exchange using all gates.

        Args:
            text: Candidate text to validate.
            context_texts: Optional conversation context for Gates B & C.

        Returns:
            Tuple of (passed, reason). passed is True if valid, False if rejected.
            reason provides a string explanation for rejection or "passed".
        """
        # Store context for gate B & C
        if context_texts:
            self._context_texts = context_texts
            # Compute context embeddings if embedder available
            if self._embedder is not None:
                self.set_context(context_texts)

        # Gate A: Rule-based
        passed_a, reason_a = self._gate_a_rules(text)
        if not passed_a:
            return False, reason_a

        # Gate B: Embedding similarity
        passed_b, reason_b = self._gate_b_embeddings(text)
        if not passed_b:
            return False, reason_b

        # Gate C: NLI cross-encoder
        passed_c, reason_c = self._gate_c_nli(text)
        if not passed_c:
            return False, reason_c

        return True, "passed"
