"""Fact verification using NLI cross-encoders.

Replaces expensive LLM self-correction passes with lightweight, high-speed
logical entailment checks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jarvis.nlp.entailment import verify_entailment_batch

if TYPE_CHECKING:
    from jarvis.contacts.contact_profile import Fact

logger = logging.getLogger(__name__)


class FactVerifier:
    """Verifies logical consistency of extracted facts against source text."""

    def __init__(self, threshold: float = 0.05) -> None:
        """Initialize verifier.

        Args:
            threshold: Minimum entailment score to keep a fact (hard rejection).
                       Defaults to 0.05 to catch obvious hallucinations while
                       being permissive of natural language variations.
        """
        self.threshold = threshold

    def verify_facts_batch(
        self,
        candidate_data: list[tuple[Fact, str]],
    ) -> tuple[list[Fact], int]:
        """Verify multiple facts against their respective source segments in one batch.

        Args:
            candidate_data: List of (Fact object, source_segment_text) pairs.

        Returns:
            Tuple of (verified_facts, rejection_count).
        """
        if not candidate_data:
            return [], 0

        # Prepare all pairs for NLI in one go
        pairs = [(text, fact.value) for fact, text in candidate_data]

        try:
            results = verify_entailment_batch(pairs, threshold=self.threshold)
        except Exception as e:
            logger.error(f"Batch NLI verification failed: {e}")
            return [f for f, _ in candidate_data], 0

        verified_facts = []
        rejection_count = 0
        for (fact, _), (is_entailed, score) in zip(candidate_data, results):
            if is_entailed:
                nli_multiplier = 0.3 + (0.7 * score)
                fact.confidence = round(fact.confidence * nli_multiplier, 3)
                verified_facts.append(fact)
            else:
                rejection_count += 1
                logger.debug("Fact rejected by NLI (score=%.3f): %s", score, fact.value)

        return verified_facts, rejection_count

    def verify_facts(
        self,
        facts: list[Fact],
        segment_text: str,
    ) -> tuple[list[Fact], int]:
        """Verify a list of facts against their source segment text.

        Args:
            facts: List of Fact objects to verify.
            segment_text: The source conversation text.

        Returns:
            Tuple of (verified_facts, rejection_count).
        """
        if not facts:
            return [], 0

        # Prepare pairs for NLI: (premise, hypothesis)
        pairs = []
        for fact in facts:
            pairs.append((segment_text, fact.value))

        try:
            results = verify_entailment_batch(pairs, threshold=self.threshold)
        except Exception as e:
            logger.error(f"NLI verification failed: {e}")
            return facts, 0

        verified_facts = []
        rejection_count = 0
        for fact, (is_entailed, score) in zip(facts, results):
            if is_entailed:
                nli_multiplier = 0.3 + (0.7 * score)
                fact.confidence = round(fact.confidence * nli_multiplier, 3)
                verified_facts.append(fact)
            else:
                rejection_count += 1
                logger.debug(f"Fact rejected by NLI (score={score:.3f}): {fact.value}")

        return verified_facts, rejection_count
