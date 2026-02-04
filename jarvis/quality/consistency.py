"""Self-consistency checking for response generation.

Ensures responses are internally consistent and consistent with
prior responses in the conversation, detecting logical contradictions
and maintaining coherent persona.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class InconsistencyType(str, Enum):
    """Types of inconsistencies that can be detected."""

    SELF_CONTRADICTION = "self_contradiction"  # Response contradicts itself
    TEMPORAL = "temporal"  # Time-related inconsistency
    FACTUAL = "factual"  # Factual inconsistency with prior responses
    TONAL = "tonal"  # Inconsistent tone/style
    PERSONA = "persona"  # Inconsistent personality/role
    NUMERIC = "numeric"  # Conflicting numbers


@dataclass
class InconsistencyIssue:
    """A detected inconsistency issue."""

    inconsistency_type: InconsistencyType
    description: str
    severity: float  # 0 (minor) to 1 (severe)
    text_span_1: str | None = None  # First conflicting text
    text_span_2: str | None = None  # Second conflicting text


@dataclass
class ConsistencyResult:
    """Result of consistency checking."""

    # Overall consistency score (0=inconsistent, 1=consistent)
    consistency_score: float
    # Whether response is self-consistent (internal)
    is_self_consistent: bool = True
    # Whether response is consistent with history
    is_history_consistent: bool = True
    # Detected issues
    issues: list[InconsistencyIssue] = field(default_factory=list)
    # Whether response passes consistency gate
    passes_gate: bool = True
    # Latency in milliseconds
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "consistency_score": round(self.consistency_score, 4),
            "is_self_consistent": self.is_self_consistent,
            "is_history_consistent": self.is_history_consistent,
            "passes_gate": self.passes_gate,
            "latency_ms": round(self.latency_ms, 2),
            "issues": [
                {
                    "type": issue.inconsistency_type.value,
                    "description": issue.description,
                    "severity": round(issue.severity, 2),
                    "span_1": issue.text_span_1,
                    "span_2": issue.text_span_2,
                }
                for issue in self.issues
            ],
        }


class SelfConsistencyChecker:
    """Checks internal consistency of a single response.

    Detects self-contradictions within the same response.
    """

    def check(self, response: str) -> tuple[bool, list[InconsistencyIssue]]:
        """Check self-consistency of response.

        Args:
            response: Response text to check

        Returns:
            Tuple of (is_consistent, list of issues)
        """
        issues: list[InconsistencyIssue] = []

        # Split into sentences
        sentences = self._split_sentences(response)

        if len(sentences) < 2:
            return True, issues

        # Check for contradictory statements
        for i, sent1 in enumerate(sentences):
            for sent2 in sentences[i + 1 :]:
                contradiction = self._check_contradiction(sent1, sent2)
                if contradiction:
                    issues.append(contradiction)

        # Check for numeric consistency
        numeric_issues = self._check_numeric_consistency(sentences)
        issues.extend(numeric_issues)

        # Check for temporal consistency
        temporal_issues = self._check_temporal_consistency(sentences)
        issues.extend(temporal_issues)

        is_consistent = len(issues) == 0
        return is_consistent, issues

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _check_contradiction(self, sent1: str, sent2: str) -> InconsistencyIssue | None:
        """Check if two sentences contradict each other."""
        s1_lower = sent1.lower()
        s2_lower = sent2.lower()

        # Pattern: "X is Y" vs "X is not Y" or "X isn't Y"
        is_pattern = r"(\w+)\s+is\s+(\w+)"
        is_not_pattern = r"(\w+)\s+(?:is not|isn't)\s+(\w+)"

        m1_is = re.search(is_pattern, s1_lower)
        m2_is_not = re.search(is_not_pattern, s2_lower)

        if m1_is and m2_is_not:
            if m1_is.group(1) == m2_is_not.group(1) and m1_is.group(2) == m2_is_not.group(2):
                return InconsistencyIssue(
                    inconsistency_type=InconsistencyType.SELF_CONTRADICTION,
                    description=f"Contradictory statements about '{m1_is.group(1)}'",
                    severity=0.8,
                    text_span_1=sent1,
                    text_span_2=sent2,
                )

        # Pattern: "will" vs "won't" for same subject
        will_pattern = r"(\w+)\s+will\s+(\w+)"
        wont_pattern = r"(\w+)\s+(?:will not|won't)\s+(\w+)"

        m1_will = re.search(will_pattern, s1_lower)
        m2_wont = re.search(wont_pattern, s2_lower)

        if m1_will and m2_wont:
            if m1_will.group(1) == m2_wont.group(1) and m1_will.group(2) == m2_wont.group(2):
                return InconsistencyIssue(
                    inconsistency_type=InconsistencyType.SELF_CONTRADICTION,
                    description=f"Contradictory future statements about '{m1_will.group(1)}'",
                    severity=0.7,
                    text_span_1=sent1,
                    text_span_2=sent2,
                )

        return None

    def _check_numeric_consistency(self, sentences: list[str]) -> list[InconsistencyIssue]:
        """Check for inconsistent numbers across sentences."""
        issues: list[InconsistencyIssue] = []

        # Extract number contexts: {context: [(number, sentence)]}
        number_contexts: dict[str, list[tuple[str, str]]] = {}

        for sentence in sentences:
            # Find numbers with surrounding context
            matches = re.finditer(r"(\w+\s+)?(\d+)(\s+\w+)?", sentence.lower())
            for match in matches:
                number = match.group(2)
                # Get context words
                before = match.group(1) or ""
                after = match.group(3) or ""
                context = f"{before.strip()}_{after.strip()}"

                if context not in number_contexts:
                    number_contexts[context] = []
                number_contexts[context].append((number, sentence))

        # Check for conflicting numbers in same context
        for context, occurrences in number_contexts.items():
            if len(occurrences) < 2:
                continue

            numbers = {num for num, _ in occurrences}
            if len(numbers) > 1:
                issues.append(
                    InconsistencyIssue(
                        inconsistency_type=InconsistencyType.NUMERIC,
                        description=f"Conflicting numbers in similar context: {numbers}",
                        severity=0.6,
                        text_span_1=occurrences[0][1],
                        text_span_2=occurrences[1][1],
                    )
                )

        return issues

    def _check_temporal_consistency(self, sentences: list[str]) -> list[InconsistencyIssue]:
        """Check for temporal inconsistencies."""
        issues: list[InconsistencyIssue] = []

        # Track tense usage
        past_sentences: list[str] = []
        future_sentences: list[str] = []

        past_markers = ["was", "were", "had", "did", "went", "came", "yesterday", "ago"]
        future_markers = ["will", "shall", "going to", "tomorrow", "next"]

        for sentence in sentences:
            s_lower = sentence.lower()
            is_past = any(marker in s_lower for marker in past_markers)
            is_future = any(marker in s_lower for marker in future_markers)

            if is_past:
                past_sentences.append(sentence)
            if is_future:
                future_sentences.append(sentence)

        # Check for conflicting time references to same event
        for past_sent in past_sentences:
            for future_sent in future_sentences:
                # Check if referring to same subject/event
                past_words = set(past_sent.lower().split())
                future_words = set(future_sent.lower().split())

                # Filter common words
                common = {"the", "a", "an", "is", "was", "will", "be", "to", "and", "or"}
                past_words -= common
                future_words -= common

                overlap = past_words & future_words
                if len(overlap) >= 2:  # Likely same topic
                    issues.append(
                        InconsistencyIssue(
                            inconsistency_type=InconsistencyType.TEMPORAL,
                            description="Conflicting past and future references to same topic",
                            severity=0.5,
                            text_span_1=past_sent,
                            text_span_2=future_sent,
                        )
                    )

        return issues


class HistoryConsistencyChecker:
    """Checks consistency with conversation history.

    Ensures responses don't contradict prior statements.
    """

    def __init__(self) -> None:
        """Initialize the history checker."""
        self._embedder: object | None = None
        self._lock = threading.Lock()

    def _ensure_embedder(self) -> object | None:
        """Lazy load embedder."""
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    try:
                        from jarvis.embeddings import get_embedder

                        self._embedder = get_embedder()
                    except Exception as e:
                        logger.warning("Failed to load embedder: %s", e)
        return self._embedder

    def check(
        self,
        response: str,
        history: list[str],
    ) -> tuple[bool, list[InconsistencyIssue]]:
        """Check consistency with conversation history.

        Args:
            response: Current response
            history: Prior messages/responses

        Returns:
            Tuple of (is_consistent, list of issues)
        """
        if not history:
            return True, []

        issues: list[InconsistencyIssue] = []

        # Check factual consistency
        factual_issues = self._check_factual_consistency(response, history)
        issues.extend(factual_issues)

        # Check persona consistency
        persona_issues = self._check_persona_consistency(response, history)
        issues.extend(persona_issues)

        is_consistent = len(issues) == 0
        return is_consistent, issues

    def _check_factual_consistency(
        self,
        response: str,
        history: list[str],
    ) -> list[InconsistencyIssue]:
        """Check for factual inconsistencies with history."""
        issues: list[InconsistencyIssue] = []

        # Extract factual statements from response and history
        response_facts = self._extract_facts(response)
        history_text = " ".join(history)
        history_facts = self._extract_facts(history_text)

        # Check for contradictions
        for r_fact in response_facts:
            for h_fact in history_facts:
                if self._facts_contradict(r_fact, h_fact):
                    issues.append(
                        InconsistencyIssue(
                            inconsistency_type=InconsistencyType.FACTUAL,
                            description="Response contradicts prior statement",
                            severity=0.7,
                            text_span_1=h_fact,
                            text_span_2=r_fact,
                        )
                    )

        return issues

    def _extract_facts(self, text: str) -> list[str]:
        """Extract factual statements from text."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        facts: list[str] = []

        for sentence in sentences:
            s = sentence.strip()
            if not s:
                continue

            # Filter out questions and commands
            if s.endswith("?"):
                continue
            if s.startswith(("Please", "Could you", "Would you")):
                continue

            # Keep declarative statements
            if any(
                pattern in s.lower()
                for pattern in ["is ", "was ", "are ", "were ", "has ", "have ", "had "]
            ):
                facts.append(s)

        return facts

    def _facts_contradict(self, fact1: str, fact2: str) -> bool:
        """Check if two facts contradict each other."""
        f1_lower = fact1.lower()
        f2_lower = fact2.lower()

        # Check for negation patterns
        # "X is Y" vs "X is not Y"
        is_match = re.search(r"(\w+)\s+is\s+(\w+)", f1_lower)
        is_not_match = re.search(r"(\w+)\s+(?:is not|isn't)\s+(\w+)", f2_lower)

        if is_match and is_not_match:
            if is_match.group(1) == is_not_match.group(1):
                if is_match.group(2) == is_not_match.group(2):
                    return True

        # Check for conflicting numbers with same subject
        f1_numbers = re.findall(r"(\w+)\s+(\d+)", f1_lower)
        f2_numbers = re.findall(r"(\w+)\s+(\d+)", f2_lower)

        for word1, num1 in f1_numbers:
            for word2, num2 in f2_numbers:
                if word1 == word2 and num1 != num2:
                    return True

        return False

    def _check_persona_consistency(
        self,
        response: str,
        history: list[str],
    ) -> list[InconsistencyIssue]:
        """Check for persona/style consistency."""
        issues: list[InconsistencyIssue] = []

        # Analyze formality level
        response_formality = self._measure_formality(response)
        history_formality = (
            sum(
                self._measure_formality(h)
                for h in history[-5:]  # Last 5 messages
            )
            / max(len(history[-5:]), 1)
        )

        # Large formality shift is inconsistent
        if abs(response_formality - history_formality) > 0.4:
            issues.append(
                InconsistencyIssue(
                    inconsistency_type=InconsistencyType.TONAL,
                    description="Significant formality shift from prior messages",
                    severity=0.3,
                )
            )

        return issues

    def _measure_formality(self, text: str) -> float:
        """Measure formality level of text (0=informal, 1=formal)."""
        text_lower = text.lower()

        # Informal markers
        informal_markers = [
            "hey",
            "hi",
            "gonna",
            "wanna",
            "gotta",
            "lol",
            "haha",
            "yeah",
            "yep",
            "nope",
            "cool",
            "awesome",
            "!",
            "?",
        ]

        # Formal markers
        formal_markers = [
            "please",
            "thank you",
            "would you",
            "could you",
            "regards",
            "sincerely",
            "appreciate",
            "furthermore",
            "however",
            "therefore",
        ]

        informal_count = sum(1 for m in informal_markers if m in text_lower)
        formal_count = sum(1 for m in formal_markers if m in text_lower)

        total = informal_count + formal_count
        if total == 0:
            return 0.5  # Neutral

        return formal_count / total


class ConsistencyChecker:
    """Main consistency checker combining self and history checks.

    Provides comprehensive consistency validation for responses.
    """

    CONSISTENCY_GATE_THRESHOLD = 0.7

    def __init__(self, gate_threshold: float | None = None) -> None:
        """Initialize the consistency checker.

        Args:
            gate_threshold: Minimum consistency score to pass gate
        """
        self._gate_threshold = gate_threshold or self.CONSISTENCY_GATE_THRESHOLD
        self._self_checker = SelfConsistencyChecker()
        self._history_checker = HistoryConsistencyChecker()

    def check_consistency(
        self,
        response: str,
        history: list[str] | None = None,
    ) -> ConsistencyResult:
        """Check consistency of response.

        Args:
            response: Response to check
            history: Optional prior messages for history consistency

        Returns:
            ConsistencyResult with scores and issues
        """
        start_time = time.perf_counter()
        all_issues: list[InconsistencyIssue] = []

        # Self-consistency check
        is_self_consistent, self_issues = self._self_checker.check(response)
        all_issues.extend(self_issues)

        # History consistency check
        is_history_consistent = True
        if history:
            is_history_consistent, history_issues = self._history_checker.check(response, history)
            all_issues.extend(history_issues)

        # Calculate consistency score
        if not all_issues:
            consistency_score = 1.0
        else:
            # Weight issues by severity
            total_severity = sum(issue.severity for issue in all_issues)
            # Normalize: max reasonable severity sum is ~3 (3 severe issues)
            consistency_score = max(0.0, 1.0 - (total_severity / 3.0))

        passes_gate = consistency_score >= self._gate_threshold

        latency_ms = (time.perf_counter() - start_time) * 1000

        return ConsistencyResult(
            consistency_score=consistency_score,
            is_self_consistent=is_self_consistent,
            is_history_consistent=is_history_consistent,
            issues=all_issues,
            passes_gate=passes_gate,
            latency_ms=latency_ms,
        )


# Global singleton
_consistency_checker: ConsistencyChecker | None = None
_checker_lock = threading.Lock()


def get_consistency_checker(gate_threshold: float = 0.7) -> ConsistencyChecker:
    """Get the global consistency checker instance.

    Args:
        gate_threshold: Minimum consistency score to pass gate

    Returns:
        Shared ConsistencyChecker instance
    """
    global _consistency_checker
    if _consistency_checker is None:
        with _checker_lock:
            if _consistency_checker is None:
                _consistency_checker = ConsistencyChecker(gate_threshold=gate_threshold)
    return _consistency_checker


def reset_consistency_checker() -> None:
    """Reset the global consistency checker instance."""
    global _consistency_checker
    with _checker_lock:
        _consistency_checker = None
