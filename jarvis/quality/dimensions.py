"""Quality dimension scoring for response evaluation.

Provides multi-dimensional quality assessment covering:
- Factual accuracy
- Coherence
- Relevance
- Tone appropriateness
- Length appropriateness
- Personalization accuracy
"""

from __future__ import annotations

import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class QualityDimension(str, Enum):
    """Quality dimensions for response evaluation."""

    FACTUAL = "factual"  # Factual accuracy (grounded in context)
    COHERENCE = "coherence"  # Logical flow and clarity
    RELEVANCE = "relevance"  # On-topic, addresses the query
    TONE = "tone"  # Appropriate tone for context
    LENGTH = "length"  # Appropriate length
    PERSONALIZATION = "personalization"  # Tailored to recipient


@dataclass
class QualityDimensionResult:
    """Result for a single quality dimension."""

    dimension: QualityDimension
    score: float  # 0 to 1
    confidence: float = 1.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension.value,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "issues": self.issues,
            "suggestions": self.suggestions,
        }


@dataclass
class MultiDimensionResult:
    """Combined result for all quality dimensions."""

    # Individual dimension results
    results: dict[QualityDimension, QualityDimensionResult] = field(default_factory=dict)
    # Overall weighted score
    overall_score: float = 0.0
    # Whether all dimensions pass their thresholds
    passes_gate: bool = True
    # Dimensions that failed
    failed_dimensions: list[QualityDimension] = field(default_factory=list)
    # Latency in milliseconds
    latency_ms: float = 0.0

    def get_score(self, dimension: QualityDimension) -> float | None:
        """Get score for a specific dimension."""
        result = self.results.get(dimension)
        return result.score if result else None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "overall_score": round(self.overall_score, 4),
            "passes_gate": self.passes_gate,
            "failed_dimensions": [d.value for d in self.failed_dimensions],
            "latency_ms": round(self.latency_ms, 2),
            "dimensions": {dim.value: result.to_dict() for dim, result in self.results.items()},
        }


class QualityDimensionScorer(ABC):
    """Abstract base class for quality dimension scorers."""

    dimension: QualityDimension
    default_threshold: float = 0.5

    @abstractmethod
    def score(
        self,
        response: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> QualityDimensionResult:
        """Score the response on this dimension.

        Args:
            response: Response to score
            context: Optional context (query, conversation history)
            **kwargs: Additional dimension-specific parameters

        Returns:
            QualityDimensionResult with score and details
        """
        ...


class FactualScorer(QualityDimensionScorer):
    """Scores factual accuracy of responses.

    Evaluates whether the response is grounded in the provided context
    and doesn't make unsubstantiated claims.
    """

    dimension = QualityDimension.FACTUAL
    default_threshold = 0.6

    def __init__(self) -> None:
        """Initialize the factual scorer."""
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

    def score(
        self,
        response: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> QualityDimensionResult:
        """Score factual accuracy."""
        issues: list[str] = []
        suggestions: list[str] = []

        if not context:
            # No context to verify against
            return QualityDimensionResult(
                dimension=self.dimension,
                score=0.5,  # Neutral - can't verify
                confidence=0.3,
                issues=["No context provided for factual verification"],
                suggestions=["Provide source context for better factual scoring"],
            )

        # Compute semantic similarity to context
        embedder = self._ensure_embedder()
        if embedder is None:
            # Fall back to keyword overlap
            score = self._keyword_factual_score(response, context)
            return QualityDimensionResult(
                dimension=self.dimension,
                score=score,
                confidence=0.6,
                issues=issues,
                suggestions=suggestions,
            )

        try:
            embeddings = embedder.encode([response, context])
            similarity = self._cosine_similarity(embeddings[0], embeddings[1])

            # Adjust score based on similarity
            score = max(0.0, min(1.0, similarity))

            if score < 0.4:
                issues.append("Response appears to diverge significantly from source context")
                suggestions.append("Ensure response is grounded in the conversation")
            elif score < 0.6:
                issues.append("Moderate factual alignment with context")

            return QualityDimensionResult(
                dimension=self.dimension,
                score=score,
                confidence=0.8,
                issues=issues,
                suggestions=suggestions,
            )
        except Exception as e:
            logger.warning("Factual scoring failed: %s", e)
            return QualityDimensionResult(
                dimension=self.dimension,
                score=0.5,
                confidence=0.3,
                issues=["Factual scoring unavailable"],
            )

    def _keyword_factual_score(self, response: str, context: str) -> float:
        """Fallback keyword-based factual scoring."""
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())

        # Filter short words
        response_words = {w for w in response_words if len(w) > 3}
        context_words = {w for w in context_words if len(w) > 3}

        if not response_words:
            return 0.5

        overlap = response_words & context_words
        return len(overlap) / len(response_words)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity."""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if norm_product == 0:
            return 0.0
        return float(dot_product / norm_product)


class CoherenceScorer(QualityDimensionScorer):
    """Scores coherence and logical flow of responses.

    Evaluates sentence-level coherence, logical transitions,
    and overall clarity.
    """

    dimension = QualityDimension.COHERENCE
    default_threshold = 0.6

    # Transition words that indicate good flow
    TRANSITION_WORDS = {
        "however",
        "therefore",
        "furthermore",
        "additionally",
        "moreover",
        "consequently",
        "nevertheless",
        "although",
        "because",
        "since",
        "thus",
        "hence",
        "accordingly",
        "also",
        "finally",
        "first",
        "second",
        "third",
        "next",
        "then",
        "meanwhile",
        "similarly",
    }

    def score(
        self,
        response: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> QualityDimensionResult:
        """Score coherence and logical flow."""
        issues: list[str] = []
        suggestions: list[str] = []

        sentences = self._split_sentences(response)

        if len(sentences) == 0:
            return QualityDimensionResult(
                dimension=self.dimension,
                score=0.0,
                confidence=1.0,
                issues=["Empty response"],
            )

        if len(sentences) == 1:
            # Single sentence - check basic quality
            score = self._score_single_sentence(sentences[0])
            return QualityDimensionResult(
                dimension=self.dimension,
                score=score,
                confidence=0.7,
                issues=issues,
                suggestions=suggestions,
            )

        # Multi-sentence coherence checks
        scores: list[float] = []

        # Check for transition words (good flow indicator)
        transition_score = self._score_transitions(response)
        scores.append(transition_score)
        if transition_score < 0.3:
            suggestions.append("Consider using transition words for better flow")

        # Check for subject consistency
        subject_score = self._score_subject_consistency(sentences)
        scores.append(subject_score)
        if subject_score < 0.4:
            issues.append("Inconsistent subjects across sentences")

        # Check for sentence length variation (too uniform = robotic)
        length_var_score = self._score_length_variation(sentences)
        scores.append(length_var_score)
        if length_var_score < 0.3:
            suggestions.append("Vary sentence length for more natural flow")

        # Check for repetition (bad coherence signal)
        repetition_score = self._score_no_repetition(sentences)
        scores.append(repetition_score)
        if repetition_score < 0.5:
            issues.append("Excessive word or phrase repetition detected")

        overall_score = sum(scores) / len(scores) if scores else 0.5

        return QualityDimensionResult(
            dimension=self.dimension,
            score=overall_score,
            confidence=0.8,
            issues=issues,
            suggestions=suggestions,
        )

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _score_single_sentence(self, sentence: str) -> float:
        """Score a single sentence for basic quality."""
        words = sentence.split()

        # Very short sentences are often incomplete
        if len(words) < 3:
            return 0.4

        # Very long sentences may be run-ons
        if len(words) > 40:
            return 0.6

        # Check for capitalization
        if not sentence[0].isupper():
            return 0.5

        # Check for ending punctuation
        if sentence.rstrip()[-1] not in ".!?":
            return 0.7

        return 0.9

    def _score_transitions(self, text: str) -> float:
        """Score use of transition words."""
        text_lower = text.lower()
        transition_count = sum(1 for w in self.TRANSITION_WORDS if w in text_lower)
        word_count = len(text.split())

        if word_count < 20:
            return 0.7  # Short text doesn't need transitions

        # Ideal: ~1 transition per 15-20 words
        ideal_transitions = word_count / 17
        ratio = min(transition_count, ideal_transitions) / ideal_transitions

        return min(1.0, ratio + 0.3)  # Bonus for having any

    def _score_subject_consistency(self, sentences: list[str]) -> float:
        """Score consistency of subjects across sentences."""
        subjects: list[str] = []

        for sentence in sentences:
            words = sentence.split()
            if words:
                # Simple heuristic: first word or second if first is article
                if words[0].lower() in ("the", "a", "an") and len(words) > 1:
                    subjects.append(words[1].lower())
                else:
                    subjects.append(words[0].lower())

        if len(subjects) < 2:
            return 1.0

        # Check for subject continuity
        unique_subjects = set(subjects)
        continuity_score = 1 - (len(unique_subjects) - 1) / len(subjects)

        return max(0.3, continuity_score)

    def _score_length_variation(self, sentences: list[str]) -> float:
        """Score variation in sentence length."""
        lengths = [len(s.split()) for s in sentences]

        if len(lengths) < 2:
            return 0.8

        mean_length = sum(lengths) / len(lengths)
        if mean_length == 0:
            return 0.5

        variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
        std_dev = variance**0.5

        # Ideal coefficient of variation: 0.3-0.5
        cv = std_dev / mean_length if mean_length > 0 else 0

        if cv < 0.1:
            return 0.4  # Too uniform
        elif cv > 0.8:
            return 0.5  # Too varied
        else:
            return 0.8 + (0.2 if 0.2 <= cv <= 0.6 else 0)

    def _score_no_repetition(self, sentences: list[str]) -> float:
        """Score absence of excessive repetition."""
        all_words = []
        for sentence in sentences:
            words = sentence.lower().split()
            all_words.extend([w for w in words if len(w) > 3])

        if not all_words:
            return 0.8

        word_counts: dict[str, int] = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Check for overly repeated words (>20% of content)
        max_repetition = max(word_counts.values()) / len(all_words) if all_words else 0

        if max_repetition > 0.3:
            return 0.3
        elif max_repetition > 0.2:
            return 0.5
        else:
            return 0.9


class RelevanceScorer(QualityDimensionScorer):
    """Scores relevance of response to query/context.

    Evaluates whether the response actually addresses what was asked.
    """

    dimension = QualityDimension.RELEVANCE
    default_threshold = 0.6

    def __init__(self) -> None:
        """Initialize the relevance scorer."""
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

    def score(
        self,
        response: str,
        context: str | None = None,
        query: str | None = None,
        **kwargs: Any,
    ) -> QualityDimensionResult:
        """Score relevance to query/context."""
        issues: list[str] = []
        suggestions: list[str] = []

        # Use query if provided, otherwise use context
        reference = query or context

        if not reference:
            return QualityDimensionResult(
                dimension=self.dimension,
                score=0.5,
                confidence=0.3,
                issues=["No query or context provided for relevance scoring"],
            )

        embedder = self._ensure_embedder()
        if embedder is None:
            # Fall back to keyword matching
            score = self._keyword_relevance_score(response, reference)
        else:
            try:
                embeddings = embedder.encode([response, reference])
                score = self._cosine_similarity(embeddings[0], embeddings[1])
            except Exception as e:
                logger.warning("Relevance scoring failed: %s", e)
                score = self._keyword_relevance_score(response, reference)

        # Adjust based on question-answer alignment
        if reference.endswith("?"):
            qa_score = self._score_question_answer(reference, response)
            score = (score + qa_score) / 2

        if score < 0.4:
            issues.append("Response may not address the query")
            suggestions.append("Ensure the response directly answers what was asked")
        elif score < 0.6:
            issues.append("Moderate relevance to query")

        return QualityDimensionResult(
            dimension=self.dimension,
            score=score,
            confidence=0.8,
            issues=issues,
            suggestions=suggestions,
        )

    def _keyword_relevance_score(self, response: str, reference: str) -> float:
        """Fallback keyword-based relevance scoring."""
        response_words = set(response.lower().split())
        reference_words = set(reference.lower().split())

        # Filter common words
        common = {"the", "a", "an", "is", "are", "was", "were", "to", "and", "or", "of", "in"}
        response_words -= common
        reference_words -= common

        if not reference_words:
            return 0.5

        overlap = response_words & reference_words
        return len(overlap) / len(reference_words)

    def _score_question_answer(self, question: str, answer: str) -> float:
        """Score question-answer alignment."""
        q_lower = question.lower()
        a_lower = answer.lower()

        # Check for question type and appropriate answer
        if q_lower.startswith(("what is", "what are", "what's")):
            # Definition questions should have "is" or descriptive content
            if "is" in a_lower or "are" in a_lower:
                return 0.8
        elif q_lower.startswith(("how do", "how can", "how to")):
            # How-to questions should have instructions
            if any(w in a_lower for w in ["first", "then", "next", "step", "you can", "try"]):
                return 0.9
        elif q_lower.startswith(("why", "what's the reason")):
            # Why questions should have explanations
            if any(w in a_lower for w in ["because", "since", "due to", "reason"]):
                return 0.9
        elif q_lower.startswith(("when", "what time")):
            # When questions should have temporal info
            if any(
                pattern in a_lower
                for pattern in [
                    r"\d",
                    "today",
                    "tomorrow",
                    "yesterday",
                    "am",
                    "pm",
                    "morning",
                    "evening",
                    "night",
                    "soon",
                    "later",
                ]
            ):
                return 0.9
        elif q_lower.startswith(("where", "what place")):
            # Where questions should have location info
            if any(w in a_lower for w in ["at", "in", "on", "near", "here", "there"]):
                return 0.8

        return 0.6  # Neutral

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity."""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if norm_product == 0:
            return 0.0
        return float(dot_product / norm_product)


class ToneScorer(QualityDimensionScorer):
    """Scores tone appropriateness for context.

    Evaluates whether the tone matches the expected register
    (formal, casual, empathetic, etc.).
    """

    dimension = QualityDimension.TONE
    default_threshold = 0.5

    # Markers for different tones
    FORMAL_MARKERS = {
        "please",
        "thank you",
        "kindly",
        "would",
        "could",
        "appreciate",
        "regards",
        "sincerely",
        "furthermore",
        "however",
        "therefore",
    }
    CASUAL_MARKERS = {
        "hey",
        "hi",
        "yeah",
        "yep",
        "nope",
        "cool",
        "awesome",
        "lol",
        "haha",
        "gonna",
        "wanna",
        "gotta",
        "sup",
        "btw",
        "tbh",
    }
    EMPATHETIC_MARKERS = {
        "sorry",
        "understand",
        "feel",
        "hope",
        "care",
        "concern",
        "difficult",
        "hard",
        "tough",
        "support",
        "help",
    }

    def score(
        self,
        response: str,
        context: str | None = None,
        expected_tone: str | None = None,
        relationship_type: str | None = None,
        **kwargs: Any,
    ) -> QualityDimensionResult:
        """Score tone appropriateness."""
        issues: list[str] = []
        suggestions: list[str] = []

        # Analyze response tone
        formality = self._measure_formality(response)
        empathy = self._measure_empathy(response)

        # Determine expected tone from context
        if expected_tone:
            expected_formality = 0.8 if expected_tone == "formal" else 0.3
        elif relationship_type:
            expected_formality = self._get_relationship_formality(relationship_type)
        elif context:
            expected_formality = self._infer_formality_from_context(context)
        else:
            expected_formality = 0.5  # Neutral

        # Score based on formality match
        formality_diff = abs(formality - expected_formality)
        formality_score = 1.0 - formality_diff

        # Check for empathy when needed
        empathy_score = 1.0
        if context and self._context_needs_empathy(context):
            if empathy < 0.3:
                issues.append("Response may lack empathy for sensitive context")
                suggestions.append("Consider acknowledging the person's feelings")
                empathy_score = 0.5

        # Combined score
        score = (formality_score * 0.6) + (empathy_score * 0.4)

        if formality_diff > 0.4:
            if formality > expected_formality:
                issues.append("Response may be too formal for this context")
                suggestions.append("Consider using a more casual tone")
            else:
                issues.append("Response may be too casual for this context")
                suggestions.append("Consider using a more professional tone")

        return QualityDimensionResult(
            dimension=self.dimension,
            score=score,
            confidence=0.7,
            issues=issues,
            suggestions=suggestions,
        )

    def _measure_formality(self, text: str) -> float:
        """Measure formality level (0=casual, 1=formal)."""
        text_lower = text.lower()

        formal_count = sum(1 for m in self.FORMAL_MARKERS if m in text_lower)
        casual_count = sum(1 for m in self.CASUAL_MARKERS if m in text_lower)

        total = formal_count + casual_count
        if total == 0:
            return 0.5  # Neutral

        return formal_count / total

    def _measure_empathy(self, text: str) -> float:
        """Measure empathy level in text."""
        text_lower = text.lower()
        empathy_count = sum(1 for m in self.EMPATHETIC_MARKERS if m in text_lower)
        word_count = len(text.split())

        if word_count == 0:
            return 0.0

        return min(1.0, empathy_count * 5 / word_count)

    def _get_relationship_formality(self, relationship: str) -> float:
        """Get expected formality for relationship type."""
        formal_relationships = {"professional", "business", "work", "colleague"}
        casual_relationships = {"friend", "family", "close", "partner"}

        if relationship.lower() in formal_relationships:
            return 0.7
        elif relationship.lower() in casual_relationships:
            return 0.3
        return 0.5

    def _infer_formality_from_context(self, context: str) -> float:
        """Infer expected formality from context."""
        return self._measure_formality(context)

    def _context_needs_empathy(self, context: str) -> bool:
        """Check if context suggests need for empathy."""
        empathy_triggers = [
            "sorry",
            "sad",
            "upset",
            "difficult",
            "hard",
            "struggle",
            "problem",
            "issue",
            "worried",
            "anxious",
            "stress",
            "lost",
            "miss",
            "hurt",
            "pain",
            "sick",
            "ill",
        ]
        context_lower = context.lower()
        return any(trigger in context_lower for trigger in empathy_triggers)


class LengthScorer(QualityDimensionScorer):
    """Scores length appropriateness.

    Evaluates whether response length is appropriate for the context
    and query type.
    """

    dimension = QualityDimension.LENGTH
    default_threshold = 0.5

    # Ideal lengths by response type
    IDEAL_LENGTHS = {
        "greeting": (5, 20),
        "question": (10, 50),
        "explanation": (30, 150),
        "summary": (20, 100),
        "default": (10, 80),
    }

    def score(
        self,
        response: str,
        context: str | None = None,
        response_type: str | None = None,
        **kwargs: Any,
    ) -> QualityDimensionResult:
        """Score length appropriateness."""
        issues: list[str] = []
        suggestions: list[str] = []

        word_count = len(response.split())

        # Determine expected length range
        if response_type and response_type in self.IDEAL_LENGTHS:
            min_len, max_len = self.IDEAL_LENGTHS[response_type]
        elif context:
            min_len, max_len = self._infer_length_from_context(context)
        else:
            min_len, max_len = self.IDEAL_LENGTHS["default"]

        # Score based on length
        if word_count < min_len:
            ratio = word_count / min_len
            score = max(0.3, ratio)
            issues.append("Response may be too brief")
            suggestions.append(f"Consider expanding to at least {min_len} words")
        elif word_count > max_len:
            ratio = max_len / word_count
            score = max(0.4, ratio)
            issues.append("Response may be too long")
            suggestions.append(f"Consider condensing to around {max_len} words")
        else:
            # Perfect length range
            score = 1.0

        return QualityDimensionResult(
            dimension=self.dimension,
            score=score,
            confidence=0.9,
            issues=issues,
            suggestions=suggestions,
        )

    def _infer_length_from_context(self, context: str) -> tuple[int, int]:
        """Infer expected length range from context."""
        context_lower = context.lower()

        # Short responses expected
        if any(
            pattern in context_lower
            for pattern in ["yes or no", "brief", "quick", "short", "one word"]
        ):
            return (3, 20)

        # Longer responses expected
        if any(
            pattern in context_lower
            for pattern in ["explain", "detail", "describe", "elaborate", "why"]
        ):
            return (30, 150)

        # Questions typically need moderate responses
        if context_lower.endswith("?"):
            return (10, 60)

        return self.IDEAL_LENGTHS["default"]


class PersonalizationScorer(QualityDimensionScorer):
    """Scores personalization accuracy.

    Evaluates whether the response is appropriately personalized
    for the recipient (using their name, referencing shared context, etc.).
    """

    dimension = QualityDimension.PERSONALIZATION
    default_threshold = 0.4

    def score(
        self,
        response: str,
        context: str | None = None,
        contact_name: str | None = None,
        contact_preferences: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> QualityDimensionResult:
        """Score personalization accuracy."""
        issues: list[str] = []
        suggestions: list[str] = []
        scores: list[float] = []

        # Check for name usage when appropriate
        if contact_name:
            name_score = self._score_name_usage(response, contact_name, context)
            scores.append(name_score)
            if name_score < 0.5 and context and "?" in context:
                suggestions.append("Consider addressing the person by name")

        # Check for context reference
        if context:
            context_score = self._score_context_reference(response, context)
            scores.append(context_score)
            if context_score < 0.4:
                issues.append("Response doesn't reference conversation context")

        # Check preferences alignment
        if contact_preferences:
            pref_score = self._score_preference_alignment(response, contact_preferences)
            scores.append(pref_score)

        # Compute overall score
        if scores:
            score = sum(scores) / len(scores)
        else:
            score = 0.5  # Neutral when no personalization info available
            issues.append("No personalization information provided")

        return QualityDimensionResult(
            dimension=self.dimension,
            score=score,
            confidence=0.6 if scores else 0.3,
            issues=issues,
            suggestions=suggestions,
        )

    def _score_name_usage(self, response: str, name: str, context: str | None) -> float:
        """Score appropriate use of contact's name."""
        response_lower = response.lower()
        name_lower = name.lower()

        # Check if name is in response
        name_in_response = name_lower in response_lower

        # Determine if name should be used
        # Questions and greetings benefit from name usage
        should_use_name = False
        if context:
            context_lower = context.lower()
            if any(pattern in context_lower for pattern in ["?", "hi", "hey", "hello", "thank"]):
                should_use_name = True

        if should_use_name and name_in_response:
            return 1.0
        elif should_use_name and not name_in_response:
            return 0.4
        elif not should_use_name and name_in_response:
            return 0.8  # Using name is generally good
        else:
            return 0.6  # Neutral

    def _score_context_reference(self, response: str, context: str) -> float:
        """Score reference to conversation context."""
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())

        # Filter common words
        common = {"the", "a", "an", "is", "are", "to", "and", "or", "of", "in", "i", "you"}
        response_words -= common
        context_words -= common

        if not context_words:
            return 0.5

        overlap = response_words & context_words
        overlap_ratio = len(overlap) / len(context_words)

        # Some overlap is good, but not too much (indicates copy)
        if overlap_ratio < 0.1:
            return 0.3  # Too little reference
        elif overlap_ratio > 0.8:
            return 0.5  # Too much copying
        else:
            return 0.8  # Good balance

    def _score_preference_alignment(self, response: str, preferences: dict[str, Any]) -> float:
        """Score alignment with contact preferences."""
        score = 1.0

        # Check formality preference
        if "formality" in preferences:
            pref_formality = preferences["formality"]
            response_formality = self._measure_formality(response)

            if abs(pref_formality - response_formality) > 0.4:
                score *= 0.6

        # Check length preference
        if "preferred_length" in preferences:
            pref_length = preferences["preferred_length"]
            actual_length = len(response.split())

            if pref_length == "short" and actual_length > 30:
                score *= 0.7
            elif pref_length == "long" and actual_length < 20:
                score *= 0.7

        return score

    def _measure_formality(self, text: str) -> float:
        """Measure formality level."""
        formal_markers = {"please", "thank", "would", "could", "appreciate"}
        casual_markers = {"hey", "yeah", "cool", "gonna", "wanna"}

        text_lower = text.lower()
        formal_count = sum(1 for m in formal_markers if m in text_lower)
        casual_count = sum(1 for m in casual_markers if m in text_lower)

        total = formal_count + casual_count
        if total == 0:
            return 0.5
        return formal_count / total


class MultiDimensionScorer:
    """Combines multiple quality dimension scorers.

    Provides comprehensive multi-dimensional quality assessment.
    """

    DEFAULT_WEIGHTS = {
        QualityDimension.FACTUAL: 0.25,
        QualityDimension.COHERENCE: 0.20,
        QualityDimension.RELEVANCE: 0.25,
        QualityDimension.TONE: 0.10,
        QualityDimension.LENGTH: 0.10,
        QualityDimension.PERSONALIZATION: 0.10,
    }

    DEFAULT_THRESHOLDS = {
        QualityDimension.FACTUAL: 0.5,
        QualityDimension.COHERENCE: 0.5,
        QualityDimension.RELEVANCE: 0.5,
        QualityDimension.TONE: 0.4,
        QualityDimension.LENGTH: 0.4,
        QualityDimension.PERSONALIZATION: 0.3,
    }

    def __init__(
        self,
        weights: dict[QualityDimension, float] | None = None,
        thresholds: dict[QualityDimension, float] | None = None,
    ) -> None:
        """Initialize the multi-dimension scorer.

        Args:
            weights: Custom weights for dimensions (must sum to 1.0)
            thresholds: Custom thresholds for dimensions
        """
        self._weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()

        # Initialize scorers
        self._scorers: dict[QualityDimension, QualityDimensionScorer] = {
            QualityDimension.FACTUAL: FactualScorer(),
            QualityDimension.COHERENCE: CoherenceScorer(),
            QualityDimension.RELEVANCE: RelevanceScorer(),
            QualityDimension.TONE: ToneScorer(),
            QualityDimension.LENGTH: LengthScorer(),
            QualityDimension.PERSONALIZATION: PersonalizationScorer(),
        }

    def score_all(
        self,
        response: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> MultiDimensionResult:
        """Score response on all dimensions.

        Args:
            response: Response to score
            context: Optional context
            **kwargs: Additional parameters for specific dimensions

        Returns:
            MultiDimensionResult with all scores
        """
        start_time = time.perf_counter()
        results: dict[QualityDimension, QualityDimensionResult] = {}
        failed_dimensions: list[QualityDimension] = []

        for dimension, scorer in self._scorers.items():
            result = scorer.score(response, context, **kwargs)
            results[dimension] = result

            # Check threshold
            threshold = self._thresholds.get(dimension, 0.5)
            if result.score < threshold:
                failed_dimensions.append(dimension)

        # Calculate weighted overall score
        weighted_sum = sum(results[dim].score * self._weights[dim] for dim in results)
        total_weight = sum(self._weights[dim] for dim in results)
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        passes_gate = len(failed_dimensions) == 0

        latency_ms = (time.perf_counter() - start_time) * 1000

        return MultiDimensionResult(
            results=results,
            overall_score=overall_score,
            passes_gate=passes_gate,
            failed_dimensions=failed_dimensions,
            latency_ms=latency_ms,
        )

    def score_dimension(
        self,
        dimension: QualityDimension,
        response: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> QualityDimensionResult:
        """Score response on a specific dimension.

        Args:
            dimension: Dimension to score
            response: Response to score
            context: Optional context
            **kwargs: Additional parameters

        Returns:
            QualityDimensionResult for the dimension
        """
        scorer = self._scorers.get(dimension)
        if scorer is None:
            raise ValueError(f"Unknown dimension: {dimension}")
        return scorer.score(response, context, **kwargs)
