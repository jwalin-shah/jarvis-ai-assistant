"""Multi-model ensemble hallucination detection.

Provides robust hallucination detection by combining multiple signals:
1. HHEM (Hallucination Evaluation Model) scores
2. NLI (Natural Language Inference) entailment checking
3. Semantic similarity between source and response
4. Token overlap analysis

Target: 95% hallucination detection rate, <100ms quality check latency.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


class HallucinationSeverity(str, Enum):
    """Severity levels for hallucination detection."""

    NONE = "none"  # No hallucination detected
    LOW = "low"  # Minor factual drift
    MEDIUM = "medium"  # Moderate hallucination
    HIGH = "high"  # Significant hallucination
    CRITICAL = "critical"  # Complete fabrication


@dataclass
class HallucinationResult:
    """Result of hallucination detection analysis."""

    # Overall hallucination score (0=grounded, 1=hallucinated)
    hallucination_score: float
    # Individual model scores
    hhem_score: float | None = None  # HHEM score (0=hallucinated, 1=grounded)
    nli_score: float | None = None  # NLI entailment score
    similarity_score: float | None = None  # Semantic similarity
    overlap_score: float | None = None  # Token overlap

    # Severity classification
    severity: HallucinationSeverity = HallucinationSeverity.NONE
    # Confidence in the result (0-1)
    confidence: float = 1.0
    # Whether the response passes quality gate (hallucination_score <= threshold)
    passes_gate: bool = True
    # Latency of the check in milliseconds
    latency_ms: float = 0.0
    # Specific issues identified
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "hallucination_score": round(self.hallucination_score, 4),
            "hhem_score": round(self.hhem_score, 4) if self.hhem_score else None,
            "nli_score": round(self.nli_score, 4) if self.nli_score else None,
            "similarity_score": round(self.similarity_score, 4) if self.similarity_score else None,
            "overlap_score": round(self.overlap_score, 4) if self.overlap_score else None,
            "severity": self.severity.value,
            "confidence": round(self.confidence, 4),
            "passes_gate": self.passes_gate,
            "latency_ms": round(self.latency_ms, 2),
            "issues": self.issues,
        }


@runtime_checkable
class HallucinationDetector(Protocol):
    """Protocol for hallucination detection implementations."""

    def detect(self, source: str, response: str) -> HallucinationResult:
        """Detect hallucination in a response given source context.

        Args:
            source: Source text/context the response should be grounded in
            response: Generated response to check

        Returns:
            HallucinationResult with scores and classification
        """
        ...

    def detect_batch(self, pairs: list[tuple[str, str]]) -> list[HallucinationResult]:
        """Batch detect hallucinations for multiple source/response pairs.

        Args:
            pairs: List of (source, response) tuples

        Returns:
            List of HallucinationResult objects
        """
        ...


class TokenOverlapAnalyzer:
    """Analyzes token overlap between source and response.

    A lightweight, fast check that doesn't require ML models.
    """

    def __init__(self, min_token_length: int = 3) -> None:
        """Initialize the analyzer.

        Args:
            min_token_length: Minimum token length to consider
        """
        self._min_token_length = min_token_length

    def compute_overlap(self, source: str, response: str) -> float:
        """Compute token overlap score.

        Args:
            source: Source text
            response: Response text

        Returns:
            Overlap score from 0 (no overlap) to 1 (complete overlap)
        """
        source_tokens = self._tokenize(source)
        response_tokens = self._tokenize(response)

        if not response_tokens:
            return 1.0  # Empty response is technically not hallucinating

        if not source_tokens:
            return 0.0  # No source = no grounding

        # Calculate Jaccard-like overlap
        overlap = source_tokens & response_tokens
        return len(overlap) / len(response_tokens)

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize text into lowercase words."""
        import re

        words = re.findall(r"\b\w+\b", text.lower())
        return {w for w in words if len(w) >= self._min_token_length}


class SemanticSimilarityChecker:
    """Checks semantic similarity between source and response.

    Uses sentence embeddings for fast similarity computation.
    """

    def __init__(self) -> None:
        """Initialize the similarity checker."""
        self._embedder: object | None = None
        self._lock = threading.Lock()

    def _ensure_embedder(self) -> object:
        """Lazy load embedder."""
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    try:
                        from jarvis.embedding_adapter import get_embedder

                        self._embedder = get_embedder()
                    except Exception as e:
                        logger.warning("Failed to load embedder: %s", e)
                        self._embedder = None
        return self._embedder

    def compute_similarity(self, source: str, response: str) -> float | None:
        """Compute semantic similarity between source and response.

        Args:
            source: Source text
            response: Response text

        Returns:
            Similarity score from 0 to 1, or None if unavailable
        """
        embedder = self._ensure_embedder()
        if embedder is None:
            return None

        try:
            # Get embeddings
            embeddings = embedder.encode([source, response])
            source_emb = embeddings[0]
            response_emb = embeddings[1]

            # Cosine similarity
            dot_product = np.dot(source_emb, response_emb)
            norm_product = np.linalg.norm(source_emb) * np.linalg.norm(response_emb)

            if norm_product == 0:
                return 0.0

            similarity = float(dot_product / norm_product)
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        except Exception as e:
            logger.warning("Similarity computation failed: %s", e)
            return None


class NLIEntailmentChecker:
    """Checks NLI entailment between source and response.

    Uses a lightweight NLI model to check if the response is entailed
    by the source text.
    """

    def __init__(self) -> None:
        """Initialize the NLI checker."""
        self._model: object | None = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> object | None:
        """Lazy load NLI model."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import CrossEncoder

                        # Use a small, fast NLI model
                        self._model = CrossEncoder(
                            "cross-encoder/nli-MiniLM2-L6-H768",
                            max_length=512,
                        )
                    except Exception as e:
                        logger.warning("Failed to load NLI model: %s", e)
                        self._model = None
        return self._model

    def check_entailment(self, source: str, response: str) -> float | None:
        """Check if response is entailed by source.

        Args:
            source: Source/premise text
            response: Response/hypothesis text

        Returns:
            Entailment score from 0 (contradiction) to 1 (entailment),
            or None if unavailable
        """
        model = self._ensure_model()
        if model is None:
            return None

        try:
            # NLI models expect (premise, hypothesis) pairs
            # Returns [contradiction, neutral, entailment] logits
            scores = model.predict([(source, response)])[0]

            # Softmax to get probabilities
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Return entailment probability (index 2)
            return float(probs[2])
        except Exception as e:
            logger.warning("NLI check failed: %s", e)
            return None


class HHEMScorer:
    """HHEM (Hallucination Evaluation Model) scorer.

    Uses the Vectara HHEM model for hallucination detection.
    """

    def __init__(self) -> None:
        """Initialize the HHEM scorer."""
        self._model: object | None = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> object | None:
        """Lazy load HHEM model."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import CrossEncoder

                        self._model = CrossEncoder("vectara/hallucination_evaluation_model")
                    except Exception as e:
                        logger.warning("Failed to load HHEM model: %s", e)
                        self._model = None
        return self._model

    def score(self, source: str, response: str) -> float | None:
        """Score hallucination using HHEM.

        Args:
            source: Source text
            response: Response text

        Returns:
            Score from 0 (hallucinated) to 1 (grounded), or None if unavailable
        """
        model = self._ensure_model()
        if model is None:
            return None

        try:
            scores = model.predict([[source, response]])
            return float(scores[0])
        except Exception as e:
            logger.warning("HHEM scoring failed: %s", e)
            return None

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float | None]:
        """Batch score hallucinations.

        Args:
            pairs: List of (source, response) tuples

        Returns:
            List of scores or None for failures
        """
        model = self._ensure_model()
        if model is None:
            return [None] * len(pairs)

        try:
            batch_pairs = [[src, resp] for src, resp in pairs]
            scores = model.predict(batch_pairs)
            return [float(s) for s in scores]
        except Exception as e:
            logger.warning("HHEM batch scoring failed: %s", e)
            return [None] * len(pairs)


class EnsembleHallucinationDetector:
    """Ensemble hallucination detector combining multiple signals.

    Combines HHEM, NLI, semantic similarity, and token overlap for
    robust hallucination detection with high recall (95%+ target).
    """

    # Weights for ensemble scoring
    DEFAULT_WEIGHTS = {
        "hhem": 0.4,  # Primary signal
        "nli": 0.3,  # Strong semantic signal
        "similarity": 0.2,  # Supporting signal
        "overlap": 0.1,  # Lightweight fallback
    }

    # Severity thresholds (hallucination_score)
    SEVERITY_THRESHOLDS = {
        HallucinationSeverity.NONE: 0.2,
        HallucinationSeverity.LOW: 0.4,
        HallucinationSeverity.MEDIUM: 0.6,
        HallucinationSeverity.HIGH: 0.8,
        # Above 0.8 = CRITICAL
    }

    def __init__(
        self,
        gate_threshold: float = 0.5,
        weights: dict[str, float] | None = None,
        enable_hhem: bool = True,
        enable_nli: bool = True,
        enable_similarity: bool = True,
        enable_overlap: bool = True,
    ) -> None:
        """Initialize the ensemble detector.

        Args:
            gate_threshold: Maximum hallucination score to pass gate (default 0.5)
            weights: Custom weights for ensemble (must sum to 1.0)
            enable_hhem: Enable HHEM scoring
            enable_nli: Enable NLI entailment checking
            enable_similarity: Enable semantic similarity
            enable_overlap: Enable token overlap analysis
        """
        self._gate_threshold = gate_threshold
        self._weights = weights or self.DEFAULT_WEIGHTS.copy()

        self._enable_hhem = enable_hhem
        self._enable_nli = enable_nli
        self._enable_similarity = enable_similarity
        self._enable_overlap = enable_overlap

        # Initialize component detectors lazily
        self._hhem: HHEMScorer | None = None
        self._nli: NLIEntailmentChecker | None = None
        self._similarity: SemanticSimilarityChecker | None = None
        self._overlap: TokenOverlapAnalyzer | None = None

    def _get_hhem(self) -> HHEMScorer:
        """Get or create HHEM scorer."""
        if self._hhem is None:
            self._hhem = HHEMScorer()
        return self._hhem

    def _get_nli(self) -> NLIEntailmentChecker:
        """Get or create NLI checker."""
        if self._nli is None:
            self._nli = NLIEntailmentChecker()
        return self._nli

    def _get_similarity(self) -> SemanticSimilarityChecker:
        """Get or create similarity checker."""
        if self._similarity is None:
            self._similarity = SemanticSimilarityChecker()
        return self._similarity

    def _get_overlap(self) -> TokenOverlapAnalyzer:
        """Get or create overlap analyzer."""
        if self._overlap is None:
            self._overlap = TokenOverlapAnalyzer()
        return self._overlap

    def detect(self, source: str, response: str) -> HallucinationResult:
        """Detect hallucination using ensemble of methods.

        Args:
            source: Source text/context
            response: Generated response to check

        Returns:
            HallucinationResult with ensemble scores and classification
        """
        start_time = time.perf_counter()
        scores: dict[str, float | None] = {}
        issues: list[str] = []

        # Collect scores from each enabled detector
        if self._enable_hhem:
            scores["hhem"] = self._get_hhem().score(source, response)
            if scores["hhem"] is not None and scores["hhem"] < 0.5:
                issues.append("Low HHEM grounding score")

        if self._enable_nli:
            scores["nli"] = self._get_nli().check_entailment(source, response)
            if scores["nli"] is not None and scores["nli"] < 0.3:
                issues.append("Response not entailed by source")

        if self._enable_similarity:
            scores["similarity"] = self._get_similarity().compute_similarity(source, response)
            if scores["similarity"] is not None and scores["similarity"] < 0.3:
                issues.append("Low semantic similarity to source")

        if self._enable_overlap:
            scores["overlap"] = self._get_overlap().compute_overlap(source, response)
            if scores["overlap"] is not None and scores["overlap"] < 0.1:
                issues.append("Minimal token overlap with source")

        # Compute ensemble hallucination score
        hallucination_score, confidence = self._compute_ensemble_score(scores)

        # Classify severity
        severity = self._classify_severity(hallucination_score)

        # Check gate
        passes_gate = hallucination_score <= self._gate_threshold

        latency_ms = (time.perf_counter() - start_time) * 1000

        return HallucinationResult(
            hallucination_score=hallucination_score,
            hhem_score=scores.get("hhem"),
            nli_score=scores.get("nli"),
            similarity_score=scores.get("similarity"),
            overlap_score=scores.get("overlap"),
            severity=severity,
            confidence=confidence,
            passes_gate=passes_gate,
            latency_ms=latency_ms,
            issues=issues,
        )

    def detect_batch(self, pairs: list[tuple[str, str]]) -> list[HallucinationResult]:
        """Batch detect hallucinations for efficiency.

        Args:
            pairs: List of (source, response) tuples

        Returns:
            List of HallucinationResult objects
        """
        if not pairs:
            return []

        start_time = time.perf_counter()
        n = len(pairs)

        # Batch HHEM scoring
        hhem_scores: list[float | None] = [None] * n
        if self._enable_hhem:
            hhem_scores = self._get_hhem().score_batch(pairs)

        # Individual scoring for other methods (can be parallelized in future)
        results: list[HallucinationResult] = []

        for i, (source, response) in enumerate(pairs):
            pair_start = time.perf_counter()
            scores: dict[str, float | None] = {"hhem": hhem_scores[i]}
            issues: list[str] = []

            if scores["hhem"] is not None and scores["hhem"] < 0.5:
                issues.append("Low HHEM grounding score")

            if self._enable_nli:
                scores["nli"] = self._get_nli().check_entailment(source, response)
                if scores["nli"] is not None and scores["nli"] < 0.3:
                    issues.append("Response not entailed by source")

            if self._enable_similarity:
                scores["similarity"] = self._get_similarity().compute_similarity(source, response)
                if scores["similarity"] is not None and scores["similarity"] < 0.3:
                    issues.append("Low semantic similarity to source")

            if self._enable_overlap:
                scores["overlap"] = self._get_overlap().compute_overlap(source, response)
                if scores["overlap"] is not None and scores["overlap"] < 0.1:
                    issues.append("Minimal token overlap with source")

            hallucination_score, confidence = self._compute_ensemble_score(scores)
            severity = self._classify_severity(hallucination_score)
            passes_gate = hallucination_score <= self._gate_threshold

            pair_latency = (time.perf_counter() - pair_start) * 1000

            results.append(
                HallucinationResult(
                    hallucination_score=hallucination_score,
                    hhem_score=scores.get("hhem"),
                    nli_score=scores.get("nli"),
                    similarity_score=scores.get("similarity"),
                    overlap_score=scores.get("overlap"),
                    severity=severity,
                    confidence=confidence,
                    passes_gate=passes_gate,
                    latency_ms=pair_latency,
                    issues=issues,
                )
            )

        total_latency = (time.perf_counter() - start_time) * 1000
        logger.debug(
            "Batch hallucination detection for %d pairs: %.2fms total",
            n,
            total_latency,
        )

        return results

    def _compute_ensemble_score(self, scores: dict[str, float | None]) -> tuple[float, float]:
        """Compute weighted ensemble hallucination score.

        Args:
            scores: Dictionary of individual scores

        Returns:
            Tuple of (hallucination_score, confidence)
        """
        weighted_sum = 0.0
        total_weight = 0.0
        available_count = 0

        for key, weight in self._weights.items():
            score = scores.get(key)
            if score is not None:
                # Convert grounding scores (higher=better) to hallucination scores
                if key in ("hhem", "nli", "similarity", "overlap"):
                    halluc_score = 1.0 - score
                else:
                    halluc_score = score

                weighted_sum += halluc_score * weight
                total_weight += weight
                available_count += 1

        if total_weight == 0:
            # No scores available - conservative default
            return 0.5, 0.0

        hallucination_score = weighted_sum / total_weight

        # Confidence based on available signals
        max_signals = len(self._weights)
        confidence = available_count / max_signals

        return hallucination_score, confidence

    def _classify_severity(self, score: float) -> HallucinationSeverity:
        """Classify hallucination severity from score.

        Args:
            score: Hallucination score (0=grounded, 1=hallucinated)

        Returns:
            HallucinationSeverity enum value
        """
        if score <= self.SEVERITY_THRESHOLDS[HallucinationSeverity.NONE]:
            return HallucinationSeverity.NONE
        elif score <= self.SEVERITY_THRESHOLDS[HallucinationSeverity.LOW]:
            return HallucinationSeverity.LOW
        elif score <= self.SEVERITY_THRESHOLDS[HallucinationSeverity.MEDIUM]:
            return HallucinationSeverity.MEDIUM
        elif score <= self.SEVERITY_THRESHOLDS[HallucinationSeverity.HIGH]:
            return HallucinationSeverity.HIGH
        else:
            return HallucinationSeverity.CRITICAL


# Global singleton instance
_hallucination_detector: EnsembleHallucinationDetector | None = None
_detector_lock = threading.Lock()


def get_hallucination_detector(
    gate_threshold: float = 0.5,
) -> EnsembleHallucinationDetector:
    """Get the global hallucination detector instance.

    Args:
        gate_threshold: Maximum hallucination score to pass gate

    Returns:
        Shared EnsembleHallucinationDetector instance
    """
    global _hallucination_detector
    if _hallucination_detector is None:
        with _detector_lock:
            if _hallucination_detector is None:
                _hallucination_detector = EnsembleHallucinationDetector(
                    gate_threshold=gate_threshold
                )
    return _hallucination_detector


def reset_hallucination_detector() -> None:
    """Reset the global hallucination detector instance."""
    global _hallucination_detector
    with _detector_lock:
        _hallucination_detector = None
