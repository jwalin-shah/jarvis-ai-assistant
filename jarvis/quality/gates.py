"""Real-time quality gates for response validation.

Provides pre-send quality checking with configurable thresholds,
automatic rewrite triggers, and user-facing quality indicators.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class GateDecision(str, Enum):
    """Quality gate decision outcomes."""

    PASS = "pass"  # Response passes all checks
    SOFT_FAIL = "soft_fail"  # Marginal quality - warn but allow
    HARD_FAIL = "hard_fail"  # Quality too low - block/rewrite
    SKIP = "skip"  # Check skipped (e.g., disabled or unavailable)


class RewriteAction(str, Enum):
    """Actions to take when quality gate fails."""

    NONE = "none"  # No action needed
    WARN = "warn"  # Show warning to user
    SUGGEST_EDIT = "suggest_edit"  # Suggest specific edits
    AUTO_REWRITE = "auto_rewrite"  # Automatically rewrite
    BLOCK = "block"  # Block the response entirely


@dataclass
class GateCheckResult:
    """Result of a single gate check."""

    gate_name: str
    decision: GateDecision
    score: float
    threshold: float
    soft_threshold: float | None = None
    latency_ms: float = 0.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class QualityGateResult:
    """Combined result of all quality gate checks."""

    # Overall decision
    decision: GateDecision
    # Recommended action
    action: RewriteAction
    # Individual gate results
    gate_results: list[GateCheckResult] = field(default_factory=list)
    # Overall quality score (weighted average)
    quality_score: float = 0.0
    # Whether response should be sent
    should_send: bool = True
    # Whether rewrite is recommended
    needs_rewrite: bool = False
    # Total latency in milliseconds
    latency_ms: float = 0.0
    # All issues aggregated
    all_issues: list[str] = field(default_factory=list)
    # All suggestions aggregated
    all_suggestions: list[str] = field(default_factory=list)
    # Quality indicator for UI
    quality_indicator: str = "good"  # "good", "fair", "poor"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "decision": self.decision.value,
            "action": self.action.value,
            "quality_score": round(self.quality_score, 4),
            "should_send": self.should_send,
            "needs_rewrite": self.needs_rewrite,
            "quality_indicator": self.quality_indicator,
            "latency_ms": round(self.latency_ms, 2),
            "all_issues": self.all_issues,
            "all_suggestions": self.all_suggestions,
            "gates": [
                {
                    "name": gr.gate_name,
                    "decision": gr.decision.value,
                    "score": round(gr.score, 4),
                    "threshold": gr.threshold,
                    "issues": gr.issues,
                }
                for gr in self.gate_results
            ],
        }


@dataclass
class QualityGateConfig:
    """Configuration for quality gates."""

    # Hallucination gate
    hallucination_enabled: bool = True
    hallucination_threshold: float = 0.5  # Max hallucination score
    hallucination_soft_threshold: float = 0.7  # Warn above this

    # Factuality gate
    factuality_enabled: bool = True
    factuality_threshold: float = 0.6  # Min factuality score
    factuality_soft_threshold: float = 0.5  # Warn below this

    # Consistency gate
    consistency_enabled: bool = True
    consistency_threshold: float = 0.6
    consistency_soft_threshold: float = 0.5

    # Grounding gate
    grounding_enabled: bool = True
    grounding_threshold: float = 0.5
    grounding_soft_threshold: float = 0.4

    # Coherence gate
    coherence_enabled: bool = True
    coherence_threshold: float = 0.5
    coherence_soft_threshold: float = 0.4

    # Relevance gate
    relevance_enabled: bool = True
    relevance_threshold: float = 0.5
    relevance_soft_threshold: float = 0.4

    # Auto-rewrite settings
    auto_rewrite_enabled: bool = False  # Disabled by default
    auto_rewrite_max_attempts: int = 2
    min_score_for_rewrite: float = 0.3  # Below this, don't even try

    # Gate weights for overall score
    weights: dict[str, float] = field(default_factory=lambda: {
        "hallucination": 0.25,
        "factuality": 0.20,
        "consistency": 0.15,
        "grounding": 0.15,
        "coherence": 0.15,
        "relevance": 0.10,
    })

    @classmethod
    def strict(cls) -> QualityGateConfig:
        """Create strict quality gate configuration."""
        return cls(
            hallucination_threshold=0.4,
            hallucination_soft_threshold=0.5,
            factuality_threshold=0.7,
            factuality_soft_threshold=0.6,
            consistency_threshold=0.7,
            consistency_soft_threshold=0.6,
            grounding_threshold=0.6,
            grounding_soft_threshold=0.5,
            coherence_threshold=0.6,
            coherence_soft_threshold=0.5,
            relevance_threshold=0.6,
            relevance_soft_threshold=0.5,
        )

    @classmethod
    def lenient(cls) -> QualityGateConfig:
        """Create lenient quality gate configuration."""
        return cls(
            hallucination_threshold=0.7,
            hallucination_soft_threshold=0.8,
            factuality_threshold=0.4,
            factuality_soft_threshold=0.3,
            consistency_threshold=0.4,
            consistency_soft_threshold=0.3,
            grounding_threshold=0.3,
            grounding_soft_threshold=0.2,
            coherence_threshold=0.4,
            coherence_soft_threshold=0.3,
            relevance_threshold=0.4,
            relevance_soft_threshold=0.3,
        )


class QualityGate:
    """Main quality gate for pre-send response validation.

    Combines multiple quality checks into a unified gate that
    determines whether a response is ready to send.

    Target: <100ms quality check latency.
    """

    def __init__(self, config: QualityGateConfig | None = None) -> None:
        """Initialize the quality gate.

        Args:
            config: Gate configuration (uses defaults if not provided)
        """
        self._config = config or QualityGateConfig()

        # Lazy-loaded checkers
        self._hallucination_detector: Any = None
        self._fact_checker: Any = None
        self._consistency_checker: Any = None
        self._grounding_checker: Any = None
        self._dimension_scorer: Any = None

        self._lock = threading.Lock()

    def _get_hallucination_detector(self) -> Any:
        """Lazy load hallucination detector."""
        if self._hallucination_detector is None:
            with self._lock:
                if self._hallucination_detector is None:
                    try:
                        from jarvis.quality.hallucination import (
                            EnsembleHallucinationDetector,
                        )

                        self._hallucination_detector = EnsembleHallucinationDetector(
                            gate_threshold=self._config.hallucination_threshold,
                            # Fast mode: only use overlap for speed
                            enable_hhem=False,  # Slow
                            enable_nli=False,  # Slow
                            enable_similarity=True,
                            enable_overlap=True,
                        )
                    except Exception as e:
                        logger.warning("Failed to load hallucination detector: %s", e)
        return self._hallucination_detector

    def _get_fact_checker(self) -> Any:
        """Lazy load fact checker."""
        if self._fact_checker is None:
            with self._lock:
                if self._fact_checker is None:
                    try:
                        from jarvis.quality.factuality import FactChecker

                        self._fact_checker = FactChecker(
                            gate_threshold=self._config.factuality_threshold
                        )
                    except Exception as e:
                        logger.warning("Failed to load fact checker: %s", e)
        return self._fact_checker

    def _get_consistency_checker(self) -> Any:
        """Lazy load consistency checker."""
        if self._consistency_checker is None:
            with self._lock:
                if self._consistency_checker is None:
                    try:
                        from jarvis.quality.consistency import ConsistencyChecker

                        self._consistency_checker = ConsistencyChecker(
                            gate_threshold=self._config.consistency_threshold
                        )
                    except Exception as e:
                        logger.warning("Failed to load consistency checker: %s", e)
        return self._consistency_checker

    def _get_grounding_checker(self) -> Any:
        """Lazy load grounding checker."""
        if self._grounding_checker is None:
            with self._lock:
                if self._grounding_checker is None:
                    try:
                        from jarvis.quality.grounding import GroundingChecker

                        self._grounding_checker = GroundingChecker(
                            gate_threshold=self._config.grounding_threshold
                        )
                    except Exception as e:
                        logger.warning("Failed to load grounding checker: %s", e)
        return self._grounding_checker

    def _get_dimension_scorer(self) -> Any:
        """Lazy load dimension scorer."""
        if self._dimension_scorer is None:
            with self._lock:
                if self._dimension_scorer is None:
                    try:
                        from jarvis.quality.dimensions import MultiDimensionScorer

                        self._dimension_scorer = MultiDimensionScorer()
                    except Exception as e:
                        logger.warning("Failed to load dimension scorer: %s", e)
        return self._dimension_scorer

    def check(
        self,
        response: str,
        source: str | list[str] | None = None,
        history: list[str] | None = None,
        **kwargs: Any,
    ) -> QualityGateResult:
        """Run all quality checks on a response.

        Args:
            response: Response to check
            source: Source context for grounding
            history: Conversation history for consistency
            **kwargs: Additional parameters for specific gates

        Returns:
            QualityGateResult with decision and details
        """
        start_time = time.perf_counter()
        gate_results: list[GateCheckResult] = []
        all_issues: list[str] = []
        all_suggestions: list[str] = []

        # Normalize source to string
        source_str = (
            " ".join(source) if isinstance(source, list) else (source or "")
        )

        # Run enabled gates
        if self._config.hallucination_enabled and source_str:
            result = self._check_hallucination(response, source_str)
            gate_results.append(result)
            all_issues.extend(result.issues)
            all_suggestions.extend(result.suggestions)

        if self._config.factuality_enabled and source_str:
            result = self._check_factuality(response, source_str)
            gate_results.append(result)
            all_issues.extend(result.issues)
            all_suggestions.extend(result.suggestions)

        if self._config.consistency_enabled:
            result = self._check_consistency(response, history)
            gate_results.append(result)
            all_issues.extend(result.issues)
            all_suggestions.extend(result.suggestions)

        if self._config.grounding_enabled and source_str:
            result = self._check_grounding(response, source_str)
            gate_results.append(result)
            all_issues.extend(result.issues)
            all_suggestions.extend(result.suggestions)

        if self._config.coherence_enabled:
            result = self._check_coherence(response, source_str)
            gate_results.append(result)
            all_issues.extend(result.issues)
            all_suggestions.extend(result.suggestions)

        if self._config.relevance_enabled and source_str:
            result = self._check_relevance(response, source_str)
            gate_results.append(result)
            all_issues.extend(result.issues)
            all_suggestions.extend(result.suggestions)

        # Determine overall decision
        decision, action = self._make_decision(gate_results)

        # Calculate overall quality score
        quality_score = self._calculate_overall_score(gate_results)

        # Determine quality indicator
        if quality_score >= 0.8:
            quality_indicator = "good"
        elif quality_score >= 0.5:
            quality_indicator = "fair"
        else:
            quality_indicator = "poor"

        # Determine send/rewrite flags
        should_send = decision != GateDecision.HARD_FAIL
        needs_rewrite = decision == GateDecision.HARD_FAIL or (
            decision == GateDecision.SOFT_FAIL
            and action in (RewriteAction.SUGGEST_EDIT, RewriteAction.AUTO_REWRITE)
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return QualityGateResult(
            decision=decision,
            action=action,
            gate_results=gate_results,
            quality_score=quality_score,
            should_send=should_send,
            needs_rewrite=needs_rewrite,
            latency_ms=latency_ms,
            all_issues=all_issues,
            all_suggestions=all_suggestions,
            quality_indicator=quality_indicator,
        )

    def check_fast(
        self,
        response: str,
        source: str | None = None,
    ) -> QualityGateResult:
        """Fast quality check using only lightweight checks.

        Designed to complete in <50ms for real-time use.

        Args:
            response: Response to check
            source: Optional source context

        Returns:
            QualityGateResult with quick assessment
        """
        start_time = time.perf_counter()
        gate_results: list[GateCheckResult] = []
        all_issues: list[str] = []

        # Only run fast checks
        if self._config.coherence_enabled:
            result = self._check_coherence(response, source)
            gate_results.append(result)
            all_issues.extend(result.issues)

        # Basic hallucination check (keyword overlap only)
        if source:
            result = self._check_hallucination_fast(response, source)
            gate_results.append(result)
            all_issues.extend(result.issues)

        decision, action = self._make_decision(gate_results)
        quality_score = self._calculate_overall_score(gate_results)

        quality_indicator = (
            "good" if quality_score >= 0.7 else ("fair" if quality_score >= 0.4 else "poor")
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return QualityGateResult(
            decision=decision,
            action=action,
            gate_results=gate_results,
            quality_score=quality_score,
            should_send=decision != GateDecision.HARD_FAIL,
            needs_rewrite=decision == GateDecision.HARD_FAIL,
            latency_ms=latency_ms,
            all_issues=all_issues,
            quality_indicator=quality_indicator,
        )

    def _check_hallucination(
        self, response: str, source: str
    ) -> GateCheckResult:
        """Check hallucination gate."""
        gate_start = time.perf_counter()

        detector = self._get_hallucination_detector()
        if detector is None:
            return GateCheckResult(
                gate_name="hallucination",
                decision=GateDecision.SKIP,
                score=0.5,
                threshold=self._config.hallucination_threshold,
                issues=["Hallucination detector unavailable"],
            )

        try:
            result = detector.detect(source, response)
            score = result.hallucination_score

            # Lower hallucination score is better
            if score <= self._config.hallucination_threshold:
                decision = GateDecision.PASS
            elif score <= self._config.hallucination_soft_threshold:
                decision = GateDecision.SOFT_FAIL
            else:
                decision = GateDecision.HARD_FAIL

            return GateCheckResult(
                gate_name="hallucination",
                decision=decision,
                score=1.0 - score,  # Invert so higher is better
                threshold=1.0 - self._config.hallucination_threshold,
                soft_threshold=1.0 - self._config.hallucination_soft_threshold,
                latency_ms=(time.perf_counter() - gate_start) * 1000,
                issues=result.issues,
            )
        except Exception as e:
            logger.warning("Hallucination check failed: %s", e)
            return GateCheckResult(
                gate_name="hallucination",
                decision=GateDecision.SKIP,
                score=0.5,
                threshold=self._config.hallucination_threshold,
                issues=[f"Check failed: {e}"],
            )

    def _check_hallucination_fast(
        self, response: str, source: str
    ) -> GateCheckResult:
        """Fast hallucination check using keyword overlap only."""
        import re

        gate_start = time.perf_counter()

        source_words = set(re.findall(r"\b\w{4,}\b", source.lower()))
        response_words = set(re.findall(r"\b\w{4,}\b", response.lower()))

        if not response_words:
            overlap = 1.0
        else:
            overlap = len(source_words & response_words) / len(response_words)

        # Higher overlap = lower hallucination risk
        score = overlap

        if score >= 1.0 - self._config.hallucination_threshold:
            decision = GateDecision.PASS
        elif score >= 1.0 - self._config.hallucination_soft_threshold:
            decision = GateDecision.SOFT_FAIL
        else:
            decision = GateDecision.HARD_FAIL

        issues = []
        if decision != GateDecision.PASS:
            issues.append("Low keyword overlap with source (potential hallucination)")

        return GateCheckResult(
            gate_name="hallucination_fast",
            decision=decision,
            score=score,
            threshold=1.0 - self._config.hallucination_threshold,
            latency_ms=(time.perf_counter() - gate_start) * 1000,
            issues=issues,
        )

    def _check_factuality(
        self, response: str, source: str
    ) -> GateCheckResult:
        """Check factuality gate."""
        gate_start = time.perf_counter()

        checker = self._get_fact_checker()
        if checker is None:
            return GateCheckResult(
                gate_name="factuality",
                decision=GateDecision.SKIP,
                score=0.5,
                threshold=self._config.factuality_threshold,
                issues=["Fact checker unavailable"],
            )

        try:
            result = checker.check_factuality(response, source)
            score = result.factuality_score

            if score >= self._config.factuality_threshold:
                decision = GateDecision.PASS
            elif score >= self._config.factuality_soft_threshold:
                decision = GateDecision.SOFT_FAIL
            else:
                decision = GateDecision.HARD_FAIL

            return GateCheckResult(
                gate_name="factuality",
                decision=decision,
                score=score,
                threshold=self._config.factuality_threshold,
                soft_threshold=self._config.factuality_soft_threshold,
                latency_ms=(time.perf_counter() - gate_start) * 1000,
                issues=result.issues,
            )
        except Exception as e:
            logger.warning("Factuality check failed: %s", e)
            return GateCheckResult(
                gate_name="factuality",
                decision=GateDecision.SKIP,
                score=0.5,
                threshold=self._config.factuality_threshold,
                issues=[f"Check failed: {e}"],
            )

    def _check_consistency(
        self, response: str, history: list[str] | None
    ) -> GateCheckResult:
        """Check consistency gate."""
        gate_start = time.perf_counter()

        checker = self._get_consistency_checker()
        if checker is None:
            return GateCheckResult(
                gate_name="consistency",
                decision=GateDecision.SKIP,
                score=0.5,
                threshold=self._config.consistency_threshold,
                issues=["Consistency checker unavailable"],
            )

        try:
            result = checker.check_consistency(response, history)
            score = result.consistency_score

            if score >= self._config.consistency_threshold:
                decision = GateDecision.PASS
            elif score >= self._config.consistency_soft_threshold:
                decision = GateDecision.SOFT_FAIL
            else:
                decision = GateDecision.HARD_FAIL

            issues = [issue.description for issue in result.issues]

            return GateCheckResult(
                gate_name="consistency",
                decision=decision,
                score=score,
                threshold=self._config.consistency_threshold,
                soft_threshold=self._config.consistency_soft_threshold,
                latency_ms=(time.perf_counter() - gate_start) * 1000,
                issues=issues,
            )
        except Exception as e:
            logger.warning("Consistency check failed: %s", e)
            return GateCheckResult(
                gate_name="consistency",
                decision=GateDecision.SKIP,
                score=0.5,
                threshold=self._config.consistency_threshold,
                issues=[f"Check failed: {e}"],
            )

    def _check_grounding(
        self, response: str, source: str
    ) -> GateCheckResult:
        """Check grounding gate."""
        gate_start = time.perf_counter()

        checker = self._get_grounding_checker()
        if checker is None:
            return GateCheckResult(
                gate_name="grounding",
                decision=GateDecision.SKIP,
                score=0.5,
                threshold=self._config.grounding_threshold,
                issues=["Grounding checker unavailable"],
            )

        try:
            result = checker.check_grounding(response, source)
            score = result.grounding_score

            if score >= self._config.grounding_threshold:
                decision = GateDecision.PASS
            elif score >= self._config.grounding_soft_threshold:
                decision = GateDecision.SOFT_FAIL
            else:
                decision = GateDecision.HARD_FAIL

            return GateCheckResult(
                gate_name="grounding",
                decision=decision,
                score=score,
                threshold=self._config.grounding_threshold,
                soft_threshold=self._config.grounding_soft_threshold,
                latency_ms=(time.perf_counter() - gate_start) * 1000,
                issues=result.issues,
            )
        except Exception as e:
            logger.warning("Grounding check failed: %s", e)
            return GateCheckResult(
                gate_name="grounding",
                decision=GateDecision.SKIP,
                score=0.5,
                threshold=self._config.grounding_threshold,
                issues=[f"Check failed: {e}"],
            )

    def _check_coherence(
        self, response: str, context: str | None
    ) -> GateCheckResult:
        """Check coherence gate."""
        gate_start = time.perf_counter()

        scorer = self._get_dimension_scorer()
        if scorer is None:
            return GateCheckResult(
                gate_name="coherence",
                decision=GateDecision.SKIP,
                score=0.5,
                threshold=self._config.coherence_threshold,
                issues=["Dimension scorer unavailable"],
            )

        try:
            from jarvis.quality.dimensions import QualityDimension

            result = scorer.score_dimension(
                QualityDimension.COHERENCE, response, context
            )
            score = result.score

            if score >= self._config.coherence_threshold:
                decision = GateDecision.PASS
            elif score >= self._config.coherence_soft_threshold:
                decision = GateDecision.SOFT_FAIL
            else:
                decision = GateDecision.HARD_FAIL

            return GateCheckResult(
                gate_name="coherence",
                decision=decision,
                score=score,
                threshold=self._config.coherence_threshold,
                soft_threshold=self._config.coherence_soft_threshold,
                latency_ms=(time.perf_counter() - gate_start) * 1000,
                issues=result.issues,
                suggestions=result.suggestions,
            )
        except Exception as e:
            logger.warning("Coherence check failed: %s", e)
            return GateCheckResult(
                gate_name="coherence",
                decision=GateDecision.SKIP,
                score=0.5,
                threshold=self._config.coherence_threshold,
                issues=[f"Check failed: {e}"],
            )

    def _check_relevance(
        self, response: str, source: str
    ) -> GateCheckResult:
        """Check relevance gate."""
        gate_start = time.perf_counter()

        scorer = self._get_dimension_scorer()
        if scorer is None:
            return GateCheckResult(
                gate_name="relevance",
                decision=GateDecision.SKIP,
                score=0.5,
                threshold=self._config.relevance_threshold,
                issues=["Dimension scorer unavailable"],
            )

        try:
            from jarvis.quality.dimensions import QualityDimension

            result = scorer.score_dimension(
                QualityDimension.RELEVANCE, response, source
            )
            score = result.score

            if score >= self._config.relevance_threshold:
                decision = GateDecision.PASS
            elif score >= self._config.relevance_soft_threshold:
                decision = GateDecision.SOFT_FAIL
            else:
                decision = GateDecision.HARD_FAIL

            return GateCheckResult(
                gate_name="relevance",
                decision=decision,
                score=score,
                threshold=self._config.relevance_threshold,
                soft_threshold=self._config.relevance_soft_threshold,
                latency_ms=(time.perf_counter() - gate_start) * 1000,
                issues=result.issues,
                suggestions=result.suggestions,
            )
        except Exception as e:
            logger.warning("Relevance check failed: %s", e)
            return GateCheckResult(
                gate_name="relevance",
                decision=GateDecision.SKIP,
                score=0.5,
                threshold=self._config.relevance_threshold,
                issues=[f"Check failed: {e}"],
            )

    def _make_decision(
        self, results: list[GateCheckResult]
    ) -> tuple[GateDecision, RewriteAction]:
        """Make overall decision based on gate results."""
        if not results:
            return GateDecision.PASS, RewriteAction.NONE

        # Count decisions
        hard_fails = sum(1 for r in results if r.decision == GateDecision.HARD_FAIL)
        soft_fails = sum(1 for r in results if r.decision == GateDecision.SOFT_FAIL)
        # passes = sum(1 for r in results if r.decision == GateDecision.PASS)

        # Any hard fail = overall hard fail
        if hard_fails > 0:
            if self._config.auto_rewrite_enabled:
                return GateDecision.HARD_FAIL, RewriteAction.AUTO_REWRITE
            else:
                return GateDecision.HARD_FAIL, RewriteAction.SUGGEST_EDIT

        # Multiple soft fails = overall soft fail with warning
        if soft_fails > 1:
            return GateDecision.SOFT_FAIL, RewriteAction.SUGGEST_EDIT

        # Single soft fail = warn
        if soft_fails > 0:
            return GateDecision.SOFT_FAIL, RewriteAction.WARN

        # All pass
        return GateDecision.PASS, RewriteAction.NONE

    def _calculate_overall_score(self, results: list[GateCheckResult]) -> float:
        """Calculate weighted overall quality score."""
        if not results:
            return 1.0

        weights = self._config.weights
        weighted_sum = 0.0
        total_weight = 0.0

        for result in results:
            # Skip results that were skipped
            if result.decision == GateDecision.SKIP:
                continue

            gate_name = result.gate_name.replace("_fast", "")
            weight = weights.get(gate_name, 0.1)
            weighted_sum += result.score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.5

        return weighted_sum / total_weight


# Global singleton
_quality_gate: QualityGate | None = None
_gate_lock = threading.Lock()


def get_quality_gate(config: QualityGateConfig | None = None) -> QualityGate:
    """Get the global quality gate instance.

    Args:
        config: Optional gate configuration

    Returns:
        Shared QualityGate instance
    """
    global _quality_gate
    if _quality_gate is None:
        with _gate_lock:
            if _quality_gate is None:
                _quality_gate = QualityGate(config)
    return _quality_gate


def reset_quality_gate() -> None:
    """Reset the global quality gate instance."""
    global _quality_gate
    with _gate_lock:
        _quality_gate = None
