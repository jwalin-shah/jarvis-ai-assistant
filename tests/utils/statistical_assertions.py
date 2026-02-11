"""Statistical assertions for ML model testing.

Provides confidence thresholds, tolerance ranges, and statistical validation
for classifier outputs. Use these instead of brittle exact-match assertions.

Usage:
    from tests.utils.statistical_assertions import (
        assert_confidence_in_range,
        assert_category_in_set,
        StatisticalValidator,
    )

    def test_classifier():
        result = classify("hello")
        assert_confidence_in_range(result.confidence, 0.80, 1.0, tolerance=0.05)
        assert_category_in_set(result.category, {"greeting", "statement"})
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ConfidenceThreshold:
    """Confidence threshold with tolerance for statistical validation.

    Attributes:
        min_confidence: Minimum acceptable confidence
        max_confidence: Maximum expected confidence (usually 1.0)
        tolerance: Acceptable deviation from range boundaries
    """

    min_confidence: float
    max_confidence: float = 1.0
    tolerance: float = 0.05  # ±5% tolerance

    def check(self, value: float) -> bool:
        """Check if value is within threshold with tolerance."""
        return self.min_confidence - self.tolerance <= value <= self.max_confidence + self.tolerance

    def __str__(self) -> str:
        return (
            f"[{self.min_confidence:.2f}±{self.tolerance:.2f}, "
            f"{self.max_confidence:.2f}±{self.tolerance:.2f}]"
        )


# Predefined thresholds for common scenarios
CONFIDENCE_THRESHOLDS: dict[str, ConfidenceThreshold] = {
    # High confidence: well-defined patterns
    "acknowledge_exact": ConfidenceThreshold(0.90, 1.0, tolerance=0.02),
    "emotion_exact": ConfidenceThreshold(0.85, 1.0, tolerance=0.03),
    "closing_exact": ConfidenceThreshold(0.90, 1.0, tolerance=0.02),
    # Medium confidence: ambiguous cases
    "question_ambiguous": ConfidenceThreshold(0.60, 0.90, tolerance=0.10),
    "statement_general": ConfidenceThreshold(0.50, 0.80, tolerance=0.10),
    "request_contextual": ConfidenceThreshold(0.60, 0.85, tolerance=0.08),
    # Low confidence: fallback cases
    "fallback_any": ConfidenceThreshold(0.30, 0.70, tolerance=0.15),
    "unknown_default": ConfidenceThreshold(0.40, 0.60, tolerance=0.10),
}


class StatisticalValidator:
    """Validator for statistical properties of classifier outputs.

    Use this to validate consistency and accuracy across multiple runs
    or test cases, rather than relying on single deterministic checks.
    """

    def __init__(self, min_samples: int = 10, max_variance: float = 0.01):
        """Initialize validator.

        Args:
            min_samples: Minimum samples for statistical validation
            max_variance: Maximum acceptable variance in confidence scores
        """
        self.min_samples = min_samples
        self.max_variance = max_variance

    def validate_consistency(
        self,
        classifier: Callable[[str], T],
        text: str,
        num_runs: int = 5,
    ) -> tuple[bool, str]:
        """Validate that classifier produces consistent results.

        Args:
            classifier: Function that takes text and returns result with
                       .category and .confidence attributes
            text: Input text to classify
            num_runs: Number of times to run classification

        Returns:
            Tuple of (is_consistent, message)
        """
        results = [classifier(text) for _ in range(num_runs)]

        # Check category consistency
        categories = [r.category for r in results]
        unique_categories = set(categories)
        if len(unique_categories) > 1:
            return False, f"Inconsistent categories: {unique_categories}"

        # Check confidence variance
        confidences = [r.confidence for r in results]
        if len(confidences) > 1:
            variance = statistics.variance(confidences)
            if variance > self.max_variance:
                return (
                    False,
                    f"High confidence variance: {variance:.4f} (max {self.max_variance:.4f})",
                )

        mean_conf = statistics.mean(confidences)
        return True, f"Consistent: {categories[0]} @ {mean_conf:.2f}"

    def validate_distribution(
        self,
        classifier: Callable[[str], T],
        test_cases: list[tuple[str, str]],
        min_accuracy: float = 0.80,
    ) -> tuple[float, list[str]]:
        """Validate classifier accuracy across test distribution.

        Args:
            classifier: Classification function
            test_cases: List of (text, expected_category) tuples
            min_accuracy: Minimum required accuracy

        Returns:
            Tuple of (accuracy, list of failure messages)
        """
        if not test_cases:
            return 0.0, ["No test cases provided"]

        correct = 0
        failures = []

        for text, expected in test_cases:
            result = classifier(text)
            if result.category == expected:
                correct += 1
            else:
                failures.append(
                    f"'{text[:50]}...': expected {expected}, "
                    f"got {result.category} (conf: {result.confidence:.2f})"
                )

        accuracy = correct / len(test_cases)
        return accuracy, failures

    def validate_confidence_calibration(
        self,
        classifier: Callable[[str], T],
        test_cases: list[tuple[str, str]],
    ) -> tuple[bool, dict]:
        """Validate that confidence scores are well-calibrated.

        Well-calibrated means: predictions with confidence X are
        correct approximately X% of the time.

        Args:
            classifier: Classification function
            test_cases: List of (text, expected_category) tuples

        Returns:
            Tuple of (is_calibrated, calibration report dict)
        """
        buckets: dict[str, list[tuple[float, bool]]] = {
            "0.5-0.6": [],
            "0.6-0.7": [],
            "0.7-0.8": [],
            "0.8-0.9": [],
            "0.9-1.0": [],
        }

        for text, expected in test_cases:
            result = classifier(text)
            is_correct = result.category == expected

            # Assign to bucket
            conf = result.confidence
            if conf < 0.6:
                buckets["0.5-0.6"].append((conf, is_correct))
            elif conf < 0.7:
                buckets["0.6-0.7"].append((conf, is_correct))
            elif conf < 0.8:
                buckets["0.7-0.8"].append((conf, is_correct))
            elif conf < 0.9:
                buckets["0.8-0.9"].append((conf, is_correct))
            else:
                buckets["0.9-1.0"].append((conf, is_correct))

        # Calculate calibration error for each bucket
        report: dict[str, dict] = {}
        total_calibration_error = 0.0
        num_buckets = 0

        for bucket_name, bucket_results in buckets.items():
            if not bucket_results:
                continue

            avg_confidence = sum(c for c, _ in bucket_results) / len(bucket_results)
            accuracy = sum(1 for _, correct in bucket_results if correct) / len(bucket_results)
            calibration_error = abs(avg_confidence - accuracy)

            report[bucket_name] = {
                "count": len(bucket_results),
                "avg_confidence": avg_confidence,
                "accuracy": accuracy,
                "calibration_error": calibration_error,
            }

            total_calibration_error += calibration_error
            num_buckets += 1

        avg_calibration_error = total_calibration_error / num_buckets if num_buckets > 0 else 1.0
        is_calibrated = avg_calibration_error < 0.1  # < 10% calibration error

        return is_calibrated, {
            "buckets": report,
            "avg_calibration_error": avg_calibration_error,
            "is_calibrated": is_calibrated,
        }


def assert_confidence_in_range(
    actual: float,
    expected_min: float,
    expected_max: float,
    tolerance: float = 0.05,
    message: str = "",
):
    """Assert confidence is within expected range with tolerance.

    Args:
        actual: Actual confidence value
        expected_min: Minimum expected confidence
        expected_max: Maximum expected confidence
        tolerance: Acceptable deviation from range
        message: Optional message prefix

    Raises:
        AssertionError: If confidence outside tolerated range
    """
    effective_min = max(0.0, expected_min - tolerance)
    effective_max = min(1.0, expected_max + tolerance)

    if not (effective_min <= actual <= effective_max):
        prefix = f"{message}: " if message else ""
        raise AssertionError(
            f"{prefix}Confidence {actual:.3f} outside range "
            f"[{expected_min:.3f}, {expected_max:.3f}] ±{tolerance:.3f}"
        )


def assert_category_in_set(
    actual: str,
    allowed: set[str],
    probability_threshold: float = 0.95,
):
    """Assert category is in allowed set.

    Args:
        actual: Predicted category
        allowed: Set of acceptable categories
        probability_threshold: Not used for single assertion (for API consistency)

    Raises:
        AssertionError: If category not in allowed set
    """
    if actual not in allowed:
        raise AssertionError(f"Category '{actual}' not in allowed set: {sorted(allowed)}")


def assert_top_k_accuracy(
    classifier: Callable[[str], list[tuple[str, float]]],
    text: str,
    expected_in_top_k: set[str],
    k: int = 3,
):
    """Assert expected categories appear in top-k predictions.

    Args:
        classifier: Function returning sorted list of (category, confidence)
        text: Input text
        expected_in_top_k: Categories that must appear in top-k
        k: Number of top predictions to check

    Raises:
        AssertionError: If expected categories not in top-k
    """
    predictions = classifier(text)
    top_k = {cat for cat, _ in predictions[:k]}

    missing = expected_in_top_k - top_k
    if missing:
        raise AssertionError(f"Expected categories {missing} not in top-{k}: {top_k}")


def assert_probabilistic_range(
    values: list[float],
    expected_mean: float,
    tolerance: float = 0.1,
    confidence: float = 0.95,
) -> bool:
    """Assert that a list of values has expected mean within tolerance.

    Uses statistical check appropriate for the confidence level.

    Args:
        values: List of observed values
        expected_mean: Expected mean value
        tolerance: Acceptable deviation from expected mean
        confidence: Confidence level for statistical test

    Returns:
        True if assertion passes

    Raises:
        AssertionError: If mean outside tolerated range
    """
    if not values:
        raise AssertionError("No values provided")

    actual_mean = statistics.mean(values)
    std_dev = statistics.stdev(values) if len(values) > 1 else 0

    # For 95% confidence, use ~2 std devs
    margin_of_error = 2 * (std_dev / (len(values) ** 0.5)) if len(values) > 1 else 0

    lower_bound = expected_mean - tolerance - margin_of_error
    upper_bound = expected_mean + tolerance + margin_of_error

    if not (lower_bound <= actual_mean <= upper_bound):
        raise AssertionError(
            f"Mean {actual_mean:.3f} outside range "
            f"[{lower_bound:.3f}, {upper_bound:.3f}] "
            f"(expected {expected_mean:.3f} ± {tolerance:.3f}, "
            f"n={len(values)}, std={std_dev:.3f})"
        )

    return True
