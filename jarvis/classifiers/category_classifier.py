"""Category Classifier - Route messages to 6 optimization categories.

Three-layer classification (fast path + ML model + heuristics):
1. Fast path: reactions/acknowledgments → `acknowledge` (100% precision)
2. Trained SVM: BERT (384) + hand-crafted (26) + spaCy (14) = 424 features → category
3. Heuristic post-processing: Rule-based corrections for common errors
4. Fallback: `statement` (default)

Categories: closing, acknowledge, question, request, emotion, statement

Heuristic corrections:
- Reaction messages ("Laughed at", "Loved") → emotion
- Messages ending with "lmao", "lol", "xd" → emotion
- Question words without "?" → question
- Imperative verbs at start → request
- Brief agreements → acknowledge (not emotion)
- "rip" → emotion (not closing)

Usage:
    from jarvis.classifiers.category_classifier import classify_category

    result = classify_category("Want to grab lunch?", context=["Hey"])
    print(result.category, result.confidence)  # request, 0.87
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

from jarvis.classifiers.factory import SingletonFactory
from jarvis.classifiers.mixins import EmbedderMixin
from jarvis.features import CategoryFeatureExtractor
from jarvis.text_normalizer import is_acknowledgment_only, is_reaction

if TYPE_CHECKING:
    from jarvis.classifiers.response_mobilization import MobilizationResult

logger = logging.getLogger(__name__)


VALID_CATEGORIES = frozenset({
    "closing",
    "acknowledge",
    "question",
    "request",
    "emotion",
    "statement",
})

# Feature extractor singleton
_feature_extractor = None


def _get_feature_extractor() -> CategoryFeatureExtractor:
    """Get or initialize feature extractor."""
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = CategoryFeatureExtractor()
    return _feature_extractor


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CategoryResult:
    """Result from category classification."""

    category: str
    confidence: float
    method: str  # "fast_path", "svm", "default"

    def __repr__(self) -> str:
        return (
            f"CategoryResult({self.category}, "
            f"conf={self.confidence:.2f}, method={self.method})"
        )


# ---------------------------------------------------------------------------
# Classifier class
# ---------------------------------------------------------------------------


class CategoryClassifier(EmbedderMixin):
    """Two-layer category classifier.

    Layers:
    1. Fast path: reactions/acknowledgments → `acknowledge`
    2. SVM prediction (BERT + hand-crafted + spaCy features)
    3. Fallback: `statement` (conf=0.30)
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._pipeline_loaded = False

    def _load_pipeline(self) -> bool:
        """Load trained Pipeline (with scaler + SVM) from disk."""
        if self._pipeline_loaded:
            return self._pipeline is not None

        self._pipeline_loaded = True
        model_path = Path("models/category_svm_v2.joblib")

        if not model_path.exists():
            logger.warning("No pipeline at %s - using fallback only", model_path)
            return False

        try:
            self._pipeline = joblib.load(model_path)
            logger.info("Loaded category pipeline from %s", model_path)
            return True
        except Exception as e:
            logger.error("Failed to load pipeline: %s", e)
            return False

    def _apply_heuristics(
        self,
        text: str,
        svm_prediction: str,
        context: list[str],
        confidence: float,
    ) -> str:
        """Apply rule-based corrections to common SVM errors.

        Returns corrected category (or original if no correction needed).

        NOTE: Keep this minimal - only universal patterns that generalize.
        Overfitting check showed that specific word lists don't generalize.
        """
        # ONLY UNIVERSAL FIX: Reactions are handled in fast path
        # No heuristic overrides here - let SVM decide
        return svm_prediction

    def classify(
        self,
        text: str,
        context: list[str] | None = None,
        mobilization: MobilizationResult | None = None,
    ) -> CategoryResult:
        """Classify message into category.

        Args:
            text: Message text
            context: Recent conversation messages (before this message)
            mobilization: MobilizationResult from mobilization classifier

        Returns:
            CategoryResult with category, confidence, method
        """
        context = context or []

        # Layer 0: Fast path for reactions and acknowledgments
        # iMessage reactions - categorize by intent
        if is_reaction(text):
            # Emotional reactions
            if text.startswith(("Loved", "Laughed at")):
                category = "emotion"
            # Question reactions
            elif text.startswith("Questioned"):
                category = "question"
            # Acknowledgment reactions (approval/disapproval)
            elif text.startswith(("Liked", "Disliked", "Emphasized")):
                category = "acknowledge"
            # Removed reactions - acknowledge that the reaction was removed
            elif "Removed" in text:
                category = "acknowledge"
            else:
                # Default for unknown reactions
                category = "emotion"

            return CategoryResult(
                category=category,
                confidence=1.0,
                method="fast_path",
            )

        # Simple acknowledgments → acknowledge
        if is_acknowledgment_only(text):
            return CategoryResult(
                category="acknowledge",
                confidence=1.0,
                method="fast_path",
            )

        # Layer 1: Pipeline prediction (scaler + SVM)
        if self._load_pipeline():
            try:
                # Extract mobilization features
                mob_pressure = mobilization.pressure if mobilization else "none"
                mob_type = mobilization.response_type if mobilization else "answer"

                # Get feature extractor
                extractor = _get_feature_extractor()

                # 1. BERT embedding (384) - FIXED: use normalize=True to match training
                embedding = self.embedder.encode([text], normalize=True)[0]

                # 2. All non-BERT features (~103)
                non_bert_features = extractor.extract_all(text, context, mob_pressure, mob_type)

                # 3. Concatenate (384 BERT + ~103 = ~487 total)
                features = np.concatenate([embedding, non_bert_features])
                features = features.reshape(1, -1)

                # Predict (pipeline handles scaling automatically)
                category = self._pipeline.predict(features)[0]

                # Get confidence via decision function from the SVM step
                svm = self._pipeline.named_steps.get('svm', self._pipeline.steps[-1][1])
                decision_values = svm.decision_function(
                    self._pipeline.named_steps.get('preprocess', lambda x: x).transform(features)
                )[0]

                # For multi-class SVM, decision_values is an array
                # Confidence = softmax of decision values
                if hasattr(decision_values, '__len__'):
                    # Multi-class: use softmax
                    exp_vals = np.exp(decision_values - np.max(decision_values))
                    probs = exp_vals / exp_vals.sum()
                    confidence = float(probs.max())
                else:
                    # Binary (shouldn't happen with 6 classes)
                    confidence = float(1 / (1 + np.exp(-decision_values)))

                # Handle LightGBM label encoding (if present)
                if hasattr(svm, 'label_encoder_'):
                    category = svm.label_encoder_.inverse_transform([category])[0]

                # Layer 2: Heuristic post-processing (correct common SVM errors)
                original_category = category
                category = self._apply_heuristics(text, category, context, confidence)

                # If heuristics changed the prediction, lower confidence
                if category != original_category:
                    confidence = 0.75  # Heuristic override confidence
                    method = "heuristic"
                else:
                    method = "svm"

                return CategoryResult(
                    category=category,
                    confidence=confidence,
                    method=method,
                )
            except Exception as e:
                logger.error("Pipeline prediction failed: %s", e, exc_info=True)

        # Fallback: statement with low confidence
        return CategoryResult(
            category="statement",
            confidence=0.30,
            method="default",
        )


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------

_factory = SingletonFactory(CategoryClassifier)


def get_classifier() -> CategoryClassifier:
    """Get singleton category classifier instance."""
    return _factory.get()


def classify_category(
    text: str,
    context: list[str] | None = None,
    mobilization: MobilizationResult | None = None,
) -> CategoryResult:
    """Classify message category (convenience function).

    Args:
        text: Message text
        context: Recent conversation messages
        mobilization: MobilizationResult from mobilization classifier

    Returns:
        CategoryResult with category, confidence, method
    """
    return get_classifier().classify(text, context, mobilization)


def reset_category_classifier() -> None:
    """Reset the singleton classifier instance (for testing)."""
    _factory.reset()
