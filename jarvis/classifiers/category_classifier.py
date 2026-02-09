"""Category Classifier - Route messages to 6 optimization categories.

Three-layer classification (fast path + ML model + heuristics):
1. Fast path: reactions/acknowledgments → `acknowledge` (100% precision)
2. Trained LightGBM: BERT (384) + context BERT (384) + hand-crafted (147) = 915 features → category
3. Heuristic post-processing: Rule-based corrections for common errors
4. Fallback: `statement` (default)

Categories: acknowledge, closing, emotion, question, request, statement

**Zero-Context-at-Inference Strategy**:
Model trained WITH context features (915 dims), but context BERT embedding (indices 384:768)
is ZEROED at inference. Context features during training act as "auxiliary supervision" -
they help the model learn better representations in the other 531 features.
Result: F1 0.7111 (samples) vs 0.7021 without context in training.

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

import hashlib
import logging
import time
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

# Category order from training (for predict_proba index mapping).
# WARNING: This order MUST match the trained model's mlb.classes_ (alphabetical).
# Used as fallback only when self._mlb is None (missing from model artifact).
CATEGORIES = [
    "acknowledge",
    "closing",
    "emotion",
    "question",
    "request",
    "statement",
]

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
    method: str  # "fast_path", "lightgbm", "heuristic", "default"

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
    2. LightGBM prediction (BERT + context BERT + hand-crafted + spaCy features)
       - Context BERT is ZEROED at inference (auxiliary supervision strategy)
    3. Fallback: `statement` (conf=0.30)
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._mlb = None
        self._pipeline_loaded = False
        # Classification cache: hash(text) -> (CategoryResult, timestamp)
        # TTL of 60 seconds to avoid stale results during prefetch + actual request
        self._classification_cache: dict[str, tuple[CategoryResult, float]] = {}
        self._cache_ttl = 60.0  # seconds
        self._cache_max_size = 1000

    def _cache_put(self, key: str, result: CategoryResult) -> None:
        """Cache a result, evicting oldest entries if over max size."""
        now = time.time()
        if len(self._classification_cache) >= self._cache_max_size:
            # Evict oldest 10% to avoid evicting on every insert
            entries = sorted(self._classification_cache.items(), key=lambda x: x[1][1])
            for k, _ in entries[: self._cache_max_size // 10]:
                del self._classification_cache[k]
        self._classification_cache[key] = (result, now)

    def _load_pipeline(self) -> bool:
        """Load trained Pipeline (with scaler + LightGBM) from disk.

        Model: OneVsRestClassifier(LGBMClassifier) trained with 915 features.
        Strategy: Trained WITH context embeddings, zeroed at inference.
        """
        if self._pipeline_loaded:
            return self._pipeline is not None

        self._pipeline_loaded = True
        _project_root = Path(__file__).resolve().parent.parent.parent
        model_path = _project_root / "models" / "category_multilabel_lightgbm_hardclass.joblib"

        if not model_path.exists():
            logger.warning("No pipeline at %s - using fallback only", model_path)
            return False

        try:
            # Model is saved as dict with 'model' and 'mlb' keys
            model_dict = joblib.load(model_path)
            self._pipeline = model_dict['model']
            self._mlb = model_dict.get('mlb')

            # Validate loaded classes match expected categories
            if self._mlb is not None:
                loaded_cats = set(self._mlb.classes_)
                if loaded_cats != VALID_CATEGORIES:
                    logger.error(
                        "Model categories %s != expected %s", loaded_cats, VALID_CATEGORIES
                    )
                    self._pipeline = None
                    self._mlb = None
                    return False
                logger.info(
                    "Loaded category pipeline from %s (classes: %s)",
                    model_path, list(self._mlb.classes_),
                )
            else:
                logger.warning("No mlb in model artifact, using hardcoded CATEGORIES")
                logger.info("Loaded category pipeline from %s", model_path)

            return True
        except Exception as e:
            logger.error("Failed to load pipeline: %s", e)
            return False

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

        # Check cache first (hash of text only, not context, for prefetch coherence)
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._classification_cache:
            cached_result, cached_time = self._classification_cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_result

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

            result = CategoryResult(
                category=category,
                confidence=1.0,
                method="fast_path",
            )
            # Cache the result
            self._cache_put(cache_key, result)
            return result

        # Simple acknowledgments → acknowledge
        if is_acknowledgment_only(text):
            result = CategoryResult(
                category="acknowledge",
                confidence=1.0,
                method="fast_path",
            )
            # Cache the result
            self._cache_put(cache_key, result)
            return result

        # Layer 1: Pipeline prediction (scaler + LightGBM)
        if self._load_pipeline():
            try:
                # Extract mobilization features
                mob_pressure = mobilization.pressure if mobilization else "none"
                mob_type = mobilization.response_type if mobilization else "answer"

                # Get feature extractor
                extractor = _get_feature_extractor()

                # 1. Current message BERT embedding (384) - use normalize=True to match training
                embedding = self.embedder.encode([text], normalize=True)[0]

                # 2. Context BERT embedding (384) - ALWAYS ZERO at inference
                # Zero-context-at-inference strategy: model trained WITH context for auxiliary
                # supervision, but we zero it out at inference for better generalization.
                # The 3 hand-crafted context stats in non-BERT features still use context.
                context_embedding = np.zeros(384, dtype=np.float32)

                # 3. All non-BERT features (147) - still pass context for hand-crafted features
                # These include 3 context stats + context_lexical_overlap (minimal contribution)
                non_bert_features = extractor.extract_all(text, context, mob_pressure, mob_type)

                # 4. Concatenate: [current_bert(384) + context_bert(384) + non_bert(147)] = 915
                features = np.concatenate([embedding, context_embedding, non_bert_features])
                features = features.reshape(1, -1)

                # Predict (pipeline handles scaling automatically)
                # Model is OneVsRestClassifier(LGBMClassifier) - returns probabilities directly
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "X does not have valid feature names")
                    proba = self._pipeline.predict_proba(features)[0]

                # Get category and confidence
                category_idx = int(np.argmax(proba))
                if self._mlb is not None:
                    classes = self._mlb.classes_
                else:
                    logger.warning("Using hardcoded CATEGORIES fallback (mlb not loaded)")
                    classes = CATEGORIES
                category = classes[category_idx]
                confidence = float(proba[category_idx])

                method = "lightgbm"

                result = CategoryResult(
                    category=category,
                    confidence=confidence,
                    method=method,
                )
                # Cache the result
                self._cache_put(cache_key, result)
                return result
            except Exception as e:
                logger.error("Pipeline prediction failed: %s", e, exc_info=True)

        # Fallback: statement with low confidence
        result = CategoryResult(
            category="statement",
            confidence=0.30,
            method="default",
        )
        # Cache the result
        self._cache_put(cache_key, result)
        return result


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
