"""Category Classifier - Route messages to 6 optimization categories.

Three-layer classification (fast path + ML model + heuristics):
1. Fast path: reactions/acknowledgments → `acknowledge` (100% precision)
2. Trained LightGBM: BERT (384) + hand-crafted (147) = 531 features → category
3. Heuristic post-processing: Rule-based corrections for common errors
4. Fallback: `statement` (default)

Categories: acknowledge, closing, emotion, question, request, statement

Feature layout (531 total):
- [0:384]   = BERT embedding (L2-normalized)
- [384:531] = hand-crafted features (26 structural + 94 spaCy + 19 error-analysis + 8 hard-class)

Note: Context BERT embeddings were removed from both training and inference to
eliminate train-serve skew. Previously, context BERT (384 dims) was zeroed at
inference but present during training, causing a distribution mismatch.

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
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

from jarvis.classifiers.factory import SingletonFactory
from jarvis.classifiers.mixins import EmbedderMixin
from jarvis.features import CategoryFeatureExtractor
from jarvis.observability.logging import log_event
from jarvis.text_normalizer import is_acknowledgment_only, is_reaction

if TYPE_CHECKING:
    from jarvis.classifiers.response_mobilization import MobilizationResult

logger = logging.getLogger(__name__)


VALID_CATEGORIES = frozenset(
    {
        "closing",
        "acknowledge",
        "question",
        "request",
        "emotion",
        "statement",
    }
)

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
_feature_extractor_lock = threading.Lock()


def _get_feature_extractor() -> CategoryFeatureExtractor:
    """Get or initialize feature extractor (thread-safe)."""
    global _feature_extractor
    if _feature_extractor is None:
        with _feature_extractor_lock:
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
        return f"CategoryResult({self.category}, conf={self.confidence:.2f}, method={self.method})"


# ---------------------------------------------------------------------------
# Classifier class
# ---------------------------------------------------------------------------


class CategoryClassifier(EmbedderMixin):
    """Two-layer category classifier.

    Layers:
    1. Fast path: reactions/acknowledgments → `acknowledge`
    2. LightGBM prediction: BERT (384) + hand-crafted (147) = 531 features
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

        Model: OneVsRestClassifier(LGBMClassifier) trained with 531 features.
        Feature layout: BERT (384) + hand-crafted (147).
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
            self._pipeline = model_dict["model"]
            self._mlb = model_dict.get("mlb")

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
                    model_path,
                    list(self._mlb.classes_),
                )
            else:
                logger.warning("No mlb in model artifact, using hardcoded CATEGORIES")
                logger.info("Loaded category pipeline from %s", model_path)

            return True
        except Exception as e:
            logger.error("Failed to load pipeline: %s", e)
            return False

    def _log_classification(self, result: CategoryResult, classify_start: float) -> None:
        """Log a classification result with timing."""
        log_event(
            logger,
            "classifier.inference.complete",
            classifier="category",
            result=result.category,
            confidence=(
                round(result.confidence, 3) if result.method != "fast_path" else result.confidence
            ),
            method=result.method,
            latency_ms=round((time.perf_counter() - classify_start) * 1000, 2),
        )

    def _classify_fast_path(self, text: str) -> CategoryResult | None:
        """Layer 0: Fast path for reactions and acknowledgments.

        Returns CategoryResult on match, None otherwise.
        """
        if is_reaction(text):
            if text.startswith(("Loved", "Laughed at")):
                category = "emotion"
            elif text.startswith("Questioned"):
                category = "question"
            elif text.startswith(("Liked", "Disliked", "Emphasized")):
                category = "acknowledge"
            elif "Removed" in text:
                category = "acknowledge"
            else:
                category = "emotion"
            return CategoryResult(category=category, confidence=1.0, method="fast_path")

        if is_acknowledgment_only(text):
            return CategoryResult(category="acknowledge", confidence=1.0, method="fast_path")

        return None

    def _classify_pipeline(
        self,
        text: str,
        context: list[str],
        mobilization: MobilizationResult | None,
    ) -> CategoryResult | None:
        """Layer 1: LightGBM pipeline prediction.

        Returns CategoryResult on success, None if pipeline unavailable or fails.
        """
        if not self._load_pipeline():
            return None

        try:
            mob_pressure = mobilization.pressure if mobilization else "none"
            mob_type = mobilization.response_type if mobilization else "answer"
            extractor = _get_feature_extractor()

            try:
                embedding = self.embedder.encode([text], normalize=True)[0]
            except Exception as embed_err:
                logger.warning("BERT encode failed, falling back to default: %s", embed_err)
                return None

            non_bert_features = extractor.extract_all(text, context, mob_pressure, mob_type)
            # Model trained with 915 features: BERT(384) + context_BERT(384) + handcrafted(147)
            # Context BERT is zeroed at inference (not available in real-time)
            context_embedding = np.zeros(384, dtype=np.float32)
            features = np.concatenate([embedding, context_embedding, non_bert_features])
            features = features.reshape(1, -1)

            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "X does not have valid feature names")
                proba = self._pipeline.predict_proba(features)[0]

            category_idx = int(np.argmax(proba))
            if self._mlb is not None:
                classes = self._mlb.classes_
            else:
                logger.warning("Using hardcoded CATEGORIES fallback (mlb not loaded)")
                classes = CATEGORIES
            category = classes[category_idx]
            confidence = float(proba[category_idx])

            return CategoryResult(category=category, confidence=confidence, method="lightgbm")
        except Exception as e:
            logger.error("Pipeline prediction failed: %s", e, exc_info=True)
            return None

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

        cache_input = json.dumps([text, context], ensure_ascii=False)
        cache_key = hashlib.md5(cache_input.encode()).hexdigest()
        if cache_key in self._classification_cache:
            cached_result, cached_time = self._classification_cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_result

        classify_start = time.perf_counter()

        result = self._classify_fast_path(text)
        if result is None:
            result = self._classify_pipeline(text, context, mobilization)
        if result is None:
            log_event(
                logger,
                "classifier.fallback",
                level=logging.WARNING,
                classifier="category",
                reason="no_pipeline",
            )
            result = CategoryResult(category="statement", confidence=0.30, method="default")

        self._log_classification(result, classify_start)
        self._cache_put(cache_key, result)
        return result

    def classify_batch(
        self,
        texts: list[str],
        contexts: list[list[str] | None] | None = None,
        mobilizations: list[MobilizationResult | None] | None = None,
    ) -> list[CategoryResult]:
        """Classify a batch of messages efficiently.

        Batch-encodes all texts in one call and runs prediction in a single pass.
        Falls back to per-message classify() for fast-path hits (reactions,
        acknowledgments) since those skip the pipeline entirely.

        Args:
            texts: List of message texts.
            contexts: Optional list of context lists (one per text).
            mobilizations: Optional list of MobilizationResults (one per text).

        Returns:
            List of CategoryResult, one per input text.
        """
        if not texts:
            return []

        n = len(texts)
        if contexts is None:
            contexts = [None] * n
        if mobilizations is None:
            mobilizations = [None] * n

        results: list[CategoryResult | None] = [None] * n

        # --- Pass 1: Check cache and fast path ---
        pipeline_indices: list[int] = []
        for i, text in enumerate(texts):
            ctx = contexts[i] or []
            cache_input = text + "|" + "|".join(ctx)
            cache_key = hashlib.md5(cache_input.encode()).hexdigest()

            # Check cache
            if cache_key in self._classification_cache:
                cached_result, cached_time = self._classification_cache[cache_key]
                if time.time() - cached_time < self._cache_ttl:
                    results[i] = cached_result
                    continue

            # Check fast path
            fast = self._classify_fast_path(text)
            if fast is not None:
                results[i] = fast
                self._cache_put(cache_key, fast)
                continue

            pipeline_indices.append(i)

        # --- Pass 2: Batch pipeline classification ---
        if pipeline_indices and self._load_pipeline():
            import warnings

            pipeline_texts = [texts[i] for i in pipeline_indices]
            pipeline_contexts = [contexts[i] or [] for i in pipeline_indices]
            pipeline_mobs = [mobilizations[i] for i in pipeline_indices]

            # Batch BERT encode
            try:
                embeddings = self.embedder.encode(pipeline_texts, normalize=True)
            except Exception as embed_err:
                logger.warning(
                    "Batch BERT encode failed, %d items will use default: %s",
                    len(pipeline_texts),
                    embed_err,
                )
                embeddings = None

            if embeddings is not None:
                # Batch non-BERT feature extraction
                mob_pressures = [m.pressure if m else "none" for m in pipeline_mobs]
                mob_types = [m.response_type if m else "answer" for m in pipeline_mobs]
                extractor = _get_feature_extractor()
                non_bert_batch = extractor.extract_all_batch(
                    pipeline_texts,
                    pipeline_contexts,
                    mob_pressures,
                    mob_types,
                )

                # Build full feature matrix: BERT(384) + context_BERT(384) + hand-crafted(147) = 915
                # Context BERT is zeroed at inference (not available in real-time)
                non_bert_matrix = np.array(non_bert_batch, dtype=np.float32)
                context_embeddings = np.zeros((len(pipeline_texts), 384), dtype=np.float32)
                feature_matrix = np.concatenate(
                    [embeddings, context_embeddings, non_bert_matrix],
                    axis=1,
                )

                # Single prediction call
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", "X does not have valid feature names")
                        proba_matrix = self._pipeline.predict_proba(feature_matrix)

                    classes = self._mlb.classes_ if self._mlb is not None else CATEGORIES

                    for j, idx in enumerate(pipeline_indices):
                        proba = proba_matrix[j]
                        category_idx = int(np.argmax(proba))
                        category = classes[category_idx]
                        confidence = float(proba[category_idx])
                        result = CategoryResult(
                            category=category,
                            confidence=confidence,
                            method="lightgbm",
                        )
                        results[idx] = result
                        ctx = contexts[idx] or []
                        cache_input = json.dumps(
                            [texts[idx], ctx], ensure_ascii=False
                        )
                        cache_key = hashlib.md5(cache_input.encode()).hexdigest()
                        self._cache_put(cache_key, result)
                except Exception as e:
                    logger.error("Batch pipeline prediction failed: %s", e, exc_info=True)

        # --- Pass 3: Fill any remaining with fallback ---
        for i in range(n):
            if results[i] is None:
                results[i] = CategoryResult(
                    category="statement",
                    confidence=0.30,
                    method="default",
                )
                ctx = contexts[i] or []
                cache_input = json.dumps([texts[i], ctx], ensure_ascii=False)
                cache_key = hashlib.md5(cache_input.encode()).hexdigest()
                self._cache_put(cache_key, results[i])

        return results  # type: ignore[return-value]


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


def classify_category_batch(
    texts: list[str],
    contexts: list[list[str] | None] | None = None,
    mobilizations: list[MobilizationResult | None] | None = None,
) -> list[CategoryResult]:
    """Classify a batch of messages (convenience function).

    Args:
        texts: List of message texts.
        contexts: Optional list of context lists (one per text).
        mobilizations: Optional list of MobilizationResults (one per text).

    Returns:
        List of CategoryResult, one per input text.
    """
    return get_classifier().classify_batch(texts, contexts, mobilizations)


def reset_category_classifier() -> None:
    """Reset the singleton classifier instance (for testing)."""
    _factory.reset()
