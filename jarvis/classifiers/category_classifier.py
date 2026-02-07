"""Category Classifier - Route messages to optimization categories.

Layered classification (fast-to-slow, exit on high confidence):
1. Structural patterns (regex) - bare "?", single emoji, emotional distress
2. Mobilization mapping - reuse MobilizationResult for quick routing
3. SVM model (if trained) - embed message, extract features, predict
4. Centroid verification - catch SVM errors
5. Default: social

Categories: clarify, warm, brief, social
Professional tone is handled as an orthogonal modifier via detect_tone().

Usage:
    from jarvis.classifiers.category_classifier import classify_category

    result = classify_category("Want to grab lunch?", context=["Hey"])
    print(result.category, result.confidence)  # brief, 0.87
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from jarvis.classifiers.factory import SingletonFactory
from jarvis.classifiers.mixins import CentroidMixin, EmbedderMixin
from jarvis.classifiers.patterns import StructuralPatternMatcher

if TYPE_CHECKING:
    from jarvis.classifiers.response_mobilization import MobilizationResult

logger = logging.getLogger(__name__)

VALID_CATEGORIES = frozenset({
    "clarify",
    "warm",
    "brief",
    "social",
})

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CategoryResult:
    """Result from category classification."""

    category: str
    confidence: float
    method: str  # "structural", "mobilization", "svm", "centroid", "default"

    def __repr__(self) -> str:
        return (
            f"CategoryResult({self.category}, "
            f"conf={self.confidence:.2f}, method={self.method})"
        )


# ---------------------------------------------------------------------------
# Structural patterns (Layer 1)
# ---------------------------------------------------------------------------

STRUCTURAL_PATTERNS: list[tuple[str, str, float]] = [
    # Clarify: bare punctuation or single emoji
    (r"^[?!.]{1,3}$", "clarify", 0.95),
    (
        r"^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U00002600-\U000026FF]{1,2}$",
        "clarify",
        0.90,
    ),
]

_structural_matcher = StructuralPatternMatcher(STRUCTURAL_PATTERNS)

# Emotional support patterns (checked separately for clarity)
EMOTIONAL_PATTERN = re.compile(
    r"\b(i('m| am) (so )?(sad|depressed|anxious|stressed|overwhelmed|scared|hurt|"
    r"lonely|heartbroken|devastated|miserable|struggling)|"
    r"i (just )?(lost|broke up|got fired|got dumped|failed)|"
    r"i can'?t (take|handle|cope|deal|do this|stop crying)|"
    r"(funeral|grief|passed away|died|death|suicide|self-harm|panic attack))",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Hand-crafted feature extraction (matches label_soc_categories.py)
# ---------------------------------------------------------------------------

EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
    r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]"
)

ABBREVIATION_RE = re.compile(
    r"\b(lol|lmao|omg|wtf|brb|btw|smh|tbh|imo|idk|ngl|fr|rn|ong|nvm|wya|hmu|"
    r"fyi|asap|dm|irl|fomo|goat|sus|bet|cap|no cap)\b",
    re.IGNORECASE,
)

PROFESSIONAL_KEYWORDS_RE = re.compile(
    r"\b(meeting|deadline|project|report|schedule|conference|presentation|"
    r"budget|client|invoice|proposal)\b",
    re.IGNORECASE,
)


def _extract_hand_crafted(
    text: str,
    context: list[str],
    mobilization_pressure: str,
    mobilization_type: str,
) -> np.ndarray:
    """Extract hand-crafted features matching training pipeline."""
    features: list[float] = []

    # Message structure (5)
    features.append(float(len(text)))
    features.append(float(len(text.split())))
    features.append(float(text.count("?")))
    features.append(float(text.count("!")))
    features.append(float(len(EMOJI_RE.findall(text))))

    # Mobilization one-hots (7)
    for level in ("high", "medium", "low", "none"):
        features.append(1.0 if mobilization_pressure == level else 0.0)
    for rtype in ("commitment", "answer", "emotional"):
        features.append(1.0 if mobilization_type == rtype else 0.0)

    # Tone flags (2)
    features.append(1.0 if PROFESSIONAL_KEYWORDS_RE.search(text) else 0.0)
    features.append(1.0 if ABBREVIATION_RE.search(text) else 0.0)

    # Context features (3)
    features.append(float(len(context)))
    avg_ctx_len = float(np.mean([len(m) for m in context])) if context else 0.0
    features.append(avg_ctx_len)
    features.append(1.0 if len(context) == 0 else 0.0)

    # Style features (2)
    words = text.split()
    total_words = len(words)
    abbr_count = len(ABBREVIATION_RE.findall(text))
    features.append(abbr_count / max(total_words, 1))
    capitalized = sum(1 for w in words[1:] if w[0].isupper()) if len(words) > 1 else 0
    features.append(capitalized / max(len(words) - 1, 1))

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Classifier class
# ---------------------------------------------------------------------------


class CategoryClassifier(EmbedderMixin, CentroidMixin):
    """Multi-layer category classifier.

    Layers (fast-to-slow):
    1. Structural patterns (regex)
    2. Mobilization mapping
    3. SVM prediction (if model exists)
    4. Centroid verification
    5. Default: social
    """

    def __init__(self) -> None:
        from jarvis.config import get_category_classifier_path

        self._model_path: Path = get_category_classifier_path()
        self._svm_model = None
        self._svm_loaded = False
        self._metadata: dict | None = None

    def _load_svm(self) -> bool:
        """Load trained SVM model from disk."""
        if self._svm_loaded:
            return self._svm_model is not None

        self._svm_loaded = True
        model_path = self._model_path / "svm_model.joblib"
        if not model_path.exists():
            logger.debug("No SVM model at %s", model_path)
            return False

        try:
            import joblib

            self._svm_model = joblib.load(model_path)
            # Load metadata for feature dims
            meta_path = self._model_path / "metadata.json"
            if meta_path.exists():
                import json

                self._metadata = json.loads(meta_path.read_text())
            logger.info("Loaded category SVM from %s", model_path)
            return True
        except Exception as e:
            logger.warning("Failed to load SVM model: %s", e)
            return False

    def classify(
        self,
        text: str,
        context: list[str] | None = None,
        mobilization: MobilizationResult | None = None,
    ) -> CategoryResult:
        """Classify a message into a category.

        Args:
            text: The message text to classify.
            context: Recent conversation messages (before this message).
            mobilization: Pre-computed mobilization result (avoids re-computing).

        Returns:
            CategoryResult with category, confidence, and method.
        """
        if not text or not text.strip():
            return CategoryResult("clarify", 0.95, "structural")

        context = context or []

        # Layer 1: Structural patterns
        label, conf = _structural_matcher.match(text)
        if label is not None:
            return CategoryResult(label, conf, "structural")

        # Check emotional patterns
        if EMOTIONAL_PATTERN.search(text):
            return CategoryResult("warm", 0.90, "structural")

        # Layer 2: Mobilization mapping
        if mobilization is not None:
            result = self._from_mobilization(mobilization)
            if result is not None:
                return result

        # Layer 3: SVM prediction
        if self._load_svm() and self._svm_model is not None:
            return self._predict_svm(text, context, mobilization)

        # Layer 5: Default
        return CategoryResult("social", 0.30, "default")

    def _from_mobilization(
        self, mobilization: MobilizationResult
    ) -> CategoryResult | None:
        """Map mobilization result to category (Layer 2).

        Only returns for high-confidence mappings.
        """
        from jarvis.classifiers.response_mobilization import (
            ResponsePressure,
            ResponseType,
        )

        p = mobilization.pressure
        t = mobilization.response_type

        if p == ResponsePressure.HIGH and t in (
            ResponseType.COMMITMENT,
            ResponseType.ANSWER,
        ):
            return CategoryResult("brief", 0.80, "mobilization")

        if p == ResponsePressure.MEDIUM and t == ResponseType.EMOTIONAL:
            return CategoryResult("warm", 0.80, "mobilization")

        if p == ResponsePressure.NONE and t == ResponseType.CLOSING:
            return CategoryResult("clarify", 0.75, "mobilization")

        # Don't route LOW/OPTIONAL via mobilization -- too ambiguous
        return None

    def _predict_svm(
        self,
        text: str,
        context: list[str],
        mobilization: MobilizationResult | None,
    ) -> CategoryResult:
        """Predict category using the trained SVM (Layer 3+4)."""
        # Compute mobilization if not provided
        if mobilization is None:
            from jarvis.classifiers.response_mobilization import (
                classify_response_pressure,
            )

            mobilization = classify_response_pressure(text)

        mob_pressure = mobilization.pressure.value
        mob_type = mobilization.response_type.value

        # Embed the message
        embedding = self.embedder.encode([text], normalize=True)[0]

        # Extract hand-crafted features
        hc = _extract_hand_crafted(text, context, mob_pressure, mob_type)

        # Combine features
        features = np.concatenate([embedding, hc]).reshape(1, -1)

        # SVM prediction
        predicted = self._svm_model.predict(features)[0]

        # Confidence from decision function distance
        try:
            decision = self._svm_model.decision_function(features)
            if decision.ndim == 1:
                confidence = float(min(abs(decision.max()), 1.0))
            else:
                confidence = float(min(abs(decision[0].max()), 1.0))
        except Exception:
            confidence = 0.60

        # Layer 4: Centroid verification
        self._load_centroids()
        if self.centroids_available:
            final_label, centroid_conf, verified = self._verify_with_centroids(
                embedding, predicted
            )
            if not verified:
                return CategoryResult(final_label, centroid_conf, "centroid")

        return CategoryResult(predicted, confidence, "svm")


# ---------------------------------------------------------------------------
# Singleton and public API
# ---------------------------------------------------------------------------

_factory: SingletonFactory[CategoryClassifier] = SingletonFactory(CategoryClassifier)


def get_category_classifier() -> CategoryClassifier:
    """Get the singleton CategoryClassifier instance."""
    return _factory.get()


def reset_category_classifier() -> None:
    """Reset the singleton (for testing)."""
    _factory.reset()


def classify_category(
    text: str,
    context: list[str] | None = None,
    mobilization: MobilizationResult | None = None,
) -> CategoryResult:
    """Classify a message into an optimization category.

    Convenience function using the singleton classifier.

    Args:
        text: Message text.
        context: Recent conversation messages.
        mobilization: Pre-computed mobilization result.

    Returns:
        CategoryResult with category, confidence, and method.
    """
    return get_category_classifier().classify(text, context, mobilization)


__all__ = [
    "CategoryResult",
    "CategoryClassifier",
    "classify_category",
    "get_category_classifier",
    "reset_category_classifier",
]
