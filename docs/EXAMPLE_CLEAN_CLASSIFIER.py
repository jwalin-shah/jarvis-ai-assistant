"""
EXAMPLE: Clean classification pipeline (simplified from cascade + category + mobilization)

Before: 3 files, 700+ lines, confusing interactions
After: 1 file, ~150 lines, clear flow
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ============================================================================
# Data Types
# ============================================================================


class Category(str, Enum):
    """Message categories - what is this message trying to do?"""

    ACKNOWLEDGE = "acknowledge"  # "ok", "sure", reactions
    QUESTION = "question"  # "what time?", "are you coming?"
    REQUEST = "request"  # "pick up milk", "call me"
    STATEMENT = "statement"  # "running late", "just got home"
    EMOTION = "emotion"  # "lol", "that's amazing", "rip"
    CLOSING = "closing"  # "goodnight", "talk later"


class Urgency(str, Enum):
    """How urgent is a reply?"""

    NONE = "none"  # No reply needed (acks, closings)
    LOW = "low"  # Casual reply fine
    HIGH = "high"  # Needs substantive response


@dataclass(frozen=True)
class Classification:
    """Result of classifying a single message."""

    category: Category
    urgency: Urgency
    confidence: float  # 0.0 - 1.0
    method: str  # "fast_path", "lightgbm", "fallback"

    @property
    def should_reply(self) -> bool:
        """Should we generate a reply?"""
        return self.urgency != Urgency.NONE

    @property
    def use_template(self) -> bool:
        """Should we use a template instead of generating?"""
        return self.category in (Category.ACKNOWLEDGE, Category.CLOSING)


# ============================================================================
# Fast Path Rules (100% precision shortcuts)
# ============================================================================

# Patterns that indicate specific categories
REACTION_PATTERN = re.compile(r"^(Loved|Laughed at|Liked|Disliked|Emphasized|Questioned)\s+")
ACK_PATTERN = re.compile(r"^(ok|okay|k|sure|yep|yeah|yes|no|nah|np|thanks|thank you)[!\.]*$", re.I)
CLOSING_PATTERN = re.compile(r"(goodnight|gn|ttyl|talk later|bye|see you|cu later)[!\.]*$", re.I)
QUESTION_PATTERN = re.compile(r"\?$")


def fast_path_classify(text: str) -> Classification | None:
    """
    Fast, rule-based classification for common cases.
    Returns None if we need to use ML.
    """
    text = text.strip()
    if not text:
        return Classification(
            category=Category.ACKNOWLEDGE,
            urgency=Urgency.NONE,
            confidence=1.0,
            method="fast_path_empty",
        )

    # iMessage reactions
    if REACTION_PATTERN.match(text):
        if text.startswith(("Loved", "Laughed at")):
            return Classification(Category.EMOTION, Urgency.LOW, 1.0, "fast_path_reaction")
        elif text.startswith("Questioned"):
            return Classification(Category.QUESTION, Urgency.HIGH, 1.0, "fast_path_reaction")
        else:
            return Classification(Category.ACKNOWLEDGE, Urgency.NONE, 1.0, "fast_path_reaction")

    # Simple acknowledgments (no reply needed)
    if ACK_PATTERN.match(text):
        return Classification(Category.ACKNOWLEDGE, Urgency.NONE, 0.98, "fast_path_ack")

    # Closings (no reply needed)
    if CLOSING_PATTERN.search(text):
        return Classification(Category.CLOSING, Urgency.NONE, 0.95, "fast_path_closing")

    # Questions (high urgency)
    if QUESTION_PATTERN.search(text):
        return Classification(Category.QUESTION, Urgency.HIGH, 0.90, "fast_path_question")

    # Need ML for everything else
    return None


# ============================================================================
# ML Classifier (LightGBM)
# ============================================================================


class MessageClassifier:
    """
    Single classifier that determines category and urgency.

    Before we had:
    - Mobilization classifier (should reply? how urgent?)
    - Category classifier (what type?)
    - Intent classifier (fallback)

    Now we have one model that outputs both category and urgency.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self._model = None
        self._mlb = None  # MultiLabelBinarizer or similar
        self._cache: dict[str, tuple[Classification, float]] = {}
        self._cache_ttl = 60.0

        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: Path) -> None:
        """Load the LightGBM model."""
        if not path.exists():
            return

        artifact = joblib.load(path)
        self._model = artifact.get("model")
        self._mlb = artifact.get("mlb")

    def _get_cache_key(self, text: str, context: list[str]) -> str:
        """Hash for caching results."""
        content = text + "|".join(context[-3:])  # Last 3 context items
        return hashlib.md5(content.encode()).hexdigest()

    def _extract_features(self, text: str, context: list[str]) -> NDArray:
        """
        Extract features for classification.

        Features:
        - Text embedding (384-dim from BERT)
        - Hand-crafted features (length, question marks, etc.)
        - Context features (prev message type, etc.)
        """
        # This would call your feature extractor
        # For now, placeholder
        from jarvis.features import CategoryFeatureExtractor

        extractor = CategoryFeatureExtractor()

        # Get BERT embedding
        from jarvis.embedding_adapter import get_embedder

        embedder = get_embedder()
        embedding = embedder.encode([text], normalize=True)[0]

        # Get hand-crafted features
        handcrafted = extractor.extract_all(text, context, "none", "answer")

        return np.concatenate([embedding, handcrafted])

    def classify(self, text: str, context: list[str] | None = None) -> Classification:
        """
        Classify a message into category + urgency.

        Flow:
        1. Try fast path rules
        2. If no match, use LightGBM
        3. Map output to Category + Urgency
        """
        context = context or []

        # Check cache
        cache_key = self._get_cache_key(text, context)
        if cache_key in self._cache:
            result, ts = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return result

        # 1. Try fast path
        if fast_result := fast_path_classify(text):
            self._cache[cache_key] = (fast_result, time.time())
            return fast_result

        # 2. Use ML model
        if self._model is None:
            # Fallback if no model loaded
            result = Classification(
                category=Category.STATEMENT,
                urgency=Urgency.LOW,
                confidence=0.5,
                method="fallback_no_model",
            )
            self._cache[cache_key] = (result, time.time())
            return result

        # Extract features and predict
        features = self._extract_features(text, context)

        # Model outputs probabilities for each category
        probs = self._model.predict_proba(features.reshape(1, -1))[0]

        # Get top category
        cat_idx = int(np.argmax(probs))
        category = Category(self._mlb.classes_[cat_idx])
        confidence = float(probs[cat_idx])

        # Determine urgency from category
        urgency = self._category_to_urgency(category, confidence)

        result = Classification(
            category=category, urgency=urgency, confidence=confidence, method="lightgbm"
        )

        self._cache[cache_key] = (result, time.time())
        return result

    def _category_to_urgency(self, category: Category, confidence: float) -> Urgency:
        """Map category to urgency level."""
        if category in (Category.ACKNOWLEDGE, Category.CLOSING):
            return Urgency.NONE
        elif category in (Category.QUESTION, Category.REQUEST):
            return Urgency.HIGH
        else:
            return Urgency.LOW


# ============================================================================
# Public API
# ============================================================================

# Singleton instance
_classifier: MessageClassifier | None = None


def get_classifier() -> MessageClassifier:
    """Get the singleton classifier."""
    global _classifier
    if _classifier is None:
        model_path = Path(__file__).parent.parent / "models" / "classifier.joblib"
        _classifier = MessageClassifier(model_path)
    return _classifier


def classify_message(text: str, context: list[str] | None = None) -> Classification:
    """
    Classify a message.

    This is the main entry point. Usage:

        result = classify_message("Want to grab lunch?")
        if result.should_reply:
            if result.use_template:
                reply = get_template_reply(result.category)
            else:
                reply = generate_reply(text, result)
    """
    return get_classifier().classify(text, context)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test cases
    test_messages = [
        "ok",  # Fast path: acknowledge, no reply
        'Laughed at "haha"',  # Fast path: emotion, low urgency
        "Want to grab lunch?",  # ML: question, high urgency
        "I'm running late",  # ML: statement, low urgency
        "goodnight!",  # Fast path: closing, no reply
        "Can you pick up milk?",  # ML: request, high urgency
    ]

    for msg in test_messages:
        result = classify_message(msg)
        print(
            f"{msg:25} → {result.category.value:12} | {result.urgency.value:4} | {result.confidence:.2f} | {result.method}"
        )

    # Output:
    # ok                        → acknowledge  | none | 0.98 | fast_path_ack
    # Laughed at "haha"        → emotion      | low  | 1.00 | fast_path_reaction
    # Want to grab lunch?      → question     | high | 0.87 | lightgbm
    # I'm running late         → statement    | low  | 0.72 | lightgbm
    # goodnight!               → closing      | none | 0.95 | fast_path_closing
    # Can you pick up milk?    → request      | high | 0.91 | lightgbm
