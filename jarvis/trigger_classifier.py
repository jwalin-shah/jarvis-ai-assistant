"""Trigger Classifier - Structural patterns + Trained SVM.

Classifies incoming messages (triggers) to determine what type of response is needed.

5-Label Classification Scheme:
- COMMITMENT: Invitations, requests (need AGREE/DECLINE/DEFER)
- QUESTION: Yes/no questions, info questions (need YES/NO or INFO answer)
- REACTION: Good news, bad news, reaction prompts (need emotional response)
- SOCIAL: Greetings, acknowledgments (need greeting/ack back)
- STATEMENT: Neutral info sharing (need acknowledgment or follow-up)

Strategy:
1. Structural patterns (fast regex) - high precision matches
2. Trained SVM classifier (optional) - for ambiguous cases
3. Default to STATEMENT - when unsure

Usage:
    from jarvis.trigger_classifier import classify_trigger, TriggerType

    result = classify_trigger("Want to grab lunch?")
    print(result.trigger_type)  # TriggerType.COMMITMENT
    print(result.confidence)    # 0.95
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Trigger Types (5-Label Scheme)
# =============================================================================


class TriggerType(str, Enum):
    """Types of incoming messages based on what response they need."""

    # Commitment - invitations and requests (need AGREE/DECLINE/DEFER)
    COMMITMENT = "commitment"

    # Question - yes/no and info questions (need answer)
    QUESTION = "question"

    # Reaction - emotional content (need emotional response)
    REACTION = "reaction"

    # Social - greetings and acknowledgments
    SOCIAL = "social"

    # Statement - neutral information sharing
    STATEMENT = "statement"

    # Fallback
    UNKNOWN = "unknown"


# Default model path
DEFAULT_MODEL_PATH = Path.home() / ".jarvis" / "trigger_classifier_model"


# Per-class SVM thresholds (tuned based on per-class performance)
# Higher thresholds for important/hard classes, lower for classes with strong structural patterns
PER_CLASS_SVM_THRESHOLDS: dict[TriggerType, float] = {
    TriggerType.COMMITMENT: 0.50,  # Most important class - need high confidence
    TriggerType.QUESTION: 0.35,    # Moderate - "?" helps structural
    TriggerType.REACTION: 0.40,    # Medium - emotional words help
    TriggerType.SOCIAL: 0.25,      # Low - structural patterns very strong (tapbacks, greetings)
    TriggerType.STATEMENT: 0.40,   # Medium - fallback category
}

# Default threshold for unknown types
DEFAULT_SVM_THRESHOLD = 0.35


# What response types are valid for each trigger type
TRIGGER_TO_RESPONSE_TYPES: dict[TriggerType, list[str]] = {
    TriggerType.COMMITMENT: ["AGREE", "DECLINE", "DEFER", "QUESTION"],
    TriggerType.QUESTION: ["YES", "NO", "ANSWER", "DEFER"],
    TriggerType.REACTION: ["REACT_POSITIVE", "REACT_SYMPATHY", "QUESTION", "ACKNOWLEDGE"],
    TriggerType.SOCIAL: ["GREETING", "ACKNOWLEDGE"],
    TriggerType.STATEMENT: ["ACKNOWLEDGE", "REACT", "QUESTION"],
    TriggerType.UNKNOWN: ["ANSWER", "QUESTION", "ACKNOWLEDGE"],
}


# =============================================================================
# Structural Patterns (High Precision)
# =============================================================================

# Order matters! More specific patterns first.
STRUCTURAL_PATTERNS: list[tuple[re.Pattern, TriggerType, float]] = [
    # === SOCIAL: Tapbacks (check first - these are metadata) ===
    (re.compile(r'^(Liked|Loved|Laughed at|Emphasized|Questioned|Disliked)\s+["\u201c\u201d]', re.I),
     TriggerType.SOCIAL, 0.95),

    # === SOCIAL: Greetings (exact matches) ===
    (re.compile(r"^(hey+|hi+|hello+|yo+|sup|what'?s up|wassup|hiya|howdy|hola)[\s!?\u200d\U0001f64b\U0001f600-\U0001f64f]*$", re.I),
     TriggerType.SOCIAL, 0.95),
    (re.compile(r"^what'?s up\s+(homie|dude|friend|bro|man|guys?)[\s!?]*$", re.I),
     TriggerType.SOCIAL, 0.90),
    (re.compile(r"^how (are|r) (you|u|ya)[\s!?]*$", re.I),
     TriggerType.SOCIAL, 0.95),
    (re.compile(r"^how('?s| is) it going[\s!?]*$", re.I),
     TriggerType.SOCIAL, 0.95),
    (re.compile(r"^(good|gm|gn)\s*(night|morning|evening|afternoon)?['\s!?]*$", re.I),
     TriggerType.SOCIAL, 0.95),
    (re.compile(r"^(happy|merry)\s+(thanksgiving|christmas|birthday|holiday|new year|easter)", re.I),
     TriggerType.SOCIAL, 0.90),
    (re.compile(r"^(what'?s good|wsg|wsup|whaddup)[\s!?]*$", re.I),
     TriggerType.SOCIAL, 0.90),
    (re.compile(r"^love (you|u|ya)[\s!]*$", re.I),
     TriggerType.SOCIAL, 0.85),

    # === SOCIAL: Acknowledgments (exact matches) ===
    (re.compile(r"^(ok|okay|k+|sure|bet|got it|sounds good|cool|alright|aight|word)[\s!.]*$", re.I),
     TriggerType.SOCIAL, 0.95),
    (re.compile(r"^(yes+|yea+|yeah+|yup|yep|nah+|nope|true|for sure|all\s*right|could be)[\s!.]*$", re.I),
     TriggerType.SOCIAL, 0.95),
    (re.compile(r"^(thanks|thank you|thx|ty|appreciate it)[\s\w!.]*$", re.I),
     TriggerType.SOCIAL, 0.95),
    (re.compile(r"^(lol|lmao|haha+|hehe+|😂|🤣|💀)+[\s!]*$", re.I),
     TriggerType.SOCIAL, 0.90),
    (re.compile(r"^(ik|i know)[\s!]*(haha+)?[\s!]*$", re.I),
     TriggerType.SOCIAL, 0.90),
    (re.compile(r"^(ofc|of course|def|definitely|obviously|obvi)[\s!.]*$", re.I),
     TriggerType.SOCIAL, 0.90),
    (re.compile(r"^(right+|i know right|ikr)[\s!]*$", re.I),
     TriggerType.SOCIAL, 0.90),
    (re.compile(r"^(have fun|enjoy|nice|noice|dope|sick|lit|fire)[\s!.]*$", re.I),
     TriggerType.SOCIAL, 0.85),

    # === COMMITMENT: Invitations (asking to do something together) ===
    (re.compile(r"\b(wanna|want to|down to|dtf|tryna|trying to)\s+.*(hang|chill|go|come|grab|get|play|watch|do)\b.*\?", re.I),
     TriggerType.COMMITMENT, 0.95),
    (re.compile(r"^(wanna|want to|down to)\s+\w+.*\?", re.I),
     TriggerType.COMMITMENT, 0.90),
    (re.compile(r"\b(you|u)\s+(free|available|busy|down)\s*(today|tonight|tomorrow|tmrw|later|this weekend|rn)?\s*\?", re.I),
     TriggerType.COMMITMENT, 0.95),
    (re.compile(r"^(let'?s|lets)\s+(go|hang|chill|grab|get|do|play|watch|call|meet)\b", re.I),
     TriggerType.COMMITMENT, 0.85),
    (re.compile(r"\bcome (over|through|thru|hang|chill)\b.*\?", re.I),
     TriggerType.COMMITMENT, 0.90),
    (re.compile(r"^(pull up|slide|bool|link)\b", re.I),
     TriggerType.COMMITMENT, 0.85),

    # === COMMITMENT: Requests (asking someone to do something FOR you) ===
    (re.compile(r"^(can|could|would|will)\s+(you|u)\s+(send|check|grab|get|pick|help|call|text|set|setup|bring)\b", re.I),
     TriggerType.COMMITMENT, 0.90),
    (re.compile(r"^(can|could|would|will)\s+(you|u)\s+(please|pls|plz)\b", re.I),
     TriggerType.COMMITMENT, 0.90),
    (re.compile(r"^(please|pls|plz)\s+(send|check|help|call|grab|get|pick|bring|remind)\b", re.I),
     TriggerType.COMMITMENT, 0.85),
    (re.compile(r"\b(pick me up|drop me off)\b", re.I),
     TriggerType.COMMITMENT, 0.90),
    (re.compile(r"^(tell|ask|remind)\s+(everyone|them|him|her|me)\b", re.I),
     TriggerType.COMMITMENT, 0.85),

    # === QUESTION: Info questions (what/when/where/who/how) ===
    (re.compile(r"^(what|what'?s)\s+(time|day|the plan|up|going on|happening)\b", re.I),
     TriggerType.QUESTION, 0.95),
    (re.compile(r"^what\s+(did|do|does|are|is|was|were)\s+", re.I),
     TriggerType.QUESTION, 0.90),
    (re.compile(r"^whatchu\s+", re.I),
     TriggerType.QUESTION, 0.85),
    (re.compile(r"^how('?s|s)?\s+\w+", re.I),
     TriggerType.QUESTION, 0.85),
    (re.compile(r"^(when|where|who|which|why)\s+.+\?$", re.I),
     TriggerType.QUESTION, 0.90),
    (re.compile(r"\b(what time|how long|how much|how many)\b.*\?", re.I),
     TriggerType.QUESTION, 0.95),
    (re.compile(r"\bwhat\s+.{5,}\?$", re.I),
     TriggerType.QUESTION, 0.80),

    # === QUESTION: Yes/No questions ===
    (re.compile(r"^(do|does|did|is|are|was|were|have|has|can|could|will|would|should)\s+(you|u|we|they|i|it|he|she)\b.*\?", re.I),
     TriggerType.QUESTION, 0.85),

    # === REACTION: Reaction prompts (wanting your reaction) ===
    (re.compile(r"^(omg|oh my god)\b.+(\?|!{2,}|wtf|crazy|insane)", re.I),
     TriggerType.REACTION, 0.85),
    (re.compile(r"\b(did you (see|hear|watch)|have you seen)\b.*\?", re.I),
     TriggerType.REACTION, 0.85),
    (re.compile(r"\b(can you believe|isn't that|wasn't that)\b.*\?", re.I),
     TriggerType.REACTION, 0.85),
    (re.compile(r"\b(that'?s|thats)\s+(dope|sick|crazy|wild|insane|fire|lit|awesome|cool)\b", re.I),
     TriggerType.REACTION, 0.80),
    (re.compile(r"^(holy|oh)\s+(shit|fuck|crap|damn|my god)\b", re.I),
     TriggerType.REACTION, 0.80),

    # === REACTION: Good news ===
    (re.compile(r"\b(i got (the job|accepted|promoted|in|hired)|i passed|i made it|we won|i'm engaged|i'm pregnant)\b", re.I),
     TriggerType.REACTION, 0.85),
    (re.compile(r"^(great news|good news|finally|so excited|so happy)[!:\s]", re.I),
     TriggerType.REACTION, 0.80),

    # === REACTION: Bad news ===
    (re.compile(r"\b(i lost my (wallet|keys|phone|job)|i failed|i got fired|i'm sick|someone died|passed away)\b", re.I),
     TriggerType.REACTION, 0.85),
    (re.compile(r"^(so sad|so upset|terrible news|awful news|bad news)\b", re.I),
     TriggerType.REACTION, 0.80),
    (re.compile(r"^(fuck+|shit+|damn+|ugh+|omfg|fml)[\s!.]*$", re.I),
     TriggerType.REACTION, 0.80),

    # === FALLBACK: Any question mark = QUESTION ===
    (re.compile(r"\?\s*$"), TriggerType.QUESTION, 0.60),
]


# =============================================================================
# Classification Result
# =============================================================================


@dataclass
class TriggerClassification:
    """Result from classifying a trigger message."""

    trigger_type: TriggerType
    confidence: float
    method: str  # 'structural', 'svm', 'fallback'
    valid_response_types: list[str]

    @property
    def is_commitment(self) -> bool:
        """True if this trigger expects a commitment response (agree/decline/defer)."""
        return self.trigger_type == TriggerType.COMMITMENT

    @property
    def is_question(self) -> bool:
        """True if this trigger needs an answer."""
        return self.trigger_type == TriggerType.QUESTION

    @property
    def is_reaction(self) -> bool:
        """True if this trigger needs an emotional/reactive response."""
        return self.trigger_type == TriggerType.REACTION

    @property
    def is_social(self) -> bool:
        """True if this trigger is a greeting or acknowledgment."""
        return self.trigger_type == TriggerType.SOCIAL


# =============================================================================
# Trigger Classifier
# =============================================================================


class HybridTriggerClassifier:
    """Classifier for incoming messages using structural patterns + SVM.

    Strategy:
    1. Check structural patterns (fast, high precision)
    2. If no match, try trained SVM classifier (if available)
    3. Fallback to STATEMENT if still unsure
    """

    # Centroid verification thresholds (experimental feature)
    CENTROID_VERIFY_THRESHOLD = 0.4  # Minimum similarity to accept centroid verification
    CENTROID_MARGIN = 0.15  # Margin by which another class must beat predicted class

    def __init__(
        self,
        model_path: Path | str | None = None,
        use_centroid_verification: bool = False,
    ):
        """Initialize the trigger classifier.

        Args:
            model_path: Path to trained model directory. Defaults to ~/.jarvis/trigger_classifier_model/
            use_centroid_verification: If True, verify SVM predictions with centroid distance.
                This is an experimental feature that may improve accuracy for some cases.
        """
        self._embedder = None
        self._lock = threading.Lock()

        # Trained SVM model
        self._svm = None
        self._svm_labels: list[str] | None = None
        self._svm_loaded = False
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        # Centroid verification (experimental)
        self._use_centroid_verification = use_centroid_verification
        self._centroids: dict[str, np.ndarray] | None = None
        self._centroids_loaded = False

        # Try to load the trained model
        self._load_trained_model()

    def _get_embedder(self):
        """Get or create embedder."""
        if self._embedder is None:
            from jarvis.embedding_adapter import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def _load_centroids(self) -> None:
        """Load centroids from file if available.

        Centroids are the mean embeddings for each class, computed during training.
        Used for optional centroid verification to catch edge cases.
        """
        if self._centroids_loaded:
            return

        centroids_path = self._model_path / "centroids.npy"
        if not centroids_path.exists():
            logger.debug("Centroids not found at %s", centroids_path)
            self._centroids_loaded = True
            return

        try:
            data = np.load(centroids_path, allow_pickle=True).item()
            self._centroids = {
                label: np.array(centroid) for label, centroid in data.items()
            }
            self._centroids_loaded = True
            logger.info("Loaded centroids for %d trigger classes", len(self._centroids))
        except Exception as e:
            logger.warning("Failed to load centroids: %s", e)
            self._centroids_loaded = True

    def _verify_with_centroid(
        self,
        embedding: np.ndarray,
        predicted_type: TriggerType,
    ) -> tuple[TriggerType, float, bool]:
        """Verify SVM prediction using centroid distance.

        If the embedding is closer to another class's centroid by a significant
        margin, override the SVM prediction. This catches edge cases where the
        SVM is confident but semantically wrong.

        Args:
            embedding: Normalized text embedding.
            predicted_type: SVM's predicted trigger type.

        Returns:
            Tuple of (final_type, confidence, was_verified).
            was_verified=True if prediction was confirmed, False if overridden.
        """
        if self._centroids is None:
            return predicted_type, 0.0, True

        # Compute similarity to all centroids
        similarities = {}
        for label, centroid in self._centroids.items():
            # Cosine similarity (vectors are normalized)
            sim = float(np.dot(embedding, centroid))
            similarities[label] = sim

        predicted_sim = similarities.get(predicted_type.value, 0.0)
        best_label = max(similarities, key=similarities.get)
        best_sim = similarities[best_label]

        # Decision logic:
        # 1. If predicted class has high similarity -> confirm
        # 2. If another class is significantly closer -> override
        # 3. Otherwise -> confirm prediction

        if predicted_sim >= self.CENTROID_VERIFY_THRESHOLD:
            # Prediction confirmed by centroid proximity
            return predicted_type, predicted_sim, True

        if best_sim - predicted_sim > self.CENTROID_MARGIN:
            # Another class is significantly closer - override
            try:
                override_type = TriggerType(best_label)
                logger.debug(
                    "Centroid override: %s -> %s (sim: %.2f vs %.2f)",
                    predicted_type.value, best_label, best_sim, predicted_sim
                )
                return override_type, best_sim, False
            except ValueError:
                pass

        # Default: trust SVM prediction
        return predicted_type, predicted_sim, True

    def _load_trained_model(self) -> None:
        """Load the trained SVM classifier if available."""
        svm_path = self._model_path / "svm.pkl"
        config_path = self._model_path / "config.json"

        if not svm_path.exists() or not config_path.exists():
            logger.debug("Trained model not found at %s", self._model_path)
            return

        try:
            with open(svm_path, "rb") as f:
                self._svm = pickle.load(f)
            with open(config_path) as f:
                config = json.load(f)
                self._svm_labels = config.get("labels", [])

            self._svm_loaded = True
            logger.info("Loaded trained trigger classifier from %s", self._model_path)
        except Exception as e:
            logger.warning("Failed to load trained model: %s", e)
            self._svm = None
            self._svm_labels = None

    def _get_svm_threshold(self, trigger_type: TriggerType) -> float:
        """Get the SVM confidence threshold for a trigger type.

        Uses per-class thresholds tuned based on performance analysis.
        Higher thresholds for hard/important classes (COMMITMENT),
        lower for classes with strong structural patterns (SOCIAL).
        """
        return PER_CLASS_SVM_THRESHOLDS.get(trigger_type, DEFAULT_SVM_THRESHOLD)

    def _match_svm(self, text: str) -> tuple[TriggerType | None, float, str]:
        """Match using trained SVM classifier.

        Uses per-class confidence thresholds - higher for important classes
        like COMMITMENT (0.50), lower for SOCIAL (0.25) where structural
        patterns are strong.

        Returns:
            Tuple of (trigger_type, confidence, method) or (None, 0, 'svm').
            method is 'svm' or 'svm+centroid' if centroid verification was used.
        """
        if not self._svm_loaded or self._svm is None:
            return None, 0.0, "svm"

        try:
            embedder = self._get_embedder()
            embedding = embedder.encode([text], normalize=True)[0]

            # Get prediction and probability
            probs = self._svm.predict_proba(embedding.reshape(1, -1))[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            label = self._svm_labels[pred_idx]

            # Map label to TriggerType
            try:
                trigger_type = TriggerType(label)
            except ValueError:
                logger.warning("Unknown SVM label: %s", label)
                return None, confidence, "svm"

            # Use per-class threshold
            threshold = self._get_svm_threshold(trigger_type)
            if confidence < threshold:
                return None, confidence, "svm"

            # Optional: verify with centroid distance
            method = "svm"
            if self._use_centroid_verification:
                self._load_centroids()
                if self._centroids is not None:
                    verified_type, _, was_verified = self._verify_with_centroid(
                        embedding, trigger_type
                    )
                    if not was_verified:
                        # Centroid overrode SVM prediction
                        trigger_type = verified_type
                        method = "svm+centroid"

            return trigger_type, confidence, method

        except Exception as e:
            logger.warning("SVM classification failed: %s", e)
            return None, 0.0, "svm"

    def _match_structural(self, text: str) -> tuple[TriggerType | None, float]:
        """Match against structural patterns.

        Returns:
            Tuple of (trigger_type, confidence) or (None, 0) if no match.
        """
        text = text.strip()

        for pattern, trigger_type, confidence in STRUCTURAL_PATTERNS:
            if pattern.search(text):
                return trigger_type, confidence

        return None, 0.0

    def classify(self, text: str, use_svm: bool = True) -> TriggerClassification:
        """Classify a trigger message.

        Args:
            text: The incoming message to classify.
            use_svm: Whether to use the trained SVM classifier.

        Returns:
            TriggerClassification with type, confidence, and valid response types.
        """
        if not text or not text.strip():
            return TriggerClassification(
                trigger_type=TriggerType.UNKNOWN,
                confidence=0.0,
                method="empty",
                valid_response_types=TRIGGER_TO_RESPONSE_TYPES[TriggerType.UNKNOWN],
            )

        # Step 1: Try structural patterns (highest precision)
        struct_type, struct_conf = self._match_structural(text)

        if struct_type is not None and struct_conf >= 0.85:
            # High confidence structural match - use it
            return TriggerClassification(
                trigger_type=struct_type,
                confidence=struct_conf,
                method="structural",
                valid_response_types=TRIGGER_TO_RESPONSE_TYPES[struct_type],
            )

        # Step 2: Try trained SVM classifier (if available)
        if use_svm and self._svm_loaded:
            svm_type, svm_conf, svm_method = self._match_svm(text)

            # _match_svm already applies per-class threshold, so if svm_type is not None,
            # the confidence already passed the threshold check
            if svm_type is not None:
                # If structural had a partial match, check if they agree
                if struct_type is not None and struct_type == svm_type:
                    # They agree - boost confidence
                    return TriggerClassification(
                        trigger_type=struct_type,
                        confidence=max(struct_conf, svm_conf),
                        method="structural+svm",
                        valid_response_types=TRIGGER_TO_RESPONSE_TYPES[struct_type],
                    )

                # Use SVM prediction (with or without centroid verification)
                return TriggerClassification(
                    trigger_type=svm_type,
                    confidence=svm_conf,
                    method=svm_method,
                    valid_response_types=TRIGGER_TO_RESPONSE_TYPES[svm_type],
                )

        # Step 3: Use low-confidence structural match if available
        if struct_type is not None:
            return TriggerClassification(
                trigger_type=struct_type,
                confidence=struct_conf,
                method="structural_low",
                valid_response_types=TRIGGER_TO_RESPONSE_TYPES[struct_type],
            )

        # Step 4: Fallback to STATEMENT
        return TriggerClassification(
            trigger_type=TriggerType.STATEMENT,
            confidence=0.3,
            method="fallback",
            valid_response_types=TRIGGER_TO_RESPONSE_TYPES[TriggerType.STATEMENT],
        )


# =============================================================================
# Singleton Access
# =============================================================================

_classifier: HybridTriggerClassifier | None = None
_lock = threading.Lock()


def get_trigger_classifier(model_path: Path | str | None = None) -> HybridTriggerClassifier:
    """Get the singleton trigger classifier.

    Args:
        model_path: Optional path to trained model. Only used on first call.

    Returns:
        The singleton HybridTriggerClassifier instance.
    """
    global _classifier
    if _classifier is None:
        with _lock:
            if _classifier is None:
                _classifier = HybridTriggerClassifier(model_path=model_path)
    return _classifier


def reset_trigger_classifier() -> None:
    """Reset the singleton classifier (useful for testing or reloading model)."""
    global _classifier
    with _lock:
        _classifier = None


def classify_trigger(text: str, use_svm: bool = True) -> TriggerClassification:
    """Convenience function to classify a trigger message.

    Args:
        text: The message to classify.
        use_svm: Whether to use the trained SVM classifier if available.

    Returns:
        TriggerClassification with type, confidence, and valid response types.
    """
    return get_trigger_classifier().classify(text, use_svm=use_svm)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TriggerType",
    "TRIGGER_TO_RESPONSE_TYPES",
    "PER_CLASS_SVM_THRESHOLDS",
    "DEFAULT_SVM_THRESHOLD",
    "TriggerClassification",
    "HybridTriggerClassifier",
    "get_trigger_classifier",
    "reset_trigger_classifier",
    "classify_trigger",
]
