"""Hybrid Trigger Classifier - Structural patterns + Trained SVM + Centroid verification.

Classifies incoming messages (triggers) to determine what type of response is needed.
Uses a multi-layer approach for robust classification.

Strategy:
1. Structural patterns (fast regex) - high precision matches (~95% when fires)
2. Trained SVM classifier (optional) - 71% accuracy on 5 merged classes
3. Centroid verification (light ML) - for ambiguous cases
4. Fallback to STATEMENT - when unsure

The trained SVM uses 5 merged classes that map back to fine-grained types:
- acknowledgment -> GREETING, ACKNOWLEDGMENT
- action -> INVITATION, REQUEST
- emotional -> GOOD_NEWS, BAD_NEWS, REACTION_PROMPT
- question -> INFO_QUESTION, YN_QUESTION
- statement -> STATEMENT

Usage:
    from jarvis.trigger_classifier import classify_trigger, TriggerType

    result = classify_trigger("Want to grab lunch?")
    print(result.trigger_type)  # TriggerType.INVITATION
    print(result.confidence)    # 0.95
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jarvis.embedding_adapter import Embedder

logger = logging.getLogger(__name__)


# =============================================================================
# Trigger Types
# =============================================================================


class TriggerType(str, Enum):
    """Types of incoming messages based on what response they need."""

    # Commitment triggers - need yes/no/maybe options
    INVITATION = "invitation"      # "Want to hang out?"
    REQUEST = "request"            # "Can you pick me up?"
    YN_QUESTION = "yn_question"    # "Did you finish?"

    # Info triggers - need informational response
    INFO_QUESTION = "info_question"  # "What time?", "Where?"

    # Reaction triggers - need emotional/reactive response
    GOOD_NEWS = "good_news"        # "I got the job!"
    BAD_NEWS = "bad_news"          # "My car broke down"
    REACTION_PROMPT = "reaction"   # "omg did you see that??"

    # Simple triggers - need simple acknowledgment
    STATEMENT = "statement"        # Sharing info
    GREETING = "greeting"          # "Hey!"
    ACKNOWLEDGMENT = "ack"         # "ok", "thanks"

    # Fallback
    UNKNOWN = "unknown"


# Mapping from merged SVM classes to fine-grained trigger types
# The SVM predicts 5 classes, which map to multiple possible TriggerTypes
MERGED_TO_TRIGGER_TYPES: dict[str, list[TriggerType]] = {
    "acknowledgment": [TriggerType.ACKNOWLEDGMENT, TriggerType.GREETING],
    "action": [TriggerType.INVITATION, TriggerType.REQUEST],
    "emotional": [TriggerType.REACTION_PROMPT, TriggerType.GOOD_NEWS, TriggerType.BAD_NEWS],
    "question": [TriggerType.INFO_QUESTION, TriggerType.YN_QUESTION],
    "statement": [TriggerType.STATEMENT],
}

# Default model path
DEFAULT_MODEL_PATH = Path.home() / ".jarvis" / "trigger_classifier_model"


# What response types are valid for each trigger type
TRIGGER_TO_RESPONSE_TYPES: dict[TriggerType, list[str]] = {
    TriggerType.INVITATION: ["AGREE", "DECLINE", "DEFER", "QUESTION"],
    TriggerType.REQUEST: ["AGREE", "DECLINE", "DEFER", "QUESTION"],
    TriggerType.YN_QUESTION: ["YES", "NO", "MAYBE", "ANSWER"],
    TriggerType.INFO_QUESTION: ["ANSWER", "DEFER", "QUESTION"],
    TriggerType.GOOD_NEWS: ["REACT_POSITIVE", "QUESTION", "ACKNOWLEDGE"],
    TriggerType.BAD_NEWS: ["REACT_SYMPATHY", "QUESTION", "ACKNOWLEDGE"],
    TriggerType.REACTION_PROMPT: ["REACT_POSITIVE", "QUESTION", "ANSWER"],
    TriggerType.STATEMENT: ["ACKNOWLEDGE", "REACT", "QUESTION"],
    TriggerType.GREETING: ["GREETING", "QUESTION"],
    TriggerType.ACKNOWLEDGMENT: ["ACKNOWLEDGE"],
    TriggerType.UNKNOWN: ["ANSWER", "QUESTION", "ACKNOWLEDGE"],
}


# =============================================================================
# Structural Patterns (High Precision)
# =============================================================================

# Order matters! More specific patterns first.
STRUCTURAL_PATTERNS: list[tuple[re.Pattern, TriggerType, float]] = [
    # === TAPBACKS (check first - these are metadata not real messages) ===
    (re.compile(r'^(Liked|Loved|Laughed at|Emphasized|Questioned|Disliked)\s+["\u201c\u201d]', re.I),
     TriggerType.ACKNOWLEDGMENT, 0.95),

    # === GREETINGS (exact matches) ===
    (re.compile(r"^(hey|hi|hello|yo|sup|what'?s up|wassup|hiya|howdy)[\s!?]*$", re.I),
     TriggerType.GREETING, 0.95),

    # === ACKNOWLEDGMENTS (exact matches) ===
    (re.compile(r"^(ok|okay|k|kk|sure|bet|got it|sounds good|cool|alright|aight|word)[\s!.]*$", re.I),
     TriggerType.ACKNOWLEDGMENT, 0.95),
    (re.compile(r"^(thanks|thank you|thx|ty|appreciate it)[\s!.]*$", re.I),
     TriggerType.ACKNOWLEDGMENT, 0.95),
    (re.compile(r"^(lol|lmao|haha+|hehe+|😂|🤣|💀)+[\s!]*$", re.I),
     TriggerType.ACKNOWLEDGMENT, 0.90),

    # === INVITATIONS (asking to do something together) ===
    # Must have question-like structure AND social intent
    (re.compile(r"\b(wanna|want to|down to|dtf|tryna|trying to)\s+.*(hang|chill|go|come|grab|get|play|watch|do)\b.*\?", re.I),
     TriggerType.INVITATION, 0.95),
    (re.compile(r"^(wanna|want to|down to)\s+\w+.*\?", re.I),
     TriggerType.INVITATION, 0.90),
    (re.compile(r"\b(you|u)\s+(free|available|busy|down)\s*(today|tonight|tomorrow|tmrw|later|this weekend|rn)?\s*\?", re.I),
     TriggerType.INVITATION, 0.95),
    (re.compile(r"^(let'?s|lets)\s+(go|hang|chill|grab|get|do|play|watch)\b", re.I),
     TriggerType.INVITATION, 0.85),
    (re.compile(r"\bcome (over|through|thru|hang|chill)\b.*\?", re.I),
     TriggerType.INVITATION, 0.90),

    # === REQUESTS (asking someone to do something FOR you) ===
    (re.compile(r"^(can|could|would|will)\s+(you|u)\s+\w+", re.I),
     TriggerType.REQUEST, 0.85),
    (re.compile(r"^(please|pls|plz)\s+\w+", re.I),
     TriggerType.REQUEST, 0.85),
    (re.compile(r"\b(pick me up|drop me off|send me|get me|help me)\b", re.I),
     TriggerType.REQUEST, 0.90),
    (re.compile(r"\b(lmk|let me know)\s+(if|when|what)\b", re.I),
     TriggerType.REQUEST, 0.80),

    # === INFO QUESTIONS (what/when/where/who/how) ===
    (re.compile(r"^(what|what'?s)\s+(time|day|the plan|up|going on|happening)\b", re.I),
     TriggerType.INFO_QUESTION, 0.95),
    (re.compile(r"^(when|where|who|which|how)\s+\w+", re.I),
     TriggerType.INFO_QUESTION, 0.90),
    (re.compile(r"\b(what time|how long|how much|how many)\b.*\?", re.I),
     TriggerType.INFO_QUESTION, 0.95),

    # === YES/NO QUESTIONS ===
    (re.compile(r"^(do|does|did|is|are|was|were|have|has|can|could|will|would|should)\s+(you|u|we|they|i|it|he|she)\b.*\?", re.I),
     TriggerType.YN_QUESTION, 0.85),

    # === GOOD NEWS ===
    (re.compile(r"\b(i got|i passed|i made|i won|we won|i'm engaged|i'm pregnant)\b", re.I),
     TriggerType.GOOD_NEWS, 0.85),
    (re.compile(r"\b(finally|just got|so excited|so happy|great news|good news)\b", re.I),
     TriggerType.GOOD_NEWS, 0.75),

    # === BAD NEWS ===
    (re.compile(r"\b(i lost|i failed|i got fired|i'm sick|someone died|passed away)\b", re.I),
     TriggerType.BAD_NEWS, 0.85),
    (re.compile(r"\b(so sad|so upset|terrible|awful|bad news|unfortunately)\b", re.I),
     TriggerType.BAD_NEWS, 0.75),

    # === REACTION PROMPTS (wanting your reaction to something) ===
    (re.compile(r"^(omg|oh my god|dude|bro|yo)\b.+(\?|!{2,}|wtf|crazy|insane)", re.I),
     TriggerType.REACTION_PROMPT, 0.85),
    (re.compile(r"\b(did you (see|hear|watch)|have you seen)\b", re.I),
     TriggerType.REACTION_PROMPT, 0.85),
    (re.compile(r"\b(can you believe|isn't that|wasn't that|how crazy)\b", re.I),
     TriggerType.REACTION_PROMPT, 0.85),

    # === VENTING (emotional expression, subset of bad news) ===
    (re.compile(r"^(fuck+|shit+|damn+|ugh+|omfg|fml)\b", re.I),
     TriggerType.BAD_NEWS, 0.80),  # Treat venting as bad news for response purposes

    # === FALLBACK: Any question mark = YN_QUESTION ===
    (re.compile(r"\?\s*$"), TriggerType.YN_QUESTION, 0.60),
]


# =============================================================================
# Centroid Examples (for ML verification)
# =============================================================================

CENTROID_EXAMPLES: dict[TriggerType, list[str]] = {
    TriggerType.INVITATION: [
        "Want to grab lunch?",
        "Wanna hang out later?",
        "Down to play basketball?",
        "Are you free tonight?",
        "Let's go get food",
        "Come over to my place",
        "Tryna chill this weekend?",
        "You busy tomorrow?",
    ],
    TriggerType.REQUEST: [
        "Can you pick me up?",
        "Could you send me that file?",
        "Please call me back",
        "Can you help me with this?",
        "Would you mind checking?",
        "Let me know when you're done",
        "Can you grab me a coffee?",
    ],
    TriggerType.YN_QUESTION: [
        "Did you finish the report?",
        "Is it raining outside?",
        "Are you coming to the party?",
        "Have you eaten yet?",
        "Did you see my message?",
        "Is everything okay?",
        "Do you have the keys?",
    ],
    TriggerType.INFO_QUESTION: [
        "What time is the meeting?",
        "Where are we going?",
        "When does it start?",
        "How much does it cost?",
        "Who else is coming?",
        "Which restaurant?",
        "How do I get there?",
    ],
    TriggerType.GOOD_NEWS: [
        "I got the job!",
        "We won the game!",
        "I passed the exam!",
        "I'm engaged!",
        "Just got promoted!",
        "Finally finished the project!",
        "Great news about the deal!",
    ],
    TriggerType.BAD_NEWS: [
        "I lost my wallet",
        "My car broke down",
        "I failed the test",
        "I'm so stressed",
        "This sucks",
        "Ugh worst day ever",
        "I got rejected",
    ],
    TriggerType.REACTION_PROMPT: [
        "Omg did you see that play??",
        "Dude that's insane!",
        "Bro can you believe this?",
        "Have you seen the news?",
        "Yo watch this video",
        "Did you hear what happened?",
    ],
    TriggerType.STATEMENT: [
        "I'm on my way",
        "The meeting got moved to 3pm",
        "Just finished work",
        "I'll be there soon",
        "Got the tickets",
        "Running a bit late",
    ],
    TriggerType.GREETING: [
        "Hey!",
        "Hi there",
        "What's up?",
        "Yo",
        "Hello!",
        "Sup",
    ],
    TriggerType.ACKNOWLEDGMENT: [
        "Ok",
        "Sounds good",
        "Got it",
        "Thanks!",
        "Bet",
        "Cool",
        "Lol",
    ],
}


# =============================================================================
# Classification Result
# =============================================================================


@dataclass
class TriggerClassification:
    """Result from classifying a trigger message."""

    trigger_type: TriggerType
    confidence: float
    method: str  # 'structural', 'centroid', 'fallback'
    valid_response_types: list[str]

    @property
    def is_commitment(self) -> bool:
        """True if this trigger expects a commitment response (yes/no/maybe)."""
        return self.trigger_type in {
            TriggerType.INVITATION,
            TriggerType.REQUEST,
            TriggerType.YN_QUESTION
        }

    @property
    def needs_info(self) -> bool:
        """True if this trigger needs an informational response."""
        return self.trigger_type == TriggerType.INFO_QUESTION

    @property
    def is_emotional(self) -> bool:
        """True if this trigger needs an emotional/reactive response."""
        return self.trigger_type in {
            TriggerType.GOOD_NEWS,
            TriggerType.BAD_NEWS,
            TriggerType.REACTION_PROMPT
        }


# =============================================================================
# Hybrid Trigger Classifier
# =============================================================================


class HybridTriggerClassifier:
    """Hybrid classifier for incoming messages using structural + ML approach.

    Strategy:
    1. Check structural patterns (fast, high precision)
    2. If no match, try trained SVM classifier (if available)
    3. If no match or low confidence, verify with centroid similarity
    4. Fallback to STATEMENT if still unsure
    """

    CENTROID_THRESHOLD = 0.6  # Minimum similarity to trust centroid
    SVM_THRESHOLD = 0.5  # Minimum probability to trust SVM

    def __init__(self, model_path: Path | str | None = None):
        self._embedder = None
        self._centroids: dict[TriggerType, np.ndarray] | None = None
        self._lock = threading.Lock()

        # Trained SVM model
        self._svm = None
        self._svm_labels: list[str] | None = None
        self._svm_loaded = False
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        # Try to load the trained model
        self._load_trained_model()

    def _get_embedder(self):
        """Get or create embedder."""
        if self._embedder is None:
            from jarvis.embedding_adapter import get_embedder
            self._embedder = get_embedder()
        return self._embedder

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
            with open(config_path, "r") as f:
                config = json.load(f)
                self._svm_labels = config.get("labels", [])

            self._svm_loaded = True
            logger.info("Loaded trained trigger classifier from %s", self._model_path)
        except Exception as e:
            logger.warning("Failed to load trained model: %s", e)
            self._svm = None
            self._svm_labels = None

    def _match_svm(self, text: str) -> tuple[TriggerType | None, float, str]:
        """Match using trained SVM classifier.

        Returns:
            Tuple of (trigger_type, confidence, merged_class) or (None, 0, "").
        """
        if not self._svm_loaded or self._svm is None:
            return None, 0.0, ""

        try:
            embedder = self._get_embedder()
            embedding = embedder.encode([text], normalize=True)

            # Get prediction and probability
            probs = self._svm.predict_proba(embedding)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            if confidence < self.SVM_THRESHOLD:
                return None, confidence, ""

            merged_class = self._svm_labels[pred_idx]

            # Map merged class to best trigger type
            possible_types = MERGED_TO_TRIGGER_TYPES.get(merged_class, [TriggerType.STATEMENT])

            # Return the first (most common) type for the merged class
            trigger_type = possible_types[0] if possible_types else TriggerType.STATEMENT

            return trigger_type, confidence, merged_class

        except Exception as e:
            logger.warning("SVM classification failed: %s", e)
            return None, 0.0, ""

    def _ensure_centroids(self) -> None:
        """Compute and cache centroids for each trigger type."""
        if self._centroids is not None:
            return

        with self._lock:
            if self._centroids is not None:
                return

            try:
                embedder = self._get_embedder()
                centroids = {}

                for trigger_type, examples in CENTROID_EXAMPLES.items():
                    # Compute embeddings for all examples
                    embeddings = embedder.encode(examples, normalize=True)
                    # Compute centroid (mean embedding)
                    centroid = np.mean(embeddings, axis=0)
                    # Normalize
                    centroid = centroid / np.linalg.norm(centroid)
                    centroids[trigger_type] = centroid.astype(np.float32)

                self._centroids = centroids
                logger.info("Computed centroids for %d trigger types", len(centroids))

            except Exception as e:
                logger.warning("Failed to compute centroids: %s", e)

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

    def _match_centroid(self, text: str) -> tuple[TriggerType | None, float]:
        """Match against centroids using embedding similarity.

        Returns:
            Tuple of (trigger_type, similarity) or (None, 0) if no good match.
        """
        try:
            self._ensure_centroids()

            if not self._centroids:
                return None, 0.0

            embedder = self._get_embedder()
            embedding = embedder.encode([text], normalize=True)[0]

            # Find most similar centroid
            best_type = None
            best_sim = -1.0

            for trigger_type, centroid in self._centroids.items():
                sim = float(np.dot(embedding, centroid))
                if sim > best_sim:
                    best_sim = sim
                    best_type = trigger_type

            if best_sim >= self.CENTROID_THRESHOLD:
                return best_type, best_sim

            return None, best_sim

        except Exception as e:
            logger.warning("Centroid matching failed: %s", e)
            return None, 0.0

    def classify(
        self,
        text: str,
        use_centroid: bool = True,
        use_svm: bool = True
    ) -> TriggerClassification:
        """Classify a trigger message.

        Args:
            text: The incoming message to classify.
            use_centroid: Whether to use centroid verification for ambiguous cases.
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
            svm_type, svm_conf, merged_class = self._match_svm(text)

            if svm_type is not None and svm_conf >= self.SVM_THRESHOLD:
                # If structural had a partial match, check if they agree
                if struct_type is not None:
                    # Check if struct_type is compatible with merged_class
                    compatible_types = MERGED_TO_TRIGGER_TYPES.get(merged_class, [])
                    if struct_type in compatible_types:
                        # They agree - use structural type with boosted confidence
                        return TriggerClassification(
                            trigger_type=struct_type,
                            confidence=max(struct_conf, svm_conf),
                            method="structural+svm",
                            valid_response_types=TRIGGER_TO_RESPONSE_TYPES[struct_type],
                        )

                # Use SVM prediction
                return TriggerClassification(
                    trigger_type=svm_type,
                    confidence=svm_conf,
                    method=f"svm:{merged_class}",
                    valid_response_types=TRIGGER_TO_RESPONSE_TYPES[svm_type],
                )

        # Step 3: Try centroid verification
        if use_centroid:
            cent_type, cent_conf = self._match_centroid(text)

            if cent_type is not None and cent_conf >= self.CENTROID_THRESHOLD:
                # Good centroid match
                # If structural also had a match, prefer structural if they agree
                if struct_type == cent_type:
                    return TriggerClassification(
                        trigger_type=struct_type,
                        confidence=max(struct_conf, cent_conf),
                        method="structural+centroid",
                        valid_response_types=TRIGGER_TO_RESPONSE_TYPES[struct_type],
                    )

                # Centroid match only
                return TriggerClassification(
                    trigger_type=cent_type,
                    confidence=cent_conf,
                    method="centroid",
                    valid_response_types=TRIGGER_TO_RESPONSE_TYPES[cent_type],
                )

        # Step 4: Use low-confidence structural match if available
        if struct_type is not None:
            return TriggerClassification(
                trigger_type=struct_type,
                confidence=struct_conf,
                method="structural_low",
                valid_response_types=TRIGGER_TO_RESPONSE_TYPES[struct_type],
            )

        # Step 5: Fallback to STATEMENT
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
    "MERGED_TO_TRIGGER_TYPES",
    "TriggerClassification",
    "HybridTriggerClassifier",
    "get_trigger_classifier",
    "reset_trigger_classifier",
    "classify_trigger",
]
