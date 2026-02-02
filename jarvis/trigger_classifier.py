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

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from jarvis.classifiers import (
    CentroidMixin,
    EmbedderMixin,
    SingletonFactory,
    StructuralPatternMatcher,
    SVMModelMixin,
)
from jarvis.config import get_config, get_trigger_classifier_path

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


def _get_default_model_path() -> Path:
    """Get the default model path based on configured embedding model."""
    return get_trigger_classifier_path()


def _get_svm_thresholds() -> dict[TriggerType, float]:
    """Build per-class SVM thresholds from config.

    Higher thresholds for important/hard classes, lower for classes with strong structural patterns.
    """
    cfg = get_config().classifier_thresholds
    return {
        TriggerType.COMMITMENT: cfg.trigger_svm_commitment,
        TriggerType.QUESTION: cfg.trigger_svm_question,
        TriggerType.REACTION: cfg.trigger_svm_reaction,
        TriggerType.SOCIAL: cfg.trigger_svm_social,
        TriggerType.STATEMENT: cfg.trigger_svm_statement,
    }


def _get_default_svm_threshold() -> float:
    """Get default SVM threshold from config for unknown types."""
    return get_config().classifier_thresholds.trigger_svm_default


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
    (
        re.compile(
            r'^(Liked|Loved|Laughed at|Emphasized|Questioned|Disliked)\s+["\u201c\u201d]', re.I
        ),
        TriggerType.SOCIAL,
        0.95,
    ),
    # === SOCIAL: Greetings (exact matches) ===
    (
        re.compile(
            r"^(hey+|hi+|hello+|yo+|sup|what'?s up|wassup|hiya|howdy|hola)"
            r"[\s!?\u200d\U0001f64b\U0001f600-\U0001f64f]*$",
            re.I,
        ),
        TriggerType.SOCIAL,
        0.95,
    ),
    (
        re.compile(r"^what'?s up\s+(homie|dude|friend|bro|man|guys?)[\s!?]*$", re.I),
        TriggerType.SOCIAL,
        0.90,
    ),
    (re.compile(r"^how (are|r) (you|u|ya)[\s!?]*$", re.I), TriggerType.SOCIAL, 0.95),
    (re.compile(r"^how('?s| is) it going[\s!?]*$", re.I), TriggerType.SOCIAL, 0.95),
    (
        re.compile(r"^(good|gm|gn)\s*(night|morning|evening|afternoon)?['\s!?]*$", re.I),
        TriggerType.SOCIAL,
        0.95,
    ),
    (
        re.compile(
            r"^(happy|merry)\s+(thanksgiving|christmas|birthday|holiday|new year|easter)", re.I
        ),
        TriggerType.SOCIAL,
        0.90,
    ),
    (re.compile(r"^(what'?s good|wsg|wsup|whaddup)[\s!?]*$", re.I), TriggerType.SOCIAL, 0.90),
    (re.compile(r"^love (you|u|ya)[\s!]*$", re.I), TriggerType.SOCIAL, 0.85),
    # === SOCIAL: Acknowledgments (exact matches) ===
    (
        re.compile(
            r"^(ok|okay|k+|sure|bet|got it|sounds good|cool|alright|aight|word)[\s!.]*$", re.I
        ),
        TriggerType.SOCIAL,
        0.95,
    ),
    (
        re.compile(
            r"^(yes+|yea+|yeah+|yup|yep|nah+|nope|true|for sure|all\s*right|could be)[\s!.]*$", re.I
        ),
        TriggerType.SOCIAL,
        0.95,
    ),
    (
        re.compile(r"^(thanks|thank you|thx|ty|appreciate it)[\s\w!.]*$", re.I),
        TriggerType.SOCIAL,
        0.95,
    ),
    (re.compile(r"^(lol|lmao|haha+|hehe+|😂|🤣|💀)+[\s!]*$", re.I), TriggerType.SOCIAL, 0.90),
    (re.compile(r"^(ik|i know)[\s!]*(haha+)?[\s!]*$", re.I), TriggerType.SOCIAL, 0.90),
    (
        re.compile(r"^(ofc|of course|def|definitely|obviously|obvi)[\s!.]*$", re.I),
        TriggerType.SOCIAL,
        0.90,
    ),
    (re.compile(r"^(right+|i know right|ikr)[\s!]*$", re.I), TriggerType.SOCIAL, 0.90),
    (
        re.compile(r"^(have fun|enjoy|nice|noice|dope|sick|lit|fire)[\s!.]*$", re.I),
        TriggerType.SOCIAL,
        0.85,
    ),
    # === COMMITMENT: Invitations (asking to do something together) ===
    (
        re.compile(
            r"\b(wanna|want to|down to|dtf|tryna|trying to)\s+.*"
            r"(hang|chill|go|come|grab|get|play|watch|do)\b.*\?",
            re.I,
        ),
        TriggerType.COMMITMENT,
        0.95,
    ),
    (re.compile(r"^(wanna|want to|down to)\s+\w+.*\?", re.I), TriggerType.COMMITMENT, 0.90),
    (
        re.compile(
            r"\b(you|u)\s+(free|available|busy|down)\s*"
            r"(today|tonight|tomorrow|tmrw|later|this weekend|rn)?\s*\?",
            re.I,
        ),
        TriggerType.COMMITMENT,
        0.95,
    ),
    (
        re.compile(r"^(let'?s|lets)\s+(go|hang|chill|grab|get|do|play|watch|call|meet)\b", re.I),
        TriggerType.COMMITMENT,
        0.85,
    ),
    (
        re.compile(r"\bcome (over|through|thru|hang|chill)\b.*\?", re.I),
        TriggerType.COMMITMENT,
        0.90,
    ),
    (re.compile(r"^(pull up|slide|bool|link)\b", re.I), TriggerType.COMMITMENT, 0.85),
    # === COMMITMENT: Requests (asking someone to do something FOR you) ===
    (
        re.compile(
            r"^(can|could|would|will)\s+(you|u)\s+(send|check|grab|get|pick|help|call|text|set|setup|bring)\b",
            re.I,
        ),
        TriggerType.COMMITMENT,
        0.90,
    ),
    (
        re.compile(r"^(can|could|would|will)\s+(you|u)\s+(please|pls|plz)\b", re.I),
        TriggerType.COMMITMENT,
        0.90,
    ),
    (
        re.compile(
            r"^(please|pls|plz)\s+(send|check|help|call|grab|get|pick|bring|remind)\b", re.I
        ),
        TriggerType.COMMITMENT,
        0.85,
    ),
    (re.compile(r"\b(pick me up|drop me off)\b", re.I), TriggerType.COMMITMENT, 0.90),
    (
        re.compile(r"^(tell|ask|remind)\s+(everyone|them|him|her|me)\b", re.I),
        TriggerType.COMMITMENT,
        0.85,
    ),
    # === QUESTION: Info questions (what/when/where/who/how) ===
    (
        re.compile(r"^(what|what'?s)\s+(time|day|the plan|up|going on|happening)\b", re.I),
        TriggerType.QUESTION,
        0.95,
    ),
    (re.compile(r"^what\s+(did|do|does|are|is|was|were)\s+", re.I), TriggerType.QUESTION, 0.90),
    (re.compile(r"^whatchu\s+", re.I), TriggerType.QUESTION, 0.85),
    (re.compile(r"^how('?s|s)?\s+\w+", re.I), TriggerType.QUESTION, 0.85),
    (re.compile(r"^(when|where|who|which|why)\s+.+\?$", re.I), TriggerType.QUESTION, 0.90),
    (
        re.compile(r"\b(what time|how long|how much|how many)\b.*\?", re.I),
        TriggerType.QUESTION,
        0.95,
    ),
    (re.compile(r"\bwhat\s+.{5,}\?$", re.I), TriggerType.QUESTION, 0.80),
    # === QUESTION: Yes/No questions ===
    (
        re.compile(
            r"^(do|does|did|is|are|was|were|have|has|can|could|will|would|should)\s+(you|u|we|they|i|it|he|she)\b.*\?",
            re.I,
        ),
        TriggerType.QUESTION,
        0.85,
    ),
    # === REACTION: Reaction prompts (wanting your reaction) ===
    (
        re.compile(r"^(omg|oh my god)\b.+(\?|!{2,}|wtf|crazy|insane)", re.I),
        TriggerType.REACTION,
        0.85,
    ),
    (
        re.compile(r"\b(did you (see|hear|watch)|have you seen)\b.*\?", re.I),
        TriggerType.REACTION,
        0.85,
    ),
    (
        re.compile(r"\b(can you believe|isn't that|wasn't that)\b.*\?", re.I),
        TriggerType.REACTION,
        0.85,
    ),
    (
        re.compile(
            r"\b(that'?s|thats)\s+(dope|sick|crazy|wild|insane|fire|lit|awesome|cool)\b", re.I
        ),
        TriggerType.REACTION,
        0.80,
    ),
    (re.compile(r"^(holy|oh)\s+(shit|fuck|crap|damn|my god)\b", re.I), TriggerType.REACTION, 0.80),
    # === REACTION: Good news ===
    (
        re.compile(
            r"\b(i got (the job|accepted|promoted|in|hired)|i passed|i made it|"
            r"we won|i'm engaged|i'm pregnant)\b",
            re.I,
        ),
        TriggerType.REACTION,
        0.85,
    ),
    (
        re.compile(r"^(great news|good news|finally|so excited|so happy)[!:\s]", re.I),
        TriggerType.REACTION,
        0.80,
    ),
    # === REACTION: Bad news ===
    (
        re.compile(
            r"\b(i lost my (wallet|keys|phone|job)|i failed|i got fired|"
            r"i'm sick|someone died|passed away)\b",
            re.I,
        ),
        TriggerType.REACTION,
        0.85,
    ),
    (
        re.compile(r"^(so sad|so upset|terrible news|awful news|bad news)\b", re.I),
        TriggerType.REACTION,
        0.80,
    ),
    (re.compile(r"^(fuck+|shit+|damn+|ugh+|omfg|fml)[\s!.]*$", re.I), TriggerType.REACTION, 0.80),
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


class HybridTriggerClassifier(EmbedderMixin, SVMModelMixin, CentroidMixin):
    """Classifier for incoming messages using structural patterns + SVM.

    Strategy:
    1. Check structural patterns (fast, high precision)
    2. If no match, try trained SVM classifier (if available)
    3. Fallback to STATEMENT if still unsure

    Uses mixins for shared functionality:
    - EmbedderMixin: Lazy-loaded embedder access
    - SVMModelMixin: SVM model loading and prediction
    - CentroidMixin: Centroid-based verification
    """

    @property
    def CENTROID_VERIFY_THRESHOLD(self) -> float:  # noqa: N802
        """Centroid verification threshold from config (overrides CentroidMixin default)."""
        return get_config().classifier_thresholds.trigger_centroid_verify

    @property
    def CENTROID_MARGIN(self) -> float:  # noqa: N802
        """Centroid margin from config (overrides CentroidMixin default)."""
        return get_config().classifier_thresholds.trigger_centroid_margin

    def __init__(
        self,
        model_path: Path | str | None = None,
        use_centroid_verification: bool = False,
    ):
        """Initialize the trigger classifier.

        Args:
            model_path: Path to trained model directory.
            use_centroid_verification: If True, verify SVM predictions with centroid distance.
                This is an experimental feature that may improve accuracy for some cases.
        """
        # Set model path (used by SVMModelMixin and CentroidMixin)
        self._model_path = Path(model_path) if model_path else _get_default_model_path()

        # Centroid verification (experimental)
        self._use_centroid_verification = use_centroid_verification

        # Initialize structural pattern matcher
        self._pattern_matcher = StructuralPatternMatcher(
            [(p.pattern, t, c) for p, t, c in STRUCTURAL_PATTERNS],
            flags=re.IGNORECASE,
        )

        # Load the trained SVM model (from SVMModelMixin)
        self._load_svm()

    def _get_svm_threshold(self, trigger_type: TriggerType) -> float:
        """Get the SVM confidence threshold for a trigger type.

        Uses per-class thresholds tuned based on performance analysis.
        Higher thresholds for hard/important classes (COMMITMENT),
        lower for classes with strong structural patterns (SOCIAL).
        """
        return _get_svm_thresholds().get(trigger_type, _get_default_svm_threshold())

    def _match_svm(self, text: str) -> tuple[TriggerType | None, float, str]:
        """Match using trained SVM classifier.

        Uses per-class confidence thresholds - higher for important classes
        like COMMITMENT (0.50), lower for SOCIAL (0.25) where structural
        patterns are strong.

        Returns:
            Tuple of (trigger_type, confidence, method) or (None, 0, 'svm').
            method is 'svm' or 'svm+centroid' if centroid verification was used.
        """
        if not self.svm_available:
            return None, 0.0, "svm"

        try:
            # Compute embedding using EmbedderMixin
            embedding = self.embedder.encode([text], normalize=True)[0]

            # Get prediction using SVMModelMixin
            label, confidence = self._predict_svm(embedding)
            if label is None:
                return None, confidence, "svm"

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

            # Optional: verify with centroid distance using CentroidMixin
            method = "svm"
            if self._use_centroid_verification:
                self._load_centroids()
                if self.centroids_available:
                    final_label, _, was_verified = self._verify_with_centroids(
                        embedding, trigger_type.value
                    )
                    if not was_verified:
                        # Centroid overrode SVM prediction
                        try:
                            trigger_type = TriggerType(final_label)
                            method = "svm+centroid"
                        except ValueError:
                            pass

            return trigger_type, confidence, method

        except Exception as e:
            logger.warning("SVM classification failed: %s", e)
            return None, 0.0, "svm"

    def _match_structural(self, text: str) -> tuple[TriggerType | None, float]:
        """Match against structural patterns using the pattern matcher.

        Returns:
            Tuple of (trigger_type, confidence) or (None, 0) if no match.
        """
        return self._pattern_matcher.match(text)

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

_factory: SingletonFactory[HybridTriggerClassifier] = SingletonFactory(HybridTriggerClassifier)


def get_trigger_classifier(model_path: Path | str | None = None) -> HybridTriggerClassifier:
    """Get the singleton trigger classifier.

    Args:
        model_path: Optional path to trained model. Only used on first call.

    Returns:
        The singleton HybridTriggerClassifier instance.
    """
    # Note: model_path only affects first initialization
    # For custom path, create instance directly
    if model_path is not None and not _factory.is_initialized():
        # Reset and create with custom path
        return HybridTriggerClassifier(model_path=model_path)
    return _factory.get()


def reset_trigger_classifier() -> None:
    """Reset the singleton classifier (useful for testing or reloading model)."""
    _factory.reset()


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
    "TriggerClassification",
    "HybridTriggerClassifier",
    "get_trigger_classifier",
    "reset_trigger_classifier",
    "classify_trigger",
]
