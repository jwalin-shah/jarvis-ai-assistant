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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import spacy

from jarvis.classifiers.factory import SingletonFactory
from jarvis.classifiers.mixins import EmbedderMixin
from jarvis.text_normalizer import is_acknowledgment_only, is_reaction

if TYPE_CHECKING:
    from jarvis.classifiers.response_mobilization import MobilizationResult

logger = logging.getLogger(__name__)

# Load spaCy model lazily
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


VALID_CATEGORIES = frozenset({
    "closing",
    "acknowledge",
    "question",
    "request",
    "emotion",
    "statement",
})

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
# Feature extraction patterns
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
    """Extract 26 hand-crafted features matching training pipeline."""
    features: list[float] = []
    text_lower = text.lower()
    words = text.split()
    total_words = len(words)

    # Message structure (5)
    features.append(float(len(text)))
    features.append(float(total_words))
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
    abbr_count = len(ABBREVIATION_RE.findall(text))
    features.append(abbr_count / max(total_words, 1))
    capitalized = sum(1 for w in words[1:] if w[0].isupper()) if len(words) > 1 else 0
    features.append(capitalized / max(len(words) - 1, 1))

    # NEW: Reaction/emotion features (7)
    # 1. Is this an iMessage reaction/tapback?
    reaction_patterns = ["Laughed at", "Loved", "Liked", "Disliked", "Emphasized", "Questioned"]
    is_reaction_msg = 1.0 if any(text.startswith(p) for p in reaction_patterns) else 0.0
    features.append(is_reaction_msg)

    # 2. Emotional marker count (lmao, lol, xd, haha, bruh, rip, omg)
    emotional_markers = ["lmao", "lol", "xd", "haha", "omg", "bruh", "rip", "lmfao", "rofl"]
    emotional_count = sum(text_lower.count(marker) for marker in emotional_markers)
    features.append(float(emotional_count))

    # 3. Does message END with emotional marker?
    last_word = words[-1].lower() if words else ""
    ends_with_emotion = 1.0 if last_word in emotional_markers else 0.0
    features.append(ends_with_emotion)

    # 4. Question word at start
    question_starters = {"what", "why", "how", "when", "where", "who", "did", "do", "does", "can", "could", "would", "will", "should"}
    first_word = words[0].lower() if words else ""
    question_first = 1.0 if first_word in question_starters else 0.0
    features.append(question_first)

    # 5. Imperative verb at start
    imperative_verbs = {"make", "send", "get", "tell", "show", "give", "come", "take", "call", "help", "let"}
    imperative_first = 1.0 if first_word in imperative_verbs else 0.0
    features.append(imperative_first)

    # 6. Brief agreement phrase
    brief_agreements = {"ok", "okay", "k", "yeah", "yep", "yup", "sure", "cool", "bet", "fs", "aight"}
    is_brief_agreement = 1.0 if total_words <= 3 and any(w in brief_agreements for w in words) else 0.0
    features.append(is_brief_agreement)

    # 7. Exclamatory ending
    exclamatory = 1.0 if (text.endswith("!") or text.isupper() and total_words <= 5) else 0.0
    features.append(exclamatory)

    return np.array(features, dtype=np.float32)


def _extract_spacy_features(text: str) -> np.ndarray:
    """Extract 14 SpaCy linguistic features."""
    nlp = _get_nlp()
    doc = nlp(text)
    features = []

    # 1. has_imperative: Check for imperative verbs (VB at start)
    has_imperative = 0.0
    if len(doc) > 0 and doc[0].pos_ == "VERB" and doc[0].tag_ == "VB":
        has_imperative = 1.0
    features.append(has_imperative)

    # 2. you_modal: "can you", "could you", "would you", "will you"
    text_lower = text.lower()
    you_modal = 1.0 if any(p in text_lower for p in ["can you", "could you", "would you", "will you", "should you"]) else 0.0
    features.append(you_modal)

    # 3. request_verb: Common request verbs
    request_verbs = {"send", "give", "help", "tell", "show", "let", "call", "get", "make", "take"}
    has_request = 1.0 if any(token.lemma_ in request_verbs for token in doc) else 0.0
    features.append(has_request)

    # 4. starts_modal: Starts with modal verb
    starts_modal = 0.0
    if len(doc) > 0 and doc[0].tag_ in ("MD", "VB"):
        starts_modal = 1.0
    features.append(starts_modal)

    # 5. directive_question: Questions that are really directives
    directive_q = 1.0 if you_modal and "?" in text else 0.0
    features.append(directive_q)

    # 6. i_will: "I'll", "I will", "I'm gonna"
    i_will = 1.0 if any(p in text_lower for p in ["i'll", "i will", "i'm gonna", "ima", "imma"]) else 0.0
    features.append(i_will)

    # 7. promise_verb: Promise/commitment verbs
    promise_verbs = {"promise", "guarantee", "commit", "swear"}
    has_promise = 1.0 if any(token.lemma_ in promise_verbs for token in doc) else 0.0
    features.append(has_promise)

    # 8. first_person_count
    first_person = sum(1 for token in doc if token.text.lower() in ("i", "me", "my", "mine", "myself"))
    features.append(float(first_person))

    # 9. agreement: Agreement words
    agreement_words = {"sure", "okay", "ok", "yes", "yeah", "yep", "yup", "sounds good", "bet", "fs"}
    has_agreement = 1.0 if any(word in text_lower for word in agreement_words) else 0.0
    features.append(has_agreement)

    # 10. modal_count
    modal_count = sum(1 for token in doc if token.tag_ == "MD")
    features.append(float(modal_count))

    # 11. verb_count
    verb_count = sum(1 for token in doc if token.pos_ == "VERB")
    features.append(float(verb_count))

    # 12. second_person_count
    second_person = sum(1 for token in doc if token.text.lower() in ("you", "your", "yours", "yourself"))
    features.append(float(second_person))

    # 13. has_negation
    has_neg = 1.0 if any(token.dep_ == "neg" for token in doc) else 0.0
    features.append(has_neg)

    # 14. is_interrogative: Question indicators
    is_question = 1.0 if "?" in text or any(token.tag_ in ("WDT", "WP", "WP$", "WRB") for token in doc) else 0.0
    features.append(is_question)

    return np.array(features, dtype=np.float32)


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
        self._svm_model = None
        self._svm_loaded = False

    def _load_svm(self) -> bool:
        """Load trained SVM model from disk."""
        if self._svm_loaded:
            return self._svm_model is not None

        self._svm_loaded = True
        model_path = Path("models/category_svm_v2.joblib")

        if not model_path.exists():
            logger.warning("No SVM model at %s - using fallback only", model_path)
            return False

        try:
            self._svm_model = joblib.load(model_path)
            logger.info("Loaded category SVM from %s", model_path)
            return True
        except Exception as e:
            logger.error("Failed to load SVM: %s", e)
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
        """
        text_lower = text.lower()
        words = text_lower.split()
        first_word = words[0] if words else ""

        # Rule 1: Reaction messages → emotion (100% accuracy)
        # "Laughed at", "Loved", "Liked", "Disliked", "Emphasized", "Questioned"
        if is_reaction(text):
            return "emotion"

        # Rule 2: Emotional markers → emotion (if SVM said statement/acknowledge)
        # Only override if message is SHORT or ends with marker
        if svm_prediction in ["statement", "acknowledge"]:
            emotional_markers = {"lmao", "lol", "xd", "haha", "omg", "bruh", "rip", "lmfao"}
            last_word = words[-1] if words else ""

            # If message is short (≤5 words) and has emotional marker → emotion
            if len(words) <= 5 and any(marker in text_lower for marker in emotional_markers):
                return "emotion"

            # If message ENDS with emotional marker → emotion
            if last_word in emotional_markers:
                return "emotion"

        # Rule 3: Question words without "?" → question (if SVM said statement)
        if svm_prediction == "statement" and "?" not in text:
            question_starters = {"what", "why", "how", "when", "where", "who", "did", "do", "does"}
            if first_word in question_starters:
                return "question"

        # Rule 4: Imperative verbs → request (if SVM said statement)
        if svm_prediction == "statement":
            imperative_verbs = {"make", "send", "come", "get", "tell", "show", "give", "call", "help", "take"}
            if first_word in imperative_verbs:
                return "request"

        # Rule 5: Brief agreement with no new info → acknowledge (not emotion)
        # "yeah", "ok", "sure" etc. when SVM said emotion
        if svm_prediction == "emotion" and len(words) <= 3:
            brief_agreements = {"yeah", "yep", "ok", "okay", "sure", "cool", "bet", "fs", "aight"}
            if any(word in brief_agreements for word in words) and not any(
                marker in text_lower for marker in {"lmao", "lol", "xd", "haha"}
            ):
                return "acknowledge"

        # Rule 6: "rip" alone or at end → emotion (not closing)
        if svm_prediction == "closing":
            if text_lower.strip() == "rip" or words[-1] == "rip" if words else False:
                return "emotion"

        # No correction needed
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

        # Layer 0: Fast path for reactions/acknowledgments
        if is_reaction(text) or is_acknowledgment_only(text):
            return CategoryResult(
                category="acknowledge",
                confidence=1.0,
                method="fast_path",
            )

        # Layer 1: SVM prediction
        if self._load_svm():
            try:
                # Extract mobilization features
                mob_pressure = mobilization.pressure if mobilization else "none"
                mob_type = mobilization.response_type if mobilization else "answer"

                # 1. BERT embedding (384)
                embedding = self.embedder.encode([text], normalize=True)[0]

                # 2. Hand-crafted features (26)
                hand_crafted = _extract_hand_crafted(text, context, mob_pressure, mob_type)

                # 3. SpaCy features (14)
                spacy_feats = _extract_spacy_features(text)

                # 4. Concatenate (424 total)
                features = np.concatenate([embedding, hand_crafted, spacy_feats])
                features = features.reshape(1, -1)

                # Predict
                category = self._svm_model.predict(features)[0]

                # Get confidence via decision function
                decision_values = self._svm_model.decision_function(features)[0]

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

                # Handle LightGBM label encoding
                if hasattr(self._svm_model, 'label_encoder_'):
                    category = self._svm_model.label_encoder_.inverse_transform([category])[0]

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
                logger.error("SVM prediction failed: %s", e, exc_info=True)

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
