"""Hybrid Response Classifier - Combines structural hints with semantic verification.

Problem: The base DA classifier classifies 78% of responses as STATEMENT,
making DA-filtered retrieval ineffective.

Solution: Three-layer hybrid approach:
1. Structural hints (fast) - regex patterns suggest possible DA types
2. Centroid verification (semantic) - confirm hints using embedding distance
3. kNN fallback - for ambiguous cases with no structural hint

This combines the speed of structural patterns with semantic accuracy:
- "?" → hint: QUESTION → verify with centroid distance
- "No way!" → hint: none → but close to REACT_POSITIVE centroid → not a question

Research backing:
- Prototype-based classification (Nearest Centroid)
- Prototypical Networks (Snell et al., 2017) for few-shot learning
- Rocchio classification (classic IR)

Usage:
    from jarvis.classifiers.response_classifier import (
        HybridResponseClassifier, get_response_classifier,
    )

    classifier = get_response_classifier()
    result = classifier.classify("Yeah I'm down!")
    print(result.label)  # AGREE
    print(result.confidence)  # 0.95
    print(result.method)  # structural_verified
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from jarvis.classifiers import (
    CentroidMixin,
    EmbedderMixin,
    PatternMatcherByLabel,
    SingletonFactory,
)
from jarvis.config import get_config, get_response_classifier_path
from jarvis.text_normalizer import normalize_for_task

if TYPE_CHECKING:
    from jarvis.embedding_adapter import Embedder

logger = logging.getLogger(__name__)


def _get_default_model_path() -> Path:
    """Get the default model path based on configured embedding model."""
    return get_response_classifier_path()


# =============================================================================
# Response Type Definitions
# =============================================================================


class ResponseType(str, Enum):
    """Response dialogue act types."""

    AGREE = "AGREE"  # Positive acceptance, yes, affirmation
    DECLINE = "DECLINE"  # Rejection, no, can't do it
    DEFER = "DEFER"  # Non-committal, need to check, maybe
    ACKNOWLEDGE = "ACKNOWLEDGE"  # Simple confirmation (ok, got it)
    ANSWER = "ANSWER"  # Provides specific requested information
    QUESTION = "QUESTION"  # Asks for more information
    REACT_POSITIVE = "REACT_POSITIVE"  # Excited, happy, congratulatory
    REACT_SYMPATHY = "REACT_SYMPATHY"  # Supportive, sorry, sympathetic
    STATEMENT = "STATEMENT"  # Shares information (catch-all)
    GREETING = "GREETING"  # Greeting response


# Response types for commitment questions (invitations, requests)
COMMITMENT_RESPONSE_TYPES = frozenset(
    {
        ResponseType.AGREE,
        ResponseType.DECLINE,
        ResponseType.DEFER,
    }
)

# Trigger type -> Valid response types (dialogue structure constraints)
# From the DA classifier mappings - use to constrain possible response types
TRIGGER_TO_VALID_RESPONSES: dict[str, list[ResponseType]] = {
    # === New hybrid trigger classifier labels (TriggerType enum values) ===
    # These are coarser categories that map to the fine-grained labels below
    "commitment": [
        ResponseType.AGREE,
        ResponseType.DECLINE,
        ResponseType.DEFER,
        ResponseType.QUESTION,
    ],
    "question": [
        ResponseType.ANSWER,
        ResponseType.AGREE,
        ResponseType.DECLINE,
        ResponseType.DEFER,
        ResponseType.QUESTION,
    ],
    "reaction": [
        ResponseType.REACT_POSITIVE,
        ResponseType.REACT_SYMPATHY,
        ResponseType.QUESTION,
        ResponseType.ACKNOWLEDGE,
    ],
    "social": [ResponseType.GREETING, ResponseType.ACKNOWLEDGE, ResponseType.QUESTION],
    "statement": [
        ResponseType.ACKNOWLEDGE,
        ResponseType.REACT_POSITIVE,
        ResponseType.REACT_SYMPATHY,
        ResponseType.QUESTION,
        ResponseType.STATEMENT,
    ],
    # === Legacy fine-grained DA classifier labels (backwards compatibility) ===
    "INVITATION": [
        ResponseType.AGREE,
        ResponseType.DECLINE,
        ResponseType.DEFER,
        ResponseType.QUESTION,
    ],
    "YN_QUESTION": [
        ResponseType.AGREE,
        ResponseType.DECLINE,
        ResponseType.DEFER,
        ResponseType.ANSWER,
        ResponseType.QUESTION,
    ],
    "WH_QUESTION": [ResponseType.ANSWER, ResponseType.DEFER, ResponseType.QUESTION],
    "INFO_STATEMENT": [
        ResponseType.ACKNOWLEDGE,
        ResponseType.REACT_POSITIVE,
        ResponseType.REACT_SYMPATHY,
        ResponseType.QUESTION,
        ResponseType.STATEMENT,
    ],
    "OPINION": [
        ResponseType.AGREE,
        ResponseType.DECLINE,
        ResponseType.ACKNOWLEDGE,
        ResponseType.QUESTION,
    ],
    "REQUEST": [
        ResponseType.AGREE,
        ResponseType.DECLINE,
        ResponseType.DEFER,
        ResponseType.QUESTION,
    ],
    "GOOD_NEWS": [ResponseType.REACT_POSITIVE, ResponseType.QUESTION, ResponseType.ACKNOWLEDGE],
    "BAD_NEWS": [ResponseType.REACT_SYMPATHY, ResponseType.QUESTION, ResponseType.ACKNOWLEDGE],
    "GREETING": [ResponseType.GREETING, ResponseType.QUESTION, ResponseType.ACKNOWLEDGE],
    "ACKNOWLEDGE": [ResponseType.ACKNOWLEDGE, ResponseType.QUESTION, ResponseType.STATEMENT],
}


# =============================================================================
# iMessage Tapback Reactions (Filter First)
# =============================================================================

# Tapback reactions are iMessage metadata, not real conversational responses.
# We detect and handle them before structural patterns to avoid misclassification.
# Examples: "Liked \"message\"", "Laughed at an image", "Disliked \"message\""

# Quote patterns: straight " or curly " "
_QUOTE_PATTERN = r'["\u201c\u201d]'

TAPBACK_POSITIVE_PATTERNS = [
    re.compile(rf"^Liked\s+{_QUOTE_PATTERN}", re.IGNORECASE),
    re.compile(rf"^Loved\s+{_QUOTE_PATTERN}", re.IGNORECASE),
    re.compile(r"^Laughed at\s+", re.IGNORECASE),
]

TAPBACK_FILTERED_PATTERNS = [
    re.compile(rf"^Disliked\s+{_QUOTE_PATTERN}", re.IGNORECASE),
    re.compile(rf"^Emphasized\s+{_QUOTE_PATTERN}", re.IGNORECASE),
    re.compile(rf"^Questioned\s+{_QUOTE_PATTERN}", re.IGNORECASE),
]


# =============================================================================
# Structural Patterns (High Precision)
# =============================================================================

# IMPORTANT: These patterns are designed for HIGH PRECISION (few false positives)
# even if recall is lower. The DA classifier handles ambiguous cases.

# Pattern format: list of (pattern, is_regex) tuples
# If is_regex=False, it's a simple prefix/contains check (faster)
# If is_regex=True, it's compiled as a regex

STRUCTURAL_PATTERNS: dict[ResponseType, list[tuple[str, bool]]] = {
    ResponseType.AGREE: [
        # Strong affirmatives (exact or prefix)
        (r"^(yes|yeah|yep|yup|yea|ya|yas|yass|yess|yesss)[\s!.]*$", True),
        (r"^(sure|definitely|absolutely|of course|certainly)[\s!.]*$", True),
        (r"^(i'm down|im down|i am down|down)[\s!.]*$", True),
        (r"^(sounds good|sounds great|sounds perfect)[\s!.]*$", True),
        (r"^(let's do it|lets do it|let's go|lets go)[\s!.]*$", True),
        (r"^(i'm in|im in|count me in)[\s!.]*$", True),
        (r"^(for sure|100%|bet|deal)[\s!.]*$", True),
        (r"^(works for me|that works|perfect)[\s!.]*$", True),
        # Slang affirmatives
        (r"^(say less|say no more|less go|lfg)[\s!.]*$", True),
        # Auto-mined patterns
        (r"^(true|tru)[\s!.]*$", True),
        (r"^(exactly|that's\s+(true|facts)|this\s+is\s+true)[\s!.]*$", True),
    ],
    ResponseType.DECLINE: [
        # Strong negatives
        (r"^(no|nope|nah|naw)[\s!.]*$", True),
        (r"^(can't|cannot|cant)[\s!.,]*$", True),
        (r"^(i can't|i cannot|i cant)[\s!.,]*", True),
        (r"^(sorry|unfortunately)[\s,]+(i )?(can't|cannot|cant|won't)", True),
        (r"^(won't be able|wont be able)", True),
        (r"^(not (today|tonight|this time|gonna work))", True),
        (r"^(i('m| am) (busy|not free|unavailable))", True),
        (r"^(i'll pass|ill pass|hard pass|pass)[\s!.]*$", True),
        (r"^(rain check)", True),
        # Auto-mined patterns
        (r"^(nuh\s+uh|noooo+|not\s+rlly|not\s+really)[\s!.]*$", True),
        (r"^(prolly\s+not|probably\s+not|no\s+lol)[\s!.]*$", True),
    ],
    ResponseType.DEFER: [
        # Non-committal
        (r"^(maybe|possibly|perhaps)[\s!.]*$", True),
        (r"^(let me (check|see|think|get back))", True),
        (r"^(i'll (see|check|let you know|think about it))", True),
        (r"^(need to (check|see|think))", True),
        (r"^(not sure|unsure)[\s!.,]*$", True),
        (r"^(depends|it depends)[\s!.]*$", True),
        (r"^(we'll see|we will see)[\s!.]*$", True),
        (r"^(might|might be able)", True),
        (r"^(tbd|to be determined)[\s!.]*$", True),
        (r"^(gotta see|have to see|need to see)", True),
    ],
    ResponseType.ACKNOWLEDGE: [
        # Simple acknowledgments
        (r"^(ok|okay|k|kk|okok|okk)[\s!.]*$", True),
        (r"^(got it|gotcha|gotchu)[\s!.]*$", True),
        (r"^(alright|aight|aite|ight)[\s!.]*$", True),
        (r"^(cool|nice|great|awesome)[\s!.]*$", True),
        (r"^(noted|understood|copy|roger)[\s!.]*$", True),
        (r"^(will do|on it)[\s!.]*$", True),
        (r"^(no worries|no problem|np)[\s!.]*$", True),
        (r"^(fair enough|makes sense)[\s!.]*$", True),
        (r"^(i see|ohh|ahh|ohhh)[\s!.]*$", True),
        (r"^(word|bet)[\s!.]*$", True),  # bet as acknowledgment, not agree
        # Auto-mined patterns
        (r"^(mhm|oh\s+ok|good|oh)[\s!.]*$", True),
    ],
    ResponseType.QUESTION: [
        # Questions contain "?"
        (r"\?[\s]*$", True),  # Ends with question mark
        (r"^(what|when|where|who|why|how|which)\b", True),
        (r"^(wdym|huh|wait what)[\s!?]*$", True),
        # Auxiliary verb questions (Do you..., Are you..., Can you..., etc.)
        (
            r"^(do|does|did|are|is|was|were|can|could|will|would|should|have|has)\s+"
            r"(you|u|we|they|i)\b",
            True,
        ),
    ],
    ResponseType.REACT_POSITIVE: [
        # Positive reactions
        (r"^(congrats|congratulations)[\s!.]*", True),
        (r"^(that's (awesome|amazing|great|incredible|fantastic))[\s!.]*", True),
        (r"^(so (happy|excited|proud) for you)[\s!.]*", True),
        (r"^(omg|oh my god|no way)[\s!.]*$", True),
        (r"^(yay|woohoo|woo|ayy|ayyy)[\s!.]*$", True),
        (r"^(nice|sick|dope|fire|lit)[\s!.]*$", True),
        (r"^(let's gooo?|lfg|W|big W)[\s!.]*$", True),
        (r"^(well done|good job|killed it)[\s!.]*", True),
        # Laughter expressions (commonly misclassified as DECLINE)
        (r"^(lol|lmao|lmfao|lmfaoo+|rofl)[\s!.]*$", True),
        (
            r"^(laughing out loud|laughing my ass off|rolling on the floor)[\s!.]*$",
            True,
        ),  # Expanded slang forms
        (r"^(haha+|hehe+|hihi+|hoho+)[\s!.]*$", True),
        (r"^(ha+|he+)[\s!.]*$", True),  # "haaaaa", "heeee"
        (r"^(dying|dead|i'm dead|im dead)[\s!.]*$", True),  # "dying" as in laughing
        # Auto-mined patterns
        (r"^(thanks|thanks\s+dude|thank\s+you)[\s!.]*$", True),
        (r"^(loved\s+an\s+image)[\s!.]*$", True),  # tapback variant
        (r"^(interesting|that's\s+good|ur\s+good)[\s!.]*$", True),
        (r"^(holy|wtf|noice)[\s!.]*$", True),
        # Excited expressions with repeated letters
        (r"^(yu+h+|ye+s+|ya+s+|le+t+s?\s*go+)[\s!.]*$", True),  # yuhhhh, yesss, yasss, lessgo
        (r"^(bru+h+|bre+h+|bro+)[\s!.]*$", True),  # bruhh, brehh, brooo
        # More mined patterns
        (r"^(wow|holy\s+shit|thanks\s+bro)[\s!.]*$", True),
        # Gratitude expressions (often misclassified as DECLINE due to "can't/couldn't")
        (r"(i\s*love\s*(you|u)|ily)\b", True),  # "ily dude", "i love you"
        # "couldn't have done it without you"
        (r"(couldn'?t\s+(have\s+)?(done|made|got)\s+.*(without|w\s*out)\s*(you|u))", True),
        (r"(so\s+glad|grateful|thankful)", True),  # gratitude expressions
        (r"(you'?re\s+(the\s+)?best)", True),  # "you're the best"
        (r"(appreciate\s+(you|u|it))", True),  # appreciation
        (r"(means\s+(a\s+lot|so\s+much))", True),  # "this means a lot"
    ],
    ResponseType.REACT_SYMPATHY: [
        # Sympathy reactions
        (r"^(i'm sorry|im sorry|so sorry)", True),
        (r"^(that (sucks|stinks|blows|is rough|is terrible))", True),
        (r"^(damn|ugh|man)[\s!.]*$", True),
        (r"^(here for you|thinking of you|sending)", True),
        (r"^(hang in there|it'll be ok)", True),
        (r"^(let me know if you need)", True),
        # Auto-mined patterns
        (r"^(oh\s+no+|rip)[\s!.]*$", True),
    ],
    ResponseType.GREETING: [
        # Greetings
        (r"^(hey|hi|hello|yo|sup|hiya)[\s!.]*$", True),
        (r"^(what's up|whats up|wassup)[\s!?]*$", True),
        (r"^(good (morning|afternoon|evening))[\s!.]*$", True),
        (r"^(morning|evening)[\s!.]*$", True),
        # Auto-mined patterns
        (r"^(yooo+|good\s+night|bye+|goodbye)[\s!.]*$", True),
    ],
    # ANSWER and STATEMENT don't have good structural patterns
    # They're handled by the DA classifier fallback
}

# Compile all regex patterns once at module load
_COMPILED_PATTERNS: dict[ResponseType, list[re.Pattern]] = {}

for response_type, patterns in STRUCTURAL_PATTERNS.items():
    _COMPILED_PATTERNS[response_type] = []
    for pattern, is_regex in patterns:
        if is_regex:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                _COMPILED_PATTERNS[response_type].append(compiled)
            except re.error as e:
                logger.warning("Invalid regex pattern %s: %s", pattern, e)


# =============================================================================
# Classification Result
# =============================================================================


@dataclass
class ClassificationResult:
    """Result from response classification."""

    label: ResponseType
    confidence: float
    method: str  # 'structural', 'svm', 'hybrid', 'centroid_override'
    structural_match: bool = False
    da_label: str | None = None
    da_confidence: float | None = None


# =============================================================================
# Hybrid Response Classifier
# =============================================================================


class HybridResponseClassifier(EmbedderMixin, CentroidMixin):
    """Hybrid classifier combining structural hints with semantic verification.

    Classification strategy:
    1. Get structural hint (fast regex patterns)
    2. If hint exists, verify with centroid distance (semantic check)
    3. If no hint or verification fails, use nearest centroid
    4. Return result with method indicator

    This combines structural speed with semantic accuracy:
    - Structural hints are fast but can have false positives ("No way!" isn't DECLINE)
    - Centroid verification catches edge cases using embeddings
    - Centroid fallback handles ambiguous cases

    Uses mixins for shared functionality:
    - EmbedderMixin: Lazy-loaded embedder access
    - CentroidMixin: Centroid-based verification and classification

    Thread Safety:
        This class is thread-safe. Centroids loaded lazily with locking.
    """

    # Trigger types that allow commitment responses (AGREE/DECLINE/DEFER)
    # For other trigger types, commitment responses are filtered out
    COMMITMENT_TRIGGER_TYPES = frozenset({"INVITATION", "REQUEST", "YN_QUESTION", "OPINION"})

    def __init__(
        self,
        structural_confidence: float = 0.95,
        use_centroid_verification: bool = True,
        use_confidence_threshold: bool = True,
        use_trigger_filtering: bool = True,
    ) -> None:
        """Initialize the hybrid classifier.

        Args:
            structural_confidence: Confidence to assign structural matches.
            use_centroid_verification: If True, verify structural hints with centroids.
            use_confidence_threshold: If True, default to ANSWER when confidence is low.
            use_trigger_filtering: If True, filter response types based on trigger_da.
        """
        # Set model path for CentroidMixin
        self._model_path = _get_default_model_path()

        self._lock = threading.Lock()
        self._structural_confidence = structural_confidence
        self._use_centroid_verification = use_centroid_verification
        self._use_confidence_threshold = use_confidence_threshold
        self._use_trigger_filtering = use_trigger_filtering

        # Initialize structural pattern matcher
        self._pattern_matcher = PatternMatcherByLabel(
            STRUCTURAL_PATTERNS,
            default_confidence=structural_confidence,
        )

        # Load centroids (from CentroidMixin)
        self._load_centroids()

    @property
    def CENTROID_VERIFY_THRESHOLD(self) -> float:  # noqa: N802
        """Centroid verification threshold from config."""
        return get_config().classifier_thresholds.response_centroid_verify

    @property
    def CENTROID_MARGIN(self) -> float:  # noqa: N802
        """Centroid margin from config."""
        return get_config().classifier_thresholds.response_centroid_margin

    def _verify_with_centroid(
        self,
        text: str,
        hint_type: ResponseType,
        embedder: Embedder | None = None,
    ) -> tuple[ResponseType, float, bool]:
        """Verify a structural hint using centroid distance.

        Args:
            text: Text to classify.
            hint_type: Structural hint to verify.
            embedder: Embedder for computing text embedding.

        Returns:
            Tuple of (final_type, confidence, was_verified).
            was_verified=True if hint was confirmed, False if overridden or fallback.
        """
        if not self.centroids_available or not self._use_centroid_verification:
            # No centroids available, trust the structural hint
            return hint_type, self._structural_confidence, True

        # Get embedder
        emb = embedder if embedder is not None else self.embedder

        # Compute text embedding
        try:
            embedding = emb.encode([text], normalize=True)[0]
        except Exception as e:
            logger.warning("Failed to compute embedding: %s", e)
            return hint_type, self._structural_confidence, True

        return self._verify_with_centroids_internal(embedding, hint_type)

    def _verify_with_centroids_internal(
        self,
        embedding: np.ndarray,
        hint_type: ResponseType,
    ) -> tuple[ResponseType, float, bool]:
        """Internal helper for centroid verification."""
        # Find nearest and get predicted similarity
        nearest_label, best_sim = self._find_nearest_centroid(embedding)

        # Get similarity to the hint class
        hint_sim = 0.0
        if self._centroids and hint_type.value in self._centroids:
            hint_sim = float(np.dot(embedding, self._centroids[hint_type.value]))

        # Decision logic:
        # 1. If hint has high similarity -> confirm hint
        # 2. If another class is significantly closer -> override hint
        # 3. Otherwise -> use hint (structural patterns are high precision)

        if hint_sim >= self.CENTROID_VERIFY_THRESHOLD:
            return hint_type, min(0.95, hint_sim + 0.2), True

        if nearest_label and best_sim - hint_sim > self.CENTROID_MARGIN:
            try:
                override_type = ResponseType(nearest_label)
                logger.debug(
                    "Centroid override: %s -> %s (sim: %.2f vs %.2f)",
                    hint_type.value,
                    nearest_label,
                    best_sim,
                    hint_sim,
                )
                return override_type, best_sim, False
            except ValueError:
                pass

        return hint_type, self._structural_confidence, True

    def _detect_tapback(self, text: str) -> str | None:
        """Detect iMessage tapback reactions."""
        text_stripped = text.strip()

        for pattern in TAPBACK_POSITIVE_PATTERNS:
            if pattern.match(text_stripped):
                return "positive"

        for pattern in TAPBACK_FILTERED_PATTERNS:
            if pattern.match(text_stripped):
                return "filtered"

        return None

    def _match_structural(self, text: str) -> tuple[ResponseType | None, float]:
        """Match text against structural patterns."""
        return self._pattern_matcher.match(text)

    def _apply_confidence_threshold(
        self,
        label: ResponseType,
        confidence: float,
    ) -> tuple[ResponseType, float]:
        """Apply confidence threshold - default to ANSWER if confidence is low."""
        if not self._use_confidence_threshold:
            return label, confidence

        over_predicted_types = {
            ResponseType.DECLINE,
            ResponseType.DEFER,
            ResponseType.AGREE,
            ResponseType.ACKNOWLEDGE,
            ResponseType.REACT_POSITIVE,
            ResponseType.REACT_SYMPATHY,
        }

        thresholds = get_config().classifier_thresholds
        if label == ResponseType.DECLINE:
            threshold = thresholds.response_decline_confidence
        elif label == ResponseType.DEFER:
            threshold = thresholds.response_defer_confidence
        elif label == ResponseType.AGREE:
            threshold = thresholds.response_agree_confidence
        else:
            threshold = thresholds.response_low_confidence

        if label in over_predicted_types and confidence < threshold:
            return ResponseType.ANSWER, confidence

        return label, confidence

    def _apply_trigger_filtering(
        self,
        label: ResponseType,
        confidence: float,
        trigger_da: str | None,
    ) -> tuple[ResponseType, float]:
        """Filter response type based on trigger dialogue act."""
        if not self._use_trigger_filtering or trigger_da is None:
            return label, confidence

        if label in COMMITMENT_RESPONSE_TYPES:
            if trigger_da not in self.COMMITMENT_TRIGGER_TYPES:
                valid_responses = TRIGGER_TO_VALID_RESPONSES.get(trigger_da, [])
                if label not in valid_responses:
                    return ResponseType.ANSWER, confidence * 0.8

        return label, confidence

    def classify(
        self,
        text: str,
        embedder: Embedder | None = None,
        trigger_da: str | None = None,
    ) -> ClassificationResult:
        """Classify a response text using a hybrid approach."""
        if not text or not text.strip():
            return ClassificationResult(
                label=ResponseType.STATEMENT,
                confidence=0.0,
                method="empty",
            )

        tapback_type = self._detect_tapback(text)
        if tapback_type:
            if tapback_type == "positive":
                return ClassificationResult(
                    label=ResponseType.REACT_POSITIVE,
                    confidence=0.95,
                    method="tapback_positive",
                    structural_match=True,
                )
            else:
                return ClassificationResult(
                    label=ResponseType.ANSWER,
                    confidence=0.3,
                    method="tapback_filtered",
                    structural_match=True,
                )

        normalized = normalize_for_task(text, "classification")
        if not normalized:
            return ClassificationResult(
                label=ResponseType.STATEMENT,
                confidence=0.0,
                method="normalized_empty",
            )

        # 1. Structural hint
        structural_type, structural_conf = self._match_structural(normalized)

        if structural_type is not None:
            # 2. Centroid verification
            if self._use_centroid_verification and self.centroids_available:
                verified_type, verified_conf, was_verified = self._verify_with_centroid(
                    normalized, structural_type, embedder
                )

                method = "structural_verified" if was_verified else "centroid_override"
                return ClassificationResult(
                    label=verified_type,
                    confidence=verified_conf,
                    method=method,
                    structural_match=True,
                )
            else:
                return ClassificationResult(
                    label=structural_type,
                    confidence=structural_conf,
                    method="structural",
                    structural_match=True,
                )

        # 3. Centroid fallback
        if self.centroids_available:
            try:
                emb = embedder if embedder is not None else self.embedder
                embedding = emb.encode([normalized], normalize=True)[0]
                label_str, confidence = self._find_nearest_centroid(embedding)

                if label_str:
                    try:
                        centroid_type = ResponseType(label_str)
                        final_type, final_conf = self._apply_confidence_threshold(
                            centroid_type, confidence
                        )
                        final_type, final_conf = self._apply_trigger_filtering(
                            final_type, final_conf, trigger_da
                        )

                        return ClassificationResult(
                            label=final_type,
                            confidence=final_conf,
                            method="centroid",
                            da_label=label_str,
                            da_confidence=confidence,
                        )
                    except ValueError:
                        pass
            except Exception as e:
                logger.warning("Centroid fallback failed: %s", e)

        return ClassificationResult(
            label=ResponseType.ANSWER,
            confidence=0.5,
            method="fallback",
        )

    def classify_batch(
        self,
        texts: list[str],
        embedder: Embedder | None = None,
        batch_size: int = 256,
    ) -> list[ClassificationResult]:
        """Classify multiple response texts with batched embedding."""
        if not texts:
            return []

        emb = embedder if embedder is not None else self.embedder
        n = len(texts)
        results: list[ClassificationResult | None] = [None] * n
        needs_embedding_indices = []
        needs_embedding_texts = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = ClassificationResult(
                    label=ResponseType.STATEMENT, confidence=0.0, method="empty"
                )
                continue

            tapback_type = self._detect_tapback(text)
            if tapback_type:
                label = (
                    ResponseType.REACT_POSITIVE
                    if tapback_type == "positive"
                    else ResponseType.ANSWER
                )
                results[i] = ClassificationResult(
                    label=label, confidence=0.9, method="tapback", structural_match=True
                )
                continue

            normalized = normalize_for_task(text, "classification")
            if not normalized:
                results[i] = ClassificationResult(
                    label=ResponseType.STATEMENT, confidence=0.0, method="normalized_empty"
                )
                continue

            needs_embedding_indices.append(i)
            needs_embedding_texts.append(normalized)

        if needs_embedding_texts:
            all_embeddings = []
            for batch_start in range(0, len(needs_embedding_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(needs_embedding_texts))
                batch_embeddings = emb.encode(
                    needs_embedding_texts[batch_start:batch_end], normalize=True
                )
                all_embeddings.append(batch_embeddings)

            embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]

            for idx, orig_idx in enumerate(needs_embedding_indices):
                text = needs_embedding_texts[idx]
                embedding = embeddings[idx]

                struct_type, struct_conf = self._match_structural(text)
                if struct_type is not None:
                    final_type, final_conf, was_verified = self._verify_with_centroids_internal(
                        embedding, struct_type
                    )
                    results[orig_idx] = ClassificationResult(
                        label=final_type,
                        confidence=final_conf,
                        method="structural_verified" if was_verified else "centroid_override",
                        structural_match=True,
                    )
                else:
                    label_str, confidence = self._find_nearest_centroid(embedding)
                    if label_str:
                        try:
                            centroid_type = ResponseType(label_str)
                            final_type, final_conf = self._apply_confidence_threshold(
                                centroid_type, confidence
                            )
                            results[orig_idx] = ClassificationResult(
                                label=final_type,
                                confidence=final_conf,
                                method="centroid",
                                da_label=label_str,
                                da_confidence=confidence,
                            )
                        except ValueError:
                            results[orig_idx] = ClassificationResult(
                                label=ResponseType.ANSWER, confidence=0.5, method="fallback"
                            )
                    else:
                        results[orig_idx] = ClassificationResult(
                            label=ResponseType.ANSWER, confidence=0.5, method="fallback"
                        )

        return [r for r in results if r is not None]

    def is_commitment_response(self, result: ClassificationResult) -> bool:
        """Check if the classification is a commitment response type."""
        return result.label in COMMITMENT_RESPONSE_TYPES


# =============================================================================
# Module-level singleton factory
# =============================================================================

_factory: SingletonFactory[HybridResponseClassifier] = SingletonFactory(HybridResponseClassifier)


def get_response_classifier() -> HybridResponseClassifier:
    """Get the singleton HybridResponseClassifier instance."""
    return _factory.get()


def reset_response_classifier() -> None:
    """Reset the singleton (useful for testing)."""
    _factory.reset()
