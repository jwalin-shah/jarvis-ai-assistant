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
    from jarvis.response_classifier import HybridResponseClassifier, get_response_classifier

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
    EmbedderMixin,
    PatternMatcherByLabel,
    SingletonFactory,
    SVMModelMixin,
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


class HybridResponseClassifier(EmbedderMixin, SVMModelMixin):
    """Hybrid classifier combining structural hints with semantic verification.

    Classification strategy:
    1. Get structural hint (fast regex patterns)
    2. If hint exists, verify with centroid distance (semantic check)
    3. If no hint or verification fails, use kNN fallback
    4. Return result with method indicator

    This combines structural speed with semantic accuracy:
    - Structural hints are fast but can have false positives ("No way!" isn't DECLINE)
    - Centroid verification catches edge cases using embeddings
    - kNN fallback handles ambiguous cases

    Uses mixins for shared functionality:
    - EmbedderMixin: Lazy-loaded embedder access
    - SVMModelMixin: SVM model loading and prediction

    Thread Safety:
        This class is thread-safe. DA classifier and centroids loaded lazily with locking.
    """

    # Trigger types that allow commitment responses (AGREE/DECLINE/DEFER)
    # For other trigger types, commitment responses are filtered out
    COMMITMENT_TRIGGER_TYPES = frozenset({"INVITATION", "REQUEST", "YN_QUESTION", "OPINION"})

    def __init__(
        self,
        da_confidence_threshold: float = 0.7,
        structural_confidence: float = 0.95,
        use_centroid_verification: bool = True,
        use_confidence_threshold: bool = True,
        use_trigger_filtering: bool = True,
        use_svm: bool = True,
    ) -> None:
        """Initialize the hybrid classifier.

        Args:
            da_confidence_threshold: Below this, prefer structural match if available.
            structural_confidence: Confidence to assign structural matches.
            use_centroid_verification: If True, verify structural hints with centroids.
            use_confidence_threshold: If True, default to ANSWER when DA conf < 0.5.
            use_trigger_filtering: If True, filter response types based on trigger_da.
            use_svm: If True, use trained SVM (81.9% F1) instead of DA k-NN for Layer 3.
        """
        # Set model path for SVMModelMixin
        self._model_path = _get_default_model_path()

        self._response_centroids: dict[str, list[float]] | None = None
        self._centroid_arrays: dict[str, np.ndarray] | None = None  # Cached numpy arrays
        self._lock = threading.Lock()
        self._da_confidence_threshold = da_confidence_threshold
        self._structural_confidence = structural_confidence
        self._use_centroid_verification = use_centroid_verification
        self._use_confidence_threshold = use_confidence_threshold
        self._use_trigger_filtering = use_trigger_filtering
        self._use_svm = use_svm

        # Initialize structural pattern matcher
        self._pattern_matcher = PatternMatcherByLabel(
            STRUCTURAL_PATTERNS,
            default_confidence=structural_confidence,
        )

    @property
    def svm(self):
        """Lazy-load the trained SVM classifier (81.9% F1).

        The SVM is trained by scripts/train_response_classifier.py and provides
        better accuracy than the DA k-NN classifier.
        """
        if not self._svm_loaded:
            with self._lock:
                if not self._svm_loaded:
                    self._load_response_svm()
        return self._svm

    def _load_response_svm(self) -> None:
        """Load the trained SVM model from disk with uppercase label normalization."""
        pass  # SVM loading is handled by SVMModelMixin

    @property
    def centroids(self) -> dict[str, list[float]]:
        """Lazy-load class centroids from the DA classifier.

        Centroids are the mean embedding of all exemplars for each class.
        Used for fast approximate classification and hint verification.
        """
        if self._response_centroids is None:
            with self._lock:
                if self._response_centroids is None:
                    self._response_centroids = self._compute_centroids()
        return self._response_centroids

    @property
    def centroid_arrays(self) -> dict[str, np.ndarray]:
        """Get centroids as pre-computed numpy arrays for efficiency.

        Avoids repeated list->array conversion in classification loops.
        """
        if self._centroid_arrays is None:
            with self._lock:
                if self._centroid_arrays is None and self.centroids:
                    self._centroid_arrays = {
                        label: np.array(centroid) for label, centroid in self.centroids.items()
                    }
        return self._centroid_arrays or {}

    def _compute_centroids(self) -> dict[str, list[float]]:
        """Compute centroids from the DA classifier's exemplar embeddings.

        Returns:
            Dict mapping class label -> centroid embedding.
        """
        import numpy as np

        # Load cached centroids from SVM model directory
        centroids_file = self._model_path / "centroids.npy"

        if centroids_file.exists():
            try:
                data = np.load(centroids_file, allow_pickle=True).item()
                logger.info("Loaded cached centroids for %d classes", len(data))
                return data
            except Exception as e:
                logger.warning("Failed to load cached centroids: %s", e)

        # No centroids available - centroid verification will be skipped
        logger.debug("No centroids file found at %s", centroids_file)
        return {}

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
        import numpy as np

        if not self.centroids or not self._use_centroid_verification:
            # No centroids available, trust the structural hint
            return hint_type, self._structural_confidence, True

        # Get embedder
        if embedder is None:
            from jarvis.embedding_adapter import get_embedder

            embedder = get_embedder()

        # Compute text embedding
        try:
            embedding = embedder.encode([text], normalize=True)[0]
        except Exception as e:
            logger.warning("Failed to compute embedding: %s", e)
            return hint_type, self._structural_confidence, True

        # Compute similarity to all centroids (use cached numpy arrays)
        similarities = {}
        for label, centroid_array in self.centroid_arrays.items():
            # Cosine similarity (vectors are normalized)
            sim = float(np.dot(embedding, centroid_array))
            similarities[label] = sim

        hint_sim = similarities.get(hint_type.value, 0.0)
        best_label = max(similarities, key=similarities.get)
        best_sim = similarities[best_label]

        # Decision logic:
        # 1. If hint has high similarity -> confirm hint
        # 2. If another class is significantly closer -> override hint
        # 3. Otherwise -> use hint (structural patterns are high precision)

        thresholds = get_config().classifier_thresholds
        if hint_sim >= thresholds.response_centroid_verify:
            # Hint is confirmed by centroid
            return hint_type, min(0.95, hint_sim + 0.2), True

        if best_sim - hint_sim > thresholds.response_centroid_margin:
            # Another class is significantly closer - override
            try:
                override_type = ResponseType(best_label)
                logger.debug(
                    "Centroid override: %s -> %s (sim: %.2f vs %.2f) for '%s'",
                    hint_type.value,
                    best_label,
                    best_sim,
                    hint_sim,
                    text[:30],
                )
                return override_type, best_sim, False
            except ValueError:
                pass

        # Default: trust structural hint (high precision patterns)
        return hint_type, self._structural_confidence, True

    def _detect_tapback(self, text: str) -> str | None:
        """Detect iMessage tapback reactions.

        Tapbacks are reaction metadata ("Liked \"...\", "Laughed at...") that
        shouldn't be classified as normal responses.

        Args:
            text: Response text to check.

        Returns:
            "positive" for Liked/Loved/Laughed, "filtered" for Disliked/Emphasized/Questioned,
            or None if not a tapback.
        """
        text_stripped = text.strip()

        for pattern in TAPBACK_POSITIVE_PATTERNS:
            if pattern.match(text_stripped):
                return "positive"

        for pattern in TAPBACK_FILTERED_PATTERNS:
            if pattern.match(text_stripped):
                return "filtered"

        return None

    def _match_structural(self, text: str) -> tuple[ResponseType | None, float]:
        """Match text against structural patterns using the pattern matcher.

        Args:
            text: Response text to classify.

        Returns:
            Tuple of (matched_type, confidence) or (None, 0.0) if no match.
        """
        return self._pattern_matcher.match(text)

    # SVM label mapping: SVM uses simplified 6-label scheme, map to ResponseType
    # SVM labels: AGREE, DECLINE, DEFER, OTHER, QUESTION, REACTION
    # ResponseType: AGREE, DECLINE, DEFER, ACKNOWLEDGE, ANSWER, QUESTION,
    #               REACT_POSITIVE, REACT_SYMPATHY, STATEMENT, GREETING
    SVM_LABEL_MAP: dict[str, ResponseType] = {
        "AGREE": ResponseType.AGREE,
        "DECLINE": ResponseType.DECLINE,
        "DEFER": ResponseType.DEFER,
        "QUESTION": ResponseType.QUESTION,
        "REACTION": ResponseType.REACT_POSITIVE,  # Map to positive by default
        "OTHER": ResponseType.ANSWER,  # Catch-all maps to ANSWER
    }

    def _classify_with_svm(
        self, text: str, embedder: Embedder | None = None
    ) -> tuple[ResponseType | None, float]:
        """Classify using the trained SVM (81.9% F1).

        Args:
            text: Response text to classify.
            embedder: Optional embedder for computing embeddings.

        Returns:
            Tuple of (response_type, confidence).
        """
        if not self.svm or not self._svm_labels:
            return None, 0.0

        try:
            # Get embedder (use provided or from EmbedderMixin)
            emb = embedder if embedder is not None else self.embedder

            # Compute embedding
            embedding = emb.encode([text], normalize=True)[0]

            return self._classify_with_svm_embedding(embedding)

        except Exception as e:
            logger.warning("SVM classification failed: %s", e)
            return None, 0.0

    def _classify_with_svm_embedding(
        self, embedding: np.ndarray
    ) -> tuple[ResponseType | None, float]:
        """Classify using SVM with pre-computed embedding.

        Args:
            embedding: Pre-computed text embedding (normalized).

        Returns:
            Tuple of (response_type, confidence).
        """
        if not self.svm or not self._svm_labels:
            return None, 0.0

        try:
            # Use SVMModelMixin's _predict_svm
            label, confidence = self._predict_svm(embedding)
            if label is None:
                return None, 0.0

            # Uppercase for consistency
            label = label.upper()

            # Map SVM label to ResponseType using mapping table
            response_type = self.SVM_LABEL_MAP.get(label)
            if response_type is None:
                try:
                    response_type = ResponseType(label)
                except ValueError:
                    response_type = ResponseType.ANSWER

            return response_type, confidence

        except Exception as e:
            logger.warning("SVM classification with embedding failed: %s", e)
            return None, 0.0

    def _apply_confidence_threshold(
        self,
        label: ResponseType,
        confidence: float,
    ) -> tuple[ResponseType, float]:
        """Apply confidence threshold - default to ANSWER if confidence is low.

        This prevents over-prediction of DECLINE/DEFER/AGREE when the classifier
        isn't confident. ANSWER is the safe "catch-all" for explanations/info.

        Args:
            label: Predicted label.
            confidence: Prediction confidence.

        Returns:
            Tuple of (final_label, final_confidence).
        """
        if not self._use_confidence_threshold:
            return label, confidence

        # Only apply threshold to non-structural classifications
        # and only for labels that are commonly over-predicted
        over_predicted_types = {
            ResponseType.DECLINE,
            ResponseType.DEFER,
            ResponseType.AGREE,
            ResponseType.ACKNOWLEDGE,
            ResponseType.REACT_POSITIVE,
            ResponseType.REACT_SYMPATHY,
        }

        # Higher thresholds for frequently misclassified types
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
            logger.debug(
                "Low confidence (%.2f < %.2f) for %s, defaulting to ANSWER",
                confidence,
                threshold,
                label.value,
            )
            return ResponseType.ANSWER, confidence

        return label, confidence

    def _apply_trigger_filtering(
        self,
        label: ResponseType,
        confidence: float,
        trigger_da: str | None,
    ) -> tuple[ResponseType, float]:
        """Filter response type based on trigger dialogue act.

        Commitment responses (AGREE/DECLINE/DEFER) only make sense for
        commitment triggers (INVITATION/REQUEST/YN_QUESTION).

        Args:
            label: Predicted label.
            confidence: Prediction confidence.
            trigger_da: Trigger dialogue act type (e.g., "INVITATION").

        Returns:
            Tuple of (final_label, final_confidence).
        """
        if not self._use_trigger_filtering or trigger_da is None:
            return label, confidence

        # If label is a commitment response but trigger isn't a commitment question
        if label in COMMITMENT_RESPONSE_TYPES:
            if trigger_da not in self.COMMITMENT_TRIGGER_TYPES:
                # Check valid responses for this trigger type
                valid_responses = TRIGGER_TO_VALID_RESPONSES.get(trigger_da, [])
                if label not in valid_responses:
                    logger.debug(
                        "Filtering %s for trigger %s, defaulting to ANSWER", label.value, trigger_da
                    )
                    return ResponseType.ANSWER, confidence * 0.8

        return label, confidence

    def classify(
        self,
        text: str,
        embedder: Embedder | None = None,
        trigger_da: str | None = None,
    ) -> ClassificationResult:
        """Classify a response text using three-layer hybrid approach.

        Strategy:
        1. Structural hint (fast regex) - get candidate type
        2. Centroid verification (semantic) - verify or override hint
        3. kNN fallback (DA classifier) - for ambiguous cases

        If trigger_da is provided, uses it to constrain valid response types.

        Args:
            text: Response text to classify.
            embedder: Optional embedder for centroid verification.
            trigger_da: Optional trigger DA type for response constraints.

        Returns:
            ClassificationResult with label, confidence, and method.
        """
        if not text or not text.strip():
            return ClassificationResult(
                label=ResponseType.STATEMENT,
                confidence=0.0,
                method="empty",
            )

        # =================================================================
        # PRE-FILTER: iMessage Tapback Reactions
        # =================================================================
        # Tapbacks are reaction metadata, not real responses. Handle them first.
        tapback_type = self._detect_tapback(text)
        if tapback_type:
            if tapback_type == "positive":
                # Liked/Loved/Laughed → REACT_POSITIVE
                return ClassificationResult(
                    label=ResponseType.REACT_POSITIVE,
                    confidence=0.95,
                    method="tapback_positive",
                    structural_match=True,
                )
            else:
                # Disliked/Emphasized/Questioned → Filter to ANSWER (catch-all)
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

        # =================================================================
        # THREE-LAYER CLASSIFICATION PIPELINE
        # =================================================================

        # LAYER 1: Structural hint (fast regex patterns)
        structural_type, structural_conf = self._match_structural(normalized)

        if structural_type is not None:
            # LAYER 2: Centroid verification (semantic check)
            if self._use_centroid_verification and self.centroids:
                verified_type, verified_conf, was_verified = self._verify_with_centroid(
                    normalized, structural_type, embedder
                )

                if was_verified:
                    # Structural hint confirmed by centroid
                    return ClassificationResult(
                        label=verified_type,
                        confidence=verified_conf,
                        method="structural_verified",
                        structural_match=True,
                    )
                else:
                    # Centroid overrode structural hint (e.g., "No way!" -> REACT_POSITIVE)
                    return ClassificationResult(
                        label=verified_type,
                        confidence=verified_conf,
                        method="centroid_override",
                        structural_match=True,
                    )
            else:
                # No centroid verification - trust structural hint
                return ClassificationResult(
                    label=structural_type,
                    confidence=structural_conf,
                    method="structural",
                    structural_match=True,
                )

        # LAYER 3: No structural hint - use SVM classifier
        svm_type, svm_conf = self._classify_with_svm(normalized, embedder)
        classifier_used = "svm"

        if svm_type is not None:
            # Apply confidence threshold (Option A) - default low-confidence to ANSWER
            final_type, final_conf = self._apply_confidence_threshold(svm_type, svm_conf)

            # Apply trigger filtering (Option B) - filter invalid responses for trigger
            final_type, final_conf = self._apply_trigger_filtering(
                final_type, final_conf, trigger_da
            )

            method = classifier_used
            if final_type != svm_type:
                method = f"{classifier_used}_filtered"

            return ClassificationResult(
                label=final_type,
                confidence=final_conf,
                method=method,
                structural_match=False,
                da_label=svm_type.value,
                da_confidence=svm_conf,
            )

        # Fallback: No classification possible
        return ClassificationResult(
            label=ResponseType.ANSWER,  # ANSWER is safer than STATEMENT
            confidence=0.5,
            method="fallback",
            structural_match=False,
        )

    def classify_batch(
        self,
        texts: list[str],
        embedder: Embedder | None = None,
        batch_size: int = 256,
    ) -> list[ClassificationResult]:
        """Classify multiple response texts with batched embedding.

        This is significantly faster than calling classify() in a loop because
        embeddings are computed in batches, taking advantage of GPU/MPS acceleration.

        Args:
            texts: List of response texts to classify.
            embedder: Optional embedder for computing embeddings.
            batch_size: Batch size for embedding computation. Default 256.
                - 64-128: Safe for low-memory systems
                - 256: Good default for most Apple Silicon
                - 512-1024: If you have 16GB+ RAM

        Returns:
            List of ClassificationResults in the same order as input texts.
        """
        import numpy as np

        if not texts:
            return []

        # Get embedder once for all texts
        if embedder is None:
            from jarvis.embedding_adapter import get_embedder

            embedder = get_embedder()

        n = len(texts)
        results: list[ClassificationResult | None] = [None] * n

        # =================================================================
        # PHASE 1: Fast structural matching and tapback detection (no embedding)
        # =================================================================
        needs_embedding_indices: list[int] = []
        needs_embedding_texts: list[str] = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = ClassificationResult(
                    label=ResponseType.STATEMENT,
                    confidence=0.0,
                    method="empty",
                )
                continue

            # Check tapbacks first (fast)
            tapback_type = self._detect_tapback(text)
            if tapback_type:
                if tapback_type == "positive":
                    results[i] = ClassificationResult(
                        label=ResponseType.REACT_POSITIVE,
                        confidence=0.95,
                        method="tapback_positive",
                        structural_match=True,
                    )
                else:
                    results[i] = ClassificationResult(
                        label=ResponseType.ANSWER,
                        confidence=0.3,
                        method="tapback_filtered",
                        structural_match=True,
                    )
                continue

            normalized = normalize_for_task(text, "classification")
            if not normalized:
                results[i] = ClassificationResult(
                    label=ResponseType.STATEMENT,
                    confidence=0.0,
                    method="normalized_empty",
                )
                continue

            # This text needs embedding for full classification
            needs_embedding_indices.append(i)
            needs_embedding_texts.append(normalized)

        # If all texts were handled by fast path, return early
        if not needs_embedding_texts:
            return [r for r in results if r is not None]

        # =================================================================
        # PHASE 2: Batch compute embeddings for texts that need semantic analysis
        # =================================================================
        all_embeddings = []
        for batch_start in range(0, len(needs_embedding_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(needs_embedding_texts))
            batch_texts = needs_embedding_texts[batch_start:batch_end]
            batch_embeddings = embedder.encode(batch_texts, normalize=True)
            all_embeddings.append(batch_embeddings)

        # Combine all batches
        embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]

        # =================================================================
        # PHASE 3: Classify using pre-computed embeddings
        # =================================================================
        # Pre-compute centroid arrays for efficiency
        centroid_arrays = {}
        if self.centroids:
            for label, centroid in self.centroids.items():
                centroid_arrays[label] = np.array(centroid)

        for idx, (orig_idx, text) in enumerate(zip(needs_embedding_indices, needs_embedding_texts)):
            embedding = embeddings[idx]

            # Try structural match first
            structural_type, structural_conf = self._match_structural(text)

            if structural_type is not None:
                # Verify with centroid using pre-computed embedding
                if self._use_centroid_verification and centroid_arrays:
                    verified_type, verified_conf, was_verified = self._verify_with_embedding(
                        embedding, structural_type, centroid_arrays
                    )

                    if was_verified:
                        results[orig_idx] = ClassificationResult(
                            label=verified_type,
                            confidence=verified_conf,
                            method="structural_verified",
                            structural_match=True,
                        )
                    else:
                        # Centroid overrode - check DA for confirmation
                        da_type, da_conf = self._classify_with_embedding(embedding)

                        if da_type == verified_type:
                            results[orig_idx] = ClassificationResult(
                                label=verified_type,
                                confidence=max(verified_conf, da_conf),
                                method="centroid_override_confirmed",
                                structural_match=True,
                                da_label=da_type.value if da_type else None,
                                da_confidence=da_conf,
                            )
                        elif da_type == structural_type:
                            results[orig_idx] = ClassificationResult(
                                label=structural_type,
                                confidence=structural_conf,
                                method="structural_da_agree",
                                structural_match=True,
                                da_label=da_type.value if da_type else None,
                                da_confidence=da_conf,
                            )
                        else:
                            results[orig_idx] = ClassificationResult(
                                label=verified_type,
                                confidence=verified_conf,
                                method="centroid_override",
                                structural_match=True,
                                da_label=da_type.value if da_type else None,
                                da_confidence=da_conf,
                            )
                else:
                    results[orig_idx] = ClassificationResult(
                        label=structural_type,
                        confidence=structural_conf,
                        method="structural",
                        structural_match=True,
                    )
            else:
                # No structural hint - use SVM classifier with pre-computed embedding
                cls_type, cls_conf = self._classify_with_svm_embedding(embedding)
                classifier_used = "svm"

                if cls_type is not None:
                    final_type, final_conf = self._apply_confidence_threshold(cls_type, cls_conf)
                    # Note: trigger_da not available in batch mode
                    method = classifier_used
                    if final_type != cls_type:
                        method = f"{classifier_used}_filtered"

                    results[orig_idx] = ClassificationResult(
                        label=final_type,
                        confidence=final_conf,
                        method=method,
                        structural_match=False,
                        da_label=cls_type.value,
                        da_confidence=cls_conf,
                    )
                else:
                    results[orig_idx] = ClassificationResult(
                        label=ResponseType.ANSWER,
                        confidence=0.5,
                        method="fallback",
                        structural_match=False,
                    )

        return [r for r in results if r is not None]

    def _verify_with_embedding(
        self,
        embedding: np.ndarray,
        hint_type: ResponseType,
        centroid_arrays: dict[str, np.ndarray],
    ) -> tuple[ResponseType, float, bool]:
        """Verify a structural hint using pre-computed embedding and centroids.

        Args:
            embedding: Pre-computed text embedding (normalized).
            hint_type: Structural hint to verify.
            centroid_arrays: Dict of label -> centroid numpy array.

        Returns:
            Tuple of (final_type, confidence, was_verified).
        """
        import numpy as np

        # Compute similarity to all centroids
        similarities = {}
        for label, centroid in centroid_arrays.items():
            sim = float(np.dot(embedding, centroid))
            similarities[label] = sim

        hint_sim = similarities.get(hint_type.value, 0.0)
        best_label = max(similarities, key=similarities.get)
        best_sim = similarities[best_label]

        thresholds = get_config().classifier_thresholds
        if hint_sim >= thresholds.response_centroid_verify:
            return hint_type, min(0.95, hint_sim + 0.2), True

        if best_sim - hint_sim > thresholds.response_centroid_margin:
            try:
                override_type = ResponseType(best_label)
                return override_type, best_sim, False
            except ValueError:
                pass

        return hint_type, self._structural_confidence, True

    def is_commitment_response(self, result: ClassificationResult) -> bool:
        """Check if the classification is a commitment response type.

        Commitment responses are AGREE, DECLINE, or DEFER - the types
        we want to generate multiple options for.

        Args:
            result: Classification result to check.

        Returns:
            True if this is a commitment response type.
        """
        return result.label in COMMITMENT_RESPONSE_TYPES


# =============================================================================
# Singleton Access
# =============================================================================

_factory: SingletonFactory[HybridResponseClassifier] = SingletonFactory(HybridResponseClassifier)


def get_response_classifier() -> HybridResponseClassifier:
    """Get or create the singleton HybridResponseClassifier instance.

    Returns:
        The shared HybridResponseClassifier instance.
    """
    return _factory.get()


def reset_response_classifier() -> None:
    """Reset the singleton response classifier."""
    _factory.reset()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ResponseType",
    "COMMITMENT_RESPONSE_TYPES",
    "ClassificationResult",
    "HybridResponseClassifier",
    "get_response_classifier",
    "reset_response_classifier",
]
