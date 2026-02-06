"""Response Mobilization Classifier - Structural classification based on linguistics research.

Classifies messages by response pressure using features from conversation analysis:
- Stivers & Rossano (2010): Four response-mobilizing features
- Labov & Fanshel (1977): A-events vs B-events and epistemic territory
- Heritage (2012): Epistemic gradient and knowledge asymmetry

Response pressure is a GRADIENT, not binary. We detect:
1. Interrogative syntax (WH-words, aux inversion)
2. Punctuation ("?" as proxy for rising prosody)
3. Epistemic stance (recipient-oriented vs speaker-oriented)
4. Action type (request, invitation, assessment, telling)

Categories (ordered by response pressure):
- HIGH: Requires substantive response (accept/decline, answer, commitment)
- MEDIUM: Warrants emotional response (empathy, celebration)
- LOW: Response optional (acknowledgment acceptable, silence okay)
- NONE: No response expected (backchannel, closing)

Usage:
    from jarvis.classifiers.response_mobilization import classify_response_pressure

    result = classify_response_pressure("Can you pick me up?")
    print(result.pressure, result.response_type)  # HIGH, COMMITMENT

    result = classify_response_pressure("I wonder if they'll win")
    print(result.pressure, result.response_type)  # LOW, OPTIONAL

References:
- https://emcawiki.net/Mobilizing_response
- https://emcawiki.net/Conditional_relevance
- https://emcawiki.net/B-event_statement
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ResponsePressure(str, Enum):
    """Response pressure level based on response-mobilizing features."""

    HIGH = "high"  # Response required (questions, requests, invitations)
    MEDIUM = "medium"  # Response expected (emotional news, assessments seeking alignment)
    LOW = "low"  # Response optional (musings, opinions, tellings)
    NONE = "none"  # No response needed (backchannels, closings)


class ResponseType(str, Enum):
    """What type of response is appropriate."""

    # HIGH pressure responses
    COMMITMENT = "commitment"  # Accept/decline/defer (for requests, invitations)
    ANSWER = "answer"  # Provide information (for questions)
    CONFIRMATION = "confirmation"  # Yes/no (for B-event statements seeking confirmation)

    # MEDIUM pressure responses
    EMOTIONAL = "emotional"  # Empathy, celebration, comfort (for news/assessments)
    ALIGNMENT = "alignment"  # Agreement/disagreement (for opinions seeking alignment)

    # LOW pressure responses
    OPTIONAL = "optional"  # Acknowledgment okay, elaboration okay, silence okay

    # NONE
    CLOSING = "closing"  # Conversation closing, no response needed


@dataclass
class MobilizationResult:
    """Result from response mobilization classification."""

    pressure: ResponsePressure
    response_type: ResponseType
    confidence: float
    features: dict[str, bool]  # Which features were detected
    method: str = "structural"

    def __repr__(self) -> str:
        p = self.pressure.value
        r = self.response_type.value
        return f"MobilizationResult({p}, {r}, conf={self.confidence:.2f})"


# =============================================================================
# Feature Detection Patterns
# =============================================================================

# Interrogative syntax: WH-words at start
WH_WORDS = {"what", "where", "when", "who", "why", "how", "which", "whose"}

# Auxiliary verbs for subject-aux inversion (marks polar questions)
AUX_VERBS = {
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "should",
    "shall",
    "is",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
}

# === EPISTEMIC STANCE MARKERS ===
# These determine if utterance is oriented toward speaker's or recipient's knowledge

# Speaker-oriented (A-event markers) - LOW mobilization
# Speaker is sharing their own mental state, not accessing recipient's knowledge
SPEAKER_ORIENTED_PATTERNS = [
    r"^i wonder\b",  # "I wonder if..." - musing, not question
    r"^i'?m (wondering|curious)\b",  # "I'm wondering..." - thinking aloud
    r"^(kinda?|kind of|sort of) curious\b",  # "Kind of curious..." - hedged curiosity
    r"^curious (how|what|if|whether)\b",  # "Curious how..." - no "you"
    r"^wonder (if|what|how|why|whether)\b",  # "Wonder if..." - dropped "I"
    r"^i (think|guess|suppose|assume|imagine|bet)\b",  # Opinion markers
    r"^i (don'?t think|doubt)\b",  # Negative epistemic stance
    r"^(maybe|perhaps|probably)\b",  # Hedged assertions
]

# Recipient-oriented (B-event markers) - HIGH mobilization
# Accessing recipient's epistemic territory, expects response
RECIPIENT_ORIENTED_PATTERNS = [
    r"^(do|does|did|can|could|will|would|should) (you|u|ya)\b",  # "Can you..." "Do you..."
    r"^(are|is|were|was) (you|u|ya)\b",  # "Are you coming?"
    r"^(have|has) (you|u|ya)\b",  # "Have you seen...?"
    r"^(you|u|ya) (free|available|busy|down|coming|going)\b",  # "You free?"
    r"^do (you|u|ya) (know|think|want|need|have)\b",  # "Do you know..."
]

# === REQUEST/INVITATION PATTERNS (HIGH - COMMITMENT) ===
REQUEST_PATTERNS = [
    r"^(can|could|would|will) (you|u|ya)\s+\w+",  # "Can you help?"
    r"^(wanna|want to|down to|tryna)\s+",  # "Wanna grab lunch?"
    r"^(let'?s|lets)\s+",  # "Let's go"
    r"^(let me know|lmk)\b",  # "Let me know when..."
    r"^(text|call|send|tell|show|give|bring|get|grab|pick) me\b",  # "Text me when..."
    r"\b(pick (me|us) up|give (me|us) a ride)\b",  # "Pick me up"
    r"^(want me to|should i|shall i|can i)\b",  # Offers requiring decision
    r"^(anyone|anybody|any of y'?all|y'?all) (wanna|want to|down to)\b",  # Group invitations
]

# Imperative verbs at start (commands/requests)
IMPERATIVE_VERBS = {
    "send",
    "give",
    "bring",
    "take",
    "get",
    "grab",
    "pick",
    "call",
    "text",
    "email",
    "check",
    "look",
    "help",
    "come",
    "go",
    "tell",
    "show",
    "make",
    "let",
    "put",
    "find",
    "buy",
    "read",
    "watch",
    "meet",
    "try",
    "open",
}

# === QUESTION PATTERNS (HIGH - ANSWER) ===
# Direct information-seeking questions
INFO_QUESTION_PATTERNS = [
    r"^what (time|day|is|are|was|were|did|does|do|happened)\b",
    r"^where (is|are|did|do|does|should|can|were|was|you|u|ya)\b",  # includes "where you at"
    r"^when (is|are|did|do|does|should|can|will)\b",
    r"^who (is|are|did|was|were|does|do)\b",
    r"^how (much|many|long|far|often|do|did|does|is|are|can|should)\b",
    r"^which (one|is|are|do|does|did|should)\b",
]

# === REACTIVE PATTERNS (MEDIUM - EMOTIONAL) ===
# News/assessments that warrant emotional response
REACTIVE_PATTERNS = [
    # Exclamatory markers (multiple ! = emotional intensity)
    r"!!+",  # "I got the job!!"
    # Emotional interjections
    r"^(omg|oh my god|omfg|wow|whoa|yay|ugh|damn|shit|holy|wtf)\b",
    # News announcements
    r"^i (got|just got|finally got) (the|a|an|my)\b",  # "I got the job"
    r"^(we|they|i) (won|lost|passed|failed|made it)\b",  # "We won!"
    # Strong emotion patterns
    r"\b(i'?m so|that'?s so|so happy|so sad|so sorry|so excited)\b",
    r"\b(can'?t believe|love you|miss you|hate this)\b",
    r"^(congrats|congratulations)\b",
    r"\b(this sucks|that sucks|sucks that)\b",
    # Strong emotion emojis (multiple = emphasis)
    r"(😍|😭|🥳|😱|🎉|❤️|💕|🥰|😢){2,}",
]

# === OPINION/ASSESSMENT PATTERNS (LOW - OPTIONAL) ===
# Opinions that don't strongly seek alignment
OPINION_PATTERNS = [
    r"^no (way|chance)\s+\w+",  # "No way they win" - opinion, not reactive
    r"^(i don'?t think|doubt|not sure)\b",  # Hedged disagreement
    r"^(i think|i guess|i feel like|i bet)\b",  # Opinion markers
    r"^(probably|maybe|might be)\b",  # Hedged assertions
]

# === TELLING/STATEMENT PATTERNS (LOW - OPTIONAL) ===
# Informing without seeking response
TELLING_PATTERNS = [
    r"^i (went|did|got|saw|had|made|took|came|finished|just)\b",  # Past tense reports
    r"^(the|my|our|his|her|their) \w+ (is|was|are|were|got|has)\b",  # Third-person reports
    r"^(it|that|this) (is|was|looks|seems|sounds)\b",  # Assessments
    r"^i'?m (going|heading|leaving|about to|gonna)\b",  # Future plans (informing)
    r"^i'?ll\s+",  # "I'll be there" - commitment announcement
    r"^(fyi|btw|by the way|just so you know)\b",  # Information markers
]

# === RHETORICAL PATTERNS (LOW - OPTIONAL) ===
# Questions that don't expect literal answers
RHETORICAL_PATTERNS = [
    r"^why (do|does|would|did) \w+ (even|always|never)\b",  # "Why do dads even..."
    r"^how (do|does|did|can|could) \w+ (even|always)\b",  # "How does that even work"
    r"^who (even|actually|really)\b",  # "Who even says that"
    r"^(what|where|why|how) the (hell|heck|fuck)\b",  # Exclamatory questions
]

# === BACKCHANNEL/CLOSING PATTERNS (NONE) ===
BACKCHANNEL_WORDS = {
    # Acknowledgments
    "ok",
    "okay",
    "k",
    "kk",
    "yes",
    "yeah",
    "yea",
    "yep",
    "yup",
    "ya",
    "no",
    "nope",
    "nah",
    "sure",
    "bet",
    "word",
    "true",
    "facts",
    "copy",
    "alright",
    "aight",
    "ight",
    "gotcha",
    "got it",
    "heard",
    # Reactions
    "lol",
    "lmao",
    "haha",
    "hehe",
    "lolol",
    "😂",
    "🤣",
    "💀",
    "nice",
    "cool",
    "good",
    "great",
    "awesome",
    "dope",
    "fire",
    "lit",
    # Gratitude
    "thanks",
    "thx",
    "ty",
    "thank you",
    "np",
    "nw",
    "yw",
    # Closings
    "bye",
    "cya",
    "later",
    "ttyl",
    "gn",
    "gm",
    "see you",
    "see ya",
}

BACKCHANNEL_PHRASES = {
    "sounds good",
    "for sure",
    "of course",
    "no problem",
    "no worries",
    "all good",
    "you too",
    "same",
    "mood",
    "felt that",
    "fr",
    "real",
    "on my way",
    "omw",
    "be there soon",
    "almost there",
}

# Greetings (NONE - or just return greeting)
GREETING_PATTERNS = [
    r"^(hey|hi|hello|yo|sup|hiya|howdy|what'?s up|wassup|whaddup)!*$",
    r"^(good morning|good afternoon|good evening|good night|gm|gn)\b",
]


# =============================================================================
# Classification Functions
# =============================================================================


def _normalize_for_classification(text: str) -> str:
    """Light normalization for classification.

    Note: For best results, run full slang expansion BEFORE calling this.
    """
    if not text:
        return ""
    # Lowercase, strip, collapse whitespace
    normalized = " ".join(text.lower().split())
    return normalized


def _detect_features(text: str, text_lower: str) -> dict[str, bool]:
    """Detect all response-mobilizing features in text."""
    features = {
        "has_question_mark": False,
        "has_wh_word": False,
        "has_aux_inversion": False,
        "is_speaker_oriented": False,
        "is_recipient_oriented": False,
        "is_request": False,
        "is_imperative": False,
        "is_reactive": False,
        "is_rhetorical": False,
        "is_opinion": False,
        "is_telling": False,
        "is_backchannel": False,
        "is_greeting": False,
        "has_multiple_exclamation": False,
    }

    words = text_lower.split()
    first_word = words[0] if words else ""

    # Punctuation features
    features["has_question_mark"] = text.rstrip().endswith("?")
    features["has_multiple_exclamation"] = text.count("!") >= 2

    # Interrogative syntax
    features["has_wh_word"] = first_word.rstrip("?") in WH_WORDS

    # Aux inversion (do/does/did/can/etc at start)
    if first_word in AUX_VERBS:
        # Check if followed by subject (you, it, they, etc.)
        if len(words) > 1 and words[1] in {"you", "u", "ya", "i", "we", "they", "he", "she", "it"}:
            features["has_aux_inversion"] = True

    # Epistemic stance
    for pattern in SPEAKER_ORIENTED_PATTERNS:
        if re.match(pattern, text_lower):
            features["is_speaker_oriented"] = True
            break

    for pattern in RECIPIENT_ORIENTED_PATTERNS:
        if re.match(pattern, text_lower):
            features["is_recipient_oriented"] = True
            break

    # Request patterns
    for pattern in REQUEST_PATTERNS:
        if re.search(pattern, text_lower):
            features["is_request"] = True
            break

    # Imperative
    if first_word in IMPERATIVE_VERBS:
        features["is_imperative"] = True

    # Reactive
    for pattern in REACTIVE_PATTERNS:
        if re.search(pattern, text_lower):
            features["is_reactive"] = True
            break

    # Rhetorical
    for pattern in RHETORICAL_PATTERNS:
        if re.match(pattern, text_lower):
            features["is_rhetorical"] = True
            break

    # Opinion
    for pattern in OPINION_PATTERNS:
        if re.match(pattern, text_lower):
            features["is_opinion"] = True
            break

    # Telling
    for pattern in TELLING_PATTERNS:
        if re.match(pattern, text_lower):
            features["is_telling"] = True
            break

    # Backchannel
    stripped = text_lower.rstrip("!?.")
    if stripped in BACKCHANNEL_WORDS or stripped in BACKCHANNEL_PHRASES:
        features["is_backchannel"] = True

    # Greeting
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, text_lower):
            features["is_greeting"] = True
            break

    return features


def classify_response_pressure(text: str) -> MobilizationResult:
    """Classify message by response pressure using linguistic features.

    Based on Stivers & Rossano (2010) response mobilization framework.

    Args:
        text: Message text (ideally after slang expansion)

    Returns:
        MobilizationResult with pressure level and appropriate response type
    """
    if not text or not text.strip():
        return MobilizationResult(
            pressure=ResponsePressure.NONE,
            response_type=ResponseType.CLOSING,
            confidence=0.0,
            features={},
            method="empty",
        )

    text_lower = _normalize_for_classification(text)
    features = _detect_features(text, text_lower)

    # === NONE: Backchannels and greetings ===
    if features["is_backchannel"]:
        return MobilizationResult(
            pressure=ResponsePressure.NONE,
            response_type=ResponseType.CLOSING,
            confidence=0.95,
            features=features,
        )

    if features["is_greeting"]:
        # Greetings get greetings back, but low pressure
        return MobilizationResult(
            pressure=ResponsePressure.LOW,
            response_type=ResponseType.OPTIONAL,
            confidence=0.90,
            features=features,
        )

    # === HIGH: Requests and invitations (COMMITMENT) ===
    if features["is_request"] or features["is_imperative"]:
        return MobilizationResult(
            pressure=ResponsePressure.HIGH,
            response_type=ResponseType.COMMITMENT,
            confidence=0.95,
            features=features,
        )

    # === HIGH: Recipient-oriented questions (ANSWER or COMMITMENT) ===
    if features["is_recipient_oriented"]:
        # "Are you coming?" = commitment, "Do you know X?" = answer
        if re.match(
            r"^(are|is) (you|u|ya) (coming|going|in|down|free|busy|available)\b", text_lower
        ):
            return MobilizationResult(
                pressure=ResponsePressure.HIGH,
                response_type=ResponseType.COMMITMENT,
                confidence=0.90,
                features=features,
            )
        return MobilizationResult(
            pressure=ResponsePressure.HIGH,
            response_type=ResponseType.ANSWER,
            confidence=0.90,
            features=features,
        )

    # === HIGH: Direct questions with ? and WH-word or aux inversion ===
    if features["has_question_mark"]:
        if features["has_wh_word"] or features["has_aux_inversion"]:
            # Check for rhetorical patterns first
            if features["is_rhetorical"]:
                return MobilizationResult(
                    pressure=ResponsePressure.LOW,
                    response_type=ResponseType.OPTIONAL,
                    confidence=0.80,
                    features=features,
                )
            return MobilizationResult(
                pressure=ResponsePressure.HIGH,
                response_type=ResponseType.ANSWER,
                confidence=0.90,
                features=features,
            )
        # Has ? but no clear question syntax - could be declarative question
        # "You're coming?" - B-event statement seeking confirmation
        return MobilizationResult(
            pressure=ResponsePressure.HIGH,
            response_type=ResponseType.CONFIRMATION,
            confidence=0.75,
            features=features,
        )

    # === LOW: Speaker-oriented (musings, wonderings) ===
    if features["is_speaker_oriented"]:
        return MobilizationResult(
            pressure=ResponsePressure.LOW,
            response_type=ResponseType.OPTIONAL,
            confidence=0.90,
            features=features,
        )

    # === LOW: Rhetorical questions (no ?) ===
    if features["is_rhetorical"]:
        return MobilizationResult(
            pressure=ResponsePressure.LOW,
            response_type=ResponseType.OPTIONAL,
            confidence=0.85,
            features=features,
        )

    # === MEDIUM: Reactive content (emotional news) ===
    if features["is_reactive"] or features["has_multiple_exclamation"]:
        return MobilizationResult(
            pressure=ResponsePressure.MEDIUM,
            response_type=ResponseType.EMOTIONAL,
            confidence=0.85,
            features=features,
        )

    # === LOW: Opinions without strong alignment-seeking ===
    if features["is_opinion"]:
        return MobilizationResult(
            pressure=ResponsePressure.LOW,
            response_type=ResponseType.OPTIONAL,
            confidence=0.80,
            features=features,
        )

    # === LOW: Tellings (informing) ===
    if features["is_telling"]:
        return MobilizationResult(
            pressure=ResponsePressure.LOW,
            response_type=ResponseType.OPTIONAL,
            confidence=0.80,
            features=features,
        )

    # === WH-word without ? - depends on pattern ===
    if features["has_wh_word"] and not features["has_question_mark"]:
        # Direct info-seeking WH-questions (even without ?)
        # "What time is the game" "Where are you" "When does it start"
        for pattern in INFO_QUESTION_PATTERNS:
            if re.match(pattern, text_lower):
                return MobilizationResult(
                    pressure=ResponsePressure.HIGH,
                    response_type=ResponseType.ANSWER,
                    confidence=0.80,
                    features=features,
                )
        # Otherwise, WH-word without ? is likely rhetorical
        return MobilizationResult(
            pressure=ResponsePressure.LOW,
            response_type=ResponseType.OPTIONAL,
            confidence=0.70,
            features=features,
        )

    # === DEFAULT: LOW (most casual chat is low-pressure) ===
    return MobilizationResult(
        pressure=ResponsePressure.LOW,
        response_type=ResponseType.OPTIONAL,
        confidence=0.50,
        features=features,
        method="default",
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def get_response_pressure(text: str) -> ResponsePressure:
    """Get just the pressure level."""
    return classify_response_pressure(text).pressure


def requires_response(text: str) -> bool:
    """Check if message requires a substantive response."""
    return classify_response_pressure(text).pressure == ResponsePressure.HIGH


def response_optional(text: str) -> bool:
    """Check if response is optional (LOW or NONE pressure)."""
    return classify_response_pressure(text).pressure in {
        ResponsePressure.LOW,
        ResponsePressure.NONE,
    }


# =============================================================================
# Mapping to Legacy Categories (for backwards compatibility)
# =============================================================================


def to_legacy_category(result: MobilizationResult) -> str:
    """Map to legacy ACTIONABLE/ANSWERABLE/REACTIVE/ACKNOWLEDGEABLE categories."""
    if result.pressure == ResponsePressure.HIGH:
        if result.response_type == ResponseType.COMMITMENT:
            return "ACTIONABLE"
        return "ANSWERABLE"
    elif result.pressure == ResponsePressure.MEDIUM:
        return "REACTIVE"
    else:
        return "ACKNOWLEDGEABLE"


def classify_legacy(text: str) -> str:
    """Classify using legacy category names."""
    return to_legacy_category(classify_response_pressure(text))


# =============================================================================
# Response Option Types (for multi-option generation)
# =============================================================================


class ResponseOptionType(str, Enum):
    """Types of user responses for multi-option generation.

    These are labels for the OPTIONS we generate, not for classifying
    incoming messages.
    """

    # Commitment responses (for HIGH pressure + COMMITMENT)
    AGREE = "AGREE"  # "Yeah I'm down!"
    DECLINE = "DECLINE"  # "Can't make it, sorry"
    DEFER = "DEFER"  # "Let me check and get back to you"

    # Question responses (for HIGH pressure + ANSWER)
    YES = "YES"
    NO = "NO"
    ANSWER = "ANSWER"  # Factual answer

    # Acknowledgment responses (for LOW/NONE pressure)
    ACKNOWLEDGE = "ACKNOWLEDGE"  # "Got it", "Sounds good"

    # Emotional responses (for MEDIUM pressure + EMOTIONAL)
    REACT_POSITIVE = "REACT_POSITIVE"  # "That's awesome!"
    REACT_SYMPATHY = "REACT_SYMPATHY"  # "Sorry to hear that"

    # Social responses
    GREETING = "GREETING"  # "Hey!"


# Response options valid for each response type
COMMITMENT_RESPONSE_OPTIONS = [
    ResponseOptionType.AGREE,
    ResponseOptionType.DECLINE,
    ResponseOptionType.DEFER,
]
QUESTION_RESPONSE_OPTIONS = [
    ResponseOptionType.YES,
    ResponseOptionType.NO,
    ResponseOptionType.ANSWER,
]
EMOTIONAL_RESPONSE_OPTIONS = [ResponseOptionType.REACT_POSITIVE, ResponseOptionType.REACT_SYMPATHY]

# Mapping from (pressure, response_type) to valid response options
VALID_RESPONSE_OPTIONS: dict[tuple[ResponsePressure, ResponseType], list[ResponseOptionType]] = {
    (ResponsePressure.HIGH, ResponseType.COMMITMENT): COMMITMENT_RESPONSE_OPTIONS,
    (ResponsePressure.HIGH, ResponseType.ANSWER): QUESTION_RESPONSE_OPTIONS,
    (ResponsePressure.HIGH, ResponseType.CONFIRMATION): [
        ResponseOptionType.YES,
        ResponseOptionType.NO,
    ],
    (ResponsePressure.MEDIUM, ResponseType.EMOTIONAL): EMOTIONAL_RESPONSE_OPTIONS,
    (ResponsePressure.LOW, ResponseType.OPTIONAL): [ResponseOptionType.ACKNOWLEDGE],
    (ResponsePressure.NONE, ResponseType.CLOSING): [ResponseOptionType.ACKNOWLEDGE],
}


def get_valid_response_options(result: MobilizationResult) -> list[ResponseOptionType]:
    """Get valid response options for a classification result."""
    key = (result.pressure, result.response_type)
    return VALID_RESPONSE_OPTIONS.get(key, [ResponseOptionType.ACKNOWLEDGE])


__all__ = [
    "ResponsePressure",
    "ResponseType",
    "MobilizationResult",
    "classify_response_pressure",
    "get_response_pressure",
    "requires_response",
    "response_optional",
    "to_legacy_category",
    "classify_legacy",
]
