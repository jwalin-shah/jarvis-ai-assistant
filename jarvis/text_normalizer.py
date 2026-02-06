"""Text Normalizer - Canonical text cleaning for the extraction pipeline.

Single source of truth for all text cleaning used throughout the pipeline.
All text should pass through normalize_text() before being stored or compared.

Usage:
    from jarvis.text_normalizer import normalize_text, extract_text_features

    cleaned = normalize_text("Liked "hey there"")  # Returns empty string (reaction)
    features = extract_text_features("btw, are you free?")  # {"starts_new_topic": True, ...}
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.nlp.ner_client import Entity

# Reaction patterns - these are tapbacks in iMessage
REACTION_PATTERNS = [
    # Full quoted forms
    r'^Liked\s+".*"$',
    r'^Loved\s+".*"$',
    r'^Disliked\s+".*"$',
    r'^Laughed at\s+".*"$',
    r'^Emphasized\s+".*"$',
    r'^Questioned\s+".*"$',
    r'^Removed a like from\s+".*"$',
    r'^Removed a heart from\s+".*"$',
    r'^Removed a dislike from\s+".*"$',
    r'^Removed a laugh from\s+".*"$',
    r'^Removed an exclamation from\s+".*"$',
    r'^Removed a question mark from\s+".*"$',
    # Truncated quote forms (missing closing quote)
    r'^Liked\s+"',
    r'^Loved\s+"',
    r'^Disliked\s+"',
    r'^Laughed at\s+"',
    r'^Emphasized\s+"',
    r'^Questioned\s+"',
    r'^Removed a .* from\s+"',
]
REACTION_REGEX = re.compile("|".join(REACTION_PATTERNS), re.IGNORECASE | re.DOTALL)

# Acknowledgment phrases that are too generic to be useful alone
# NOTE: Emotional reactions (cool, nice, good, great, awesome, lol, haha) are NOT acknowledgments
# They express emotion and need context-aware LLM generation
ACKNOWLEDGMENT_PHRASES = frozenset(
    {
        # True acknowledgments - confirmations and agreements
        "ok",
        "okay",
        "k",
        "kk",
        "yes",
        "yeah",
        "yep",
        "yup",
        "yea",
        "ya",
        "no",
        "nope",
        "nah",
        "na",
        "sure",
        # Gratitude expressions
        "thanks",
        "thank you",
        "thx",
        "ty",
        "np",
        "yw",
        # Understanding confirmations
        "alright",
        "aight",
        "sounds good",
        "got it",
        "gotcha",
        "heard",
        "bet",
        "word",
        # Status updates (not emotional reactions)
        "omw",
        "otw",
        "on my way",
        "be there soon",
        # NOTE: Removed emotional reactions (cool, nice, good, great, awesome)
        # and lol/lmao/haha/hehe - these need LLM generation, not canned responses
        "see you",
        "bye",
        "later",
        "ttyl",
        "cya",
        "gn",
        "gm",
        "idk",
        "idc",
        "tbh",
        "nvm",
        "ight",
    }
)

# Topic-shift markers that indicate a new conversation thread
TOPIC_SHIFT_MARKERS = frozenset(
    {
        "btw",
        "by the way",
        "anyway",
        "anyways",
        "also",
        "oh also",
        "speaking of",
        "on another note",
        "random but",
        "unrelated",
        "unrelated but",
        "separately",
        "side note",
        "quick question",
        "actually",
        "oh wait",
    }
)

# Emoji pattern for detecting emoji-only content
EMOJI_PATTERN = re.compile(
    r"^[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001F600-\U0001F64F"
    r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\s\u200d]+$"
)

# Common auto-signatures to strip
AUTO_SIGNATURE_PATTERNS = [
    r"\n--\s*\n.*$",  # Standard email signature separator
    r"\nSent from my iPhone.*$",
    r"\nSent from my iPad.*$",
    r"\nGet Outlook for iOS.*$",
    r"\nSent via.*$",
]
AUTO_SIGNATURE_REGEX = re.compile("|".join(AUTO_SIGNATURE_PATTERNS), re.IGNORECASE | re.DOTALL)

# Garbage patterns to filter out (opt-in)
GARBAGE_PATTERNS = [
    r"__kIMFileTransferGUID",  # iMessage file transfer metadata
    r"\b(verification|security)\s+code\b",  # 2FA codes
    r"\bRx\s+(is\s+)?ready\b",  # Pharmacy alerts
    r"\b(has|have)\s+shipped\b",  # Shipping notifications
    r"\btracking\s+(number|#)\b",  # Tracking numbers
    r"^\d{4,6}$",  # Pure numeric codes (4-6 digits)
]
GARBAGE_REGEX = re.compile("|".join(GARBAGE_PATTERNS), re.IGNORECASE)

# Spam/bot message patterns (notifications, marketing, automated messages)
SPAM_PATTERNS = [
    # Opt-out/unsubscribe indicators
    r"\b(reply\s+)?stop\s+to\s+(cancel|unsubscribe|opt[\s-]?out)\b",
    r"\btext\s+stop\s+to\b",
    r"\bstop\s+to\s+end\b",
    r"\bunsubscribe\b",
    # Order/shipping notifications
    r"\byour\s+order\s+(has\s+)?(shipped|been\s+shipped|is\s+on\s+(the\s+)?way)\b",
    r"\bdelivery\s+(scheduled|expected|arriving)\b",
    r"\bpackage\s+(delivered|arriving|on\s+the\s+way)\b",
    r"\bout\s+for\s+delivery\b",
    r"\btracking\s+(number|#|info|update)\b",
    r"\border\s+#?\s*\d+\b",
    # Appointment/reminder bots
    r"\bappointment\s+(reminder|confirmed|scheduled)\b",
    r"\breminder:\s*your\s+appointment\b",
    r"\bconfirm\s+your\s+appointment\b",
    r"\breply\s+(yes|y)\s+to\s+confirm\b",
    # Account/verification bots
    r"\bconfirm\s+your\s+(account|email|phone)\b",
    r"\bverify\s+your\s+(identity|account|email)\b",
    r"\byour\s+(one-?time\s+)?code\s+is\b",
    r"\byour\s+verification\s+code\b",
    r"\benter\s+code\s*:\s*\d+\b",
    # Marketing/promotional
    r"\blimited\s+time\s+offer\b",
    r"\bexclusive\s+deal\b",
    r"\bact\s+now\b",
    r"\bdon'?t\s+miss\s+out\b",
    r"\b\d+%\s+off\b",
    r"\bfree\s+shipping\b",
    r"\buse\s+code\s*:\s*\w+\b",
    # Bank/financial notifications
    r"\baccount\s+(balance|statement|alert)\b",
    r"\btransaction\s+(alert|notification)\b",
    r"\bpayment\s+(received|processed|due)\b",
    r"\byour\s+card\s+(ending\s+in\s+\d+|was\s+charged)\b",
    # Ride/food delivery
    r"\byour\s+(uber|lyft|doordash|ubereats|grubhub|postmates)\b",
    r"\bdriver\s+(is\s+)?(on\s+the\s+way|arriving|here)\b",
    r"\byour\s+ride\s+is\b",
    r"\bfood\s+is\s+(on\s+the\s+way|ready)\b",
]
SPAM_REGEX = re.compile("|".join(SPAM_PATTERNS), re.IGNORECASE)
URL_ONLY_REGEX = re.compile(r"^https?://\S+$", re.IGNORECASE)

# AttributedBody artifact patterns (NSAttributedString metadata)
ATTRIBUTED_ARTIFACT_REGEX = re.compile(
    r"zclassnamex\\$classes|NSAttributedString|NSValue", re.IGNORECASE
)

# Entity masking patterns (heuristic)
EMAIL_REGEX = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_REGEX = re.compile(r"\b(?:\+?1[\s\-\.]?)?\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b")
HANDLE_REGEX = re.compile(r"@\w+")
URL_REGEX = re.compile(r"https?://\S+", re.IGNORECASE)

# Code replacement patterns
CODE_CONTEXT_REGEX = re.compile(r"\b(code|otp|passcode|verification|security)\b", re.IGNORECASE)
CODE_REGEX = re.compile(r"\b\d{4,8}\b")

# Heuristic entity contexts
CITY_CONTEXT_REGEX = re.compile(r"\b(in|at|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")
TEAM_CONTEXT_REGEX = re.compile(
    r"\b(vs\.?|versus|against|beat|lost to|play(?:ing)?|game)\s+([A-Z][a-zA-Z]+)\b"
)
PERSON_CONTEXT_REGEX = re.compile(
    r"\b(with|for|tell|ask|meet|saw|see|call)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)

# Emoji normalization map (small, high-signal set)
EMOJI_TOKEN_MAP = {
    "😭": "<EMOJI_CRY>",
    "😢": "<EMOJI_CRY>",
    "😂": "<EMOJI_LAUGH>",
    "🤣": "<EMOJI_LAUGH>",
    "😅": "<EMOJI_LAUGH>",
    "😡": "<EMOJI_ANGRY>",
    "😠": "<EMOJI_ANGRY>",
    "❤️": "<EMOJI_LOVE>",
    "😍": "<EMOJI_LOVE>",
    "🥳": "<EMOJI_CELEBRATE>",
    "🎉": "<EMOJI_CELEBRATE>",
    "🙏": "<EMOJI_PRAY>",
    "👍": "<EMOJI_THUMBS_UP>",
    "👎": "<EMOJI_THUMBS_DOWN>",
}

# Repeated emoji pattern (3+ of same emoji in a row)
REPEATED_EMOJI_PATTERN = re.compile(
    r"([\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001F600-\U0001F64F"
    r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF])\1{2,}"
)

# Question words for detecting questions
QUESTION_WORDS = {"who", "what", "when", "where", "why", "how", "which", "whose"}

# Pre-compiled patterns for hot path functions (avoid recompiling in loops)
_WHITESPACE_PATTERN = re.compile(r"[ \t]+")

# Zero-width and control characters to strip (invisible Unicode that breaks matching)
# - \u200b-\u200f: zero-width space, non-joiner, joiner, LTR/RTL marks
# - \uFEFF: byte order mark (BOM)
# - \u00ad: soft hyphen
# - \x00-\x08, \x0b-\x0c, \x0e-\x1f: control chars (except \t\n\r)
_INVISIBLE_CHARS_PATTERN = re.compile(r"[\u200b-\u200f\uFEFF\u00ad\x00-\x08\x0b\x0c\x0e-\x1f]")
_TRAILING_PUNCT_PATTERN = re.compile(r"[.!?,;]+$")

# Request patterns - triggers that expect substantive responses
_REQUEST_PATTERNS = [
    re.compile(r"\bcan you\b"),
    re.compile(r"\bcould you\b"),
    re.compile(r"\bwould you\b"),
    re.compile(r"\bwill you\b"),
    re.compile(r"\bdo you want\b"),
    re.compile(r"\bwanna\b"),
    re.compile(r"\blet me know\b"),
    re.compile(r"\btell me\b"),
    re.compile(r"\bwhat do you think\b"),
    re.compile(r"\bthoughts\?\s*$"),
]

# Proposal patterns - triggers expecting yes/no + elaboration
_PROPOSAL_PATTERNS = [
    re.compile(r"\bhow about\b"),
    re.compile(r"\bwhat about\b"),
    re.compile(r"\bshould we\b"),
    re.compile(r"\bshall we\b"),
    re.compile(r"\blet's\b"),
    re.compile(r"\bwe could\b"),
]

# Temporal reference patterns
TEMPORAL_PATTERNS = [
    r"\b(today|tomorrow|yesterday|tonight|later|soon)\b",
    r"\b(\d{1,2}:\d{2})\s*(am|pm)?\b",
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(this|next|last)\s+(week|month|weekend)\b",
    r"\bin\s+(\d+)\s+(minutes?|hours?|days?)\b",
    r"\bat\s+(\d{1,2})\s*(am|pm|o'?clock)?\b",
]
TEMPORAL_REGEX = re.compile("|".join(TEMPORAL_PATTERNS), re.IGNORECASE)

# Spell checker singleton (lazy loaded)
_SPELL_CHECKER = None


def _get_spell_checker():
    """Get or initialize the SymSpell spell checker."""
    global _SPELL_CHECKER
    if _SPELL_CHECKER is None:
        try:
            import importlib.resources

            from symspellpy import SymSpell

            _SPELL_CHECKER = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            # Load the built-in frequency dictionary
            dict_path = importlib.resources.files("symspellpy").joinpath(
                "frequency_dictionary_en_82_765.txt"
            )
            _SPELL_CHECKER.load_dictionary(str(dict_path), term_index=0, count_index=1)
        except Exception:
            # If SymSpell fails to load, return None
            return None
    return _SPELL_CHECKER


def _spell_correct(text: str) -> str:
    """Correct spelling errors using SymSpell.

    Note: Run AFTER slang expansion to avoid breaking abbreviations.
    """
    if not text:
        return text

    checker = _get_spell_checker()
    if checker is None:
        return text

    try:
        # Use lookup_compound for multi-word correction
        suggestions = checker.lookup_compound(text, max_edit_distance=2)
        if suggestions:
            return suggestions[0].term
        return text
    except Exception:
        return text


def normalize_text(
    text: str,
    collapse_emojis: bool = True,
    strip_signatures: bool = True,
    filter_garbage: bool = False,
    filter_attributed_artifacts: bool = False,
    drop_url_only: bool = False,
    mask_entities: bool = False,
    normalize_emojis: bool = False,
    preserve_url_domain: bool = False,
    replace_codes: bool = False,
    ner_enabled: bool = False,
    ner_model: str = "en_core_web_sm",
    expand_slang: bool = False,
    spell_check: bool = False,
) -> str:
    """Canonical text cleaning used everywhere in the pipeline.

    This is the single source of truth for text normalization. All text
    should pass through this function before being stored or compared.

    Args:
        text: Raw text to normalize.
        collapse_emojis: If True, collapse repeated emojis (3+ same emoji -> 2).
        strip_signatures: If True, remove common auto-signatures.

    Returns:
        Normalized text. Returns empty string if text is a reaction or None.
    """
    if not text:
        return ""

    # 1. Check if this is a reaction (tapback) - return empty if so
    if REACTION_REGEX.match(text.strip()):
        return ""

    # 1b. Optional garbage filtering
    if filter_garbage:
        if GARBAGE_REGEX.search(text):
            return ""
        if drop_url_only and URL_ONLY_REGEX.match(text.strip()):
            return ""

    # 1c. Optional attributedBody artifact filtering
    if filter_attributed_artifacts and ATTRIBUTED_ARTIFACT_REGEX.search(text):
        return ""

    cleaned = text

    # 1c. Remove invisible/control characters (zero-width space, BOM, etc.)
    cleaned = _INVISIBLE_CHARS_PATTERN.sub("", cleaned)

    # 2. Strip auto-signatures
    if strip_signatures:
        cleaned = AUTO_SIGNATURE_REGEX.sub("", cleaned)

    # 3. Normalize whitespace (collapse multiple spaces, but preserve newlines)
    lines = cleaned.split("\n")
    normalized_lines = []
    for line in lines:
        # Collapse multiple spaces/tabs to single space (use pre-compiled pattern)
        line = _WHITESPACE_PATTERN.sub(" ", line.strip())
        if line:
            normalized_lines.append(line)
    cleaned = "\n".join(normalized_lines)

    # 3b. URL handling
    if preserve_url_domain:
        cleaned = _replace_urls_with_domains(cleaned)

    # 3c. Replace codes (OTP, verification) with placeholder
    if replace_codes:
        cleaned = _replace_codes_with_placeholder(cleaned)

    # 3d. Mask entities (email, phone, handles, heuristic person/city/team)
    if mask_entities:
        cleaned = _mask_entities(cleaned, use_ner=ner_enabled, ner_model=ner_model)

    # 4. Collapse repeated emojis (3+ same emoji -> 2)
    if collapse_emojis:
        cleaned = REPEATED_EMOJI_PATTERN.sub(r"\1\1", cleaned)

    # 4b. Emoji normalization to tokens
    if normalize_emojis:
        cleaned = _normalize_emojis(cleaned)

    # 4c. Slang expansion for better embedding alignment
    if expand_slang:
        from jarvis.nlp.slang import expand_slang as _expand_slang

        cleaned = _expand_slang(cleaned)

    # 4d. Spelling correction (AFTER slang expansion to avoid breaking slang)
    if spell_check:
        cleaned = _spell_correct(cleaned)

    # 5. Final strip
    cleaned = cleaned.strip()

    return cleaned


def is_garbage_message(text: str, drop_url_only: bool = False) -> bool:
    """Check if text matches known garbage patterns."""
    if not text:
        return False
    if GARBAGE_REGEX.search(text):
        return True
    return bool(drop_url_only and URL_ONLY_REGEX.match(text.strip()))


def detect_language(text: str) -> str:
    """Detect the language of text.

    Uses langdetect library for language detection. Falls back to "en" if
    detection fails (e.g., text too short or ambiguous).

    Args:
        text: Text to detect language for.

    Returns:
        ISO 639-1 language code (e.g., "en", "es", "fr") or "en" on failure.
    """
    if not text or len(text.strip()) < 3:
        return "en"  # Too short to detect

    try:
        from langdetect import detect

        return detect(text)
    except Exception:
        return "en"  # Default to English on any error


def is_english(text: str, threshold: float = 0.8) -> bool:
    """Check if text is primarily English.

    Uses langdetect library with probability threshold to determine if
    text is English. Short texts (< 10 chars) are assumed to be English
    since langdetect is unreliable on short strings.

    Args:
        text: Text to check.
        threshold: Minimum probability for English (default 0.8).

    Returns:
        True if text is primarily English, False otherwise.
    """
    if not text:
        return True

    # Short texts are often slang/abbreviations - assume English
    stripped = text.strip()
    if len(stripped) < 10:
        return True

    try:
        from langdetect import detect_langs

        probs = detect_langs(stripped)
        for lang_prob in probs:
            if lang_prob.lang == "en" and lang_prob.prob >= threshold:
                return True
        # If no English detected with high probability, check if it's the top language
        if probs and probs[0].lang == "en":
            return True
        return False
    except Exception:
        return True  # Assume English if detection fails


def is_spam_message(text: str) -> bool:
    """Check if text matches known spam/bot message patterns.

    Detects automated messages from:
    - Shipping/order notifications
    - Appointment reminders
    - Account verification
    - Marketing/promotional content
    - Financial alerts
    - Ride/food delivery apps

    Args:
        text: Message text to check.

    Returns:
        True if the message appears to be spam/automated, False otherwise.
    """
    if not text:
        return False
    return bool(SPAM_REGEX.search(text))


def _replace_urls_with_domains(text: str) -> str:
    """Replace URLs with <URL:domain> tokens."""
    if not text:
        return text

    def _repl(match: re.Match) -> str:
        url = match.group(0)
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path.split("/")[0]
            if domain:
                return f"<URL:{domain.lower()}>"
        except Exception:
            pass
        return "<URL>"

    return URL_REGEX.sub(_repl, text)


def _replace_codes_with_placeholder(text: str) -> str:
    """Replace likely OTP/verification codes with <CODE> when context is present."""
    if not text:
        return text

    if CODE_CONTEXT_REGEX.search(text):
        return CODE_REGEX.sub("<CODE>", text)
    return text


def _mask_entities(text: str, use_ner: bool = False, ner_model: str = "en_core_web_trf") -> str:
    """Mask common entity types with placeholders (NER server + heuristics fallback).

    Uses the spaCy NER server (via Unix socket) for accurate entity detection,
    falling back to regex-based heuristics if the server is unavailable.

    Note: For best NER results, expand slang BEFORE calling this function.
    The normalize_text() function handles this ordering correctly.
    """
    if not text:
        return text

    masked = text

    if use_ner:
        try:
            from jarvis.nlp.ner_client import get_entities, is_service_running

            if is_service_running():
                entities = get_entities(masked)
                # Replace entities from back to front to keep offsets valid
                for ent in sorted(entities, key=lambda e: e.start, reverse=True):
                    label = ent.label
                    token = None
                    if label == "PERSON":
                        token = "<PERSON>"
                    elif label in ("GPE", "LOC"):
                        token = "<CITY>"
                    elif label == "ORG":
                        token = "<ORG>"
                    elif label == "TIME":
                        token = "<TIME>"
                    elif label == "DATE":
                        token = "<DATE>"
                    if token:
                        masked = masked[: ent.start] + token + masked[ent.end :]
        except Exception:
            # Fall back to regex masking if NER unavailable
            pass

    # Regex-based masking (always apply for things NER might miss)
    masked = EMAIL_REGEX.sub("<EMAIL>", masked)
    masked = PHONE_REGEX.sub("<PHONE>", masked)
    masked = HANDLE_REGEX.sub("<PERSON>", masked)

    # Heuristic context-based masking (only if NER didn't run or missed things)
    if not use_ner:
        masked = CITY_CONTEXT_REGEX.sub(lambda m: f"{m.group(1)} <CITY>", masked)
        masked = TEAM_CONTEXT_REGEX.sub(lambda m: f"{m.group(1)} <TEAM>", masked)
        masked = PERSON_CONTEXT_REGEX.sub(lambda m: f"{m.group(1)} <PERSON>", masked)

    return masked


def _extract_entities_from_service(text: str) -> list["Entity"]:
    if not text:
        return []

    try:
        from jarvis.nlp.ner_client import get_entities, is_service_running

        if is_service_running():
            return get_entities(text)
    except Exception:
        pass

    return []


@dataclass
class NormalizationResult:
    text: str
    entities: list["Entity"]


def normalize_for_task_with_entities(text: str, task: str) -> NormalizationResult:
    normalized = normalize_for_task(text, task)
    if not normalized:
        return NormalizationResult(text="", entities=[])

    entities = _extract_entities_from_service(normalized)
    return NormalizationResult(text=normalized, entities=entities)


def extract_entities(text: str) -> list["Entity"]:
    if not text:
        return []
    return _extract_entities_from_service(text)


def normalize_for_task(text: str, task: str) -> str:
    """Normalize text using the configured profile for a task."""
    from jarvis.config import get_config

    cfg = get_config().normalization
    profile = getattr(cfg, task, cfg.classification)

    # Filter non-English text if enabled (before normalization to save work)
    if profile.filter_non_english and not is_english(text):
        return ""

    cleaned = normalize_text(
        text,
        filter_garbage=profile.filter_garbage,
        filter_attributed_artifacts=profile.filter_attributed_artifacts,
        drop_url_only=profile.drop_url_only,
        mask_entities=profile.mask_entities,
        normalize_emojis=profile.normalize_emojis,
        preserve_url_domain=profile.preserve_url_domain,
        replace_codes=profile.replace_codes,
        ner_enabled=profile.ner_enabled,
        ner_model=profile.ner_model,
        expand_slang=profile.expand_slang,
        spell_check=profile.spell_check,
    )

    if not cleaned:
        return ""

    if len(cleaned) < profile.min_length or len(cleaned) > profile.max_length:
        return ""

    return cleaned


def _normalize_emojis(text: str) -> str:
    """Replace common emojis with semantic tokens."""
    if not text:
        return text
    normalized = text
    for emoji, token in EMOJI_TOKEN_MAP.items():
        normalized = normalized.replace(emoji, token)
    return normalized


def is_reaction(text: str) -> bool:
    """Check if text is an iMessage reaction/tapback."""
    if not text:
        return False
    return bool(REACTION_REGEX.match(text.strip()))


def is_acknowledgment_only(text: str) -> bool:
    """Check if text is only a generic acknowledgment phrase.

    These are context-dependent and not useful as standalone responses.
    """
    if not text:
        return False
    normalized = text.lower().strip()
    # Also check for just punctuation added (use pre-compiled pattern)
    stripped = _TRAILING_PUNCT_PATTERN.sub("", normalized)
    return stripped in ACKNOWLEDGMENT_PHRASES or normalized in ACKNOWLEDGMENT_PHRASES


def is_emoji_only(text: str) -> bool:
    """Check if text contains only emojis and whitespace."""
    if not text:
        return False
    return bool(EMOJI_PATTERN.match(text.strip()))


def starts_new_topic(text: str) -> bool:
    """Check if text starts with a topic-shift marker."""
    if not text:
        return False
    normalized = text.lower().strip()
    for marker in TOPIC_SHIFT_MARKERS:
        if normalized.startswith(marker):
            return True
    return False


def is_question(text: str) -> bool:
    """Check if text is a question.

    Uses both punctuation and question word detection.
    """
    if not text:
        return False
    stripped = text.strip()
    # Ends with question mark
    if stripped.endswith("?"):
        return True
    # Starts with question word (split once and reuse)
    words = stripped.split()
    first_word = words[0].lower() if words else ""
    return first_word in QUESTION_WORDS


def extract_temporal_refs(text: str) -> list[str]:
    """Extract temporal references from text."""
    if not text:
        return []
    matches = TEMPORAL_REGEX.findall(text)
    # Flatten matches (some patterns have groups)
    refs = []
    for match in matches:
        if isinstance(match, tuple):
            refs.extend([m for m in match if m])
        elif match:
            refs.append(match)
    return refs


def get_attachment_token(attachment_type: str | None) -> str:
    """Get a token representation for an attachment type.

    Args:
        attachment_type: MIME type or simple type like "image", "video", etc.

    Returns:
        Token like "<ATTACHMENT:image>" or "<ATTACHMENT:file>".
    """
    if not attachment_type:
        return "<ATTACHMENT:file>"

    attachment_type = attachment_type.lower()
    if "image" in attachment_type or attachment_type in ("jpg", "jpeg", "png", "gif", "heic"):
        return "<ATTACHMENT:image>"
    elif "video" in attachment_type or attachment_type in ("mov", "mp4", "m4v"):
        return "<ATTACHMENT:video>"
    elif "audio" in attachment_type or attachment_type in ("mp3", "m4a", "aac", "wav"):
        return "<ATTACHMENT:audio>"
    elif "pdf" in attachment_type:
        return "<ATTACHMENT:pdf>"
    else:
        return "<ATTACHMENT:file>"


@dataclass
class TextFeatures:
    """Features extracted from text for gating decisions."""

    is_reaction: bool = False
    is_emoji_only: bool = False
    is_acknowledgment: bool = False
    has_attachment_token: bool = False
    word_count: int = 0
    char_count: int = 0
    starts_new_topic: bool = False
    is_question: bool = False
    temporal_refs: list[str] | None = None
    is_short: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_reaction": self.is_reaction,
            "is_emoji_only": self.is_emoji_only,
            "is_acknowledgment": self.is_acknowledgment,
            "has_attachment_token": self.has_attachment_token,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "starts_new_topic": self.starts_new_topic,
            "is_question": self.is_question,
            "temporal_refs": self.temporal_refs or [],
            "is_short": self.is_short,
        }


def extract_text_features(text: str) -> TextFeatures:
    """Extract features from text for gating decisions.

    Args:
        text: Text to analyze (should be normalized first).

    Returns:
        TextFeatures dataclass with extracted features.
    """
    if not text:
        return TextFeatures()

    return TextFeatures(
        is_reaction=is_reaction(text),
        is_emoji_only=is_emoji_only(text),
        is_acknowledgment=is_acknowledgment_only(text),
        has_attachment_token="<ATTACHMENT:" in text,
        word_count=len(text.split()),
        char_count=len(text),
        starts_new_topic=starts_new_topic(text),
        is_question=is_question(text),
        temporal_refs=extract_temporal_refs(text),
        is_short=len(text.split()) <= 3,
    )


def trigger_expects_content(text: str) -> bool:
    """Check if a trigger expects a content-rich response.

    Triggers that are questions, requests, or proposals expect more than
    just an acknowledgment in response.

    Args:
        text: Trigger text to analyze.

    Returns:
        True if trigger expects a substantive response.
    """
    if not text:
        return False

    normalized = text.lower().strip()

    # Questions expect content
    if is_question(text):
        return True

    # Request patterns (use pre-compiled patterns)
    for pattern in _REQUEST_PATTERNS:
        if pattern.search(normalized):
            return True

    # Proposal patterns (use pre-compiled patterns)
    for pattern in _PROPOSAL_PATTERNS:
        if pattern.search(normalized):
            return True

    return False
