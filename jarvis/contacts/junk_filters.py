"""Shared junk/spam/bot message filters for fact extraction pipelines.

Used by fact extractors to avoid duplicating filtering logic.
"""

from __future__ import annotations

import re

# Pre-compiled bot detection patterns (avoid re-compiling on every call)
_BOT_HIGH_CONF = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"CVS Pharmacy",
        r"Rx Ready",
        r"Check out this job at",
        r"this is .+ from .+ (?:team|hiring)",
        r"recruiter from",
        r"currently hiring",
        r"we have reviewed your profile",
        r"perfect (?:position|fit)",
        r"exclusive offer",
        r"valued .+ customer",
        r"mail system at host",
        r"review and sign",
        r"invited to participate",
        r"ready to schedule your .+ visit at",
        r"your .+ appointment is",
        r"my name.+with .+\.\s*we have .+ open",
        r"just matched you with",
        r"your package has arrived",
        r"your order has been",
        r"decided not to move forward with your .+ application",
        r"our team is reviewing your matter",
    ]
]
_BOT_SHORT_CODE_RE = re.compile(r"SMS;-;\d{5,6}$")
_BOT_BARE_SHORT_CODE_RE = re.compile(r"\d{5,6}$")
_BOT_URL_RE = re.compile(r"https?://")
_BOT_JOB_RE = re.compile(r"\bjob\b", re.IGNORECASE)
_BOT_COMPANY_RE = re.compile(r"\b[A-Z][a-z]+\s+[A-Z]")
_BOT_APPLY_RE = re.compile(r"\bapply\b", re.IGNORECASE)
_BOT_NOW_RE = re.compile(r"\bnow\b", re.IGNORECASE)


def is_bot_message(text: str, chat_id: str = "") -> bool:
    """Detect high-confidence bot messages (spam, automated replies).

    High-confidence indicators (any 1 match = reject):
    - CVS Pharmacy, Rx Ready (pharmacy bots)
    - "Check out this job at" (LinkedIn spam)
    - Sender is 5-6 digit short code (SMS;-;898287)

    Medium-confidence (any 3 matches = reject):
    - URL + "job" + capitalized company
    - "apply" + "now"
    - >50% all-caps text
    """
    for pat in _BOT_HIGH_CONF:
        if pat.search(text):
            return True

    # 5-6 digit short codes (SMS bot senders)
    if chat_id and _BOT_SHORT_CODE_RE.match(chat_id):
        return True
    # Bare 5-6 digit chat IDs (no SMS prefix)
    if chat_id and _BOT_BARE_SHORT_CODE_RE.match(chat_id):
        return True

    medium_factors = 0

    if _BOT_URL_RE.search(text) and _BOT_JOB_RE.search(text):
        if _BOT_COMPANY_RE.search(text):
            medium_factors += 1

    if _BOT_APPLY_RE.search(text) and _BOT_NOW_RE.search(text):
        medium_factors += 1

    letters = [c for c in text if c.isalpha()]
    if letters:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if caps_ratio > 0.5:
            medium_factors += 1

    return medium_factors >= 3


_PROF_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bdear\s",
        r"hello from",
        r"reminder:\s",
        r"\bregards\b",
        r"\bsincerely\b",
        r"\bbest regards\b",
        r"i appreciate",
        r"opportunity",
        r"from\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|LLC|Ltd|Corp|Hyundai|Recruiting|Marketing|Healthcare|Hospital)",
    ]
]


def is_professional_message(text: str) -> bool:
    """Detect professional/business emails that should not be processed.

    Markers:
    - Formal greetings: "Dear", "Hello from"
    - Professional signoffs: "Regards", "Sincerely", "Best regards"
    - Business patterns: "Reminder:", recruiting language
    """
    for pat in _PROF_PATTERNS:
        if pat.search(text):
            return True
    return False


_TAPBACK_RE = re.compile(
    r"^(Loved|Liked|Disliked|Laughed at|Emphasized|Questioned|Reacted with a sticker to)"
    r"\s+(?:\u201c|an attachment\b)",
)

# Code snippet markers
_CODE_MARKERS = re.compile(
    r"(?:^|\n)\s*(?:def |class |import |from \w+ import |return |#####|```|\"\"\")",
)


_GREETING_ONLY_RE = re.compile(
    r"^(?:hey|hi|hello|yo|sup|what'?s? ?up|haha|lol|lmao|bet|nice|dope|"
    r"word|facts|true|yep|yea|yeah|ya|nah|nope|ok|okay|k|kk|"
    r"good morning|good night|gm|gn|"
    r"omg|omfg|smh|bruh|bro|dude|"
    r"same|mood|fr|ong|no cap|w|l|"
    r"(?:ha)+|(?:lo)+l|(?:lmao)+|"
    r"[\U0001f600-\U0001f64f\U0001f680-\U0001f6ff\U0001f900-\U0001f9ff"
    r"\u2600-\u26ff\u2700-\u27bf\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff]+)"
    r"[.!?\s]*$",
    re.IGNORECASE,
)


def is_greeting_only(text: str) -> bool:
    """Detect bare greetings, reactions, and filler messages with no factual content."""
    return bool(_GREETING_ONLY_RE.match(text.strip()))


_SPORTS_RE = re.compile(
    r"(?:"
    r"\b(?:won|lost|beat|defeated)\s+\d+\s*[-–]\s*\d+\b|"
    r"\b(?:touchdown|field goal|halftime|overtime|interception|fumble)\b|"
    r"\b(?:fantasy (?:football|basketball|baseball|hockey|league))\b|"
    r"\bfinal score\b|"
    r"\b(?:playoffs?|championship|super bowl|world series|march madness)\b|"
    r"\b(?:first|second|third|fourth)\s+quarter\b|"
    r"\b\d+\s*[-–]\s*\d+\s+(?:at|vs\.?)\s+(?:half|the\s+half|end)\b"
    r")",
    re.IGNORECASE,
)


def is_sports_commentary(text: str) -> bool:
    """Detect sports scores, game commentary, and fantasy sports messages."""
    return bool(_SPORTS_RE.search(text))


_SPAM_RE = re.compile(
    r"(?:"
    r"\b(?:insurance|premium|deductible|copay|coverage plan)\b.*\b(?:quote|rate|enroll)\b|"
    r"\b(?:free consultation|limited time offer|act now|don'?t miss)\b|"
    r"\b(?:unsubscribe|opt[- ]?out|reply\s+stop)\b|"
    r"\b(?:pre[- ]?approved|pre[- ]?qualified)\b.*\b(?:loan|credit|mortgage)\b|"
    r"\b(?:verify your (?:account|identity|email))\b|"
    r"\b(?:claim your|you'?ve? (?:been selected|won))\b"
    r")",
    re.IGNORECASE,
)

_GROUP_SYSTEM_RE = re.compile(
    r"(?:"
    r"\b(?:added|removed|left)\s+.+\s+(?:to|from)\s+the\s+group\b|"
    r"\bchanged\s+the\s+group\s+name\b|"
    r"\bstarted\s+(?:a|an)\s+group\s+call\b"
    r")",
    re.IGNORECASE,
)


def is_spam_message(text: str) -> bool:
    """Detect insurance, legal, recruitment spam, and marketing messages."""
    return bool(_SPAM_RE.search(text) or _GROUP_SYSTEM_RE.search(text))


def is_code_message(text: str) -> bool:
    """Detect code snippets sent via iMessage (homework, debugging, etc.).

    Markers:
    - Python keywords at line start: def, class, import, return
    - Markdown fences: triple backticks or triple quotes
    - High special-character density (>30% non-alphanumeric, non-space)
    """
    if _CODE_MARKERS.search(text):
        return True

    # High special-character density
    if len(text) >= 20:
        alnum_space = sum(1 for c in text if c.isalnum() or c.isspace())
        special_ratio = 1 - (alnum_space / len(text))
        if special_ratio > 0.3:
            return True

    return False


def is_tapback_reaction(text: str) -> bool:
    """Detect iMessage tapback reaction messages (e.g. 'Loved "some text"')."""
    return bool(_TAPBACK_RE.match(text))


def is_junk_message(text: str, chat_id: str = "", min_length: int = 5) -> bool:
    """Combined junk check: too short, bot, professional, or tapback reaction.

    Single entry point for all pre-extraction filtering.
    """
    if not text or len(text) < min_length:
        return True
    if is_tapback_reaction(text):
        return True
    if is_greeting_only(text):
        return True
    if is_code_message(text):
        return True
    if is_bot_message(text, chat_id):
        return True
    if is_professional_message(text):
        return True
    if is_sports_commentary(text):
        return True
    if is_spam_message(text):
        return True
    return False
