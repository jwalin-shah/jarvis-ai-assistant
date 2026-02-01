"""Reply Suggester - Simple, fast reply suggestions.

Uses a 3-tier approach:
1. Structural pattern detection (what type of message is this?)
2. FAISS retrieval (how do YOU respond to similar messages?)
3. Template fallback (common responses for this pattern type)

No complex classification. Just patterns + your history.

Usage:
    from jarvis.reply_suggester import get_reply_suggestions

    suggestions = get_reply_suggestions(
        message="Want to grab lunch?",
        contact_id=1,
    )
    # Returns: ["Yeah I'm down!", "Can't today", "Where at?"]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.db import JarvisDB

logger = logging.getLogger(__name__)


# =============================================================================
# Message Patterns (Simple structural detection)
# =============================================================================


class MessagePattern(str, Enum):
    """Detected message pattern types."""

    INVITATION = "invitation"  # "want to..?", "can you come?"
    YN_QUESTION = "yn_question"  # yes/no questions
    INFO_QUESTION = "info_question"  # what/when/where/who
    REACTION_PROMPT = "reaction_prompt"  # "omg did you see..?"
    VENTING = "venting"  # emotional expression
    STATEMENT = "statement"  # sharing info
    GREETING = "greeting"  # hi, hey
    ACKNOWLEDGMENT = "acknowledgment"  # ok, thanks
    UNKNOWN = "unknown"


# Pattern definitions: (regex, pattern_type)
PATTERNS: list[tuple[re.Pattern, MessagePattern]] = [
    # Invitations - asking someone to do something (with OR without ?)
    (re.compile(r"\b(want to|wanna|down to|dtf|tryna)\b", re.I), MessagePattern.INVITATION),
    (re.compile(r"^(lets|let's|lemme|lmk if you wanna)\b", re.I), MessagePattern.INVITATION),
    (re.compile(r"\b(can you|could you|would you|will you)\s+(come|make it|join|hang)\b", re.I), MessagePattern.INVITATION),
    (re.compile(r"\b(free|available|busy)\s*(today|tonight|tomorrow|tmrw|this weekend)?\s*\??", re.I), MessagePattern.INVITATION),

    # Reaction prompts - exciting news/events wanting reaction
    (re.compile(r"^(omg|oh my god|dude|bro|yo)\b.*(wtf|crazy|insane|\?)", re.I), MessagePattern.REACTION_PROMPT),
    (re.compile(r"\b(did you see|have you seen|did you hear)\b", re.I), MessagePattern.REACTION_PROMPT),
    (re.compile(r"\b(can you believe|isn't that|wasn't that)\b", re.I), MessagePattern.REACTION_PROMPT),
    (re.compile(r"\bwtf\b.*\b(happened|is going on|was that)\b", re.I), MessagePattern.REACTION_PROMPT),

    # Venting - emotional expressions
    (re.compile(r"^(fuck+|shit+|damn+|ugh+|bruh+)\b", re.I), MessagePattern.VENTING),
    (re.compile(r"\b(so (frustrated|annoyed|tired|stressed|upset))\b", re.I), MessagePattern.VENTING),
    (re.compile(r"\b(i('m| am) (done|over it|sick of))\b", re.I), MessagePattern.VENTING),

    # Info questions - what/when/where/who/how (with OR without ?)
    (re.compile(r"^(what|when|where|who|which|how)\b", re.I), MessagePattern.INFO_QUESTION),
    (re.compile(r"\b(what time|what day|how long|how much)\b", re.I), MessagePattern.INFO_QUESTION),
    (re.compile(r"\b(when'?s|where'?s|what'?s)\s+the\b", re.I), MessagePattern.INFO_QUESTION),

    # Yes/No questions
    (re.compile(r"^(do|does|did|is|are|was|were|can|could|will|would|have|has)\s+(you|u|we|they|i|it)\b", re.I), MessagePattern.YN_QUESTION),
    (re.compile(r"\?\s*$", re.I), MessagePattern.YN_QUESTION),  # Fallback: ends with ?

    # Greetings
    (re.compile(r"^(hey|hi|hello|yo|sup|what'?s up|wassup|hiya)\s*[!?]?\s*$", re.I), MessagePattern.GREETING),

    # Acknowledgments
    (re.compile(r"^(ok|okay|k|kk|sure|bet|got it|sounds good|cool|alright|thanks|ty|thx)\s*[!.]?\s*$", re.I), MessagePattern.ACKNOWLEDGMENT),

    # Statements (no question mark, declarative) - MUST BE LAST
    (re.compile(r"^[^?]+[.!]?\s*$", re.I), MessagePattern.STATEMENT),
]


def detect_pattern(message: str) -> MessagePattern:
    """Detect the message pattern type using structural rules.

    Args:
        message: The incoming message text.

    Returns:
        MessagePattern enum indicating the type.
    """
    message = message.strip()
    if not message:
        return MessagePattern.UNKNOWN

    for pattern, pattern_type in PATTERNS:
        if pattern.search(message):
            return pattern_type

    return MessagePattern.UNKNOWN


# =============================================================================
# Response Templates (Fallback options per pattern)
# =============================================================================


TEMPLATES: dict[MessagePattern, list[str]] = {
    MessagePattern.INVITATION: [
        "Yeah I'm down!",
        "Can't today, sorry",
        "Let me check and get back to you",
        "Where/when?",
    ],
    MessagePattern.YN_QUESTION: [
        "Yeah",
        "Nah",
        "Maybe, let me think",
    ],
    MessagePattern.INFO_QUESTION: [
        "Let me check",
        "Not sure, I'll find out",
        "Good question, idk",
    ],
    MessagePattern.REACTION_PROMPT: [
        "No what happened??",
        "Yooo that's crazy",
        "Wait tell me more",
        "Lmao no way",
    ],
    MessagePattern.VENTING: [
        "Damn what happened?",
        "You good?",
        "That sucks, I'm sorry",
        "Vent to me",
    ],
    MessagePattern.STATEMENT: [
        "Nice",
        "Got it",
        "Oh cool",
        "Word",
    ],
    MessagePattern.GREETING: [
        "Hey!",
        "What's up",
        "Yo",
    ],
    MessagePattern.ACKNOWLEDGMENT: [
        "👍",
        "Sounds good",
        "Cool",
    ],
    MessagePattern.UNKNOWN: [
        "Got it",
        "Interesting",
        "Tell me more",
    ],
}


# =============================================================================
# Reply Suggestion Result
# =============================================================================


@dataclass
class ReplySuggestion:
    """A single reply suggestion."""

    text: str
    source: str  # 'retrieval', 'template', 'generated'
    confidence: float

    def __str__(self) -> str:
        return self.text


@dataclass
class SuggestionResult:
    """Result from get_reply_suggestions."""

    suggestions: list[ReplySuggestion]
    pattern: MessagePattern
    similar_count: int  # How many similar messages found

    @property
    def texts(self) -> list[str]:
        """Get just the suggestion texts."""
        return [s.text for s in self.suggestions]


# =============================================================================
# Main Suggester
# =============================================================================


class ReplySuggester:
    """Generates reply suggestions using patterns + retrieval + templates.

    Strategy:
    1. Detect pattern type (structural rules)
    2. Search FAISS for similar triggers you've received
    3. Get YOUR past responses to similar messages
    4. Cluster/dedupe to get diverse options
    5. Fill remaining slots with templates
    """

    def __init__(self, db: JarvisDB | None = None):
        """Initialize the suggester.

        Args:
            db: Database instance. Uses default if None.
        """
        self._db = db
        self._index_searcher = None

    @property
    def db(self) -> JarvisDB:
        """Get or create database instance."""
        if self._db is None:
            from jarvis.db import get_db
            self._db = get_db()
        return self._db

    @property
    def index_searcher(self):
        """Get or create FAISS index searcher."""
        if self._index_searcher is None:
            from jarvis.index import TriggerIndexSearcher
            self._index_searcher = TriggerIndexSearcher(self.db)
        return self._index_searcher

    def suggest(
        self,
        message: str,
        contact_id: int | None = None,
        n_suggestions: int = 3,
        include_templates: bool = True,
    ) -> SuggestionResult:
        """Get reply suggestions for a message.

        Args:
            message: The incoming message to reply to.
            contact_id: Optional contact ID for personalization.
            n_suggestions: Number of suggestions to return.
            include_templates: Whether to include template fallbacks.

        Returns:
            SuggestionResult with suggestions and metadata.
        """
        # Step 1: Detect pattern
        pattern = detect_pattern(message)
        logger.debug("Detected pattern: %s for message: %s", pattern.value, message[:50])

        # Step 2: Search for similar triggers
        suggestions: list[ReplySuggestion] = []
        similar_count = 0

        try:
            results = self.index_searcher.search_with_pairs(
                query=message,
                k=10,
                threshold=0.5,
                prefer_recent=True,
            )
            similar_count = len(results)

            # Step 3: Get unique responses from similar triggers
            seen_responses: set[str] = set()
            for result in results:
                response = result.get("response_text", "")
                response_lower = response.lower().strip()

                # Skip if duplicate or too similar
                if response_lower in seen_responses:
                    continue
                if len(response) < 2 or len(response) > 200:
                    continue

                seen_responses.add(response_lower)
                suggestions.append(ReplySuggestion(
                    text=response,
                    source="retrieval",
                    confidence=result.get("similarity", 0.5),
                ))

                if len(suggestions) >= n_suggestions:
                    break

        except Exception as e:
            logger.warning("FAISS search failed: %s", e)

        # Step 4: Fill with templates if needed
        if include_templates and len(suggestions) < n_suggestions:
            templates = TEMPLATES.get(pattern, TEMPLATES[MessagePattern.UNKNOWN])
            for template in templates:
                if len(suggestions) >= n_suggestions:
                    break
                # Don't add template if similar response already exists
                if any(template.lower() == s.text.lower() for s in suggestions):
                    continue
                suggestions.append(ReplySuggestion(
                    text=template,
                    source="template",
                    confidence=0.7,
                ))

        return SuggestionResult(
            suggestions=suggestions[:n_suggestions],
            pattern=pattern,
            similar_count=similar_count,
        )


# =============================================================================
# Singleton Access
# =============================================================================


_suggester: ReplySuggester | None = None


def get_reply_suggester() -> ReplySuggester:
    """Get the singleton ReplySuggester instance."""
    global _suggester
    if _suggester is None:
        _suggester = ReplySuggester()
    return _suggester


def get_reply_suggestions(
    message: str,
    contact_id: int | None = None,
    n_suggestions: int = 3,
) -> list[str]:
    """Convenience function to get reply suggestion texts.

    Args:
        message: The incoming message.
        contact_id: Optional contact ID.
        n_suggestions: Number of suggestions.

    Returns:
        List of reply suggestion strings.
    """
    result = get_reply_suggester().suggest(message, contact_id, n_suggestions)
    return result.texts


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MessagePattern",
    "detect_pattern",
    "ReplySuggestion",
    "SuggestionResult",
    "ReplySuggester",
    "get_reply_suggester",
    "get_reply_suggestions",
]
