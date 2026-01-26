"""Smart reply suggestions API endpoint.

Provides quick reply suggestions based on the last received message.
Uses simple pattern matching for common response scenarios.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/suggestions", tags=["suggestions"])


class SuggestionRequest(BaseModel):
    """Request for smart reply suggestions."""

    last_message: str = Field(..., min_length=1, description="The last received message")
    num_suggestions: int = Field(default=3, ge=1, le=5, description="Number of suggestions")


class Suggestion(BaseModel):
    """A single reply suggestion."""

    text: str
    score: float


class SuggestionResponse(BaseModel):
    """Response containing reply suggestions."""

    suggestions: list[Suggestion]


# Pre-defined response patterns
# Each tuple: (keywords_to_match, response_text, base_score)
RESPONSE_PATTERNS: list[tuple[list[str], str, float]] = [
    # Questions about time/scheduling
    (["what time", "when", "what's a good time"], "What time works for you?", 0.9),
    (["are you free", "you free", "available"], "Let me check my schedule!", 0.9),
    (["can we meet", "let's meet", "want to meet"], "Sure! When works for you?", 0.85),
    # Affirmative responses
    (["sounds good", "that works", "perfect"], "Sounds good!", 0.95),
    (["ok", "okay", "k", "kk"], "Got it!", 0.9),
    (["yes", "yeah", "yep", "yup"], "Great!", 0.85),
    (["sure", "of course"], "Sure thing!", 0.85),
    # Location questions
    (["where", "location", "address", "place"], "Where should we meet?", 0.85),
    # Gratitude
    (["thank", "thanks", "ty", "thx"], "You're welcome!", 0.95),
    (["appreciate"], "Happy to help!", 0.85),
    # Running late / timing
    (["running late", "gonna be late", "be there soon"], "No worries, take your time!", 0.95),
    (["omw", "on my way", "heading"], "See you soon!", 0.9),
    # Social invitations
    (["wanna hang", "want to hang", "free tonight"], "I'm down! What's the plan?", 0.85),
    (["dinner", "lunch", "food", "eat"], "I'm in! Where were you thinking?", 0.85),
    (["coffee", "drinks", "grab a"], "Sounds great! When?", 0.85),
    # Acknowledgments
    (["got it", "understood", "makes sense"], "Perfect!", 0.9),
    (["see you", "talk later", "ttyl", "bye"], "See you!", 0.9),
    # Excitement / agreement
    (["excited", "can't wait", "looking forward"], "Me too!", 0.9),
    (["same", "me too", "agreed"], "Right?!", 0.85),
    # Generic affirmative
    (["nice", "cool", "awesome", "great", "amazing"], "Thanks!", 0.75),
    # Fallback suggestions (lower scores, always available)
    ([], "Sounds good!", 0.3),
    ([], "Got it!", 0.25),
    ([], "Thanks!", 0.2),
]


def _compute_match_score(message: str, keywords: list[str], base_score: float) -> float:
    """Compute how well a message matches the keywords.

    Args:
        message: The incoming message (lowercase)
        keywords: Keywords to look for
        base_score: Base score if any keyword matches

    Returns:
        Score from 0 to base_score based on match quality
    """
    if not keywords:
        # Fallback patterns return their base score directly
        return base_score

    message_lower = message.lower()

    # Check for exact phrase matches first (highest score)
    for keyword in keywords:
        if keyword in message_lower:
            return base_score

    # Check for word-level matches
    message_words = set(re.findall(r"\b\w+\b", message_lower))
    for keyword in keywords:
        keyword_words = set(re.findall(r"\b\w+\b", keyword.lower()))
        if keyword_words & message_words:
            return base_score * 0.7  # Partial match

    return 0.0


@router.post("", response_model=SuggestionResponse)
def get_suggestions(request: SuggestionRequest) -> SuggestionResponse:
    """Get smart reply suggestions based on the last message.

    Returns contextually appropriate quick replies ranked by relevance.
    """
    message = request.last_message.strip()
    if not message:
        return SuggestionResponse(suggestions=[])

    # Score all patterns
    scored: list[tuple[str, float]] = []
    seen_responses: set[str] = set()

    for keywords, response, base_score in RESPONSE_PATTERNS:
        if response in seen_responses:
            continue

        score = _compute_match_score(message, keywords, base_score)
        if score > 0:
            scored.append((response, score))
            seen_responses.add(response)

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Take top N
    suggestions = [
        Suggestion(text=text, score=score) for text, score in scored[: request.num_suggestions]
    ]

    logger.debug("Generated %d suggestions for message", len(suggestions))
    return SuggestionResponse(suggestions=suggestions)
