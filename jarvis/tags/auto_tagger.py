"""Auto-tagger for ML-based tag suggestions.

Provides intelligent tag suggestions based on:
- Content analysis (keywords, topics)
- User tagging history/behavior
- Contact-based defaults
- Sentiment analysis
- Time-based patterns (urgent, follow-up)

Usage:
    from jarvis.tags.auto_tagger import AutoTagger

    auto_tagger = AutoTagger(tag_manager)
    suggestions = auto_tagger.suggest_tags(chat_id, messages)
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from jarvis.tags.models import (
    AutoTagTrigger,
    RuleCondition,
    TagRule,
    TagSuggestion,
)

if TYPE_CHECKING:
    from jarvis.tags.manager import TagManager

logger = logging.getLogger(__name__)


# Keyword patterns for common tag categories
KEYWORD_PATTERNS: dict[str, list[str]] = {
    "Work": [
        r"\b(meeting|deadline|project|client|boss|office|work|presentation|report)\b",
        r"\b(schedule|calendar|monday|tuesday|wednesday|thursday|friday)\b",
        r"\b(email|memo|conference|sprint|standup|retro)\b",
    ],
    "Urgent": [
        r"\b(urgent|asap|emergency|immediately|critical|important)\b",
        r"\b(now|today|right away|as soon as possible)\b",
        r"!!!",
    ],
    "Follow Up": [
        r"\b(follow up|remind me|don't forget|remember to|later)\b",
        r"\b(get back|check in|touch base)\b",
        r"\b(tomorrow|next week|soon)\b",
    ],
    "Family": [
        r"\b(mom|dad|brother|sister|family|parent|kid|child|baby)\b",
        r"\b(grandma|grandpa|aunt|uncle|cousin)\b",
        r"\b(dinner|holiday|thanksgiving|christmas|birthday)\b",
    ],
    "Personal": [
        r"\b(friend|buddy|pal|dude|bro)\b",
        r"\b(hangout|party|weekend|fun|movie|game)\b",
        r"\b(drinks|dinner|lunch|coffee)\b",
    ],
}

# Sentiment indicators
POSITIVE_PATTERNS = [
    r"\b(thanks|thank you|appreciate|grateful|awesome|great|love|amazing)\b",
    r"\b(happy|excited|wonderful|fantastic|perfect|excellent)\b",
    r"[!]+\s*$",
    r"[:;]-?[)D]",  # Smileys
]

NEGATIVE_PATTERNS = [
    r"\b(sorry|apologize|unfortunately|problem|issue|wrong|bad)\b",
    r"\b(angry|upset|frustrated|disappointed|worried|concerned)\b",
    r"\b(can't|won't|unable|failed|error)\b",
]

NEEDS_ATTENTION_PATTERNS = [
    r"\?{2,}",  # Multiple question marks
    r"\b(help|need|please|waiting|where are you)\b",
    r"\b(respond|reply|answer|call me)\b",
    r"^[A-Z\s]{10,}$",  # All caps messages
]


@dataclass
class ContentAnalysis:
    """Results of analyzing conversation content."""

    keywords: list[str]
    topics: list[str]
    sentiment: str  # positive, negative, neutral
    sentiment_score: float
    needs_attention: bool
    question_count: int
    message_count: int
    last_message_from_me: bool
    days_since_response: int | None


class AutoTagger:
    """ML-based auto-tagger for conversation tag suggestions.

    Uses pattern matching, user behavior learning, and heuristics
    to suggest relevant tags for conversations.
    """

    def __init__(self, tag_manager: TagManager) -> None:
        """Initialize auto-tagger.

        Args:
            tag_manager: TagManager instance for database access.
        """
        self.tag_manager = tag_manager
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        for category, patterns in KEYWORD_PATTERNS.items():
            self._compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        self._positive_patterns = [re.compile(p, re.IGNORECASE) for p in POSITIVE_PATTERNS]
        self._negative_patterns = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_PATTERNS]
        self._attention_patterns = [re.compile(p, re.IGNORECASE) for p in NEEDS_ATTENTION_PATTERNS]

    def suggest_tags(
        self,
        chat_id: str,
        messages: list[dict[str, Any]],
        contact_name: str | None = None,
        limit: int = 5,
    ) -> list[TagSuggestion]:
        """Suggest tags for a conversation based on content and context.

        Args:
            chat_id: Conversation identifier.
            messages: List of message dicts with 'text', 'is_from_me', 'date' fields.
            contact_name: Display name of the contact (optional).
            limit: Maximum number of suggestions.

        Returns:
            List of TagSuggestion objects sorted by confidence.
        """
        if not messages:
            return []

        suggestions: list[TagSuggestion] = []

        # Analyze content
        analysis = self._analyze_content(messages)

        # Get existing tags for this conversation
        existing_tags = self.tag_manager.get_tags_for_conversation(chat_id)
        existing_tag_names = {t.name.lower() for t, _ in existing_tags}

        # 1. Content-based suggestions (keyword matching)
        content_suggestions = self._suggest_from_content(analysis, existing_tag_names)
        suggestions.extend(content_suggestions)

        # 2. Sentiment-based suggestions
        sentiment_suggestions = self._suggest_from_sentiment(analysis, existing_tag_names)
        suggestions.extend(sentiment_suggestions)

        # 3. Time-based suggestions (needs response, follow-up)
        time_suggestions = self._suggest_from_time_patterns(analysis, existing_tag_names)
        suggestions.extend(time_suggestions)

        # 4. Contact-based suggestions (learn from user behavior)
        if contact_name:
            contact_suggestions = self._suggest_from_contact_patterns(
                contact_name, existing_tag_names
            )
            suggestions.extend(contact_suggestions)

        # 5. Learn from user's tagging history
        history_suggestions = self._suggest_from_history(chat_id, analysis, existing_tag_names)
        suggestions.extend(history_suggestions)

        # Deduplicate and sort by confidence
        seen_names: set[str] = set()
        unique_suggestions: list[TagSuggestion] = []
        for s in sorted(suggestions, key=lambda x: x.confidence, reverse=True):
            name_lower = s.tag_name.lower()
            if name_lower not in seen_names and name_lower not in existing_tag_names:
                seen_names.add(name_lower)
                unique_suggestions.append(s)

        return unique_suggestions[:limit]

    def _analyze_content(self, messages: list[dict[str, Any]]) -> ContentAnalysis:
        """Analyze message content for patterns and sentiment."""
        all_text = " ".join(m.get("text", "") or "" for m in messages)
        recent_messages = messages[-10:] if len(messages) > 10 else messages

        # Extract keywords (simple word frequency)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", all_text.lower())
        word_counts = Counter(words)
        # Filter out common words
        common_words = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
            "was", "one", "our", "out", "day", "had", "has", "his", "how", "its",
            "may", "new", "now", "old", "see", "way", "who", "boy", "did", "get",
            "let", "put", "say", "she", "too", "use", "that", "this", "with",
            "have", "from", "they", "been", "call", "come", "could", "each",
            "find", "first", "into", "just", "know", "like", "long", "look",
            "make", "many", "more", "most", "over", "such", "take", "than",
            "them", "then", "there", "these", "thing", "think", "time", "very",
            "what", "when", "which", "will", "would", "your", "about", "after",
        }
        keywords = [w for w, c in word_counts.most_common(20) if w not in common_words][:10]

        # Detect topics from keyword patterns
        topics = []
        for category, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(all_text):
                    topics.append(category)
                    break

        # Analyze sentiment
        positive_count = sum(
            1 for p in self._positive_patterns
            for m in recent_messages if p.search(m.get("text", "") or "")
        )
        negative_count = sum(
            1 for p in self._negative_patterns
            for m in recent_messages if p.search(m.get("text", "") or "")
        )

        if positive_count > negative_count + 2:
            sentiment = "positive"
            sentiment_score = min(1.0, (positive_count - negative_count) / 10)
        elif negative_count > positive_count + 2:
            sentiment = "negative"
            sentiment_score = min(1.0, (negative_count - positive_count) / 10)
        else:
            sentiment = "neutral"
            sentiment_score = 0.5

        # Check if needs attention
        needs_attention = any(
            p.search(m.get("text", "") or "")
            for p in self._attention_patterns
            for m in recent_messages[-3:]
            if not m.get("is_from_me", False)
        )

        # Count questions
        question_count = sum(
            1 for m in messages
            if "?" in (m.get("text", "") or "") and not m.get("is_from_me", False)
        )

        # Check last message direction
        last_message_from_me = messages[-1].get("is_from_me", False) if messages else False

        # Calculate days since response
        days_since_response = None
        if messages and not last_message_from_me:
            last_date = messages[-1].get("date")
            if last_date:
                if isinstance(last_date, str):
                    last_date = datetime.fromisoformat(last_date)
                days_since_response = (datetime.now(UTC) - last_date.replace(tzinfo=UTC)).days

        return ContentAnalysis(
            keywords=keywords,
            topics=list(set(topics)),
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            needs_attention=needs_attention,
            question_count=question_count,
            message_count=len(messages),
            last_message_from_me=last_message_from_me,
            days_since_response=days_since_response,
        )

    def _suggest_from_content(
        self,
        analysis: ContentAnalysis,
        existing_names: set[str],
    ) -> list[TagSuggestion]:
        """Generate suggestions from content analysis."""
        suggestions = []

        for topic in analysis.topics:
            if topic.lower() not in existing_names:
                # Check if tag exists
                tag = self.tag_manager.get_tag_by_name(topic)
                suggestions.append(
                    TagSuggestion(
                        tag_id=tag.id if tag else None,
                        tag_name=topic,
                        confidence=0.7,
                        reason=f"Content matches '{topic}' patterns",
                        source="content",
                    )
                )

        return suggestions

    def _suggest_from_sentiment(
        self,
        analysis: ContentAnalysis,
        existing_names: set[str],
    ) -> list[TagSuggestion]:
        """Generate suggestions from sentiment analysis."""
        suggestions = []

        if analysis.sentiment == "positive" and analysis.sentiment_score > 0.6:
            if "positive" not in existing_names:
                suggestions.append(
                    TagSuggestion(
                        tag_name="Positive",
                        confidence=analysis.sentiment_score * 0.8,
                        reason="Conversation has positive sentiment",
                        source="sentiment",
                    )
                )

        if analysis.sentiment == "negative" and analysis.sentiment_score > 0.6:
            if "needs attention" not in existing_names:
                suggestions.append(
                    TagSuggestion(
                        tag_name="Needs Attention",
                        confidence=analysis.sentiment_score * 0.85,
                        reason="Conversation has negative sentiment",
                        source="sentiment",
                    )
                )

        return suggestions

    def _suggest_from_time_patterns(
        self,
        analysis: ContentAnalysis,
        existing_names: set[str],
    ) -> list[TagSuggestion]:
        """Generate suggestions from time-based patterns."""
        suggestions = []

        # Suggest "Needs Response" if there are unanswered questions
        if (
            analysis.needs_attention
            or (analysis.question_count > 0 and not analysis.last_message_from_me)
        ):
            if "needs response" not in existing_names:
                suggestions.append(
                    TagSuggestion(
                        tag_name="Needs Response",
                        confidence=0.85,
                        reason="Has unanswered questions or needs attention",
                        source="time",
                    )
                )

        # Suggest "Follow Up" if no response in 2+ days
        if analysis.days_since_response is not None and analysis.days_since_response >= 2:
            if "follow up" not in existing_names:
                suggestions.append(
                    TagSuggestion(
                        tag_name="Follow Up",
                        confidence=min(0.9, 0.6 + analysis.days_since_response * 0.1),
                        reason=f"No response in {analysis.days_since_response} days",
                        source="time",
                    )
                )

        return suggestions

    def _suggest_from_contact_patterns(
        self,
        contact_name: str,
        existing_names: set[str],
    ) -> list[TagSuggestion]:
        """Suggest tags based on contact name patterns."""
        suggestions = []
        name_lower = contact_name.lower()

        # Check for family-related names
        family_indicators = ["mom", "dad", "bro", "sis", "grandma", "grandpa", "aunt", "uncle"]
        if any(ind in name_lower for ind in family_indicators):
            if "family" not in existing_names:
                suggestions.append(
                    TagSuggestion(
                        tag_name="Family",
                        confidence=0.9,
                        reason="Contact name suggests family member",
                        source="contact",
                    )
                )

        # Check for work-related names
        work_indicators = ["boss", "manager", "ceo", "dr.", "prof", "team"]
        if any(ind in name_lower for ind in work_indicators):
            if "work" not in existing_names:
                suggestions.append(
                    TagSuggestion(
                        tag_name="Work",
                        confidence=0.8,
                        reason="Contact name suggests work relationship",
                        source="contact",
                    )
                )

        return suggestions

    def _suggest_from_history(
        self,
        chat_id: str,
        analysis: ContentAnalysis,
        existing_names: set[str],
    ) -> list[TagSuggestion]:
        """Suggest tags based on user's tagging history.

        Learns from patterns in how the user tags similar conversations.
        """
        suggestions = []

        # Get frequently used tags for similar keywords
        with self.tag_manager.connection() as conn:
            # Find tags that are often used with similar content
            if analysis.keywords:
                # Look for tags used in conversations with similar keywords
                keyword_pattern = "|".join(analysis.keywords[:5])
                cursor = conn.execute(
                    """
                    SELECT t.id, t.name, COUNT(*) as count
                    FROM tags t
                    JOIN conversation_tags ct ON t.id = ct.tag_id
                    JOIN tag_usage_history h ON t.id = h.tag_id
                    WHERE h.action = 'add'
                    AND h.context_json LIKE ?
                    AND t.name NOT IN (SELECT name FROM tags WHERE id IN (
                        SELECT tag_id FROM conversation_tags WHERE chat_id = ?
                    ))
                    GROUP BY t.id
                    ORDER BY count DESC
                    LIMIT 3
                    """,
                    (f"%{keyword_pattern}%", chat_id),
                )

                for row in cursor:
                    if row["name"].lower() not in existing_names:
                        suggestions.append(
                            TagSuggestion(
                                tag_id=row["id"],
                                tag_name=row["name"],
                                confidence=min(0.7, 0.4 + row["count"] * 0.1),
                                reason="Often used with similar content",
                                source="history",
                            )
                        )

        return suggestions

    def apply_rules(
        self,
        chat_id: str,
        messages: list[dict[str, Any]],
        trigger: str = AutoTagTrigger.ON_NEW_MESSAGE.value,
    ) -> list[tuple[TagRule, list[int]]]:
        """Apply auto-tagging rules to a conversation.

        Args:
            chat_id: Conversation identifier.
            messages: List of message dicts.
            trigger: The trigger event type.

        Returns:
            List of (rule, applied_tag_ids) tuples for rules that matched.
        """
        results = []

        # Get enabled rules for this trigger
        rules = self.tag_manager.list_tag_rules(trigger=trigger, enabled_only=True)

        for rule in rules:
            if self._evaluate_rule_conditions(rule.conditions, messages, chat_id):
                # Apply tags
                applied_tags = []
                for tag_id in rule.tag_ids:
                    if self.tag_manager.add_tag_to_conversation(
                        chat_id, tag_id, added_by=f"rule:{rule.name}", confidence=0.9
                    ):
                        applied_tags.append(tag_id)

                if applied_tags:
                    self.tag_manager.record_rule_trigger(rule.id)
                    results.append((rule, applied_tags))
                    logger.info(
                        "Rule '%s' applied tags %s to conversation %s",
                        rule.name,
                        applied_tags,
                        chat_id,
                    )

        return results

    def _evaluate_rule_conditions(
        self,
        conditions: list[RuleCondition],
        messages: list[dict[str, Any]],
        chat_id: str,
    ) -> bool:
        """Evaluate if rule conditions are met."""
        if not conditions:
            return False

        all_text = " ".join(m.get("text", "") or "" for m in messages).lower()

        for condition in conditions:
            field = condition.field
            operator = condition.operator
            value = condition.value

            if field == "last_message_text":
                target = (messages[-1].get("text", "") if messages else "").lower()
            elif field == "message_count":
                target = len(messages)
            elif field == "chat_id":
                target = chat_id
            else:
                # For other fields, use combined text
                target = all_text

            # Evaluate condition
            if operator == "contains":
                if isinstance(target, str) and value.lower() not in target:
                    return False
            elif operator == "not_contains":
                if isinstance(target, str) and value.lower() in target:
                    return False
            elif operator == "equals":
                if target != value:
                    return False
            elif operator == "greater_than":
                if not isinstance(target, (int, float)) or target <= value:
                    return False
            elif operator == "less_than":
                if not isinstance(target, (int, float)) or target >= value:
                    return False

        return True

    def record_suggestion_feedback(
        self,
        chat_id: str,
        tag_id: int,
        accepted: bool,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record user feedback on a suggestion for learning.

        Args:
            chat_id: Conversation identifier.
            tag_id: The suggested tag ID.
            accepted: Whether the user accepted the suggestion.
            context: Optional context information.
        """
        action = "suggest_accepted" if accepted else "suggest_rejected"

        with self.tag_manager.connection() as conn:
            conn.execute(
                """
                INSERT INTO tag_usage_history (tag_id, chat_id, action, context_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    tag_id,
                    chat_id,
                    action,
                    json.dumps(context) if context else None,
                    datetime.now(UTC),
                ),
            )


# Export all public symbols
__all__ = ["AutoTagger", "ContentAnalysis"]
