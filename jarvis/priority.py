"""Message Priority Scoring System for JARVIS.

Scores incoming messages by importance using ML-based classification to detect:
- Messages containing questions
- Explicit requests or action items
- Time-sensitive content
- Messages from frequent/important contacts

The scorer uses semantic similarity and keyword matching for fast, accurate
priority detection without requiring the full language model.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from contracts.imessage import Message

logger = logging.getLogger(__name__)


class PriorityLevel(Enum):
    """Priority levels for messages."""

    CRITICAL = "critical"  # Urgent, needs immediate attention
    HIGH = "high"  # Important, should respond soon
    MEDIUM = "medium"  # Normal priority
    LOW = "low"  # Can wait, informational


class PriorityReason(Enum):
    """Reasons why a message was assigned a priority level."""

    CONTAINS_QUESTION = "contains_question"
    ACTION_REQUESTED = "action_requested"
    TIME_SENSITIVE = "time_sensitive"
    IMPORTANT_CONTACT = "important_contact"
    FREQUENT_CONTACT = "frequent_contact"
    AWAITING_RESPONSE = "awaiting_response"
    MULTIPLE_MESSAGES = "multiple_messages"
    CONTAINS_URGENCY = "contains_urgency"
    NORMAL = "normal"


@dataclass
class PriorityScore:
    """Result of priority scoring for a message.

    Attributes:
        message_id: The ID of the scored message
        chat_id: The conversation ID
        score: Numerical priority score from 0.0 to 1.0
        level: Categorical priority level
        reasons: List of reasons contributing to the score
        needs_response: Whether this message likely needs a reply
        handled: Whether the user has marked this as handled
    """

    message_id: int
    chat_id: str
    score: float
    level: PriorityLevel
    reasons: list[PriorityReason] = field(default_factory=list)
    needs_response: bool = False
    handled: bool = False


@dataclass
class ContactStats:
    """Statistics about a contact for priority scoring.

    Attributes:
        identifier: Phone number or email
        message_count: Total messages from this contact
        last_message_date: When they last messaged
        avg_response_time_hours: Average time to respond to them
        is_important: Manually marked as important
    """

    identifier: str
    message_count: int = 0
    last_message_date: datetime | None = None
    avg_response_time_hours: float | None = None
    is_important: bool = False


# Question detection patterns
QUESTION_PATTERNS = [
    # Direct questions (ending with ?)
    r"\?\s*$",
    # Question words at start or after common prefixes
    r"(?:^|\s)(what|when|where|who|why|how|which|whose|whom)\s",
    r"(?:^|\s)(can|could|would|will|should|do|does|did|is|are|was|were|have|has|had)\s+(?:you|i|we|they|he|she|it)\b",
    # Indirect questions
    r"(?:wondering|curious|asking)\s+(?:if|whether|about)",
    r"(?:let me know|lmk|tell me)\s+(?:if|whether|when|what)",
    r"(?:any idea|any thoughts|any chance)",
]

# Action/request patterns
ACTION_PATTERNS = [
    # Direct requests
    r"(?:^|\s)(please|pls|plz)\s+\w+",
    r"(?:^|\s)(can you|could you|would you|will you)\s+\w+",
    r"(?:^|\s)(need you to|want you to|need your|want your)\s+",
    r"(?:^|\s)(help me|help with)\s+",
    # Imperatives
    r"(?:^)(call|text|send|bring|get|pick up|grab|buy|check|confirm|reply|respond|answer)\s+",
    # Requests for information
    r"(?:send me|give me|share|forward)\s+(?:the|your|that|this)",
    r"(?:let me know|lmk|get back to me|reply)",
    # Task delegation
    r"(?:your turn|you're up|over to you|up to you|your call)",
]

# Time-sensitive patterns
TIME_SENSITIVE_PATTERNS = [
    # Urgency keywords
    r"(?:^|\s)(urgent|urgently|asap|immediately|right now|right away)\b",
    r"(?:^|\s)(important|critical|crucial|essential|priority)\b",
    # Time constraints
    r"(?:by|before|until|no later than)\s+(?:today|tonight|tomorrow|\d{1,2}(?::\d{2})?)",
    r"(?:^|\s)(today|tonight|this morning|this afternoon|this evening)\b",
    r"(?:^|\s)(soon|quickly|fast)\b",
    r"(?:^|\s)(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    # Deadlines
    r"(?:deadline|due|expires|ends)\s+",
    r"(?:running out|last chance|final|closing)",
    # Time-related questions
    r"what time|when (?:can|should|will|do)",
]

# Urgency/importance keywords
URGENCY_KEYWORDS = {
    "urgent",
    "asap",
    "emergency",
    "important",
    "critical",
    "immediately",
    "right away",
    "right now",
    "help",
    "sos",
    "911",
    "need",
    "must",
    "hurry",
}

# Pre-compiled regex patterns at module level (avoids recompilation per instance)
_COMPILED_QUESTION_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in QUESTION_PATTERNS)
_COMPILED_ACTION_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in ACTION_PATTERNS)
_COMPILED_TIME_SENSITIVE_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in TIME_SENSITIVE_PATTERNS
)


class MessagePriorityScorer:
    """Scores messages by importance using ML and heuristics.

    Uses a combination of:
    1. Pattern matching for questions, actions, and time-sensitivity
    2. Semantic similarity for intent detection
    3. Contact frequency/importance tracking
    4. Message context analysis

    Attributes:
        QUESTION_WEIGHT: Weight for question detection (0.25)
        ACTION_WEIGHT: Weight for action/request detection (0.30)
        TIME_SENSITIVE_WEIGHT: Weight for time-sensitivity (0.20)
        CONTACT_WEIGHT: Weight for contact importance (0.15)
        CONTEXT_WEIGHT: Weight for contextual factors (0.10)
    """

    QUESTION_WEIGHT = 0.25
    ACTION_WEIGHT = 0.30
    TIME_SENSITIVE_WEIGHT = 0.20
    CONTACT_WEIGHT = 0.15
    CONTEXT_WEIGHT = 0.10

    # Thresholds for priority levels
    CRITICAL_THRESHOLD = 0.8
    HIGH_THRESHOLD = 0.6
    MEDIUM_THRESHOLD = 0.3

    def __init__(self) -> None:
        """Initialize the priority scorer with lazy-loaded embeddings."""
        self._sentence_model: Any | None = None
        self._intent_embeddings: dict[str, np.ndarray] | None = None
        self._lock = threading.Lock()

        # Contact statistics cache
        self._contact_stats: dict[str, ContactStats] = {}
        self._important_contacts: set[str] = set()

        # Handled items tracking
        self._handled_items: set[tuple[str, int]] = set()  # (chat_id, message_id)

        # Use pre-compiled module-level regex patterns (avoids recompilation per instance)
        self._question_patterns = _COMPILED_QUESTION_PATTERNS
        self._action_patterns = _COMPILED_ACTION_PATTERNS
        self._time_sensitive_patterns = _COMPILED_TIME_SENSITIVE_PATTERNS

    def _get_sentence_model(self) -> Any:
        """Get the sentence transformer model for semantic similarity.

        Reuses the model from models.templates to avoid loading it twice.

        Returns:
            The loaded SentenceTransformer model

        Raises:
            SentenceModelError: If model cannot be loaded
        """
        if self._sentence_model is not None:
            return self._sentence_model

        from models.templates import SentenceModelError, _get_sentence_model

        try:
            self._sentence_model = _get_sentence_model()
            return self._sentence_model
        except SentenceModelError:
            logger.warning("Failed to load sentence model for priority scoring")
            raise

    def _ensure_embeddings_computed(self) -> None:
        """Compute and cache embeddings for priority-related intents.

        Caches embeddings for:
        - Question intent examples
        - Action/request intent examples
        - Urgency intent examples
        """
        if self._intent_embeddings is not None:
            return

        with self._lock:
            if self._intent_embeddings is not None:
                return

            try:
                model = self._get_sentence_model()
            except ImportError:
                # SentenceTransformer or related dependencies not installed
                logger.warning("Sentence transformer not installed, using pattern matching only")
                self._intent_embeddings = {}
                return
            except OSError as e:
                # File system errors (model files inaccessible, disk issues)
                logger.warning(
                    "Sentence model unavailable (I/O error: %s), using pattern matching only", e
                )
                self._intent_embeddings = {}
                return
            except (RuntimeError, ValueError) as e:
                # Model loading failures (corrupted model, invalid config)
                logger.warning(
                    "Sentence model unavailable (loading error: %s), using pattern matching only",
                    e,
                )
                self._intent_embeddings = {}
                return
            except Exception:
                # Last resort catch-all for truly unexpected errors during model loading
                # (e.g., memory exhaustion, network timeouts). Fall back to pattern
                # matching only mode to ensure the system remains functional.
                logger.exception(
                    "Sentence model unavailable (unexpected error), using pattern matching only"
                )
                self._intent_embeddings = {}
                return

            # Intent examples for semantic similarity
            intent_examples = {
                "question": [
                    "can you help me with this",
                    "what do you think about",
                    "when are you available",
                    "where should we meet",
                    "how does this work",
                    "do you have time",
                    "is this okay with you",
                    "would you be able to",
                    "any thoughts on this",
                    "what's your opinion",
                ],
                "action": [
                    "please send me the document",
                    "can you pick up groceries",
                    "need you to call me back",
                    "don't forget to",
                    "make sure you",
                    "remember to",
                    "i need you to",
                    "your turn to",
                    "please respond",
                    "let me know when you're done",
                ],
                "urgent": [
                    "this is urgent",
                    "need help right now",
                    "emergency situation",
                    "please respond asap",
                    "this can't wait",
                    "time sensitive matter",
                    "critical issue",
                    "need immediate response",
                    "very important",
                    "deadline approaching",
                ],
            }

            self._intent_embeddings = {}
            for intent_type, examples in intent_examples.items():
                embeddings = model.encode(examples, convert_to_numpy=True)
                # Store centroid (mean embedding) normalized for cosine similarity
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                self._intent_embeddings[intent_type] = centroid

            logger.debug("Computed priority intent embeddings")

    def _get_text_embedding(self, text: str) -> np.ndarray | None:
        """Compute normalized embedding for text (cached per call site).

        Args:
            text: Text to embed.

        Returns:
            Normalized embedding array, or None if encoding fails.
        """
        if not text:
            return None

        try:
            model = self._get_sentence_model()
            embedding = model.encode([text], convert_to_numpy=True)[0]
            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except (ValueError, TypeError) as e:
            logger.debug("Text embedding failed: %s", e)
        except (RuntimeError, OSError) as e:
            logger.debug("Text embedding failed (runtime/I/O): %s", e)
        except Exception:
            logger.warning("Text embedding unavailable", exc_info=True)
        return None

    def _detect_question(
        self, text: str, embedding: np.ndarray | None = None
    ) -> tuple[bool, float]:
        """Detect if message contains a question.

        Args:
            text: Message text to analyze.
            embedding: Pre-computed text embedding (optional, avoids recomputation).

        Returns:
            Tuple of (is_question, confidence)
        """
        if not text:
            return False, 0.0

        # Pattern matching
        for pattern in self._question_patterns:
            if pattern.search(text):
                return True, 0.9

        # Semantic similarity if model available
        if self._intent_embeddings and "question" in self._intent_embeddings:
            if embedding is not None:
                similarity = float(np.dot(embedding, self._intent_embeddings["question"]))
                if similarity > 0.6:
                    return True, similarity

        return False, 0.0

    def _detect_action_request(
        self, text: str, embedding: np.ndarray | None = None
    ) -> tuple[bool, float]:
        """Detect if message contains an action request.

        Args:
            text: Message text to analyze.
            embedding: Pre-computed text embedding (optional, avoids recomputation).

        Returns:
            Tuple of (is_action, confidence)
        """
        if not text:
            return False, 0.0

        # Pattern matching
        for pattern in self._action_patterns:
            if pattern.search(text):
                return True, 0.9

        # Semantic similarity if model available
        if self._intent_embeddings and "action" in self._intent_embeddings:
            if embedding is not None:
                similarity = float(np.dot(embedding, self._intent_embeddings["action"]))
                if similarity > 0.6:
                    return True, similarity

        return False, 0.0

    def _detect_time_sensitive(
        self, text: str, embedding: np.ndarray | None = None
    ) -> tuple[bool, float]:
        """Detect if message is time-sensitive.

        Args:
            text: Message text to analyze.
            embedding: Pre-computed text embedding (optional, avoids recomputation).

        Returns:
            Tuple of (is_time_sensitive, confidence)
        """
        if not text:
            return False, 0.0

        text_lower = text.lower()

        # Check urgency keywords first (highest confidence)
        words = set(re.findall(r"\b\w+\b", text_lower))
        if words & URGENCY_KEYWORDS:
            return True, 0.95

        # Pattern matching
        for pattern in self._time_sensitive_patterns:
            if pattern.search(text):
                return True, 0.85

        # Semantic similarity if model available
        if self._intent_embeddings and "urgent" in self._intent_embeddings:
            if embedding is not None:
                similarity = float(np.dot(embedding, self._intent_embeddings["urgent"]))
                if similarity > 0.65:
                    return True, similarity

        return False, 0.0

    def _get_contact_score(self, sender: str) -> tuple[float, list[PriorityReason]]:
        """Get priority score based on contact importance.

        Args:
            sender: Contact identifier (phone/email)

        Returns:
            Tuple of (score, reasons)
        """
        reasons: list[PriorityReason] = []
        score = 0.0

        # Check if contact is marked as important
        if sender in self._important_contacts:
            score += 0.5
            reasons.append(PriorityReason.IMPORTANT_CONTACT)

        # Check contact stats
        stats = self._contact_stats.get(sender)
        if stats:
            # Frequent contacts get higher priority
            if stats.message_count > 50:
                score += 0.3
                reasons.append(PriorityReason.FREQUENT_CONTACT)
            elif stats.message_count > 20:
                score += 0.2
                reasons.append(PriorityReason.FREQUENT_CONTACT)

        return min(score, 1.0), reasons

    def score_message(
        self,
        message: Message,
        recent_messages: list[Message] | None = None,
    ) -> PriorityScore:
        """Score a message's priority.

        Args:
            message: The message to score
            recent_messages: Optional list of recent messages for context

        Returns:
            PriorityScore with score, level, and reasons
        """
        # Initialize embeddings if needed
        try:
            self._ensure_embeddings_computed()
        except ImportError:
            # SentenceTransformer not installed - continue with pattern matching only
            pass
        except (OSError, RuntimeError, ValueError):
            # Model loading or embedding computation failed due to I/O, state, or config errors.
            # Continue with pattern matching only, logging is done in _ensure_embeddings_computed.
            pass
        except Exception:
            logger.warning("Unexpected error initializing embeddings, using pattern matching only")

        reasons: list[PriorityReason] = []
        component_scores: dict[str, float] = {}

        text = message.text or ""

        # Compute embedding ONCE for all semantic detection methods
        # This avoids 3x encoding overhead for the same text
        text_embedding = self._get_text_embedding(text) if self._intent_embeddings else None

        # 1. Detect questions
        is_question, question_conf = self._detect_question(text, text_embedding)
        if is_question:
            component_scores["question"] = question_conf
            reasons.append(PriorityReason.CONTAINS_QUESTION)

        # 2. Detect action requests
        is_action, action_conf = self._detect_action_request(text, text_embedding)
        if is_action:
            component_scores["action"] = action_conf
            reasons.append(PriorityReason.ACTION_REQUESTED)

        # 3. Detect time sensitivity
        is_time_sensitive, time_conf = self._detect_time_sensitive(text, text_embedding)
        if is_time_sensitive:
            component_scores["time_sensitive"] = time_conf
            reasons.append(PriorityReason.TIME_SENSITIVE)

        # 4. Contact importance
        contact_score, contact_reasons = self._get_contact_score(message.sender)
        if contact_score > 0:
            component_scores["contact"] = contact_score
            reasons.extend(contact_reasons)

        # 5. Contextual factors
        context_score = 0.0
        if recent_messages:
            # Check for multiple unanswered messages from same sender
            sender_messages = [m for m in recent_messages if m.sender == message.sender]
            unanswered = [m for m in sender_messages if not m.is_from_me]
            if len(unanswered) >= 3:
                context_score += 0.5
                reasons.append(PriorityReason.MULTIPLE_MESSAGES)

            # Check if we haven't responded in a while
            my_messages = [m for m in recent_messages if m.is_from_me]
            if my_messages and sender_messages:
                last_my_msg = max(my_messages, key=lambda m: m.date)
                last_their_msg = max(sender_messages, key=lambda m: m.date)
                if last_their_msg.date > last_my_msg.date:
                    # They messaged after our last message
                    delta = datetime.now(tz=UTC) - last_my_msg.date
                    hours_since = delta.total_seconds() / 3600
                    if hours_since > 4:
                        context_score += 0.3
                        reasons.append(PriorityReason.AWAITING_RESPONSE)

        if context_score > 0:
            component_scores["context"] = min(context_score, 1.0)

        # Calculate weighted score
        total_score = 0.0
        total_score += component_scores.get("question", 0.0) * self.QUESTION_WEIGHT
        total_score += component_scores.get("action", 0.0) * self.ACTION_WEIGHT
        total_score += component_scores.get("time_sensitive", 0.0) * self.TIME_SENSITIVE_WEIGHT
        total_score += component_scores.get("contact", 0.0) * self.CONTACT_WEIGHT
        total_score += component_scores.get("context", 0.0) * self.CONTEXT_WEIGHT

        # Normalize to 0-1 range
        total_score = min(total_score, 1.0)

        # Determine priority level
        if total_score >= self.CRITICAL_THRESHOLD:
            level = PriorityLevel.CRITICAL
        elif total_score >= self.HIGH_THRESHOLD:
            level = PriorityLevel.HIGH
        elif total_score >= self.MEDIUM_THRESHOLD:
            level = PriorityLevel.MEDIUM
        else:
            level = PriorityLevel.LOW
            if not reasons:
                reasons.append(PriorityReason.NORMAL)

        # Determine if response is needed
        needs_response = (
            is_question
            or is_action
            or PriorityReason.AWAITING_RESPONSE in reasons
            or PriorityReason.MULTIPLE_MESSAGES in reasons
        )

        # Check if already handled
        handled = (message.chat_id, message.id) in self._handled_items

        return PriorityScore(
            message_id=message.id,
            chat_id=message.chat_id,
            score=total_score,
            level=level,
            reasons=reasons,
            needs_response=needs_response,
            handled=handled,
        )

    def score_messages(
        self,
        messages: list[Message],
    ) -> list[PriorityScore]:
        """Score multiple messages and sort by priority.

        Args:
            messages: List of messages to score

        Returns:
            List of PriorityScores sorted by score descending
        """
        scores = []
        for message in messages:
            if message.is_from_me or message.is_system_message:
                continue  # Skip messages from self and system messages

            # Get recent messages for context
            recent = [m for m in messages if m.chat_id == message.chat_id]
            score = self.score_message(message, recent)
            scores.append(score)

        # Sort by score descending, then by needs_response
        scores.sort(key=lambda s: (s.score, s.needs_response), reverse=True)
        return scores

    def mark_handled(self, chat_id: str, message_id: int) -> None:
        """Mark a message as handled by the user.

        Args:
            chat_id: Conversation ID
            message_id: Message ID
        """
        with self._lock:
            self._handled_items.add((chat_id, message_id))
        logger.debug("Marked message %d in chat %s as handled", message_id, chat_id)

    def unmark_handled(self, chat_id: str, message_id: int) -> None:
        """Unmark a message as handled.

        Args:
            chat_id: Conversation ID
            message_id: Message ID
        """
        self._handled_items.discard((chat_id, message_id))
        logger.debug("Unmarked message %d in chat %s as handled", message_id, chat_id)

    def mark_contact_important(self, identifier: str, important: bool = True) -> None:
        """Mark a contact as important or not.

        Args:
            identifier: Contact phone number or email
            important: Whether to mark as important
        """
        if important:
            self._important_contacts.add(identifier)
        else:
            self._important_contacts.discard(identifier)

    def update_contact_stats(
        self,
        identifier: str,
        message_count: int,
        last_message_date: datetime | None = None,
    ) -> None:
        """Update contact statistics for priority scoring.

        Args:
            identifier: Contact phone number or email
            message_count: Total message count
            last_message_date: Date of last message
        """
        if identifier not in self._contact_stats:
            self._contact_stats[identifier] = ContactStats(identifier=identifier)

        stats = self._contact_stats[identifier]
        stats.message_count = message_count
        if last_message_date:
            stats.last_message_date = last_message_date

    def get_handled_count(self) -> int:
        """Get the number of handled items.

        Returns:
            Count of handled items
        """
        return len(self._handled_items)

    def clear_handled(self) -> None:
        """Clear all handled items."""
        self._handled_items.clear()
        logger.debug("Cleared all handled items")

    def clear_cache(self) -> None:
        """Clear cached embeddings.

        Call this to force recomputation of embeddings.
        """
        with self._lock:
            self._intent_embeddings = None
            self._sentence_model = None
            logger.debug("Priority scorer cache cleared")


# Module-level singleton instance
_scorer: MessagePriorityScorer | None = None
_scorer_lock = threading.Lock()


def get_priority_scorer() -> MessagePriorityScorer:
    """Get the singleton MessagePriorityScorer instance.

    Returns:
        The shared MessagePriorityScorer instance
    """
    global _scorer
    if _scorer is None:
        with _scorer_lock:
            if _scorer is None:
                _scorer = MessagePriorityScorer()
    return _scorer


def reset_priority_scorer() -> None:
    """Reset the singleton MessagePriorityScorer instance.

    Call this to create a fresh scorer on next access.
    """
    global _scorer
    with _scorer_lock:
        _scorer = None
    logger.debug("Priority scorer singleton reset")
