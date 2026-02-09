"""Thread-aware conversation analysis for JARVIS.

Provides thread detection, topic classification, state tracking,
and user role identification for improved reply generation and navigation.

Analyzes messages to group them into logical conversation threads using:
- Semantic similarity between consecutive messages
- Time gaps between messages
- Topic shifts detected via embeddings
- Explicit reply references (reply_to_id)
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from contracts.imessage import Message
    from jarvis.embedding_adapter import CachedEmbedder

logger = logging.getLogger(__name__)


class ThreadingMethod(Enum):
    """Methods used to determine thread boundaries."""

    REPLY_REFERENCE = "reply_reference"  # Explicit reply_to_id link
    SEMANTIC_SIMILARITY = "semantic_similarity"  # Similar message content
    TIME_GAP = "time_gap"  # Long pause between messages
    TOPIC_SHIFT = "topic_shift"  # Detected topic change


class ThreadTopic(Enum):
    """Types of conversation thread topics."""

    PLANNING = "planning"  # Making plans, scheduling
    LOGISTICS = "logistics"  # Coordinating details, times, locations
    CATCHING_UP = "catching_up"  # General conversation, catching up
    EMOTIONAL_SUPPORT = "emotional_support"  # Support, sympathy, encouragement
    QUICK_EXCHANGE = "quick_exchange"  # Brief back-and-forth, acknowledgments
    INFORMATION = "information"  # Sharing/requesting information
    DECISION_MAKING = "decision_making"  # Making choices together
    CELEBRATION = "celebration"  # Congratulations, good news
    UNKNOWN = "unknown"


class ThreadState(Enum):
    """Current state of a conversation thread."""

    OPEN_QUESTION = "open_question"  # Waiting for an answer
    AWAITING_RESPONSE = "awaiting_response"  # General expectation of reply
    IN_DISCUSSION = "in_discussion"  # Active back-and-forth
    CONCLUDED = "concluded"  # Conversation naturally ended
    STALE = "stale"  # No recent activity


class UserRole(Enum):
    """User's role in the conversation thread."""

    INITIATOR = "initiator"  # Started the thread/topic
    RESPONDER = "responder"  # Responding to other's initiation
    PARTICIPANT = "participant"  # Equal participant in ongoing exchange


@dataclass
class Thread:
    """A logical conversation thread.

    Attributes:
        thread_id: Unique identifier for this thread
        messages: List of message IDs in this thread
        topic_label: Auto-generated topic label for the thread
        start_time: Timestamp of first message
        end_time: Timestamp of last message
        participant_count: Number of unique participants
        message_count: Number of messages in thread
    """

    thread_id: str
    messages: list[int] = field(default_factory=list)
    topic_label: str = ""
    start_time: datetime | None = None
    end_time: datetime | None = None
    participant_count: int = 0
    message_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert thread to dictionary for API responses."""
        return {
            "thread_id": self.thread_id,
            "messages": self.messages,
            "topic_label": self.topic_label,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "participant_count": self.participant_count,
            "message_count": self.message_count,
        }


@dataclass
class ThreadedMessage:
    """A message with thread information attached.

    Attributes:
        message: The original message
        thread_id: ID of the thread this message belongs to
        thread_position: Position within the thread (0-indexed)
        is_thread_start: True if this message starts a new thread
        threading_reason: Why this message was grouped into its thread
    """

    message: Message
    thread_id: str
    thread_position: int = 0
    is_thread_start: bool = False
    threading_reason: ThreadingMethod = ThreadingMethod.SEMANTIC_SIMILARITY


@dataclass
class ThreadContext:
    """Analyzed context for a conversation thread.

    Attributes:
        messages: The messages in the thread
        topic: Detected thread topic
        state: Current thread state
        user_role: User's role in the thread
        confidence: Confidence score for topic classification (0.0-1.0)
        relevant_messages: Subset of messages most relevant to current context
        action_items: Any detected action items or commitments
        participants_count: Number of participants (1 = DM, >1 = group)
    """

    messages: list[Message]
    topic: ThreadTopic
    state: ThreadState
    user_role: UserRole
    confidence: float
    relevant_messages: list[Message] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    participants_count: int = 1


@dataclass
class ThreadedReplyConfig:
    """Configuration for thread-aware reply generation.

    Attributes:
        max_response_length: Suggested max response length based on thread type
        response_style: Suggested response style (concise/detailed/empathetic)
        include_action_items: Whether to include action items in response
        suggest_follow_up: Whether to suggest follow-up questions
    """

    max_response_length: int
    response_style: str
    include_action_items: bool
    suggest_follow_up: bool


@dataclass
class ThreadingConfig:
    """Configuration for thread detection.

    Attributes:
        time_gap_threshold_minutes: Minutes of silence to start new thread
        semantic_similarity_threshold: Minimum similarity to group messages (0-1)
        min_thread_messages: Minimum messages to form a thread
        max_thread_duration_hours: Maximum thread duration before forcing split
        use_semantic_analysis: Whether to use ML-based semantic similarity
    """

    time_gap_threshold_minutes: int = 30
    semantic_similarity_threshold: float = 0.4
    min_thread_messages: int = 2
    max_thread_duration_hours: int = 24
    use_semantic_analysis: bool = True


# Topic classification patterns with example phrases
TOPIC_PATTERNS: dict[ThreadTopic, list[str]] = {
    ThreadTopic.PLANNING: [
        "let's plan",
        "when should we",
        "what day works",
        "are you free",
        "want to meet",
        "let's get together",
        "should we do",
        "thinking about",
        "how about we",
        "planning to",
        "want to grab",
        "let's do",
        "wanna hang",
        "wanna go",
        "want to come",
        "are you coming",
        "can you make it",
        "will you be there",
        "planning",
        "schedule",
    ],
    ThreadTopic.LOGISTICS: [
        "what time",
        "where should",
        "what address",
        "how do I get",
        "meet at",
        "pick you up",
        "be there by",
        "running late",
        "on my way",
        "almost there",
        "just left",
        "eta",
        "where are you",
        "directions",
        "parking",
        "which entrance",
        "I'm here",
        "just arrived",
        "waiting",
    ],
    ThreadTopic.EMOTIONAL_SUPPORT: [
        "I'm sorry",
        "that's tough",
        "I understand",
        "here for you",
        "feel better",
        "thinking of you",
        "so sorry to hear",
        "hang in there",
        "I'm here",
        "let me know if you need",
        "sending love",
        "you've got this",
        "proud of you",
        "it'll be okay",
        "don't worry",
        "feeling down",
        "bad day",
        "stressed",
        "anxious",
        "worried about",
        "upset",
        "sad",
        "frustrated",
    ],
    ThreadTopic.CATCHING_UP: [
        "how are you",
        "what's new",
        "long time",
        "miss you",
        "how have you been",
        "what's up",
        "how's it going",
        "catch up",
        "tell me about",
        "what have you been up to",
        "how's life",
        "how was your",
        "what did you do",
        "anything new",
        "how's everything",
        "good to hear from",
    ],
    ThreadTopic.QUICK_EXCHANGE: [
        "ok",
        "okay",
        "sounds good",
        "got it",
        "thanks",
        "thank you",
        "cool",
        "great",
        "perfect",
        "nice",
        "lol",
        "haha",
        "yes",
        "no",
        "yep",
        "nope",
        "sure",
        "np",
        "kk",
        "ttyl",
        "bye",
    ],
    ThreadTopic.INFORMATION: [
        "do you know",
        "can you tell me",
        "what is",
        "where is",
        "who is",
        "how do",
        "what's the",
        "I need",
        "looking for",
        "have you heard",
        "did you see",
        "fyi",
        "just so you know",
        "heads up",
        "reminder",
        "don't forget",
        "btw",
        "wanted to let you know",
    ],
    ThreadTopic.DECISION_MAKING: [
        "what do you think",
        "should I",
        "which one",
        "your opinion",
        "help me decide",
        "can't decide",
        "what would you",
        "do you prefer",
        "better option",
        "pros and cons",
        "advice",
        "recommend",
        "suggest",
        "your thoughts",
        "vote",
        "choose",
    ],
    ThreadTopic.CELEBRATION: [
        "congratulations",
        "congrats",
        "so happy for",
        "amazing news",
        "great news",
        "celebrate",
        "proud of you",
        "you did it",
        "well done",
        "awesome",
        "exciting",
        "happy birthday",
        "cheers",
        "woohoo",
        "yay",
    ],
}

# Patterns indicating questions or expectation of response
QUESTION_PATTERNS = [
    r"\?$",  # Ends with question mark
    r"^(what|where|when|who|why|how|which|can|could|would|will|do|does|did|is|are|was|were)\b",
    r"\b(let me know|lmk|thoughts\??|wdyt|wyt)\b",
]

# Patterns indicating thread conclusion
CONCLUSION_PATTERNS = [
    r"\b(sounds good|perfect|great|thanks|bye|later|ttyl|see you|talk soon|cya)\b",
    r"\b(got it|understood|will do|on it)\b",
    r"^(ok|okay|k|kk)$",
]

# Response configuration based on thread topic
TOPIC_RESPONSE_CONFIG: dict[ThreadTopic, ThreadedReplyConfig] = {
    ThreadTopic.LOGISTICS: ThreadedReplyConfig(
        max_response_length=50,
        response_style="concise",
        include_action_items=True,
        suggest_follow_up=False,
    ),
    ThreadTopic.QUICK_EXCHANGE: ThreadedReplyConfig(
        max_response_length=30,
        response_style="brief",
        include_action_items=False,
        suggest_follow_up=False,
    ),
    ThreadTopic.EMOTIONAL_SUPPORT: ThreadedReplyConfig(
        max_response_length=150,
        response_style="empathetic",
        include_action_items=False,
        suggest_follow_up=True,
    ),
    ThreadTopic.PLANNING: ThreadedReplyConfig(
        max_response_length=100,
        response_style="detailed",
        include_action_items=True,
        suggest_follow_up=True,
    ),
    ThreadTopic.CATCHING_UP: ThreadedReplyConfig(
        max_response_length=100,
        response_style="warm",
        include_action_items=False,
        suggest_follow_up=True,
    ),
    ThreadTopic.INFORMATION: ThreadedReplyConfig(
        max_response_length=100,
        response_style="clear",
        include_action_items=False,
        suggest_follow_up=False,
    ),
    ThreadTopic.DECISION_MAKING: ThreadedReplyConfig(
        max_response_length=120,
        response_style="thoughtful",
        include_action_items=True,
        suggest_follow_up=True,
    ),
    ThreadTopic.CELEBRATION: ThreadedReplyConfig(
        max_response_length=80,
        response_style="enthusiastic",
        include_action_items=False,
        suggest_follow_up=False,
    ),
    ThreadTopic.UNKNOWN: ThreadedReplyConfig(
        max_response_length=80,
        response_style="natural",
        include_action_items=False,
        suggest_follow_up=False,
    ),
}


class ThreadAnalyzer:
    """Analyzes conversation threads for logical grouping and context-aware reply generation.

    Uses a combination of heuristics (time, references) and ML (embeddings) to group messages
    and identify thread properties like topic and state.
    """

    TOPIC_CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, config: ThreadingConfig | None = None) -> None:
        """Initialize the thread analyzer.

        Args:
            config: Threading configuration options
        """
        self.config = config or ThreadingConfig()
        self._sentence_model: CachedEmbedder | None = None
        self._topic_embeddings: dict[ThreadTopic, np.ndarray] | None = None
        self._lock = threading.Lock()
        # LRU cache for embeddings with 1000 entry limit (per CODE_REVIEW.md #15)
        # Prevents unbounded memory growth on 8GB systems
        self._embeddings_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_maxsize = 1000

    def _get_sentence_model(self) -> CachedEmbedder | None:
        """Get the sentence transformer model with lazy loading."""
        if self._sentence_model is not None:
            return self._sentence_model

        with self._lock:
            if self._sentence_model is not None:
                return self._sentence_model

            try:
                from models.templates import _get_sentence_model

                self._sentence_model = _get_sentence_model()
                return self._sentence_model
            except Exception as e:
                logger.debug("Failed to load sentence model for threading: %s", e)
                return None

    def _ensure_embeddings_computed(self) -> bool:
        """Compute and cache embeddings for topic patterns."""
        if self._topic_embeddings is not None:
            return True

        # Get model BEFORE acquiring lock to avoid deadlock
        # (_get_sentence_model also acquires self._lock)
        model = self._get_sentence_model()
        if model is None:
            return False

        with self._lock:
            if self._topic_embeddings is not None:
                return True

            topic_embeddings: dict[ThreadTopic, np.ndarray] = {}
            for topic, patterns in TOPIC_PATTERNS.items():
                embeddings = model.encode(patterns, convert_to_numpy=True)
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                topic_embeddings[topic] = centroid

            self._topic_embeddings = topic_embeddings
            return True

    def _compute_embedding(self, text: str) -> np.ndarray | None:
        """Compute embedding for a text string with LRU caching.

        Cache is bounded to 1000 entries to prevent memory growth on 8GB systems.
        Uses LRU eviction: most recently accessed entries are kept.
        """
        if not text or not text.strip():
            return None

        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embeddings_cache:
            # Move to end (mark as recently used) for LRU eviction
            self._embeddings_cache.move_to_end(cache_key)
            return self._embeddings_cache[cache_key]

        try:
            model = self._get_sentence_model()
            if not model:
                return None
            embeddings = model.encode([text], convert_to_numpy=True)
            embedding: np.ndarray = embeddings[0]
            embedding = embedding / np.linalg.norm(embedding)

            # Store in cache with LRU eviction
            self._embeddings_cache[cache_key] = embedding
            if len(self._embeddings_cache) > self._cache_maxsize:
                # Remove oldest (least recently used) entry
                self._embeddings_cache.popitem(last=False)

            return embedding
        except Exception as e:
            logger.debug("Failed to compute embedding for text: %s", e)
            return None

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        emb1 = self._compute_embedding(text1)
        emb2 = self._compute_embedding(text2)

        if emb1 is None or emb2 is None:
            return 0.0

        return max(0.0, min(1.0, float(np.dot(emb1, emb2))))

    def _generate_thread_id(self, chat_id: str, start_message_id: int) -> str:
        """Generate a unique thread ID."""
        data = f"{chat_id}:{start_message_id}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def analyze_threads(self, messages: list[Message], chat_id: str = "") -> list[Thread]:
        """Group messages into logical threads."""
        if not messages:
            return []

        sorted_messages = sorted(messages, key=lambda m: m.date)
        threads: list[Thread] = []
        threads_by_id: dict[str, Thread] = {}  # O(1) lookup by thread_id
        current_msgs: list[Message] = []
        current_start: datetime | None = None
        reply_threads: dict[int, str] = {}

        for i, msg in enumerate(sorted_messages):
            if msg.reply_to_id is not None and msg.reply_to_id in reply_threads:
                tid = reply_threads[msg.reply_to_id]
                # O(1) dict lookup instead of O(n) linear search
                thread = threads_by_id.get(tid)
                if thread is not None:
                    thread.messages.append(msg.id)
                    thread.message_count += 1
                    thread.end_time = msg.date
                    reply_threads[msg.id] = tid
                continue

            if not current_msgs:
                current_msgs = [msg]
                current_start = msg.date
            else:
                # current_start is guaranteed to be set when current_msgs is non-empty
                assert current_start is not None
                should_start, _ = self._should_start_new_thread(
                    msg, sorted_messages[i - 1], current_start
                )
                if should_start:
                    new_thread = self._create_thread(current_msgs, chat_id, reply_threads)
                    threads.append(new_thread)
                    threads_by_id[new_thread.thread_id] = new_thread
                    current_msgs = [msg]
                    current_start = msg.date
                else:
                    current_msgs.append(msg)

        if current_msgs:
            new_thread = self._create_thread(current_msgs, chat_id, reply_threads)
            threads.append(new_thread)
            threads_by_id[new_thread.thread_id] = new_thread

        return threads

    def _should_start_new_thread(
        self, current: Message, previous: Message, thread_start: datetime
    ) -> tuple[bool, ThreadingMethod]:
        if current.reply_to_id is not None:
            return False, ThreadingMethod.REPLY_REFERENCE

        if current.date - previous.date > timedelta(minutes=self.config.time_gap_threshold_minutes):
            return True, ThreadingMethod.TIME_GAP

        if current.date - thread_start > timedelta(hours=self.config.max_thread_duration_hours):
            return True, ThreadingMethod.TIME_GAP

        if self.config.use_semantic_analysis:
            sim = self._compute_similarity(previous.text or "", current.text or "")
            if sim < self.config.semantic_similarity_threshold:
                return True, ThreadingMethod.TOPIC_SHIFT

        return False, ThreadingMethod.SEMANTIC_SIMILARITY

    def _create_thread(
        self, msgs: list[Message], chat_id: str, reply_map: dict[int, str]
    ) -> Thread:
        tid = self._generate_thread_id(chat_id, msgs[0].id)
        for m in msgs:
            reply_map[m.id] = tid

        participants = {m.sender for m in msgs}
        return Thread(
            thread_id=tid,
            messages=[m.id for m in msgs],
            topic_label=self._detect_topic_label(msgs),
            start_time=msgs[0].date,
            end_time=msgs[-1].date,
            participant_count=len(participants),
            message_count=len(msgs),
        )

    def _detect_topic_label(self, msgs: list[Message]) -> str:
        """Detect a human-readable topic label for a thread.

        Looks for specific keywords in message text to provide more
        descriptive labels like "Meeting Plans" or "Dinner Plans".
        """
        if not msgs:
            return "General"

        # Combine message text for keyword search
        combined = " ".join((m.text or "").lower() for m in msgs)

        # Check for specific plan types first
        if "meeting" in combined or "schedule" in combined:
            return "Meeting Plans"
        if "dinner" in combined:
            return "Dinner Plans"
        if "lunch" in combined:
            return "Lunch Plans"
        if "breakfast" in combined:
            return "Breakfast Plans"
        if "coffee" in combined:
            return "Coffee Plans"
        if "party" in combined or "celebrate" in combined:
            return "Party Plans"
        if "trip" in combined or "travel" in combined:
            return "Travel Plans"

        # Fall back to topic-based label
        ctx = self.analyze(msgs)
        return ctx.topic.value.replace("_", " ").title()

    def get_threaded_messages(self, messages: list[Message], chat_id: str) -> list[ThreadedMessage]:
        """Get messages with thread information attached.

        Analyzes messages and returns them wrapped with thread context
        including thread_id, position within thread, and thread start markers.

        Args:
            messages: List of messages to analyze
            chat_id: ID of the chat these messages belong to

        Returns:
            List of ThreadedMessage objects with thread info attached
        """
        if not messages:
            return []

        # First, analyze to get the threads
        threads = self.analyze_threads(messages, chat_id)

        # Build a map from message_id to thread info
        message_to_thread: dict[int, tuple[str, int, bool, ThreadingMethod]] = {}
        for thread in threads:
            for position, msg_id in enumerate(thread.messages):
                is_start = position == 0
                message_to_thread[msg_id] = (
                    thread.thread_id,
                    position,
                    is_start,
                    ThreadingMethod.TIME_GAP if is_start else ThreadingMethod.SEMANTIC_SIMILARITY,
                )

        # Wrap each message with thread info
        result = []
        for msg in messages:
            if msg.id in message_to_thread:
                thread_id, position, is_start, reason = message_to_thread[msg.id]
                result.append(
                    ThreadedMessage(
                        message=msg,
                        thread_id=thread_id,
                        thread_position=position,
                        is_thread_start=is_start,
                        threading_reason=reason,
                    )
                )
            else:
                # Message not in any thread - create standalone
                tid = self._generate_thread_id(chat_id, msg.id)
                result.append(
                    ThreadedMessage(
                        message=msg,
                        thread_id=tid,
                        thread_position=0,
                        is_thread_start=True,
                        threading_reason=ThreadingMethod.TIME_GAP,
                    )
                )

        return result

    def analyze(self, messages: list[Message]) -> ThreadContext:
        """Analyze a specific thread context."""
        if not messages:
            return ThreadContext(
                [], ThreadTopic.UNKNOWN, ThreadState.CONCLUDED, UserRole.PARTICIPANT, 0.0
            )

        topic, conf = self._detect_topic(messages)
        state = self._detect_state(messages)
        role = self._detect_user_role(messages)
        relevant = self._get_relevant_messages(messages, topic)
        items = self._extract_action_items(messages)
        parts = {m.sender for m in messages if not m.is_from_me}

        return ThreadContext(
            messages, topic, state, role, conf, relevant, items, max(1, len(parts))
        )

    def _detect_topic(self, messages: list[Message]) -> tuple[ThreadTopic, float]:
        recent = messages[-10:]
        combined = " ".join(m.text.lower() for m in recent if m.text)

        pattern_scores = {
            t: sum(1 for p in pats if p.lower() in combined) for t, pats in TOPIC_PATTERNS.items()
        }
        best_p = max(pattern_scores, key=lambda k: pattern_scores[k])
        if pattern_scores[best_p] >= 2:
            return best_p, min(0.9, 0.5 + pattern_scores[best_p] * 0.1)

        if self._ensure_embeddings_computed() and self._topic_embeddings:
            emb = self._compute_embedding(combined)
            if emb is not None:
                sims = {t: float(np.dot(emb, c)) for t, c in self._topic_embeddings.items()}
                best_t = max(sims, key=lambda k: sims[k])
                if sims[best_t] >= self.TOPIC_CONFIDENCE_THRESHOLD:
                    return best_t, sims[best_t]

        if pattern_scores[best_p] >= 1:
            return (best_p, 0.3 + pattern_scores[best_p] * 0.1)
        return (ThreadTopic.UNKNOWN, 0.0)

    def _detect_state(self, messages: list[Message]) -> ThreadState:
        if not messages:
            return ThreadState.CONCLUDED
        last = messages[-1]
        text = (last.text or "").lower()
        if any(re.search(p, text, re.I) for p in QUESTION_PATTERNS):
            return ThreadState.AWAITING_RESPONSE if last.is_from_me else ThreadState.OPEN_QUESTION
        if any(re.search(p, text, re.I) for p in CONCLUSION_PATTERNS):
            return ThreadState.CONCLUDED
        return ThreadState.IN_DISCUSSION

    def _detect_user_role(self, messages: list[Message]) -> UserRole:
        if not messages:
            return UserRole.PARTICIPANT
        if messages[0].is_from_me:
            return UserRole.INITIATOR
        my_count = sum(1 for m in messages if m.is_from_me)
        ratio = my_count / len(messages)
        if ratio > 0.6:
            return UserRole.INITIATOR
        if ratio < 0.3:
            return UserRole.RESPONDER
        return UserRole.PARTICIPANT

    def _get_relevant_messages(self, messages: list[Message], topic: ThreadTopic) -> list[Message]:
        if topic == ThreadTopic.QUICK_EXCHANGE:
            return messages[-3:]
        if topic == ThreadTopic.LOGISTICS:
            logistics_words = ["time", "where", "address", "meet", "late", "here"]
            rel = [
                m
                for m in messages[-10:]
                if m.text and any(w in m.text.lower() for w in logistics_words)
            ]
            return list(dict.fromkeys(rel + messages[-3:]))
        if topic in (ThreadTopic.PLANNING, ThreadTopic.DECISION_MAKING):
            return messages[-7:]
        return messages[-5:]

    def _extract_action_items(self, messages: list[Message]) -> list[str]:
        pats = [
            r"(?:I'll|i'll|I will|i will)\s+(.+?)(?:\.|$)",
            r"(?:can you|could you)\s+(.+?)(?:\?|$)",
            r"(?:don't forget to|remember to)\s+(.+?)(?:\.|$)",
            r"(?:please|pls)\s+(.+?)(?:\.|$)",
            r"(?:need to|have to)\s+(.+?)(?:\.|$)",
        ]
        items = []
        for m in messages[-10:]:
            if not m.text:
                continue
            for p in pats:
                for match in re.findall(p, m.text, re.I):
                    s = match.strip()
                    if 5 < len(s) < 100:
                        items.append(s)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_items: list[str] = []
        for x in items:
            lower_x = x.lower()
            if lower_x not in seen:
                seen.add(lower_x)
                unique_items.append(x)
        return unique_items[:5]

    def get_response_config(self, context: ThreadContext) -> ThreadedReplyConfig:
        """Get recommended response configuration."""
        return TOPIC_RESPONSE_CONFIG.get(context.topic, TOPIC_RESPONSE_CONFIG[ThreadTopic.UNKNOWN])

    def clear_cache(self) -> None:
        """Clear cached data."""
        with self._lock:
            self._embeddings_cache.clear()
            self._topic_embeddings = None
            logger.debug("Thread analyzer cache cleared")


_analyzer: ThreadAnalyzer | None = None
_analyzer_lock = threading.Lock()


def get_thread_analyzer(config: ThreadingConfig | None = None) -> ThreadAnalyzer:
    """Get singleton ThreadAnalyzer."""
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                _analyzer = ThreadAnalyzer(config)
    return _analyzer


def reset_thread_analyzer() -> None:
    """Reset singleton."""
    global _analyzer
    with _analyzer_lock:
        if _analyzer:
            _analyzer.clear_cache()
        _analyzer = None
    logger.debug("Thread analyzer singleton reset")
