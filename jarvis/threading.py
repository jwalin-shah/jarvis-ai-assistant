"""Conversation Threading System for JARVIS.

Analyzes messages to group them into logical conversation threads using:
- Semantic similarity between consecutive messages
- Time gaps between messages
- Topic shifts detected via embeddings
- Explicit reply references (reply_to_id)

Thread detection helps users understand conversation flow and navigate
long conversations more effectively.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from contracts.imessage import Message

logger = logging.getLogger(__name__)


class ThreadingMethod(Enum):
    """Methods used to determine thread boundaries."""

    REPLY_REFERENCE = "reply_reference"  # Explicit reply_to_id link
    SEMANTIC_SIMILARITY = "semantic_similarity"  # Similar message content
    TIME_GAP = "time_gap"  # Long pause between messages
    TOPIC_SHIFT = "topic_shift"  # Detected topic change


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


class ThreadAnalyzer:
    """Analyzes messages to group them into logical conversation threads.

    Uses a combination of time-based heuristics, semantic similarity,
    and explicit reply references to identify thread boundaries.

    Thread-safe with lazy initialization of the sentence model.

    Example:
        >>> analyzer = ThreadAnalyzer()
        >>> threads = analyzer.analyze_threads(messages)
        >>> for thread in threads:
        ...     print(f"Thread {thread.thread_id}: {thread.topic_label}")
    """

    def __init__(self, config: ThreadingConfig | None = None) -> None:
        """Initialize the thread analyzer.

        Args:
            config: Threading configuration options
        """
        self.config = config or ThreadingConfig()
        self._sentence_model: Any | None = None
        self._lock = threading.Lock()
        self._embeddings_cache: dict[str, np.ndarray] = {}

    def _get_sentence_model(self) -> Any:
        """Get the sentence transformer model with lazy loading.

        Reuses the model from models.templates to avoid loading it twice.

        Returns:
            The loaded SentenceTransformer model

        Raises:
            Exception: If model cannot be loaded
        """
        if self._sentence_model is not None:
            return self._sentence_model

        with self._lock:
            if self._sentence_model is not None:
                return self._sentence_model

            try:
                from models.templates import _get_sentence_model

                self._sentence_model = _get_sentence_model()
                logger.info("Loaded sentence model for thread analysis")
                return self._sentence_model
            except Exception:
                logger.warning("Failed to load sentence model for threading")
                raise

    def _compute_embedding(self, text: str) -> np.ndarray | None:
        """Compute embedding for a text string with caching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if failed
        """
        if not text or not text.strip():
            return None

        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embeddings_cache:
            return self._embeddings_cache[cache_key]

        try:
            model = self._get_sentence_model()
            embedding = model.encode([text], convert_to_numpy=True)[0]
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            self._embeddings_cache[cache_key] = embedding
            return embedding
        except Exception:
            return None

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0-1)
        """
        emb1 = self._compute_embedding(text1)
        emb2 = self._compute_embedding(text2)

        if emb1 is None or emb2 is None:
            return 0.0

        # Cosine similarity with normalized vectors is just dot product
        similarity = float(np.dot(emb1, emb2))
        return max(0.0, min(1.0, similarity))

    def _generate_thread_id(self, chat_id: str, start_message_id: int) -> str:
        """Generate a unique thread ID.

        Args:
            chat_id: The conversation ID
            start_message_id: ID of the first message in thread

        Returns:
            Unique thread identifier
        """
        # Create a stable hash-based ID
        data = f"{chat_id}:{start_message_id}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _detect_topic_label(self, messages: list[Message]) -> str:
        """Generate a topic label for a thread based on message content.

        Uses keyword extraction to identify the main topic.

        Args:
            messages: Messages in the thread

        Returns:
            Topic label string
        """
        if not messages:
            return "General"

        # Combine message texts
        texts = [m.text for m in messages if m.text and not m.is_system_message]
        if not texts:
            return "General"

        combined = " ".join(texts[:10])  # Use first 10 messages

        # Common topic patterns
        topic_patterns = [
            ("meeting", "Meeting Plans"),
            ("dinner", "Dinner Plans"),
            ("lunch", "Lunch Plans"),
            ("call", "Phone Call"),
            ("zoom", "Video Call"),
            ("tomorrow", "Future Plans"),
            ("tonight", "Tonight's Plans"),
            ("weekend", "Weekend Plans"),
            ("party", "Party/Event"),
            ("birthday", "Birthday"),
            ("work", "Work Discussion"),
            ("project", "Project Discussion"),
            ("question", "Q&A"),
            ("help", "Help Request"),
            ("thanks", "Gratitude"),
            ("sorry", "Apology"),
            ("photo", "Photos"),
            ("video", "Videos"),
            ("link", "Shared Links"),
            ("address", "Location/Address"),
            ("time", "Scheduling"),
            ("money", "Financial"),
            ("buy", "Shopping"),
            ("order", "Orders"),
            ("flight", "Travel"),
            ("hotel", "Travel"),
            ("trip", "Travel"),
        ]

        combined_lower = combined.lower()
        for keyword, label in topic_patterns:
            if keyword in combined_lower:
                return label

        # Default: use first few words of first message
        first_text = texts[0][:50] if texts else "General"
        if len(first_text) > 30:
            first_text = first_text[:30] + "..."
        return first_text

    def _should_start_new_thread(
        self,
        current: Message,
        previous: Message,
        current_thread_start: datetime,
    ) -> tuple[bool, ThreadingMethod]:
        """Determine if current message should start a new thread.

        Args:
            current: Current message being processed
            previous: Previous message
            current_thread_start: When the current thread started

        Returns:
            Tuple of (should_start_new, reason)
        """
        # Check for explicit reply reference to a different thread
        if current.reply_to_id is not None:
            # This is part of an existing thread via reply
            return False, ThreadingMethod.REPLY_REFERENCE

        # Check time gap
        time_diff = current.date - previous.date
        gap_threshold = timedelta(minutes=self.config.time_gap_threshold_minutes)

        if time_diff > gap_threshold:
            logger.debug(
                "New thread due to time gap: %s minutes",
                time_diff.total_seconds() / 60,
            )
            return True, ThreadingMethod.TIME_GAP

        # Check max thread duration
        thread_duration = current.date - current_thread_start
        max_duration = timedelta(hours=self.config.max_thread_duration_hours)

        if thread_duration > max_duration:
            logger.debug(
                "New thread due to max duration: %s hours",
                thread_duration.total_seconds() / 3600,
            )
            return True, ThreadingMethod.TIME_GAP

        # Check semantic similarity if enabled
        if self.config.use_semantic_analysis:
            try:
                similarity = self._compute_similarity(previous.text, current.text)
                if similarity < self.config.semantic_similarity_threshold:
                    logger.debug(
                        "New thread due to topic shift: similarity %.3f",
                        similarity,
                    )
                    return True, ThreadingMethod.TOPIC_SHIFT
            except Exception:
                # Fall back to time-based only if semantic analysis fails
                pass

        return False, ThreadingMethod.SEMANTIC_SIMILARITY

    def analyze_threads(
        self,
        messages: list[Message],
        chat_id: str = "",
    ) -> list[Thread]:
        """Analyze messages and group them into threads.

        Args:
            messages: List of messages sorted by date (oldest first)
            chat_id: The conversation ID for thread ID generation

        Returns:
            List of Thread objects with message groupings
        """
        if not messages:
            return []

        # Sort messages by date to ensure correct ordering
        sorted_messages = sorted(messages, key=lambda m: m.date)

        threads: list[Thread] = []
        current_thread_messages: list[Message] = []
        current_thread_start: datetime | None = None
        reply_threads: dict[int, str] = {}  # message_id -> thread_id mapping

        for i, message in enumerate(sorted_messages):
            is_first = i == 0

            # Check for reply reference
            if message.reply_to_id is not None and message.reply_to_id in reply_threads:
                # Find the thread this message replies to
                target_thread_id = reply_threads[message.reply_to_id]

                # Find the thread and add this message
                for thread in threads:
                    if thread.thread_id == target_thread_id:
                        thread.messages.append(message.id)
                        thread.message_count += 1
                        thread.end_time = message.date
                        reply_threads[message.id] = target_thread_id
                        break
                continue

            if is_first:
                # Start first thread
                current_thread_messages = [message]
                current_thread_start = message.date
            else:
                should_start, reason = self._should_start_new_thread(
                    message,
                    sorted_messages[i - 1],
                    current_thread_start or message.date,
                )

                if should_start:
                    # Finalize current thread
                    if current_thread_messages:
                        thread = self._create_thread(
                            current_thread_messages, chat_id, reply_threads
                        )
                        threads.append(thread)

                    # Start new thread
                    current_thread_messages = [message]
                    current_thread_start = message.date
                else:
                    # Add to current thread
                    current_thread_messages.append(message)

            # Track message for reply threading
            if current_thread_messages and len(current_thread_messages) == 1:
                # Will be assigned thread_id when thread is finalized
                pass

        # Finalize last thread
        if current_thread_messages:
            thread = self._create_thread(current_thread_messages, chat_id, reply_threads)
            threads.append(thread)

        logger.info(
            "Analyzed %d messages into %d threads",
            len(messages),
            len(threads),
        )

        return threads

    def _create_thread(
        self,
        messages: list[Message],
        chat_id: str,
        reply_threads: dict[int, str],
    ) -> Thread:
        """Create a Thread object from a list of messages.

        Args:
            messages: Messages in the thread
            chat_id: Conversation ID
            reply_threads: Mapping to update with thread ID for replies

        Returns:
            Thread object
        """
        thread_id = self._generate_thread_id(
            chat_id, messages[0].id if messages else 0
        )

        # Track all message IDs for reply threading
        for msg in messages:
            reply_threads[msg.id] = thread_id

        # Get unique participants
        participants = set()
        for msg in messages:
            participants.add(msg.sender)

        thread = Thread(
            thread_id=thread_id,
            messages=[m.id for m in messages],
            topic_label=self._detect_topic_label(messages),
            start_time=messages[0].date if messages else None,
            end_time=messages[-1].date if messages else None,
            participant_count=len(participants),
            message_count=len(messages),
        )

        return thread

    def get_threaded_messages(
        self,
        messages: list[Message],
        chat_id: str = "",
    ) -> list[ThreadedMessage]:
        """Get messages with thread information attached.

        Args:
            messages: List of messages
            chat_id: Conversation ID

        Returns:
            List of ThreadedMessage objects
        """
        threads = self.analyze_threads(messages, chat_id)

        # Build message ID to thread mapping
        message_to_thread: dict[int, Thread] = {}
        for thread in threads:
            for msg_id in thread.messages:
                message_to_thread[msg_id] = thread

        # Create ThreadedMessage objects
        threaded_messages: list[ThreadedMessage] = []
        for message in messages:
            thread = message_to_thread.get(message.id)
            if thread:
                position = thread.messages.index(message.id)
                threaded_msg = ThreadedMessage(
                    message=message,
                    thread_id=thread.thread_id,
                    thread_position=position,
                    is_thread_start=position == 0,
                )
                threaded_messages.append(threaded_msg)

        return threaded_messages

    def clear_cache(self) -> None:
        """Clear the embeddings cache."""
        with self._lock:
            self._embeddings_cache.clear()
            logger.debug("Thread analyzer cache cleared")


# Module-level singleton instance
_analyzer: ThreadAnalyzer | None = None
_analyzer_lock = threading.Lock()


def get_thread_analyzer(config: ThreadingConfig | None = None) -> ThreadAnalyzer:
    """Get the singleton ThreadAnalyzer instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        The shared ThreadAnalyzer instance
    """
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                _analyzer = ThreadAnalyzer(config)
    return _analyzer


def reset_thread_analyzer() -> None:
    """Reset the singleton ThreadAnalyzer instance.

    Call this to create a fresh analyzer on next access.
    """
    global _analyzer
    with _analyzer_lock:
        if _analyzer is not None:
            _analyzer.clear_cache()
        _analyzer = None
    logger.debug("Thread analyzer singleton reset")
