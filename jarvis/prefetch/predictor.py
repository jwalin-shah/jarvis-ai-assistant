"""Prediction strategies for speculative prefetching.

Predicts what the user will need next based on various signals:
- Contact frequency (prefetch likely contacts)
- Time-of-day patterns (prefetch morning routine)
- Conversation continuation (prefetch follow-ups)
- Recent message context (prefetch related content)
- Access patterns (prefetch frequently accessed)

Usage:
    predictor = PrefetchPredictor()
    predictions = predictor.predict_next()
    for pred in predictions:
        executor.schedule_prefetch(pred)
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Path to iMessage database (read-only access)
CHAT_DB_PATH = Path.home() / "Library" / "Messages" / "chat.db"

# Apple epoch offset (2001-01-01 in Unix timestamp)
APPLE_EPOCH_OFFSET = 978307200


class PredictionType(str, Enum):
    """Types of predictions."""

    DRAFT_REPLY = "draft_reply"  # Pre-generate draft for likely conversation
    EMBEDDING = "embedding"  # Pre-compute embeddings for predicted messages
    CONTACT_PROFILE = "contact_profile"  # Pre-load contact data
    SEARCH_RESULTS = "search_results"  # Pre-compute common search results
    MODEL_WARM = "model_warm"  # Warm up model weights
    VEC_INDEX = "vec_index"  # Pre-load vec search index


class PredictionPriority(int, Enum):
    """Priority levels for predictions."""

    CRITICAL = 100  # Must prefetch immediately
    HIGH = 75  # Should prefetch soon
    MEDIUM = 50  # Prefetch if resources available
    LOW = 25  # Prefetch opportunistically
    BACKGROUND = 10  # Only when idle


@dataclass
class Prediction:
    """A prediction of what to prefetch."""

    type: PredictionType
    priority: PredictionPriority
    confidence: float  # 0-1 confidence score
    key: str  # Cache key for the prediction
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""  # Human-readable reason
    ttl_seconds: float = 300.0  # How long the prefetched data should be cached
    tags: list[str] = field(default_factory=list)  # Tags for invalidation
    estimated_cost_ms: int = 100  # Estimated prefetch cost in milliseconds

    @property
    def score(self) -> float:
        """Combined score for prioritization."""
        return self.priority.value * self.confidence


@dataclass
class AccessPattern:
    """Tracks access patterns for a specific key/type."""

    key: str
    access_times: list[float] = field(default_factory=list)
    access_count: int = 0
    last_access: float = 0.0

    def record_access(self) -> None:
        """Record an access."""
        now = time.time()
        self.access_times.append(now)
        self.access_count += 1
        self.last_access = now
        # Keep only last 100 access times
        if len(self.access_times) > 100:
            self.access_times = self.access_times[-100:]

    @property
    def frequency(self) -> float:
        """Calculate access frequency (accesses per hour)."""
        if len(self.access_times) < 2:
            return 0.0
        duration = self.access_times[-1] - self.access_times[0]
        if duration < 1:
            return 0.0
        return (len(self.access_times) - 1) / (duration / 3600)


class PredictionStrategy(ABC):
    """Base class for prediction strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""

    @abstractmethod
    def predict(self, context: PredictionContext) -> list[Prediction]:
        """Generate predictions.

        Args:
            context: Current prediction context.

        Returns:
            List of predictions sorted by score (highest first).
        """


@dataclass
class PredictionContext:
    """Context for generating predictions."""

    current_time: datetime = field(default_factory=datetime.now)
    current_hour: int = field(default_factory=lambda: datetime.now().hour)
    current_day_of_week: int = field(default_factory=lambda: datetime.now().weekday())
    active_chat_id: str | None = None
    recent_chat_ids: list[str] = field(default_factory=list)
    recent_searches: list[str] = field(default_factory=list)
    ui_focus: str | None = None  # What UI element is focused
    battery_level: float = 1.0  # 0-1, for power-aware prefetching
    memory_available_mb: int = 1000  # Available memory


class ContactFrequencyStrategy(PredictionStrategy):
    """Predict based on contact message frequency.

    Users tend to respond to frequent contacts quickly, so
    prefetch drafts for high-frequency conversations.
    """

    @property
    def name(self) -> str:
        return "contact_frequency"

    def __init__(self, lookback_days: int = 7, top_n: int = 5) -> None:
        self._lookback_days = lookback_days
        self._top_n = top_n
        self._contact_scores: dict[str, float] = {}
        self._last_update: float = 0.0
        self._update_interval = 3600  # 1 hour

    def predict(self, context: PredictionContext) -> list[Prediction]:
        predictions: list[Prediction] = []

        # Update contact scores if stale
        if time.time() - self._last_update > self._update_interval:
            self._update_contact_scores()

        # Get top contacts not in recent chats (those are already hot)
        recent_set = set(context.recent_chat_ids[:3])  # Exclude very recent
        candidates = [
            (chat_id, score)
            for chat_id, score in self._contact_scores.items()
            if chat_id not in recent_set
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        for chat_id, score in candidates[: self._top_n]:
            # Normalize score to 0-1
            confidence = min(score / 100, 1.0)

            predictions.append(
                Prediction(
                    type=PredictionType.DRAFT_REPLY,
                    priority=PredictionPriority.MEDIUM,
                    confidence=confidence,
                    key=f"draft:{chat_id}",
                    params={"chat_id": chat_id},
                    reason=f"High frequency contact (score: {score:.1f})",
                    ttl_seconds=600,  # 10 minutes
                    tags=[f"chat:{chat_id}", "draft"],
                    estimated_cost_ms=500,
                )
            )

            # Also prefetch contact profile
            predictions.append(
                Prediction(
                    type=PredictionType.CONTACT_PROFILE,
                    priority=PredictionPriority.LOW,
                    confidence=confidence * 0.8,
                    key=f"contact:{chat_id}",
                    params={"chat_id": chat_id},
                    reason="Contact profile for frequent contact",
                    ttl_seconds=1800,  # 30 minutes
                    tags=[f"chat:{chat_id}", "contact"],
                    estimated_cost_ms=50,
                )
            )

        return predictions

    def _update_contact_scores(self) -> None:
        """Update contact frequency scores from chat.db."""
        if not CHAT_DB_PATH.exists():
            return

        conn = None
        try:
            conn = sqlite3.connect(
                f"file:{CHAT_DB_PATH}?mode=ro",
                uri=True,
                timeout=5.0,
            )
            conn.row_factory = sqlite3.Row

            # Calculate cutoff timestamp
            cutoff = datetime.now() - timedelta(days=self._lookback_days)
            cutoff_ns = int((cutoff.timestamp() - APPLE_EPOCH_OFFSET) * 1_000_000_000)

            cursor = conn.execute(
                """
                SELECT
                    chat.guid as chat_id,
                    COUNT(*) as msg_count,
                    MAX(message.date) as last_msg
                FROM message
                JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
                JOIN chat ON chat_message_join.chat_id = chat.ROWID
                WHERE message.date > ?
                  AND message.is_from_me = 0
                GROUP BY chat.guid
                ORDER BY msg_count DESC
                LIMIT 50
                """,
                (cutoff_ns,),
            )

            self._contact_scores.clear()
            max_count = 1  # Avoid division by zero

            for row in cursor.fetchall():
                chat_id = row["chat_id"]
                msg_count = row["msg_count"]
                max_count = max(max_count, msg_count)
                # Score = message count normalized, with recency boost
                last_msg_ns = row["last_msg"]
                if last_msg_ns:
                    last_msg_ts = (last_msg_ns / 1_000_000_000) + APPLE_EPOCH_OFFSET
                    hours_ago = (time.time() - last_msg_ts) / 3600
                    recency_boost = max(0, 1 - hours_ago / 24)  # Full boost within 24h
                else:
                    recency_boost = 0

                self._contact_scores[chat_id] = (msg_count / max_count) * 50 + recency_boost * 50

            self._last_update = time.time()

        except Exception as e:
            logger.debug(f"Error updating contact scores: {e}")
        finally:
            if conn is not None:
                conn.close()


class TimeOfDayStrategy(PredictionStrategy):
    """Predict based on time-of-day patterns.

    Learn when users typically interact with specific contacts
    and prefetch during those times.
    """

    @property
    def name(self) -> str:
        return "time_of_day"

    def __init__(self) -> None:
        # hour -> list of (chat_id, count)
        self._hourly_patterns: dict[int, Counter[str]] = defaultdict(Counter)
        self._last_update: float = 0.0
        self._update_interval = 3600  # 1 hour

    def predict(self, context: PredictionContext) -> list[Prediction]:
        predictions: list[Prediction] = []

        # Update patterns if stale
        if time.time() - self._last_update > self._update_interval:
            self._update_patterns()

        current_hour = context.current_hour
        # Also check adjacent hours for fuzzy matching
        hours_to_check = [current_hour, (current_hour + 1) % 24, (current_hour - 1) % 24]

        chat_scores: Counter[str] = Counter()
        for hour in hours_to_check:
            weight = 1.0 if hour == current_hour else 0.5
            for chat_id, count in self._hourly_patterns[hour].items():
                chat_scores[chat_id] += count * weight

        # Get top candidates
        for chat_id, score in chat_scores.most_common(3):
            if score < 2:  # Minimum threshold
                continue

            confidence = min(score / 20, 0.9)  # Cap at 0.9

            predictions.append(
                Prediction(
                    type=PredictionType.DRAFT_REPLY,
                    priority=PredictionPriority.MEDIUM,
                    confidence=confidence,
                    key=f"draft:tod:{chat_id}",
                    params={"chat_id": chat_id},
                    reason=f"Time-of-day pattern (hour={current_hour}, score={score:.1f})",
                    ttl_seconds=1800,  # 30 minutes
                    tags=[f"chat:{chat_id}", "draft", "tod"],
                    estimated_cost_ms=500,
                )
            )

        return predictions

    def _update_patterns(self) -> None:
        """Update time-of-day patterns from chat.db."""
        if not CHAT_DB_PATH.exists():
            return

        conn = None
        try:
            conn = sqlite3.connect(
                f"file:{CHAT_DB_PATH}?mode=ro",
                uri=True,
                timeout=5.0,
            )
            conn.row_factory = sqlite3.Row

            # Look at last 30 days
            cutoff = datetime.now() - timedelta(days=30)
            cutoff_ns = int((cutoff.timestamp() - APPLE_EPOCH_OFFSET) * 1_000_000_000)

            cursor = conn.execute(
                """
                SELECT
                    chat.guid as chat_id,
                    message.date as msg_date
                FROM message
                JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
                JOIN chat ON chat_message_join.chat_id = chat.ROWID
                WHERE message.date > ?
                  AND message.is_from_me = 0
                ORDER BY message.date DESC
                LIMIT 5000
                """,
                (cutoff_ns,),
            )

            self._hourly_patterns.clear()
            for row in cursor.fetchall():
                chat_id = row["chat_id"]
                msg_date_ns = row["msg_date"]
                if msg_date_ns:
                    unix_ts = (msg_date_ns / 1_000_000_000) + APPLE_EPOCH_OFFSET
                    hour = datetime.fromtimestamp(unix_ts).hour
                    self._hourly_patterns[hour][chat_id] += 1

            self._last_update = time.time()

        except Exception as e:
            logger.debug(f"Error updating time-of-day patterns: {e}")
        finally:
            if conn is not None:
                conn.close()


class ConversationContinuationStrategy(PredictionStrategy):
    """Predict based on conversation continuation patterns.

    When a user receives a message, they often respond within
    a few minutes. Prefetch drafts for active conversations.
    """

    @property
    def name(self) -> str:
        return "conversation_continuation"

    def __init__(
        self,
        active_window_minutes: int = 30,
        response_probability_threshold: float = 0.3,
        max_tracked_chats: int = 1000,
    ) -> None:
        self._active_window = active_window_minutes * 60  # Convert to seconds
        self._threshold = response_probability_threshold
        self._max_tracked_chats = max_tracked_chats  # Prevent unbounded memory growth
        self._recent_messages: dict[str, list[float]] = defaultdict(list)  # chat_id -> timestamps

    def record_message(self, chat_id: str, is_from_me: bool) -> None:
        """Record an incoming or outgoing message.

        Args:
            chat_id: Chat identifier.
            is_from_me: Whether the message was sent by user.
        """
        now = time.time()
        self._recent_messages[chat_id].append(now)
        # Keep only recent messages
        cutoff = now - self._active_window
        self._recent_messages[chat_id] = [
            ts for ts in self._recent_messages[chat_id] if ts > cutoff
        ]

    def predict(self, context: PredictionContext) -> list[Prediction]:
        predictions: list[Prediction] = []
        now = time.time()

        # Cleanup stale chat entries to prevent unbounded growth
        cutoff = now - self._active_window
        stale_keys = [
            cid for cid, ts in self._recent_messages.items() if not ts or max(ts) < cutoff
        ]
        for cid in stale_keys:
            del self._recent_messages[cid]

        # Enforce max tracked chats limit - remove oldest inactive chats
        if len(self._recent_messages) > self._max_tracked_chats:
            # Sort by most recent message timestamp, keep most active chats
            sorted_chats = sorted(
                self._recent_messages.items(),
                key=lambda x: max(x[1]) if x[1] else 0,
                reverse=True,
            )
            # Keep only the most recent max_tracked_chats
            chats_to_keep = {cid for cid, _ in sorted_chats[: self._max_tracked_chats]}
            chats_to_remove = [cid for cid in self._recent_messages if cid not in chats_to_keep]
            for cid in chats_to_remove:
                del self._recent_messages[cid]

        for chat_id, timestamps in self._recent_messages.items():
            if not timestamps:
                continue

            # Calculate activity score based on recency and frequency
            cutoff = now - self._active_window
            recent = [ts for ts in timestamps if ts > cutoff]

            if not recent:
                continue

            # More recent = higher score
            most_recent = max(recent)
            recency_score = 1 - (now - most_recent) / self._active_window

            # More messages = higher score
            frequency_score = min(len(recent) / 10, 1.0)

            confidence = recency_score * 0.7 + frequency_score * 0.3

            if confidence < self._threshold:
                continue

            # Very recent messages get higher priority
            if now - most_recent < 300:  # Within 5 minutes
                priority = PredictionPriority.HIGH
            elif now - most_recent < 900:  # Within 15 minutes
                priority = PredictionPriority.MEDIUM
            else:
                priority = PredictionPriority.LOW

            predictions.append(
                Prediction(
                    type=PredictionType.DRAFT_REPLY,
                    priority=priority,
                    confidence=confidence,
                    key=f"draft:cont:{chat_id}",
                    params={"chat_id": chat_id},
                    reason=f"Active conversation (recency={recency_score:.2f})",
                    ttl_seconds=300,  # 5 minutes (short for active conversations)
                    tags=[f"chat:{chat_id}", "draft", "active"],
                    estimated_cost_ms=500,
                )
            )

        return predictions


class RecentContextStrategy(PredictionStrategy):
    """Predict based on recent message context.

    Analyze recent messages to predict related content that
    might be needed (embeddings, search results, etc.).
    """

    @property
    def name(self) -> str:
        return "recent_context"

    def __init__(self) -> None:
        self._recent_contexts: dict[str, list[str]] = {}  # chat_id -> recent texts

    def record_context(self, chat_id: str, text: str) -> None:
        """Record message text for context analysis.

        Args:
            chat_id: Chat identifier.
            text: Message text.
        """
        if chat_id not in self._recent_contexts:
            self._recent_contexts[chat_id] = []
        self._recent_contexts[chat_id].append(text)
        # Keep only last 10 messages
        self._recent_contexts[chat_id] = self._recent_contexts[chat_id][-10:]

    def predict(self, context: PredictionContext) -> list[Prediction]:
        predictions: list[Prediction] = []

        # Focus on active chat if any
        if context.active_chat_id and context.active_chat_id in self._recent_contexts:
            recent_texts = self._recent_contexts[context.active_chat_id]
            if recent_texts:
                # Predict embedding for combined context (last 3 messages)
                predictions.append(
                    Prediction(
                        type=PredictionType.EMBEDDING,
                        priority=PredictionPriority.HIGH,
                        confidence=0.9,
                        key=f"embed:ctx:{context.active_chat_id}",
                        params={
                            "chat_id": context.active_chat_id,
                            "texts": recent_texts[-3:],
                        },
                        reason="Active conversation context",
                        ttl_seconds=180,  # 3 minutes
                        tags=[f"chat:{context.active_chat_id}", "embedding"],
                        estimated_cost_ms=100,
                    )
                )

        # Also predict for recent chats
        for chat_id in context.recent_chat_ids[:3]:
            if chat_id in self._recent_contexts:
                recent_texts = self._recent_contexts[chat_id]
                if recent_texts:
                    predictions.append(
                        Prediction(
                            type=PredictionType.EMBEDDING,
                            priority=PredictionPriority.MEDIUM,
                            confidence=0.7,
                            key=f"embed:ctx:{chat_id}",
                            params={
                                "chat_id": chat_id,
                                "texts": recent_texts[-3:],
                            },
                            reason="Recent conversation context",
                            ttl_seconds=300,  # 5 minutes
                            tags=[f"chat:{chat_id}", "embedding"],
                            estimated_cost_ms=100,
                        )
                    )

        return predictions


class UIFocusStrategy(PredictionStrategy):
    """Predict based on UI focus/hover events.

    When user focuses on a conversation in the UI, prefetch
    draft replies before they start typing.
    """

    @property
    def name(self) -> str:
        return "ui_focus"

    def __init__(self, focus_threshold_ms: int = 500) -> None:
        self._focus_threshold = focus_threshold_ms / 1000
        self._focus_times: dict[str, float] = {}  # chat_id -> focus start time
        self._hover_times: dict[str, float] = {}  # chat_id -> hover start time

    def record_focus(self, chat_id: str) -> None:
        """Record focus on a chat.

        Args:
            chat_id: Chat identifier.
        """
        self._focus_times[chat_id] = time.time()

    def record_hover(self, chat_id: str) -> None:
        """Record hover over a chat.

        Args:
            chat_id: Chat identifier.
        """
        self._hover_times[chat_id] = time.time()

    def predict(self, context: PredictionContext) -> list[Prediction]:
        predictions: list[Prediction] = []
        now = time.time()

        # Check focused chat
        if context.active_chat_id:
            focus_time = self._focus_times.get(context.active_chat_id)
            if focus_time and (now - focus_time) >= self._focus_threshold:
                predictions.append(
                    Prediction(
                        type=PredictionType.DRAFT_REPLY,
                        priority=PredictionPriority.CRITICAL,  # User is actively looking
                        confidence=0.95,
                        key=f"draft:focus:{context.active_chat_id}",
                        params={"chat_id": context.active_chat_id},
                        reason="User focused on chat",
                        ttl_seconds=120,  # 2 minutes
                        tags=[f"chat:{context.active_chat_id}", "draft", "focus"],
                        estimated_cost_ms=500,
                    )
                )

        # Check hovered chats
        for chat_id, hover_time in self._hover_times.items():
            if chat_id == context.active_chat_id:
                continue  # Already handled above

            if (now - hover_time) < 5.0:  # Recent hover
                predictions.append(
                    Prediction(
                        type=PredictionType.DRAFT_REPLY,
                        priority=PredictionPriority.HIGH,
                        confidence=0.7,
                        key=f"draft:hover:{chat_id}",
                        params={"chat_id": chat_id},
                        reason="User hovered over chat",
                        ttl_seconds=180,  # 3 minutes
                        tags=[f"chat:{chat_id}", "draft", "hover"],
                        estimated_cost_ms=500,
                    )
                )

        return predictions


class ModelWarmingStrategy(PredictionStrategy):
    """Predict need for model warming.

    Keep models warm during active usage periods to
    avoid cold start latency.
    """

    @property
    def name(self) -> str:
        return "model_warming"

    def __init__(self, idle_threshold_seconds: int = 300) -> None:
        self._idle_threshold = idle_threshold_seconds
        self._last_activity: float = time.time()
        self._models_warm: set[str] = set()

    def record_activity(self) -> None:
        """Record user activity."""
        self._last_activity = time.time()

    def predict(self, context: PredictionContext) -> list[Prediction]:
        predictions: list[Prediction] = []
        now = time.time()

        # Check if we're in active period
        idle_time = now - self._last_activity
        if idle_time > self._idle_threshold:
            return []  # Too idle, don't waste resources

        # Predict LLM warming if not warm
        if "llm" not in self._models_warm:
            predictions.append(
                Prediction(
                    type=PredictionType.MODEL_WARM,
                    priority=PredictionPriority.HIGH,
                    confidence=0.95,
                    key="warm:llm",
                    params={"model_type": "llm"},
                    reason="Keep LLM warm during active period",
                    ttl_seconds=600,  # 10 minutes
                    tags=["model", "llm"],
                    estimated_cost_ms=2000,  # LLM warmup is expensive
                )
            )

        # Predict embedding model warming
        if "embeddings" not in self._models_warm:
            predictions.append(
                Prediction(
                    type=PredictionType.MODEL_WARM,
                    priority=PredictionPriority.MEDIUM,
                    confidence=0.9,
                    key="warm:embeddings",
                    params={"model_type": "embeddings"},
                    reason="Keep embedding model warm",
                    ttl_seconds=600,
                    tags=["model", "embeddings"],
                    estimated_cost_ms=500,
                )
            )

        # Battery-aware: skip expensive warming on low battery
        if context.battery_level < 0.2:
            return [p for p in predictions if p.estimated_cost_ms < 500]

        return predictions

    def mark_warm(self, model_type: str) -> None:
        """Mark a model as warm."""
        self._models_warm.add(model_type)

    def mark_cold(self, model_type: str) -> None:
        """Mark a model as cold."""
        self._models_warm.discard(model_type)


class PrefetchPredictor:
    """Main predictor that combines multiple strategies.

    Usage:
        predictor = PrefetchPredictor()
        predictor.record_message("chat123", "Hello", is_from_me=False)
        predictions = predictor.predict()
        for pred in predictions:
            if pred.score > 50:
                executor.schedule(pred)
    """

    def __init__(
        self,
        max_predictions: int = 20,
        min_confidence: float = 0.3,
    ) -> None:
        """Initialize predictor with default strategies.

        Args:
            max_predictions: Maximum predictions to return.
            min_confidence: Minimum confidence threshold.
        """
        self._max_predictions = max_predictions
        self._min_confidence = min_confidence
        self._lock = threading.RLock()

        # Initialize strategies
        self._strategies: list[PredictionStrategy] = [
            ContactFrequencyStrategy(),
            TimeOfDayStrategy(),
            ConversationContinuationStrategy(),
            RecentContextStrategy(),
            UIFocusStrategy(),
            ModelWarmingStrategy(),
        ]

        # Access pattern tracking
        self._access_patterns: dict[str, AccessPattern] = {}
        self._context = PredictionContext()

    def predict(self, context: PredictionContext | None = None) -> list[Prediction]:
        """Generate predictions from all strategies.

        Args:
            context: Optional prediction context. Uses default if not provided.

        Returns:
            List of predictions sorted by score (highest first).
        """
        with self._lock:
            ctx = context or self._context
            all_predictions: list[Prediction] = []

            for strategy in self._strategies:
                try:
                    preds = strategy.predict(ctx)
                    all_predictions.extend(preds)
                except Exception as e:
                    logger.warning(f"Strategy {strategy.name} failed: {e}")

            # Filter by confidence
            all_predictions = [p for p in all_predictions if p.confidence >= self._min_confidence]

            # Deduplicate by key (keep highest score)
            seen_keys: dict[str, Prediction] = {}
            for pred in all_predictions:
                if pred.key not in seen_keys or pred.score > seen_keys[pred.key].score:
                    seen_keys[pred.key] = pred

            # Deduplicate DRAFT_REPLY by chat_id (keep highest score per chat)
            seen_drafts: dict[str, Prediction] = {}
            non_drafts: list[Prediction] = []
            for pred in seen_keys.values():
                if pred.type == PredictionType.DRAFT_REPLY:
                    cid = pred.params.get("chat_id", "")
                    if cid not in seen_drafts or pred.score > seen_drafts[cid].score:
                        seen_drafts[cid] = pred
                else:
                    non_drafts.append(pred)

            deduped = non_drafts + list(seen_drafts.values())

            # Sort by score
            final_predictions = sorted(deduped, key=lambda p: p.score, reverse=True)

            return final_predictions[: self._max_predictions]

    def record_message(self, chat_id: str, text: str, is_from_me: bool) -> None:
        """Record a message for prediction context.

        Args:
            chat_id: Chat identifier.
            text: Message text.
            is_from_me: Whether message was from user.
        """
        with self._lock:
            # Update conversation continuation strategy
            for strategy in self._strategies:
                if isinstance(strategy, ConversationContinuationStrategy):
                    strategy.record_message(chat_id, is_from_me)
                elif isinstance(strategy, RecentContextStrategy):
                    strategy.record_context(chat_id, text)
                elif isinstance(strategy, ModelWarmingStrategy):
                    strategy.record_activity()

            # Update recent chat IDs
            if chat_id in self._context.recent_chat_ids:
                self._context.recent_chat_ids.remove(chat_id)
            self._context.recent_chat_ids.insert(0, chat_id)
            self._context.recent_chat_ids = self._context.recent_chat_ids[:20]

    def record_focus(self, chat_id: str) -> None:
        """Record UI focus on a chat.

        Args:
            chat_id: Chat identifier.
        """
        with self._lock:
            for strategy in self._strategies:
                if isinstance(strategy, UIFocusStrategy):
                    strategy.record_focus(chat_id)
            self._context.active_chat_id = chat_id

    def record_hover(self, chat_id: str) -> None:
        """Record UI hover on a chat.

        Args:
            chat_id: Chat identifier.
        """
        with self._lock:
            for strategy in self._strategies:
                if isinstance(strategy, UIFocusStrategy):
                    strategy.record_hover(chat_id)

    def record_search(self, query: str) -> None:
        """Record a search query.

        Args:
            query: Search query.
        """
        with self._lock:
            if query not in self._context.recent_searches:
                self._context.recent_searches.insert(0, query)
                self._context.recent_searches = self._context.recent_searches[:10]

    def record_access(self, key: str) -> None:
        """Record access to a cached item.

        Args:
            key: Cache key accessed.
        """
        with self._lock:
            if key not in self._access_patterns:
                self._access_patterns[key] = AccessPattern(key=key)
            self._access_patterns[key].record_access()

    def update_context(
        self,
        battery_level: float | None = None,
        memory_available_mb: int | None = None,
    ) -> None:
        """Update prediction context.

        Args:
            battery_level: Current battery level (0-1).
            memory_available_mb: Available memory in MB.
        """
        with self._lock:
            if battery_level is not None:
                self._context.battery_level = battery_level
            if memory_available_mb is not None:
                self._context.memory_available_mb = memory_available_mb
            self._context.current_time = datetime.now()
            self._context.current_hour = datetime.now().hour
            self._context.current_day_of_week = datetime.now().weekday()

    def get_strategy(self, name: str) -> PredictionStrategy | None:
        """Get a strategy by name.

        Args:
            name: Strategy name.

        Returns:
            Strategy instance or None if not found.
        """
        for strategy in self._strategies:
            if strategy.name == name:
                return strategy
        return None


from jarvis.utils.singleton import thread_safe_singleton


@thread_safe_singleton
def get_predictor() -> PrefetchPredictor:
    """Get or create singleton predictor instance."""
    return PrefetchPredictor()


def reset_predictor() -> None:
    """Reset singleton predictor."""
    get_predictor.reset()  # type: ignore[attr-defined]
