"""Tests for conversation threading module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from jarvis.threading import (
    Thread,
    ThreadAnalyzer,
    ThreadedMessage,
    ThreadingConfig,
    ThreadingMethod,
    get_thread_analyzer,
    reset_thread_analyzer,
)


class TestThread:
    """Tests for Thread dataclass."""

    def test_thread_creation(self) -> None:
        """Test basic thread creation."""
        thread = Thread(
            thread_id="test123",
            messages=[1, 2, 3],
            topic_label="Test Topic",
            message_count=3,
        )
        assert thread.thread_id == "test123"
        assert thread.messages == [1, 2, 3]
        assert thread.topic_label == "Test Topic"
        assert thread.message_count == 3

    def test_thread_defaults(self) -> None:
        """Test thread default values."""
        thread = Thread(thread_id="test456")
        assert thread.messages == []
        assert thread.topic_label == ""
        assert thread.start_time is None
        assert thread.end_time is None
        assert thread.participant_count == 0
        assert thread.message_count == 0

    def test_thread_to_dict(self) -> None:
        """Test thread serialization to dict."""
        now = datetime.now()
        thread = Thread(
            thread_id="test789",
            messages=[10, 20],
            topic_label="Dinner Plans",
            start_time=now,
            end_time=now + timedelta(hours=1),
            participant_count=2,
            message_count=2,
        )
        result = thread.to_dict()
        assert result["thread_id"] == "test789"
        assert result["messages"] == [10, 20]
        assert result["topic_label"] == "Dinner Plans"
        assert result["participant_count"] == 2
        assert result["message_count"] == 2
        assert result["start_time"] is not None
        assert result["end_time"] is not None


class TestThreadingConfig:
    """Tests for ThreadingConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ThreadingConfig()
        assert config.time_gap_threshold_minutes == 30
        assert config.semantic_similarity_threshold == 0.4
        assert config.min_thread_messages == 2
        assert config.max_thread_duration_hours == 24
        assert config.use_semantic_analysis is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ThreadingConfig(
            time_gap_threshold_minutes=60,
            semantic_similarity_threshold=0.6,
            use_semantic_analysis=False,
        )
        assert config.time_gap_threshold_minutes == 60
        assert config.semantic_similarity_threshold == 0.6
        assert config.use_semantic_analysis is False


class TestThreadAnalyzer:
    """Tests for ThreadAnalyzer.

    Note: _get_sentence_model is patched to return None in tests that create
    threads, because _detect_topic_label -> analyze() -> _ensure_embeddings_computed()
    tries to load BAAI/bge-small-en-v1.5 from HuggingFace cache, which isn't
    available in CI. These are unit tests for threading logic, not embedding integration.
    """

    def test_analyzer_creation(self) -> None:
        """Test analyzer creation with default config."""
        analyzer = ThreadAnalyzer()
        assert analyzer.config is not None
        assert analyzer.config.time_gap_threshold_minutes == 30

    def test_analyzer_with_custom_config(self) -> None:
        """Test analyzer creation with custom config."""
        config = ThreadingConfig(time_gap_threshold_minutes=45)
        analyzer = ThreadAnalyzer(config)
        assert analyzer.config.time_gap_threshold_minutes == 45

    def test_analyze_empty_messages(self) -> None:
        """Test analyzing empty message list."""
        analyzer = ThreadAnalyzer()
        threads = analyzer.analyze_threads([], "chat123")
        assert threads == []

    @patch.object(ThreadAnalyzer, "_get_sentence_model", return_value=None)
    def test_analyze_single_message(self, _mock_model) -> None:
        """Test analyzing a single message."""
        analyzer = ThreadAnalyzer()
        message = self._create_mock_message(
            id=1,
            text="Hello!",
            date=datetime.now(),
        )
        threads = analyzer.analyze_threads([message], "chat123")
        assert len(threads) == 1
        assert threads[0].message_count == 1
        assert 1 in threads[0].messages

    @patch.object(ThreadAnalyzer, "_get_sentence_model", return_value=None)
    def test_analyze_messages_within_time_gap(self, _mock_model) -> None:
        """Test that messages within time gap are grouped together."""
        config = ThreadingConfig(
            time_gap_threshold_minutes=30,
            use_semantic_analysis=False,
        )
        analyzer = ThreadAnalyzer(config)

        now = datetime.now()
        messages = [
            self._create_mock_message(id=1, text="Hi", date=now),
            self._create_mock_message(id=2, text="Hello", date=now + timedelta(minutes=5)),
            self._create_mock_message(id=3, text="How are you?", date=now + timedelta(minutes=10)),
        ]

        threads = analyzer.analyze_threads(messages, "chat123")
        assert len(threads) == 1
        assert threads[0].message_count == 3

    @patch.object(ThreadAnalyzer, "_get_sentence_model", return_value=None)
    def test_analyze_messages_split_by_time_gap(self, _mock_model) -> None:
        """Test that messages split by time gap form separate threads."""
        config = ThreadingConfig(
            time_gap_threshold_minutes=30,
            use_semantic_analysis=False,
        )
        analyzer = ThreadAnalyzer(config)

        now = datetime.now()
        messages = [
            self._create_mock_message(id=1, text="Morning!", date=now),
            self._create_mock_message(id=2, text="Good morning", date=now + timedelta(minutes=5)),
            # Gap of 2 hours - should create new thread
            self._create_mock_message(id=3, text="Afternoon!", date=now + timedelta(hours=2)),
        ]

        threads = analyzer.analyze_threads(messages, "chat123")
        assert len(threads) == 2
        assert threads[0].message_count == 2
        assert threads[1].message_count == 1

    def test_thread_id_generation(self) -> None:
        """Test that thread IDs are generated consistently."""
        analyzer = ThreadAnalyzer()
        thread_id = analyzer._generate_thread_id("chat123", 456)
        # Thread ID should be a hex string
        assert isinstance(thread_id, str)
        assert len(thread_id) == 16

        # Same inputs should generate same ID
        thread_id2 = analyzer._generate_thread_id("chat123", 456)
        assert thread_id == thread_id2

        # Different inputs should generate different IDs
        thread_id3 = analyzer._generate_thread_id("chat456", 456)
        assert thread_id != thread_id3

    @patch.object(ThreadAnalyzer, "_get_sentence_model", return_value=None)
    def test_detect_topic_label(self, _mock_model) -> None:
        """Test topic label detection."""
        analyzer = ThreadAnalyzer()

        # Test meeting detection
        messages = [self._create_mock_message(id=1, text="Let's schedule a meeting")]
        label = analyzer._detect_topic_label(messages)
        assert label == "Meeting Plans"

        # Test dinner detection
        messages = [self._create_mock_message(id=1, text="Want to grab dinner?")]
        label = analyzer._detect_topic_label(messages)
        assert label == "Dinner Plans"

        # Test empty messages
        label = analyzer._detect_topic_label([])
        assert label == "General"

    @patch.object(ThreadAnalyzer, "_get_sentence_model", return_value=None)
    def test_get_threaded_messages(self, _mock_model) -> None:
        """Test getting messages with thread info."""
        config = ThreadingConfig(use_semantic_analysis=False)
        analyzer = ThreadAnalyzer(config)

        now = datetime.now()
        messages = [
            self._create_mock_message(id=1, text="Hi", date=now),
            self._create_mock_message(id=2, text="Hello", date=now + timedelta(minutes=5)),
        ]

        threaded = analyzer.get_threaded_messages(messages, "chat123")
        assert len(threaded) == 2
        assert all(isinstance(tm, ThreadedMessage) for tm in threaded)
        assert threaded[0].is_thread_start is True
        assert threaded[1].is_thread_start is False
        assert threaded[0].thread_position == 0
        assert threaded[1].thread_position == 1

    def test_clear_cache(self) -> None:
        """Test cache clearing."""
        analyzer = ThreadAnalyzer()
        # Just ensure it doesn't raise
        analyzer.clear_cache()

    def _create_mock_message(
        self,
        id: int,
        text: str,
        date: datetime | None = None,
        sender: str = "user1",
        is_from_me: bool = False,
        reply_to_id: int | None = None,
    ) -> MagicMock:
        """Create a mock Message object."""
        message = MagicMock()
        message.id = id
        message.text = text
        message.date = date or datetime.now()
        message.sender = sender
        message.sender_name = sender
        message.chat_id = "chat123"
        message.is_from_me = is_from_me
        message.is_system_message = False
        message.reply_to_id = reply_to_id
        message.attachments = []
        message.reactions = []
        return message


class TestThreadingSingleton:
    """Tests for singleton pattern."""

    def test_get_thread_analyzer_returns_same_instance(self) -> None:
        """Test singleton returns same instance."""
        reset_thread_analyzer()
        analyzer1 = get_thread_analyzer()
        analyzer2 = get_thread_analyzer()
        assert analyzer1 is analyzer2

    def test_reset_thread_analyzer(self) -> None:
        """Test singleton reset creates new instance."""
        reset_thread_analyzer()
        analyzer1 = get_thread_analyzer()
        reset_thread_analyzer()
        analyzer2 = get_thread_analyzer()
        assert analyzer1 is not analyzer2


class TestThreadingMethod:
    """Tests for ThreadingMethod enum."""

    def test_threading_methods_exist(self) -> None:
        """Test all threading methods are defined."""
        assert ThreadingMethod.REPLY_REFERENCE.value == "reply_reference"
        assert ThreadingMethod.SEMANTIC_SIMILARITY.value == "semantic_similarity"
        assert ThreadingMethod.TIME_GAP.value == "time_gap"
        assert ThreadingMethod.TOPIC_SHIFT.value == "topic_shift"
