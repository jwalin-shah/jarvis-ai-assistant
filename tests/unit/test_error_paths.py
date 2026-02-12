"""Error path tests for JARVIS.

Tests failure scenarios across critical modules to prove the system degrades
gracefully, not catastrophically. Each test verifies:
1. The error is handled gracefully (no crash)
2. Appropriate fallback behavior occurs
3. Error is logged or propagated correctly
4. The system can recover (next request works)
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.contracts.pipeline import (
    CategoryType,
    ClassificationResult,
    GenerationResponse,
    IntentType,
    MessageContext,
    UrgencyLevel,
)
from jarvis.errors import (
    DatabaseError,
    ErrorCode,
    JarvisError,
    ModelGenerationError,
    ModelLoadError,
    iMessageAccessError,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_db():
    """Create a mock JarvisDB."""
    db = MagicMock()
    db.init_schema = MagicMock()
    db.get_contact = MagicMock(return_value=None)
    db.get_contact_by_chat_id = MagicMock(return_value=None)
    db.get_stats = MagicMock(return_value={"pairs": 0, "contacts": 0})
    db.get_active_index = MagicMock(return_value=None)
    return db


@pytest.fixture
def mock_generator():
    """Create a mock MLXGenerator that can simulate failures."""
    generator = MagicMock()
    generator.is_loaded = MagicMock(return_value=True)
    mock_response = MagicMock()
    mock_response.text = "Generated response"
    generator.generate = MagicMock(return_value=mock_response)
    return generator


@pytest.fixture
def mock_context_service():
    """Create a mock ContextService that can simulate failures."""
    svc = MagicMock()
    svc.get_contact = MagicMock(return_value=None)
    svc.search_examples = MagicMock(return_value=[])
    svc.get_relationship_profile = MagicMock(return_value=("", ""))
    svc.fetch_conversation_context = MagicMock(return_value=[])
    return svc


@pytest.fixture
def basic_classification():
    """A basic classification result for tests."""
    return ClassificationResult(
        intent=IntentType.QUESTION,
        category=CategoryType.FULL_RESPONSE,
        urgency=UrgencyLevel.MEDIUM,
        confidence=0.8,
        requires_knowledge=False,
        metadata={
            "category_name": "question",
            "mobilization_pressure": "medium",
            "category_confidence": 0.8,
        },
    )


@pytest.fixture
def basic_message_context():
    """A basic MessageContext for tests."""
    return MessageContext(
        chat_id="chat123",
        message_text="What time is dinner?",
        is_from_me=False,
        timestamp=datetime.utcnow(),
        metadata={"thread": ["Hey!", "What's up?"]},
    )


@pytest.fixture(autouse=True)
def mock_health_check():
    """Mock health check so tests don't fail on system resource checks."""
    with patch("jarvis.generation.can_use_llm", return_value=(True, "ok")):
        yield


# =============================================================================
# 1. Database Connection Failures
# =============================================================================


class TestDBConnectionFailures:
    """Tests for database connection failures during routing and search."""

    def test_db_connection_failure_during_contact_lookup(self, mock_db, mock_generator):
        """Contact lookup failure should not crash routing; proceeds without contact."""
        from jarvis.router import ReplyRouter

        mock_db.get_contact.side_effect = sqlite3.OperationalError("database is locked")
        mock_db.get_contact_by_chat_id.side_effect = sqlite3.OperationalError("database is locked")

        # Mock at service level to verify router handles None contact gracefully.
        with patch("jarvis.services.context_service.ContextService") as MockCS:  # noqa: N806
            mock_cs = MockCS.return_value
            mock_cs.get_contact.return_value = None
            mock_cs.search_examples.return_value = []
            mock_cs.get_relationship_profile.return_value = ("", "")

            router = ReplyRouter(db=mock_db, generator=mock_generator)
            router._context_service = mock_cs

            # Mock reply_service to return a valid response
            mock_reply_svc = MagicMock()
            mock_reply_svc.generate_reply.return_value = GenerationResponse(
                response="Sure thing!",
                confidence=0.7,
                metadata={"type": "generated", "similarity_score": 0.0},
            )
            router._reply_service = mock_reply_svc

            result = router.route(incoming="What time is dinner?", chat_id="chat123")

            assert "response" in result
            assert result["type"] in ("generated", "clarify")

    def test_db_connection_failure_in_jarvisdb_connection(self, tmp_path):
        """JarvisDB should raise on corrupt/inaccessible database, not silently corrupt."""
        from jarvis.db import JarvisDB

        # Create a file that is not a valid SQLite database
        bad_db_path = tmp_path / "corrupt.db"
        bad_db_path.write_text("this is not a database")

        db = JarvisDB(db_path=bad_db_path)

        with pytest.raises((sqlite3.DatabaseError, sqlite3.OperationalError)):
            db.init_schema()

        # Cleanup
        db.close()

    def test_db_close_and_reopen(self, tmp_path):
        """After closing, database should be reusable on next connection."""
        from jarvis.db import JarvisDB

        db_path = tmp_path / "test.db"
        db = JarvisDB(db_path=db_path)
        db.init_schema()

        # Use and close
        with db.connection() as conn:
            conn.execute("SELECT 1")
        db.close()

        # Reopen (new instance, same path)
        db2 = JarvisDB(db_path=db_path)
        with db2.connection() as conn:
            cursor = conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            )
            row = cursor.fetchone()
            assert row is not None
        db2.close()


# =============================================================================
# 2. Model Loading Failures
# =============================================================================


class TestModelLoadFailures:
    """Tests for model load failures (file not found, corrupt, memory pressure)."""

    def test_generator_returns_fallback_on_load_failure(self):
        """MLXGenerator should return fallback response when model load fails."""
        from unittest.mock import patch

        from contracts.models import GenerationRequest as ModelGenRequest
        from models.generator import MLXGenerator

        mock_loader = MagicMock()
        mock_loader.is_loaded.return_value = False
        mock_loader.load.return_value = False  # Load fails

        generator = MLXGenerator(loader=mock_loader, skip_templates=True)

        request = ModelGenRequest(prompt="Hello, how are you?")
        # Bypass memory pressure check to test the load-failure path specifically
        with patch("models.generator.should_skip_model_load", return_value=False):
            response = generator.generate(request)

        # Should return a fallback, not crash
        assert response is not None
        assert response.finish_reason == "fallback"
        assert "load_failed" in response.error
        assert response.model_name == "fallback"

    def test_generator_returns_fallback_on_memory_pressure(self):
        """MLXGenerator should return fallback when memory pressure is high."""
        from contracts.models import GenerationRequest as ModelGenRequest
        from models.generator import MLXGenerator

        mock_loader = MagicMock()
        mock_loader.is_loaded.return_value = False

        generator = MLXGenerator(loader=mock_loader, skip_templates=True)

        with patch("models.generator.should_skip_model_load", return_value=True):
            request = ModelGenRequest(prompt="Hello")
            response = generator.generate(request)

        assert response is not None
        assert response.finish_reason == "fallback"
        assert "memory_pressure" in response.error
        assert response.model_name == "fallback"
        # Model load should NOT have been attempted
        mock_loader.load.assert_not_called()

    def test_generator_returns_fallback_on_generation_exception(self):
        """MLXGenerator should catch generation exceptions and return fallback."""
        from contracts.models import GenerationRequest as ModelGenRequest
        from models.generator import MLXGenerator

        mock_loader = MagicMock()
        mock_loader.is_loaded.return_value = True
        mock_loader.has_draft_model = False
        mock_loader.has_prompt_cache = False
        mock_loader.generate_sync.side_effect = RuntimeError("Metal GPU assertion failed")

        generator = MLXGenerator(loader=mock_loader, skip_templates=True)
        generator._prompt_builder = MagicMock(build=MagicMock(return_value="formatted prompt"))

        request = ModelGenRequest(prompt="Hello")
        response = generator.generate(request)

        assert response is not None
        assert response.finish_reason == "error"
        assert response.model_name == "fallback"

    def test_generator_unloads_on_failure_if_loaded_for_call(self):
        """If model was loaded for this call and generation fails, unload it."""
        from contracts.models import GenerationRequest as ModelGenRequest
        from models.generator import MLXGenerator

        mock_loader = MagicMock()
        mock_loader.is_loaded.return_value = False
        mock_loader.load.return_value = True
        mock_loader.has_draft_model = False
        mock_loader.has_prompt_cache = False
        mock_loader.generate_sync.side_effect = RuntimeError("GPU crash")

        generator = MLXGenerator(loader=mock_loader, skip_templates=True)
        generator._prompt_builder = MagicMock(build=MagicMock(return_value="formatted"))

        with patch("models.generator.should_skip_model_load", return_value=False):
            request = ModelGenRequest(prompt="Test")
            response = generator.generate(request)

        # Should unload model to free memory after failure
        mock_loader.unload.assert_called_once()
        assert response.finish_reason == "error"


# =============================================================================
# 3. Embedding Service Failures
# =============================================================================


class TestEmbeddingServiceFailures:
    """Tests for embedding service timeout/failure scenarios."""

    def test_semantic_search_returns_empty_when_no_messages(self, tmp_path):
        """SemanticSearcher should return empty results when reader has no messages."""
        from jarvis.search.semantic_search import EmbeddingCache, SemanticSearcher

        mock_reader = MagicMock()
        mock_reader.get_messages.return_value = []
        mock_reader.search.return_value = []

        cache = EmbeddingCache(cache_path=tmp_path / "cache.db")
        searcher = SemanticSearcher(reader=mock_reader, cache=cache)

        results = searcher.search("dinner plans")
        assert results == []

        cache.close()

    def test_corrupt_cached_embedding_does_not_crash(self, tmp_path):
        """Corrupt cached embeddings should not crash the cache layer."""
        from jarvis.search.semantic_search import EmbeddingCache

        cache = EmbeddingCache(cache_path=tmp_path / "cache.db")

        # Store a valid embedding
        valid_emb = np.random.randn(384).astype(np.float32)
        cache.set(message_id=1, chat_id="chat1", text_hash="abc", embedding=valid_emb)

        # Read it back successfully
        result = cache.get(1)
        assert result is not None
        assert result.shape == (384,)

        # Now directly corrupt the blob in the DB
        with cache._lock:
            conn = cache._get_connection()
            conn.execute(
                "UPDATE embeddings SET embedding = ? WHERE message_id = ?",
                (b"corrupt_data_not_float32", 1),
            )
            conn.commit()

        # Reading corrupt data: numpy will interpret the bytes, possibly wrong shape.
        # The important thing is it does not raise an unhandled exception.
        result = cache.get(1)
        assert result is not None

        cache.close()

    def test_embedding_cache_handles_concurrent_access(self, tmp_path):
        """EmbeddingCache should handle concurrent reads/writes without corruption."""
        from jarvis.search.semantic_search import EmbeddingCache

        cache = EmbeddingCache(cache_path=tmp_path / "embed.db")
        errors: list[Exception] = []

        def writer(start_id: int) -> None:
            try:
                for i in range(20):
                    emb = np.random.randn(384).astype(np.float32)
                    cache.set(
                        message_id=start_id + i,
                        chat_id="chat1",
                        text_hash=f"hash_{start_id + i}",
                        embedding=emb,
                    )
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(20):
                    cache.stats()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(100,)),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Concurrent access errors: {errors}"
        cache.close()


# =============================================================================
# 4. Socket Server Malformed Request Handling
# =============================================================================


class TestSocketServerMalformedRequests:
    """Tests for socket server handling of malformed and invalid requests."""

    def test_process_invalid_json(self):
        """Server should return parse error for invalid JSON."""
        from jarvis.socket_server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False, preload_models=False, enable_prefetch=False
        )

        result = asyncio.run(server._process_message("this is not json"))

        parsed = json.loads(result)
        assert "error" in parsed
        assert parsed["error"]["code"] == -32700  # PARSE_ERROR

    def test_process_missing_method(self):
        """Server should return invalid request for missing method."""
        from jarvis.socket_server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False, preload_models=False, enable_prefetch=False
        )

        msg = json.dumps({"jsonrpc": "2.0", "params": {}, "id": 1})
        result = asyncio.run(server._process_message(msg))

        parsed = json.loads(result)
        assert "error" in parsed
        assert parsed["error"]["code"] == -32600  # INVALID_REQUEST

    def test_process_unknown_method(self):
        """Server should return method not found for unknown methods."""
        from jarvis.socket_server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False, preload_models=False, enable_prefetch=False
        )

        msg = json.dumps({"jsonrpc": "2.0", "method": "nonexistent_method", "params": {}, "id": 1})
        result = asyncio.run(server._process_message(msg))

        parsed = json.loads(result)
        assert "error" in parsed
        assert parsed["error"]["code"] == -32601  # METHOD_NOT_FOUND

    def test_process_non_dict_request(self):
        """Server should return invalid request for non-dict JSON."""
        from jarvis.socket_server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False, preload_models=False, enable_prefetch=False
        )

        msg = json.dumps([1, 2, 3])
        result = asyncio.run(server._process_message(msg))

        parsed = json.loads(result)
        assert "error" in parsed
        assert parsed["error"]["code"] == -32600  # INVALID_REQUEST

    def test_process_invalid_params_type(self):
        """Server should handle TypeError from invalid param types gracefully."""
        from jarvis.socket_server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False, preload_models=False, enable_prefetch=False
        )

        # ping takes no params but we send extra keys
        msg = json.dumps(
            {"jsonrpc": "2.0", "method": "ping", "params": {"unexpected": True}, "id": 1}
        )
        result = asyncio.run(server._process_message(msg))

        parsed = json.loads(result)
        # Either succeeds (ping ignores extra params) or returns typed error.
        # Must NOT crash the server.
        assert "result" in parsed or "error" in parsed

    def test_ping_works_without_models(self):
        """Ping should work even when models are not loaded."""
        from jarvis.socket_server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False, preload_models=False, enable_prefetch=False
        )

        msg = json.dumps({"jsonrpc": "2.0", "method": "ping", "params": {}, "id": 1})
        result = asyncio.run(server._process_message(msg))

        parsed = json.loads(result)
        assert "result" in parsed
        assert parsed["result"]["status"] == "ok"
        assert parsed["result"]["models_ready"] is False

    def test_rate_limiter_allows_normal_traffic(self):
        """Rate limiter should allow normal traffic patterns."""
        from jarvis.socket_server import RateLimiter

        limiter = RateLimiter(max_requests=10, window_seconds=1.0)

        for _ in range(10):
            assert limiter.is_allowed("client1") is True

    def test_rate_limiter_blocks_burst(self):
        """Rate limiter should block after exceeding burst limit."""
        from jarvis.socket_server import RateLimiter

        limiter = RateLimiter(max_requests=5, window_seconds=1.0)

        for _ in range(5):
            limiter.is_allowed("client1")

        assert limiter.is_allowed("client1") is False

    def test_rate_limiter_per_client_isolation(self):
        """Rate limiter should track clients independently."""
        from jarvis.socket_server import RateLimiter

        limiter = RateLimiter(max_requests=3, window_seconds=1.0)

        for _ in range(3):
            limiter.is_allowed("client1")
        assert limiter.is_allowed("client1") is False

        # client2 should still have quota
        assert limiter.is_allowed("client2") is True


# =============================================================================
# 5. Reply Service Missing Context
# =============================================================================


class TestReplyServiceMissingContext:
    """Tests for reply service behavior with missing context data."""

    def test_generate_reply_with_empty_message(self, mock_db, mock_generator):
        """Empty message should return clarify response, not crash."""
        from jarvis.reply_service import ReplyService

        service = ReplyService(db=mock_db, generator=mock_generator)

        context = MessageContext(
            chat_id="chat123",
            message_text="",
            is_from_me=False,
            timestamp=datetime.utcnow(),
        )
        classification = ClassificationResult(
            intent=IntentType.UNKNOWN,
            category=CategoryType.FULL_RESPONSE,
            urgency=UrgencyLevel.LOW,
            confidence=0.3,
            requires_knowledge=False,
            metadata={"category_name": "question", "mobilization_pressure": "low"},
        )

        result = service.generate_reply(context=context, classification=classification)

        assert result is not None
        assert result.metadata.get("type") == "clarify"
        assert result.metadata.get("reason") == "empty_message"

    def test_generate_reply_with_whitespace_only_message(self, mock_db, mock_generator):
        """Whitespace-only message should also return clarify response."""
        from jarvis.reply_service import ReplyService

        service = ReplyService(db=mock_db, generator=mock_generator)

        context = MessageContext(
            chat_id="chat123",
            message_text="   \n\t  ",
            is_from_me=False,
            timestamp=datetime.utcnow(),
        )
        classification = ClassificationResult(
            intent=IntentType.UNKNOWN,
            category=CategoryType.FULL_RESPONSE,
            urgency=UrgencyLevel.LOW,
            confidence=0.3,
            requires_knowledge=False,
            metadata={"category_name": "question"},
        )

        result = service.generate_reply(context=context, classification=classification)

        assert result is not None
        assert result.metadata.get("reason") == "empty_message"

    def test_reply_service_with_no_search_results(
        self, mock_db, mock_generator, basic_classification, basic_message_context
    ):
        """Reply service should work fine with no search results (no RAG context)."""
        from jarvis.reply_service import ReplyService

        mock_response = MagicMock()
        mock_response.text = "Sounds good!"
        mock_generator.generate.return_value = mock_response

        service = ReplyService(db=mock_db, generator=mock_generator)

        mock_cs = MagicMock()
        mock_cs.get_contact.return_value = None
        mock_cs.search_examples.return_value = []
        mock_cs.get_relationship_profile.return_value = ("", "")
        mock_cs.fetch_conversation_context.return_value = []
        service._context_service = mock_cs

        result = service.generate_reply(
            context=basic_message_context,
            classification=basic_classification,
            search_results=[],
            thread=[],
        )

        assert result is not None
        assert isinstance(result.response, str)

    def test_reply_service_llm_unavailable_returns_fallback(
        self, mock_db, mock_generator, basic_classification, basic_message_context
    ):
        """When LLM is unavailable (health check fails), return fallback response."""
        from jarvis.reply_service import ReplyService

        service = ReplyService(db=mock_db, generator=mock_generator)

        mock_cs = MagicMock()
        mock_cs.get_contact.return_value = None
        mock_cs.search_examples.return_value = []
        mock_cs.get_relationship_profile.return_value = ("", "")
        service._context_service = mock_cs

        with patch("jarvis.generation.can_use_llm", return_value=(False, "memory_critical")):
            result = service.generate_reply(
                context=basic_message_context,
                classification=basic_classification,
                search_results=[],
                thread=[],
            )

        assert result is not None
        assert result.metadata.get("type") == "fallback"
        assert "memory" in result.metadata.get("reason", "").lower()

    def test_reply_service_generation_exception_returns_graceful_error(
        self, mock_db, mock_generator, basic_classification, basic_message_context
    ):
        """When LLM generation raises an exception, return graceful error response."""
        from jarvis.reply_service import ReplyService

        mock_generator.generate.side_effect = RuntimeError("Metal GPU assertion failed")

        service = ReplyService(db=mock_db, generator=mock_generator)

        mock_cs = MagicMock()
        mock_cs.get_contact.return_value = None
        mock_cs.search_examples.return_value = []
        mock_cs.get_relationship_profile.return_value = ("", "")
        mock_cs.fetch_conversation_context.return_value = []
        service._context_service = mock_cs

        result = service.generate_reply(
            context=basic_message_context,
            classification=basic_classification,
            search_results=[],
            thread=["Hey!", "What's up?"],
        )

        assert result is not None
        # Should return error response, not propagate exception
        assert result.metadata.get("reason") == "generation_error"
        assert "trouble" in result.response.lower() or result.response != ""


# =============================================================================
# 6. Router Error Paths
# =============================================================================


class TestRouterErrorPaths:
    """Tests for router-level error handling and edge cases."""

    def test_route_empty_message(self, mock_db, mock_generator):
        """Router should handle empty message gracefully."""
        from jarvis.router import ReplyRouter

        router = ReplyRouter(db=mock_db, generator=mock_generator)

        mock_reply_svc = MagicMock()
        mock_reply_svc.generate_reply.return_value = GenerationResponse(
            response="I received an empty message. Could you tell me what you need?",
            confidence=0.2,
            metadata={"type": "clarify", "reason": "empty_message", "similarity_score": 0.0},
        )
        router._reply_service = mock_reply_svc

        result = router.route(incoming="")

        assert result is not None
        assert result["type"] == "clarify"

    def test_route_message_context_with_empty_text(self, mock_db, mock_generator):
        """route_message should handle empty text via MessageContext."""
        from jarvis.router import ReplyRouter

        router = ReplyRouter(db=mock_db, generator=mock_generator)

        context = MessageContext(
            chat_id="chat1",
            message_text="",
            is_from_me=False,
            timestamp=datetime.utcnow(),
            metadata={},
        )

        result = router.route_message(context)

        assert result is not None
        assert result.metadata.get("type") == "clarify"
        assert result.metadata.get("reason") == "empty_message"

    def test_to_legacy_response_format(self):
        """_to_legacy_response should produce correct dict from GenerationResponse."""
        from jarvis.router import ReplyRouter

        response = GenerationResponse(
            response="Hello!",
            confidence=0.85,
            metadata={
                "type": "generated",
                "similarity_score": 0.75,
                "reason": "",
                "category": "question",
            },
        )

        result = ReplyRouter._to_legacy_response(response)

        assert result["type"] == "generated"
        assert result["response"] == "Hello!"
        assert result["confidence"] in ("high", "medium", "low")
        assert "similarity_score" in result

    def test_analyze_complexity_edge_cases(self):
        """_analyze_complexity should handle edge cases without crashing."""
        from jarvis.router import ReplyRouter

        assert ReplyRouter._analyze_complexity("") == 0.0
        assert ReplyRouter._analyze_complexity("ok") > 0.0
        assert ReplyRouter._analyze_complexity("Hello, how are you?") > 0.0
        # Very long text
        long_text = "word " * 500
        result = ReplyRouter._analyze_complexity(long_text)
        assert 0.0 <= result <= 1.0

    def test_build_thread_context_with_malformed_messages(self):
        """_build_thread_context should handle malformed message objects."""
        from jarvis.router import ReplyRouter

        messages = [
            {"text": "Hello", "is_from_me": True},
            {"text": "", "is_from_me": False},  # empty text
            {},  # completely empty
            {"text": "World", "is_from_me": False, "sender_name": "Alice"},
        ]

        result = ReplyRouter._build_thread_context(messages)

        # Should not crash, and should produce some thread entries
        assert result is not None
        assert isinstance(result, list)

    def test_prefetch_cache_check_handles_missing_manager(self):
        """_check_prefetch_cache should return None when no prefetch manager."""
        from jarvis.router import ReplyRouter

        result = ReplyRouter._check_prefetch_cache("chat123")
        assert result is None

        result = ReplyRouter._check_prefetch_cache(None)
        assert result is None


# =============================================================================
# 7. Semantic Search Error Paths
# =============================================================================


class TestSemanticSearchErrorPaths:
    """Tests for semantic search with no index, empty queries, etc."""

    def test_search_with_empty_query(self, tmp_path):
        """Empty query should return empty results."""
        from jarvis.search.semantic_search import EmbeddingCache, SemanticSearcher

        mock_reader = MagicMock()
        cache = EmbeddingCache(cache_path=tmp_path / "cache.db")
        searcher = SemanticSearcher(reader=mock_reader, cache=cache)

        assert searcher.search("") == []
        assert searcher.search("   ") == []

        cache.close()

    def test_search_with_no_messages_in_db(self, tmp_path):
        """Search against empty message DB should return empty results."""
        from jarvis.search.semantic_search import EmbeddingCache, SemanticSearcher

        mock_reader = MagicMock()
        mock_reader.get_messages.return_value = []
        mock_reader.search.return_value = []

        cache = EmbeddingCache(cache_path=tmp_path / "cache.db")
        searcher = SemanticSearcher(reader=mock_reader, cache=cache)

        results = searcher.search("dinner plans", limit=10)
        assert results == []

        cache.close()

    def test_embedding_cache_stats_on_empty_cache(self, tmp_path):
        """Cache stats should work on an empty cache."""
        from jarvis.search.semantic_search import EmbeddingCache

        cache = EmbeddingCache(cache_path=tmp_path / "cache.db")
        stats = cache.stats()

        assert stats["embedding_count"] == 0
        assert stats["size_bytes"] == 0
        assert stats["size_mb"] == 0.0

        cache.close()

    def test_embedding_cache_get_nonexistent(self, tmp_path):
        """Getting a nonexistent embedding should return None."""
        from jarvis.search.semantic_search import EmbeddingCache

        cache = EmbeddingCache(cache_path=tmp_path / "cache.db")

        result = cache.get(message_id=99999)
        assert result is None

        batch_result = cache.get_batch([1, 2, 3])
        assert batch_result == {}

        cache.close()

    def test_embedding_cache_invalidation(self, tmp_path):
        """Invalidation should remove entries without errors."""
        from jarvis.search.semantic_search import EmbeddingCache

        cache = EmbeddingCache(cache_path=tmp_path / "cache.db")

        # Store an embedding
        emb = np.random.randn(384).astype(np.float32)
        cache.set(message_id=1, chat_id="chat1", text_hash="abc", embedding=emb)
        assert cache.get(1) is not None

        # Invalidate
        cache.invalidate(1)
        assert cache.get(1) is None

        # Invalidate non-existent (should not crash)
        cache.invalidate(9999)

        # Invalidate by chat
        emb2 = np.random.randn(384).astype(np.float32)
        cache.set(message_id=2, chat_id="chat2", text_hash="def", embedding=emb2)
        cache.invalidate_chat("chat2")
        assert cache.get(2) is None

        cache.close()

    def test_search_similar_to_message_with_no_text(self, tmp_path):
        """search_similar_to_message with no text should return empty."""
        from jarvis.search.semantic_search import EmbeddingCache, SemanticSearcher

        mock_reader = MagicMock()
        cache = EmbeddingCache(cache_path=tmp_path / "cache.db")
        searcher = SemanticSearcher(reader=mock_reader, cache=cache)

        mock_message = MagicMock()
        mock_message.text = None

        results = searcher.search_similar_to_message(mock_message)
        assert results == []

        cache.close()


# =============================================================================
# 8. Error Hierarchy Behavior
# =============================================================================


class TestErrorHierarchy:
    """Tests that the error hierarchy works correctly for error propagation."""

    def test_jarvis_error_base(self):
        """JarvisError should be catchable by base class."""
        err = JarvisError("test error")
        assert str(err) == "test error"
        assert err.code == ErrorCode.UNKNOWN
        assert err.details == {}

    def test_model_errors_are_jarvis_errors(self):
        """Model errors should be catchable as JarvisError."""
        err = ModelLoadError("model not found", model_path="/missing/model")
        assert isinstance(err, JarvisError)
        assert err.code == ErrorCode.MDL_LOAD_FAILED
        assert err.details["model_path"] == "/missing/model"

    def test_db_error_preserves_cause(self):
        """DatabaseError should preserve the original cause."""
        original = sqlite3.OperationalError("disk I/O error")
        err = DatabaseError("Query failed", cause=original)
        assert err.cause is original
        assert err.__cause__ is original

    def test_error_to_dict_for_api_response(self):
        """to_dict should produce valid API error structure."""
        err = ModelGenerationError(
            "Generation timed out",
            model_name="lfm-1.2b",
            timeout_seconds=60.0,
        )
        d = err.to_dict()
        assert d["error"] == "ModelGenerationError"
        assert d["code"] == ErrorCode.MDL_TIMEOUT.value
        assert "detail" in d

    def test_imessage_error_with_permission_instructions(self):
        """iMessageAccessError should include permission instructions."""
        err = iMessageAccessError(
            "Cannot access chat.db",
            requires_permission=True,
            db_path="/Users/test/Library/Messages/chat.db",
        )
        assert err.details["requires_permission"] is True
        assert "permission_instructions" in err.details
        assert len(err.details["permission_instructions"]) > 0

    def test_convenience_error_factories(self):
        """Convenience functions should create properly typed errors."""
        from jarvis.errors import (
            model_generation_timeout,
            model_not_found,
            model_out_of_memory,
            validation_required,
            validation_type_error,
        )

        err1 = model_not_found("/missing/path")
        assert isinstance(err1, ModelLoadError)
        assert err1.code == ErrorCode.MDL_NOT_FOUND

        err2 = model_out_of_memory("lfm-1.2b", available_mb=1024, required_mb=2048)
        assert isinstance(err2, ModelLoadError)
        assert err2.details["available_mb"] == 1024

        err3 = model_generation_timeout("lfm-1.2b", timeout_seconds=60.0)
        assert isinstance(err3, ModelGenerationError)
        assert err3.code == ErrorCode.MDL_TIMEOUT

        err4 = validation_required("chat_id")
        assert err4.code == ErrorCode.VAL_MISSING_REQUIRED

        err5 = validation_type_error("limit", "abc", "int")
        assert err5.code == ErrorCode.VAL_TYPE_ERROR


# =============================================================================
# 9. Recovery After Failure
# =============================================================================


class TestRecoveryAfterFailure:
    """Tests proving the system can recover after a failure (next request works)."""

    def test_generator_recovers_after_generation_failure(self):
        """After a generation failure, the next generation should work."""
        from contracts.models import GenerationRequest as ModelGenRequest
        from models.generator import MLXGenerator

        mock_loader = MagicMock()
        mock_loader.is_loaded.return_value = True
        mock_loader.has_draft_model = False
        mock_loader.has_prompt_cache = False

        mock_result_ok = MagicMock()
        mock_result_ok.text = "Good response"
        mock_result_ok.tokens_generated = 5

        mock_loader.generate_sync.side_effect = [
            RuntimeError("GPU crash"),  # First call fails
            mock_result_ok,  # Second call succeeds
        ]

        generator = MLXGenerator(loader=mock_loader, skip_templates=True)
        generator._prompt_builder = MagicMock(build=MagicMock(return_value="formatted"))

        # First call: should get fallback
        request = ModelGenRequest(prompt="Hello")
        response1 = generator.generate(request)
        assert response1.finish_reason == "error"

        # Second call: should succeed
        response2 = generator.generate(request)
        assert response2.text == "Good response"
        assert response2.finish_reason == "stop"

    def test_reply_service_recovers_after_llm_failure(
        self, mock_db, basic_classification, basic_message_context
    ):
        """After LLM failure, next reply should work if LLM recovers."""
        from jarvis.reply_service import ReplyService

        mock_gen = MagicMock()
        mock_gen.is_loaded.return_value = True

        mock_ok_response = MagicMock()
        mock_ok_response.text = "Recovered response"
        mock_gen.generate.side_effect = [
            RuntimeError("Temporary GPU error"),
            mock_ok_response,
        ]

        service = ReplyService(db=mock_db, generator=mock_gen)

        mock_cs = MagicMock()
        mock_cs.get_contact.return_value = None
        mock_cs.search_examples.return_value = []
        mock_cs.get_relationship_profile.return_value = ("", "")
        mock_cs.fetch_conversation_context.return_value = []
        service._context_service = mock_cs

        # First call: should get error response (graceful)
        result1 = service.generate_reply(
            context=basic_message_context,
            classification=basic_classification,
            search_results=[],
            thread=[],
        )
        assert result1.metadata.get("reason") == "generation_error"

        # Second call: should recover
        result2 = service.generate_reply(
            context=basic_message_context,
            classification=basic_classification,
            search_results=[],
            thread=[],
        )
        assert result2.response == "Recovered response"

    def test_embedding_cache_recovers_after_write_failure(self, tmp_path):
        """EmbeddingCache should recover after a write failure."""
        from jarvis.search.semantic_search import EmbeddingCache

        cache = EmbeddingCache(cache_path=tmp_path / "recover.db")

        # Normal write
        emb = np.random.randn(384).astype(np.float32)
        cache.set(message_id=1, chat_id="c1", text_hash="h1", embedding=emb)
        assert cache.get(1) is not None

        # Simulate a write with wrong shape (may succeed but store bad data)
        try:
            cache.set(
                message_id=2,
                chat_id="c1",
                text_hash="h2",
                embedding=np.array([1, 2, 3]),  # Wrong shape
            )
        except Exception:
            pass  # Expected to fail or produce garbage

        # Next write should work fine
        emb3 = np.random.randn(384).astype(np.float32)
        cache.set(message_id=3, chat_id="c1", text_hash="h3", embedding=emb3)
        result = cache.get(3)
        assert result is not None
        assert result.shape == (384,)

        cache.close()


# =============================================================================
# 10. Confidence Computation Edge Cases
# =============================================================================


class TestConfidenceEdgeCases:
    """Tests for ReplyService confidence computation with unusual inputs."""

    def test_confidence_with_parrot_response(self):
        """Reply that parrots input should get penalized confidence."""
        from jarvis.classifiers.response_mobilization import ResponsePressure
        from jarvis.reply_service import ReplyService

        confidence, label = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.9,
            example_diversity=1.0,
            reply_length=5,
            reply_text="What time is dinner?",
            incoming_text="What time is dinner?",
        )

        # Parroting penalty: 0.85 * 0.5 = ~0.425 -> low
        assert confidence < 0.5
        assert label == "low"

    def test_confidence_with_uncertain_signal(self):
        """Very short uncertain reply with high pressure should be low confidence."""
        from jarvis.classifiers.response_mobilization import ResponsePressure
        from jarvis.reply_service import ReplyService

        confidence, label = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=0.3,
            example_diversity=0.2,
            reply_length=1,
            reply_text="?",
            incoming_text="Can you pick me up?",
        )

        assert confidence < 0.5

    def test_confidence_clamp_to_valid_range(self):
        """Confidence should always be between 0.0 and 1.0."""
        from jarvis.classifiers.response_mobilization import ResponsePressure
        from jarvis.reply_service import ReplyService

        confidence, label = ReplyService._compute_confidence(
            pressure=ResponsePressure.HIGH,
            rag_similarity=1.0,
            example_diversity=1.0,
            reply_length=20,
            reply_text="Great, I'll be there at 6pm!",
            rerank_score=0.99,
        )

        assert 0.0 <= confidence <= 1.0

    def test_example_diversity_edge_cases(self):
        """Example diversity computation should handle edge cases."""
        from jarvis.reply_service import ReplyService

        assert ReplyService._compute_example_diversity([]) == 0.0

        # All same trigger
        same = [{"trigger_text": "hello"}, {"trigger_text": "hello"}]
        assert ReplyService._compute_example_diversity(same) == 0.5

        # All unique
        unique = [{"trigger_text": "a"}, {"trigger_text": "b"}, {"trigger_text": "c"}]
        assert ReplyService._compute_example_diversity(unique) == 1.0


# =============================================================================
# 11. Socket Server Batch Error Handling
# =============================================================================


class TestSocketServerBatchErrors:
    """Tests for batch RPC error handling."""

    def test_batch_with_empty_requests(self):
        """Batch with empty list should return empty results."""
        from jarvis.socket_server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False, preload_models=False, enable_prefetch=False
        )

        msg = json.dumps({"jsonrpc": "2.0", "method": "batch", "params": {"requests": []}, "id": 1})
        result = asyncio.run(server._process_message(msg))

        parsed = json.loads(result)
        assert "result" in parsed
        assert parsed["result"]["results"] == []

    def test_batch_with_too_many_requests(self):
        """Batch over limit should return error."""
        from jarvis.socket_server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False, preload_models=False, enable_prefetch=False
        )

        requests = [{"method": "ping", "id": i} for i in range(51)]
        msg = json.dumps(
            {"jsonrpc": "2.0", "method": "batch", "params": {"requests": requests}, "id": 1}
        )
        result = asyncio.run(server._process_message(msg))

        parsed = json.loads(result)
        assert "error" in parsed

    def test_batch_with_mixed_valid_invalid(self):
        """Batch with mix of valid and invalid methods handles each independently."""
        from jarvis.socket_server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False, preload_models=False, enable_prefetch=False
        )

        requests = [
            {"method": "ping", "id": 1},
            {"method": "nonexistent", "id": 2},
            {"method": "ping", "id": 3},
        ]
        msg = json.dumps(
            {"jsonrpc": "2.0", "method": "batch", "params": {"requests": requests}, "id": 99}
        )
        result = asyncio.run(server._process_message(msg))

        parsed = json.loads(result)
        assert "result" in parsed
        batch_results = parsed["result"]["results"]
        assert len(batch_results) == 3

        # First and third should succeed
        assert "result" in batch_results[0]
        assert "result" in batch_results[2]

        # Second should have error
        assert "error" in batch_results[1]
        assert batch_results[1]["error"]["code"] == -32601  # METHOD_NOT_FOUND


# =============================================================================
# 12. Prefetch Disabled Graceful Handling
# =============================================================================


class TestPrefetchDisabled:
    """Tests that prefetch-related calls degrade gracefully when disabled."""

    def test_prefetch_stats_when_disabled(self):
        """Prefetch stats should indicate disabled status, not crash."""
        from jarvis.socket_server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False, preload_models=False, enable_prefetch=False
        )

        msg = json.dumps({"jsonrpc": "2.0", "method": "prefetch_stats", "params": {}, "id": 1})
        result = asyncio.run(server._process_message(msg))

        parsed = json.loads(result)
        assert "result" in parsed
        assert parsed["result"]["enabled"] is False

    def test_prefetch_focus_when_disabled(self):
        """Prefetch focus should return disabled status."""
        from jarvis.socket_server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False, preload_models=False, enable_prefetch=False
        )

        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "prefetch_focus",
                "params": {"chat_id": "c1"},
                "id": 1,
            }
        )
        result = asyncio.run(server._process_message(msg))

        parsed = json.loads(result)
        assert "result" in parsed
        assert parsed["result"]["status"] == "disabled"

    def test_prefetch_hover_when_disabled(self):
        """Prefetch hover should return disabled status."""
        from jarvis.socket_server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False, preload_models=False, enable_prefetch=False
        )

        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "prefetch_hover",
                "params": {"chat_id": "c1"},
                "id": 1,
            }
        )
        result = asyncio.run(server._process_message(msg))

        parsed = json.loads(result)
        assert "result" in parsed
        assert parsed["result"]["status"] == "disabled"

    def test_prefetch_invalidate_when_disabled(self):
        """Prefetch invalidate should return zero count when disabled."""
        from jarvis.socket_server import JarvisSocketServer

        server = JarvisSocketServer(
            enable_watcher=False, preload_models=False, enable_prefetch=False
        )

        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "prefetch_invalidate",
                "params": {"chat_id": "c1"},
                "id": 1,
            }
        )
        result = asyncio.run(server._process_message(msg))

        parsed = json.loads(result)
        assert "result" in parsed
        assert parsed["result"]["invalidated"] == 0
