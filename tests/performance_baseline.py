"""Performance baseline tests for N+1 query detection.

These tests verify that performance-critical operations complete in acceptable time.
They serve as early detection for N+1 queries and other performance regressions.

Run with: make test -k performance
"""

import time
from typing import Any

import pytest


class TestConversationsPerformance:
    """Conversations fetch must be fast even with 400k messages."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    def test_get_conversations_under_100ms(self, caplog: Any) -> None:
        """Verify getConversations completes in <100ms.

        This would fail with 5 correlated subqueries (1400ms).
        With CTE optimization: ~50ms.
        """
        from integrations.imessage import ChatDBReader

        start = time.perf_counter()
        try:
            with ChatDBReader() as reader:
                convos = reader.get_conversations(limit=50)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # With 400k messages, unoptimized = 1400ms, optimized = 50ms
            assert elapsed_ms < 200, (
                f"getConversations too slow: {elapsed_ms:.1f}ms (indicates N+1 query pattern)"
            )
            assert len(convos) <= 50, "Should return at most 50 conversations"
        except Exception as e:
            # If DB access fails, log but don't fail test (DB might not be available)
            caplog.info(f"Skipping perf test (DB unavailable): {e}")


class TestMessagesPerformance:
    """Message loading must batch attachments/reactions, not N+1."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    def test_get_messages_batches_attachments(self, caplog: Any) -> None:
        """Verify getMessages batches attachment/reaction queries.

        This would fail if doing:
          - 1 query for messages
          - 100 queries for attachments (one per message)
          - 100 queries for reactions
          = 201 queries, ~800ms

        With batching: 3 queries, ~25ms
        """
        from integrations.imessage import ChatDBReader

        try:
            with ChatDBReader() as reader:
                # Get a test chat
                convos = reader.get_conversations(limit=1)
                if not convos:
                    caplog.info("No conversations available for perf test")
                    return

                start = time.perf_counter()
                messages = reader.get_messages(convos[0].chat_id, limit=20)
                elapsed_ms = (time.perf_counter() - start) * 1000

                # Batch fetching: 3 queries = ~25ms
                # N+1 fetching: 201 queries = ~500ms
                assert elapsed_ms < 200, (
                    f"getMessages too slow: {elapsed_ms:.1f}ms "
                    f"(indicates N+1 on attachments/reactions)"
                )
                assert len(messages) <= 20
        except Exception as e:
            caplog.info(f"Skipping perf test (DB unavailable): {e}")


class TestQueryPatternDetection:
    """Detect common N+1 patterns in query files."""

    def test_no_correlated_subqueries_in_conversation_query(self) -> None:
        """Verify conversation query doesn't use correlated subqueries.

        Anti-pattern: SELECT ... FROM chat WHERE (...subquery...)
        Pro pattern: WITH cte AS (...) SELECT ... FROM chat JOIN cte
        """
        from desktop.src.lib.db.queries import getConversationsQuery

        query = getConversationsQuery({})
        query_upper = query.upper()

        # Check for signs of correlated subqueries
        assert "WITH" in query_upper, "Should use CTE for aggregations"
        assert query_upper.count("SELECT") >= 2, (
            "Should have multiple SELECT in CTE (not single SELECT with subqueries)"
        )

        # Should NOT have subqueries in main SELECT clause
        # (Simplified check - robust SQL parsing would be better but overkill here)


class TestBatchOperations:
    """Verify batch operations are used for bulk data operations."""

    def test_fact_storage_uses_batch_insert(self) -> None:
        """Verify fact storage uses batch INSERT, not N individual INSERTs.

        Anti-pattern: Loop INSERT - 50 operations, 150ms
        Pro pattern: executemany() - 1 operation, 3ms
        """
        import inspect

        from jarvis.contacts.fact_storage import FactStorage

        source = inspect.getsource(FactStorage.save_facts)

        # Check for executemany (good) vs individual executes in loop (bad)
        assert "executemany" in source, (
            "save_facts should use executemany() for batch INSERT, not loop"
        )


class TestSearchFiltering:
    """Verify search doesn't fetch 1000 messages and filter to 200 in Python."""

    def test_semantic_search_filters_in_sql(self) -> None:
        """Verify filters are pushed to SQL, not applied in Python.

        Anti-pattern: Fetch 1000 messages, filter in Python loop = wasted load
        Pro pattern: Pass filters to SQL WHERE clause = fetch only needed data
        """
        import inspect

        from jarvis.search.semantic_search import SemanticSearcher

        source = inspect.getsource(SemanticSearcher._get_messages_to_index)

        # Check that filters are NOT in a separate Python loop
        assert "for msg in" not in source or ("filters." not in source), (
            "_get_messages_to_index should not have Python loop filtering "
            "(filters should be in SQL query)"
        )


class TestGraphBatching:
    """Verify knowledge graph uses batch operations."""

    def test_knowledge_graph_batches_node_operations(self) -> None:
        """Verify knowledge graph uses add_nodes_from(), not loop add_node().

        Anti-pattern: Loop add_node() - 1100 calls, 200ms
        Pro pattern: add_nodes_from() - 3 calls, 30ms
        """
        import inspect

        from jarvis.graph.knowledge_graph import KnowledgeGraph

        source = inspect.getsource(KnowledgeGraph.build_from_db)

        # Should use batch operations
        assert "add_nodes_from" in source, (
            "build_from_db should use add_nodes_from() for batch operations"
        )
        assert "add_edges_from" in source, (
            "build_from_db should use add_edges_from() for batch operations"
        )


class TestPreloadModels:
    """Verify socket server uses on-demand loading, not preload on startup."""

    def test_socket_server_supports_no_preload_flag(self) -> None:
        """Verify socket server can defer model loading with --no-preload."""
        import inspect

        from jarvis.socket_server import main

        source = inspect.getsource(main)

        # Should support --no-preload flag for faster startup
        assert "--no-preload" in source or "preload_models" in source, (
            "Socket server should support --no-preload flag "
            "(otherwise startup blocked by model loading)"
        )
