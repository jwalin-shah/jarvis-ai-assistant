"""Tests for KnowledgeGraph: search index, query methods, and performance.

Tests build_from_db with mocked DB, verifying:
- Batch graph construction
- Inverted search index
- search_facts performance (indexed vs full scan)
- query_contact and find_connections
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from jarvis.graph.knowledge_graph import KnowledgeGraph, KnowledgeGraphData


def _make_fact_row(contact_id, category, subject, predicate, value, confidence):
    """Create a mock fact row with dict-like access."""
    row = MagicMock()
    row.keys.return_value = [
        "contact_id", "category", "subject", "predicate", "value", "confidence"
    ]
    data = {
        "contact_id": contact_id,
        "category": category,
        "subject": subject,
        "predicate": predicate,
        "value": value,
        "confidence": confidence,
    }
    row.__getitem__ = lambda self, key: data[key]
    row.__contains__ = lambda self, key: key in data
    return row


def _make_profile_row(contact_id, contact_name, relationship, message_count):
    """Create a mock profile row with dict-like access."""
    row = MagicMock()
    row.keys.return_value = [
        "contact_id", "contact_name", "relationship", "message_count"
    ]
    data = {
        "contact_id": contact_id,
        "contact_name": contact_name,
        "relationship": relationship,
        "message_count": message_count,
    }
    row.__getitem__ = lambda self, key: data[key]
    row.__contains__ = lambda self, key: key in data
    return row


@pytest.fixture
def mock_db():
    """Mock the jarvis DB with sample profiles and facts."""
    profiles = [
        _make_profile_row("alice", "Alice Smith", "friend", 200),
        _make_profile_row("bob", "Bob Jones", "work", 150),
        _make_profile_row("carol", "Carol Lee", "family", 300),
    ]

    facts = [
        _make_fact_row("alice", "location", "San Francisco", "lives_in", "", 0.9),
        _make_fact_row("alice", "work", "Google", "works_at", "Engineer", 0.85),
        _make_fact_row("alice", "preference", "hiking", "enjoys", "", 0.7),
        _make_fact_row("bob", "location", "San Francisco", "lives_in", "", 0.8),
        _make_fact_row("bob", "work", "Apple", "works_at", "Designer", 0.9),
        _make_fact_row("carol", "relationship", "Alice Smith", "is_sister_of", "", 0.95),
        _make_fact_row("carol", "location", "New York", "lives_in", "", 0.8),
        _make_fact_row("carol", "preference", "cooking", "enjoys", "", 0.75),
    ]

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    # Set up execute to return different results based on query
    def execute_side_effect(query):
        result = MagicMock()
        if "contact_profiles" in query:
            result.fetchall.return_value = profiles
        elif "contact_facts" in query:
            result.fetchall.return_value = facts
        else:
            result.fetchall.return_value = []
        return result

    mock_conn.execute.side_effect = execute_side_effect

    mock_db_instance = MagicMock()
    mock_db_instance.connection.return_value = mock_conn

    return mock_db_instance


@pytest.fixture
def kg(mock_db):
    """Build a KnowledgeGraph from mocked DB."""
    with patch("jarvis.db.get_db", return_value=mock_db):
        graph = KnowledgeGraph()
        graph.build_from_db()
    return graph


class TestKnowledgeGraphBuild:
    """Test graph construction from DB."""

    def test_builds_contact_nodes(self, kg: KnowledgeGraph) -> None:
        """Contact profiles become contact nodes."""
        contacts = [
            nid for nid, attrs in kg.graph.nodes(data=True)
            if attrs.get("node_type") == "contact"
        ]
        assert len(contacts) == 3
        assert "alice" in contacts
        assert "bob" in contacts
        assert "carol" in contacts

    def test_builds_entity_nodes(self, kg: KnowledgeGraph) -> None:
        """Unique subjects become entity nodes."""
        entities = [
            nid for nid, attrs in kg.graph.nodes(data=True)
            if attrs.get("node_type") == "entity"
        ]
        # san francisco, google, hiking, apple, alice smith, new york, cooking = 7
        assert len(entities) == 7

    def test_deduplicates_entities(self, kg: KnowledgeGraph) -> None:
        """Same subject from different contacts creates one entity node."""
        # Both alice and bob live in San Francisco
        sf_nodes = [
            nid for nid in kg.graph.nodes()
            if nid == "entity:san francisco"
        ]
        assert len(sf_nodes) == 1

    def test_builds_edges(self, kg: KnowledgeGraph) -> None:
        """Each fact creates an edge from contact to entity."""
        assert kg.graph.number_of_edges() == 8  # 8 facts = 8 edges

    def test_edge_attributes(self, kg: KnowledgeGraph) -> None:
        """Edges carry predicate and confidence as attributes."""
        edges = list(kg.graph.edges("alice", data=True))
        work_edges = [e for e in edges if e[2].get("edge_type") == "works_at"]
        assert len(work_edges) == 1
        assert work_edges[0][2]["weight"] == 0.85
        assert "Engineer" in work_edges[0][2]["label"]

    def test_entity_colors(self, kg: KnowledgeGraph) -> None:
        """Entity nodes get category-appropriate colors."""
        sf_attrs = kg.graph.nodes["entity:san francisco"]
        assert sf_attrs["color"] == "#4ECDC4"  # location = teal


class TestSearchIndex:
    """Test inverted search index."""

    def test_index_built(self, kg: KnowledgeGraph) -> None:
        """Search index is populated during build."""
        assert len(kg._search_index) > 0
        assert len(kg._edge_cache) == 8  # One per edge

    def test_token_lookup(self, kg: KnowledgeGraph) -> None:
        """Index maps tokens to edge references."""
        # "francisco" should appear in index (from "san francisco")
        assert "francisco" in kg._search_index
        refs = kg._search_index["francisco"]
        assert len(refs) == 2  # alice and bob both live in SF

    def test_search_by_entity(self, kg: KnowledgeGraph) -> None:
        """search_facts finds facts by entity name."""
        results = kg.search_facts("Google")
        assert len(results) == 1
        assert results[0]["source"] == "alice"
        assert results[0]["edge_type"] == "works_at"

    def test_search_by_location(self, kg: KnowledgeGraph) -> None:
        """search_facts finds all contacts in a location."""
        results = kg.search_facts("San Francisco")
        assert len(results) == 2
        sources = {r["source"] for r in results}
        assert sources == {"alice", "bob"}

    def test_search_case_insensitive(self, kg: KnowledgeGraph) -> None:
        """Search is case-insensitive."""
        results_lower = kg.search_facts("google")
        results_upper = kg.search_facts("GOOGLE")
        assert len(results_lower) == len(results_upper) == 1

    def test_search_no_results(self, kg: KnowledgeGraph) -> None:
        """Search returns empty list for non-matching query."""
        results = kg.search_facts("nonexistent")
        assert results == []

    def test_search_empty_query(self, kg: KnowledgeGraph) -> None:
        """Empty query returns empty list."""
        assert kg.search_facts("") == []
        assert kg.search_facts("   ") == []

    def test_search_limit(self, kg: KnowledgeGraph) -> None:
        """Search respects limit parameter."""
        results = kg.search_facts("lives", limit=1)
        assert len(results) <= 1

    def test_search_on_empty_graph(self) -> None:
        """Search on unbuilt graph returns empty."""
        kg = KnowledgeGraph()
        assert kg.search_facts("anything") == []


class TestQueryMethods:
    """Test query_contact and find_connections."""

    def test_query_contact_facts(self, kg: KnowledgeGraph) -> None:
        """query_contact returns all facts for a contact."""
        result = kg.query_contact("alice")
        assert result["contact_id"] == "alice"
        assert result["label"] == "Alice Smith"
        assert len(result["facts"]) == 3  # SF, Google, hiking

    def test_query_contact_not_found(self, kg: KnowledgeGraph) -> None:
        """query_contact returns empty for unknown contact."""
        result = kg.query_contact("unknown")
        assert result["facts"] == []
        assert result["connections"] == []

    def test_find_connections_shared_entity(self, kg: KnowledgeGraph) -> None:
        """find_connections returns contacts sharing an entity."""
        contacts = kg.find_connections("San Francisco")
        assert set(contacts) == {"alice", "bob"}

    def test_find_connections_not_found(self, kg: KnowledgeGraph) -> None:
        """find_connections returns empty for unknown entity."""
        assert kg.find_connections("Mars") == []


class TestToGraphData:
    """Test serialization to KnowledgeGraphData."""

    def test_to_graph_data(self, kg: KnowledgeGraph) -> None:
        """to_graph_data returns serializable data."""
        data = kg.to_graph_data()
        assert isinstance(data, KnowledgeGraphData)
        assert data.metadata["total_nodes"] == 10  # 3 contacts + 7 entities
        assert data.metadata["total_edges"] == 8
        assert data.metadata["contact_count"] == 3
        assert data.metadata["entity_count"] == 7

    def test_to_graph_data_empty(self) -> None:
        """to_graph_data on empty graph returns empty data."""
        kg = KnowledgeGraph()
        data = kg.to_graph_data()
        assert data.nodes == []
        assert data.edges == []

    def test_node_serialization(self, kg: KnowledgeGraph) -> None:
        """Nodes have expected fields for API consumption."""
        data = kg.to_graph_data()
        contact_nodes = [n for n in data.nodes if n["node_type"] == "contact"]
        assert len(contact_nodes) == 3
        alice = next(n for n in contact_nodes if n["id"] == "alice")
        assert alice["label"] == "Alice Smith"
        assert alice["relationship_type"] == "friend"
        assert alice["message_count"] == 200


class TestSearchPerformance:
    """Performance tests for search operations."""

    def _build_large_kg(self, num_contacts: int, facts_per_contact: int):
        """Build a KG with many contacts and facts for perf testing."""
        profiles = [
            _make_profile_row(f"contact_{i}", f"Contact {i}", "friend", i * 10)
            for i in range(num_contacts)
        ]
        facts = []
        for i in range(num_contacts):
            for j in range(facts_per_contact):
                facts.append(_make_fact_row(
                    f"contact_{i}",
                    "preference",
                    f"entity_{i}_{j}",
                    "enjoys",
                    f"value_{j}",
                    0.8,
                ))

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        def execute_side_effect(query):
            result = MagicMock()
            if "contact_profiles" in query:
                result.fetchall.return_value = profiles
            elif "contact_facts" in query:
                result.fetchall.return_value = facts
            else:
                result.fetchall.return_value = []
            return result

        mock_conn.execute.side_effect = execute_side_effect

        mock_db_inst = MagicMock()
        mock_db_inst.connection.return_value = mock_conn

        with patch("jarvis.db.get_db", return_value=mock_db_inst):
            kg = KnowledgeGraph()
            kg.build_from_db()
        return kg

    def test_build_performance(self) -> None:
        """Graph build with 100 contacts x 10 facts completes in <500ms."""
        start = time.perf_counter()
        kg = self._build_large_kg(100, 10)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert kg.graph.number_of_nodes() > 0
        assert kg.graph.number_of_edges() == 1000
        assert elapsed_ms < 500, f"Build too slow: {elapsed_ms:.1f}ms (should be <500ms)"

    def test_indexed_search_performance(self) -> None:
        """Indexed search on 5000 edges completes in <5ms."""
        kg = self._build_large_kg(500, 10)

        # Search for a specific entity
        start = time.perf_counter()
        for _ in range(100):
            kg.search_facts("entity_0_0")
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        assert elapsed_ms < 5, f"Search too slow: {elapsed_ms:.2f}ms avg (should be <5ms)"

    def test_search_faster_than_full_scan(self) -> None:
        """Indexed search is faster than iterating all edges."""
        kg = self._build_large_kg(500, 10)

        # Time indexed search
        start = time.perf_counter()
        for _ in range(50):
            kg.search_facts("entity_250_5")
        indexed_ms = (time.perf_counter() - start) * 1000

        # Time manual full scan (simulating old behavior)
        start = time.perf_counter()
        query_lower = "entity_250_5"
        for _ in range(50):
            results = []
            for src, tgt, attrs in kg.graph.edges(data=True):
                label = attrs.get("label", "").lower()
                tgt_label = kg.graph.nodes.get(tgt, {}).get("label", "").lower()
                if query_lower in label or query_lower in tgt_label:
                    results.append({"source": src, "target": tgt})
        scan_ms = (time.perf_counter() - start) * 1000

        # Indexed should be meaningfully faster
        assert indexed_ms < scan_ms, (
            f"Index ({indexed_ms:.1f}ms) not faster than scan ({scan_ms:.1f}ms)"
        )
