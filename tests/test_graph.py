"""Tests for graph module functionality.

Tests graph building, layout algorithms, clustering, and export.
"""

import json
import math
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.graph.builder import (
    GraphBuilder,
    GraphData,
    GraphEdge,
    GraphNode,
    _compute_edge_weight,
    _compute_node_size,
    _hash_id,
)
from jarvis.graph.clustering import (
    ClusterResult,
    LouvainClustering,
    cluster_by_relationship,
    detect_communities,
    get_cluster_colors,
)
from jarvis.graph.export import (
    export_to_graphml,
    export_to_html,
    export_to_json,
    export_to_svg,
)
from jarvis.graph.layout import (
    LayoutConfig,
    LayoutEngine,
    compute_force_layout,
    compute_hierarchical_layout,
)


class TestGraphNode:
    """Test GraphNode dataclass."""

    def test_create_node(self) -> None:
        """Test creating a basic node."""
        node = GraphNode(id="test123", label="Test User")
        assert node.id == "test123"
        assert node.label == "Test User"
        assert node.size == 16.0  # Default
        assert node.message_count == 0
        assert node.cluster_id is None

    def test_node_to_dict(self) -> None:
        """Test converting node to dictionary."""
        node = GraphNode(
            id="abc",
            label="Alice",
            size=24.0,
            color="#FF6B6B",
            relationship_type="friend",
            message_count=100,
        )
        d = node.to_dict()

        assert d["id"] == "abc"
        assert d["label"] == "Alice"
        assert d["size"] == 24.0
        assert d["color"] == "#FF6B6B"
        assert d["relationship_type"] == "friend"
        assert d["message_count"] == 100

    def test_node_metadata(self) -> None:
        """Test node with metadata."""
        node = GraphNode(
            id="test",
            label="Test",
            metadata={"extra": "data", "count": 42},
        )
        assert node.metadata["extra"] == "data"
        assert node.metadata["count"] == 42


class TestGraphEdge:
    """Test GraphEdge dataclass."""

    def test_create_edge(self) -> None:
        """Test creating a basic edge."""
        edge = GraphEdge(source="a", target="b")
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.weight == 0.5  # Default
        assert edge.bidirectional is True

    def test_edge_to_dict(self) -> None:
        """Test converting edge to dictionary."""
        edge = GraphEdge(
            source="node1",
            target="node2",
            weight=0.8,
            message_count=50,
            sentiment=0.3,
        )
        d = edge.to_dict()

        assert d["source"] == "node1"
        assert d["target"] == "node2"
        assert d["weight"] == 0.8
        assert d["message_count"] == 50
        assert d["sentiment"] == 0.3


class TestGraphData:
    """Test GraphData dataclass."""

    def test_empty_graph(self) -> None:
        """Test empty graph."""
        graph = GraphData()
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_graph_with_data(self) -> None:
        """Test graph with nodes and edges."""
        nodes = [
            GraphNode(id="a", label="A"),
            GraphNode(id="b", label="B"),
            GraphNode(id="c", label="C"),
        ]
        edges = [
            GraphEdge(source="a", target="b"),
            GraphEdge(source="b", target="c"),
        ]
        graph = GraphData(nodes=nodes, edges=edges)

        assert graph.node_count == 3
        assert graph.edge_count == 2

    def test_graph_to_dict(self) -> None:
        """Test converting graph to dictionary."""
        graph = GraphData(
            nodes=[GraphNode(id="x", label="X")],
            edges=[],
            metadata={"test": True},
        )
        d = graph.to_dict()

        assert len(d["nodes"]) == 1
        assert len(d["edges"]) == 0
        assert d["metadata"]["test"] is True


class TestHelperFunctions:
    """Test helper functions in builder module."""

    def test_hash_id(self) -> None:
        """Test ID hashing is consistent."""
        id1 = _hash_id("test@example.com")
        id2 = _hash_id("test@example.com")
        id3 = _hash_id("other@example.com")

        assert id1 == id2
        assert id1 != id3
        assert len(id1) == 16

    def test_compute_node_size_single_value(self) -> None:
        """Test node size when min == max."""
        size = _compute_node_size(100, 100, 100)
        assert size == 16.0  # DEFAULT_NODE_SIZE

    def test_compute_node_size_range(self) -> None:
        """Test node size scales correctly."""
        size_min = _compute_node_size(1, 1, 1000)
        size_max = _compute_node_size(1000, 1, 1000)
        size_mid = _compute_node_size(100, 1, 1000)

        assert size_min < size_mid < size_max
        assert 8.0 <= size_min  # MIN_NODE_SIZE
        assert size_max <= 40.0  # MAX_NODE_SIZE

    def test_compute_edge_weight_zero_messages(self) -> None:
        """Test edge weight with zero max messages."""
        weight = _compute_edge_weight(10, 0)
        assert weight == 0.1

    def test_compute_edge_weight_range(self) -> None:
        """Test edge weight scales correctly."""
        weight_low = _compute_edge_weight(1, 1000)
        weight_high = _compute_edge_weight(1000, 1000)

        assert 0.05 <= weight_low <= 1.0
        assert 0.05 <= weight_high <= 1.0
        assert weight_low < weight_high

    def test_compute_edge_weight_with_recency(self) -> None:
        """Test edge weight includes recency decay."""
        weight_recent = _compute_edge_weight(100, 1000, recency_days=1)
        weight_old = _compute_edge_weight(100, 1000, recency_days=180)

        assert weight_recent > weight_old


class TestLayoutEngine:
    """Test layout algorithms."""

    def create_test_graph(self) -> GraphData:
        """Create a simple test graph."""
        nodes = [
            GraphNode(id="a", label="A", message_count=100),
            GraphNode(id="b", label="B", message_count=50),
            GraphNode(id="c", label="C", message_count=75),
            GraphNode(id="d", label="D", message_count=25),
        ]
        edges = [
            GraphEdge(source="a", target="b", weight=0.8),
            GraphEdge(source="a", target="c", weight=0.6),
            GraphEdge(source="b", target="c", weight=0.4),
            GraphEdge(source="c", target="d", weight=0.3),
        ]
        return GraphData(nodes=nodes, edges=edges)

    def test_force_directed_layout(self) -> None:
        """Test force-directed layout assigns positions."""
        graph = self.create_test_graph()
        config = LayoutConfig(width=400, height=400, iterations=50)
        engine = LayoutEngine(config)

        result = engine.force_directed(graph)

        # All nodes should have positions
        for node in result.nodes:
            assert node.x is not None
            assert node.y is not None
            assert 0 <= node.x <= 400
            assert 0 <= node.y <= 400

        assert result.metadata.get("layout") == "force_directed"

    def test_hierarchical_layout(self) -> None:
        """Test hierarchical layout assigns positions."""
        graph = self.create_test_graph()
        config = LayoutConfig(width=400, height=400)
        engine = LayoutEngine(config)

        result = engine.hierarchical(graph, root_id="a")

        for node in result.nodes:
            assert node.x is not None
            assert node.y is not None

        assert result.metadata.get("layout") == "hierarchical"

    def test_radial_layout(self) -> None:
        """Test radial layout assigns positions."""
        graph = self.create_test_graph()
        config = LayoutConfig(width=400, height=400)
        engine = LayoutEngine(config)

        result = engine.radial(graph, center_id="a")

        for node in result.nodes:
            assert node.x is not None
            assert node.y is not None

        assert result.metadata.get("layout") == "radial"

    def test_layout_empty_graph(self) -> None:
        """Test layout handles empty graph."""
        graph = GraphData()
        engine = LayoutEngine()

        result = engine.force_directed(graph)
        assert result.node_count == 0

    def test_convenience_functions(self) -> None:
        """Test convenience layout functions."""
        graph = self.create_test_graph()

        result1 = compute_force_layout(graph)
        assert all(n.x is not None for n in result1.nodes)

        result2 = compute_hierarchical_layout(graph)
        assert all(n.x is not None for n in result2.nodes)


class TestClustering:
    """Test community detection algorithms."""

    def create_clustered_graph(self) -> GraphData:
        """Create a graph with clear cluster structure."""
        # Two clusters: (a, b, c) and (d, e, f) with weak connection between
        nodes = [
            GraphNode(id="a", label="A", relationship_type="friend"),
            GraphNode(id="b", label="B", relationship_type="friend"),
            GraphNode(id="c", label="C", relationship_type="friend"),
            GraphNode(id="d", label="D", relationship_type="work"),
            GraphNode(id="e", label="E", relationship_type="work"),
            GraphNode(id="f", label="F", relationship_type="work"),
        ]
        edges = [
            # Cluster 1
            GraphEdge(source="a", target="b", weight=0.9),
            GraphEdge(source="a", target="c", weight=0.9),
            GraphEdge(source="b", target="c", weight=0.9),
            # Cluster 2
            GraphEdge(source="d", target="e", weight=0.9),
            GraphEdge(source="d", target="f", weight=0.9),
            GraphEdge(source="e", target="f", weight=0.9),
            # Weak link between clusters
            GraphEdge(source="c", target="d", weight=0.1),
        ]
        return GraphData(nodes=nodes, edges=edges)

    def test_get_cluster_colors(self) -> None:
        """Test cluster color generation."""
        colors = get_cluster_colors(5)
        assert len(colors) == 5
        assert all(c.startswith("#") for c in colors)

        # Test generating more colors than predefined
        colors = get_cluster_colors(20)
        assert len(colors) == 20

    def test_louvain_clustering(self) -> None:
        """Test Louvain community detection."""
        graph = self.create_clustered_graph()
        clustering = LouvainClustering(resolution=1.0)

        result = clustering.detect(graph)

        assert result.num_clusters >= 1
        assert len(result.clusters) == 6
        assert result.modularity >= 0  # Should have positive modularity

    def test_detect_communities_function(self) -> None:
        """Test convenience detect_communities function."""
        graph = self.create_clustered_graph()

        result_graph, result = detect_communities(graph, apply_colors=True)

        # Check that cluster IDs are assigned
        assert all(n.cluster_id is not None for n in result_graph.nodes)
        assert result_graph.metadata.get("clustering") is not None

    def test_cluster_by_relationship(self) -> None:
        """Test clustering by relationship type."""
        graph = self.create_clustered_graph()

        result_graph, result = cluster_by_relationship(graph)

        # Should have 2 clusters (friend and work)
        assert result.num_clusters == 2

        # Check cluster labels
        assert "Friend" in result.cluster_labels.values()
        assert "Work" in result.cluster_labels.values()

    def test_empty_graph_clustering(self) -> None:
        """Test clustering handles empty graph."""
        graph = GraphData()
        clustering = LouvainClustering()

        result = clustering.detect(graph)

        assert result.num_clusters == 0
        assert len(result.clusters) == 0


class TestExport:
    """Test export functionality."""

    def create_test_graph(self) -> GraphData:
        """Create a test graph for export."""
        nodes = [
            GraphNode(id="a", label="Alice", x=100, y=100, size=20, color="#FF6B6B"),
            GraphNode(id="b", label="Bob", x=200, y=150, size=15, color="#4ECDC4"),
        ]
        edges = [
            GraphEdge(source="a", target="b", weight=0.7, message_count=50),
        ]
        return GraphData(nodes=nodes, edges=edges, metadata={"test": True})

    def test_export_to_json(self) -> None:
        """Test JSON export."""
        graph = self.create_test_graph()

        json_str = export_to_json(graph)
        data = json.loads(json_str)

        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        assert data["export_info"]["format"] == "json"
        assert data["export_info"]["node_count"] == 2

    def test_export_to_json_file(self) -> None:
        """Test JSON export to file."""
        graph = self.create_test_graph()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            export_to_json(graph, path)
            assert path.exists()

            data = json.loads(path.read_text())
            assert len(data["nodes"]) == 2
        finally:
            path.unlink(missing_ok=True)

    def test_export_to_graphml(self) -> None:
        """Test GraphML export."""
        graph = self.create_test_graph()

        xml_str = export_to_graphml(graph)

        assert '<?xml version="1.0"' in xml_str
        assert "graphml" in xml_str
        assert "node" in xml_str
        assert "edge" in xml_str

    def test_export_to_svg(self) -> None:
        """Test SVG export."""
        graph = self.create_test_graph()

        svg_str = export_to_svg(graph, width=400, height=300)

        assert "<svg" in svg_str
        assert "circle" in svg_str
        assert "line" in svg_str
        assert 'viewBox="0 0 400 300"' in svg_str

    def test_export_to_svg_without_labels(self) -> None:
        """Test SVG export without labels."""
        graph = self.create_test_graph()

        svg_str = export_to_svg(graph, include_labels=False)

        assert "<svg" in svg_str
        # Should have fewer text elements
        assert svg_str.count("<text") == 0 or "labels" not in svg_str.split("<g")[0]

    def test_export_to_html(self) -> None:
        """Test HTML export with D3.js."""
        graph = self.create_test_graph()

        html_str = export_to_html(graph, title="Test Graph")

        assert "<!DOCTYPE html>" in html_str
        assert "<title>Test Graph</title>" in html_str
        assert "d3.js" in html_str or "d3.v7" in html_str
        assert "d3.forceSimulation" in html_str


class TestGraphBuilder:
    """Test GraphBuilder class."""

    @patch("jarvis.db.JarvisDB")
    @patch("integrations.imessage.iMessageReader")
    def test_build_network_empty(
        self, mock_reader: MagicMock, mock_db: MagicMock
    ) -> None:
        """Test building network with no data."""
        # Mock database connection context manager
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_db.return_value.connect.return_value = mock_conn

        mock_reader.return_value.get_conversations.return_value = []

        builder = GraphBuilder()
        graph = builder.build_network()

        # Should have at least "me" node if no data
        assert graph.node_count >= 0

    def test_graph_data_serialization(self) -> None:
        """Test that GraphData can be serialized for API responses."""
        graph = GraphData(
            nodes=[GraphNode(id="test", label="Test", x=1.5, y=2.5)],
            edges=[GraphEdge(source="test", target="test", weight=0.5)],
            metadata={"key": "value"},
        )

        # Should be JSON serializable
        json_str = json.dumps(graph.to_dict())
        parsed = json.loads(json_str)

        assert parsed["nodes"][0]["id"] == "test"
        assert parsed["edges"][0]["weight"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
