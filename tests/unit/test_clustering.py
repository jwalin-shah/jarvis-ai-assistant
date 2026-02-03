"""Unit tests for the cluster analysis module.

Tests cover the ClusterAnalyzer class, cluster creation, pair assignments,
and edge cases like insufficient data.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.clustering import (
    MIN_CLUSTER_SIZE,
    ClusterAnalyzer,
    ClusteringStats,
    run_cluster_analysis,
)
from jarvis.db import JarvisDB


class TestClusterAnalyzer:
    """Tests for ClusterAnalyzer class."""

    @pytest.fixture
    def mock_db(self, tmp_path: Path) -> JarvisDB:
        """Create a test database with sample data."""
        db_path = tmp_path / "test_jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()

        # Add a contact
        contact = db.add_contact(display_name="Test User", relationship="friend")

        # Add sample pairs - need at least MIN_PAIRS_FOR_CLUSTERING
        now = datetime.now()
        for i in range(100):
            db.add_pair(
                trigger_text=f"Test trigger {i % 10}",  # Create some repeating patterns
                response_text=f"Test response {i}",
                trigger_timestamp=now,
                response_timestamp=now,
                chat_id=f"chat_{i % 5}",
                contact_id=contact.id,
                quality_score=0.8,
            )

        return db

    @pytest.fixture
    def mock_embedder(self) -> MagicMock:
        """Create a mock embedder."""
        embedder = MagicMock()

        # Return random but deterministic embeddings
        def encode_side_effect(texts, normalize=True):
            np.random.seed(42)
            embeddings = np.random.randn(len(texts), 384).astype(np.float32)
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
            return embeddings

        embedder.encode = MagicMock(side_effect=encode_side_effect)
        return embedder

    def test_analyze_creates_clusters(self, mock_db: JarvisDB, mock_embedder: MagicMock) -> None:
        """Test that analyze() creates clusters in the database."""
        analyzer = ClusterAnalyzer(db=mock_db, embedder=mock_embedder)

        stats = analyzer.analyze(n_clusters=5, min_cluster_size=3)

        # Should have created some clusters
        assert stats.clusters_created > 0
        assert stats.pairs_clustered > 0
        assert stats.total_pairs == 100

        # Verify clusters exist in database
        clusters = mock_db.list_clusters()
        assert len(clusters) == stats.clusters_created

    def test_analyze_assigns_pairs_to_clusters(
        self, mock_db: JarvisDB, mock_embedder: MagicMock
    ) -> None:
        """Test that analyze() assigns pairs to clusters."""
        analyzer = ClusterAnalyzer(db=mock_db, embedder=mock_embedder)

        stats = analyzer.analyze(n_clusters=5, min_cluster_size=3)

        # Check that some pairs have cluster assignments
        pairs = mock_db.get_training_pairs(min_quality=0.0)
        pairs_with_clusters = [p for p in pairs if p.cluster_id is not None]

        assert len(pairs_with_clusters) > 0
        assert len(pairs_with_clusters) == stats.pairs_clustered

    def test_analyze_with_insufficient_pairs(self, tmp_path: Path) -> None:
        """Test that analyze() handles insufficient data gracefully."""
        db_path = tmp_path / "empty_jarvis.db"
        db = JarvisDB(db_path)
        db.init_schema()

        # Add only a few pairs (less than MIN_PAIRS_FOR_CLUSTERING)
        contact = db.add_contact(display_name="Test", relationship="friend")
        now = datetime.now()
        for i in range(10):
            db.add_pair(
                trigger_text=f"Test {i}",
                response_text=f"Response {i}",
                trigger_timestamp=now,
                response_timestamp=now,
                chat_id="test",
                contact_id=contact.id,
                quality_score=0.8,
            )

        mock_embedder = MagicMock()
        analyzer = ClusterAnalyzer(db=db, embedder=mock_embedder)

        stats = analyzer.analyze()

        # Should return stats with 0 clusters
        assert stats.clusters_created == 0
        assert stats.pairs_clustered == 0
        assert stats.noise_count == 10  # All pairs are "noise"

        # Embedder should not be called
        mock_embedder.encode.assert_not_called()

    def test_analyze_respects_min_cluster_size(
        self, mock_db: JarvisDB, mock_embedder: MagicMock
    ) -> None:
        """Test that clusters smaller than min_cluster_size are filtered."""
        analyzer = ClusterAnalyzer(db=mock_db, embedder=mock_embedder)

        # Use a large min_cluster_size to filter out small clusters
        stats = analyzer.analyze(n_clusters=20, min_cluster_size=50)

        # With 100 pairs and 20 clusters, most will have < 50 pairs
        # So we expect fewer clusters
        assert stats.clusters_created < 20

    def test_analyze_clears_existing_clusters(
        self, mock_db: JarvisDB, mock_embedder: MagicMock
    ) -> None:
        """Test that analyze() clears existing clusters before creating new ones."""
        # Add some existing clusters
        mock_db.add_cluster(name="OLD_CLUSTER_1", description="Old cluster")
        mock_db.add_cluster(name="OLD_CLUSTER_2", description="Old cluster")

        assert len(mock_db.list_clusters()) == 2

        analyzer = ClusterAnalyzer(db=mock_db, embedder=mock_embedder)
        analyzer.analyze(n_clusters=3)

        # Old clusters should be gone
        clusters = mock_db.list_clusters()
        cluster_names = [c.name for c in clusters]
        assert "OLD_CLUSTER_1" not in cluster_names
        assert "OLD_CLUSTER_2" not in cluster_names

    def test_cluster_name_generation(self, mock_db: JarvisDB, mock_embedder: MagicMock) -> None:
        """Test that cluster names are generated from trigger content."""
        analyzer = ClusterAnalyzer(db=mock_db, embedder=mock_embedder)
        analyzer.analyze(n_clusters=5)

        clusters = mock_db.list_clusters()
        for cluster in clusters:
            # Names should be uppercase and not just "CLUSTER_N"
            assert cluster.name is not None
            assert len(cluster.name) > 0
            assert cluster.name.isupper()


class TestClusteringStats:
    """Tests for ClusteringStats dataclass."""

    def test_stats_initialization(self) -> None:
        """Test ClusteringStats can be initialized with all fields."""
        stats = ClusteringStats(
            total_pairs=1000,
            pairs_clustered=800,
            clusters_created=10,
            noise_count=200,
            largest_cluster_size=150,
            smallest_cluster_size=50,
            avg_cluster_size=80.0,
        )

        assert stats.total_pairs == 1000
        assert stats.pairs_clustered == 800
        assert stats.clusters_created == 10
        assert stats.noise_count == 200
        assert stats.largest_cluster_size == 150
        assert stats.smallest_cluster_size == 50
        assert stats.avg_cluster_size == 80.0


class TestRunClusterAnalysis:
    """Tests for the run_cluster_analysis convenience function."""

    @patch("jarvis.clustering.ClusterAnalyzer")
    def test_run_cluster_analysis_creates_analyzer(self, mock_analyzer_class: MagicMock) -> None:
        """Test that run_cluster_analysis creates and uses a ClusterAnalyzer."""
        mock_instance = MagicMock()
        mock_instance.analyze.return_value = ClusteringStats(
            total_pairs=100,
            pairs_clustered=80,
            clusters_created=5,
            noise_count=20,
            largest_cluster_size=20,
            smallest_cluster_size=10,
            avg_cluster_size=16.0,
        )
        mock_analyzer_class.return_value = mock_instance

        stats = run_cluster_analysis(n_clusters=10, min_cluster_size=5)

        mock_analyzer_class.assert_called_once()
        mock_instance.analyze.assert_called_once_with(
            n_clusters=10,
            min_cluster_size=5,
            min_quality=0.3,
        )
        assert stats.clusters_created == 5


class TestClusterEmbeddings:
    """Tests for the internal clustering logic."""

    def test_cluster_embeddings_returns_correct_shape(self) -> None:
        """Test that _cluster_embeddings returns correct shapes."""
        # Create sample embeddings
        np.random.seed(42)
        embeddings = np.random.randn(100, 384).astype(np.float32)

        # Create analyzer with mocked db
        mock_db = MagicMock()
        analyzer = ClusterAnalyzer(db=mock_db, embedder=MagicMock())

        labels, centroids = analyzer._cluster_embeddings(embeddings, n_clusters=5)

        assert labels.shape == (100,)
        assert centroids.shape == (5, 384)
        assert set(labels).issubset(set(range(5)))

    def test_cluster_embeddings_adjusts_n_clusters(self) -> None:
        """Test that clustering adjusts n_clusters when data is small."""
        # Create very small dataset
        np.random.seed(42)
        embeddings = np.random.randn(10, 384).astype(np.float32)

        mock_db = MagicMock()
        analyzer = ClusterAnalyzer(db=mock_db, embedder=MagicMock())

        # Request more clusters than we can support
        labels, centroids = analyzer._cluster_embeddings(embeddings, n_clusters=20)

        # Should have adjusted to 2 clusters (10 samples / 5 min_size = 2)
        assert len(centroids) <= 10 // MIN_CLUSTER_SIZE


class TestClusterNameGeneration:
    """Tests for cluster name generation."""

    def test_generate_cluster_name_from_triggers(self) -> None:
        """Test that cluster names are generated from trigger content."""
        mock_db = MagicMock()
        analyzer = ClusterAnalyzer(db=mock_db, embedder=MagicMock())

        sample_triggers = [
            "Want to grab lunch tomorrow?",
            "Lunch today?",
            "Let's do lunch",
        ]

        name = analyzer._generate_cluster_name(sample_triggers, cluster_id=0)

        # Should contain common words from triggers
        assert "LUNCH" in name or "GRAB" in name or "WANT" in name

    def test_generate_cluster_name_fallback(self) -> None:
        """Test fallback naming when no meaningful words found."""
        mock_db = MagicMock()
        analyzer = ClusterAnalyzer(db=mock_db, embedder=MagicMock())

        # All stop words
        sample_triggers = ["the is a", "to be or", "it was"]

        name = analyzer._generate_cluster_name(sample_triggers, cluster_id=42)

        # Should fallback to CLUSTER_N format (no base name, just ID)
        assert "42" in name

    def test_generate_cluster_name_filters_stopwords(self) -> None:
        """Test that stop words are filtered from cluster names."""
        mock_db = MagicMock()
        analyzer = ClusterAnalyzer(db=mock_db, embedder=MagicMock())

        sample_triggers = [
            "The dinner is at seven",
            "Dinner with the family",
            "Having dinner tonight",
        ]

        name = analyzer._generate_cluster_name(sample_triggers, cluster_id=0)

        # Should contain "DINNER" but not stop words like "THE", "IS", "AT"
        assert "DINNER" in name
        assert "THE" not in name.split("_")
        assert "IS" not in name.split("_")
