"""Tests for topic discovery (jarvis/topics/topic_discovery.py)."""

from __future__ import annotations

import numpy as np

from jarvis.topics.topic_discovery import TopicCluster, TopicDiscovery, TopicResult, _tokenize


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = _tokenize("I love eating sushi for dinner")
        assert "sushi" in tokens
        assert "dinner" in tokens
        assert "love" in tokens
        # Stopwords removed
        assert "for" not in tokens

    def test_short_words_excluded(self):
        tokens = _tokenize("go to be")
        assert tokens == []

    def test_case_insensitive(self):
        tokens = _tokenize("SUSHI dinner LUNCH")
        assert "sushi" in tokens
        assert "dinner" in tokens
        assert "lunch" in tokens


class TestTopicDiscovery:
    def _make_cluster_embeddings(
        self, n_per_cluster: int = 30, n_clusters: int = 3, dim: int = 384
    ) -> tuple[np.ndarray, list[str]]:
        """Generate synthetic embeddings with clear clusters."""
        rng = np.random.RandomState(42)
        embeddings = []
        texts = []

        cluster_topics = [
            ["dinner", "restaurant", "sushi", "food", "eating", "lunch", "pizza"],
            ["meeting", "project", "deadline", "office", "presentation", "work", "report"],
            ["hiking", "camping", "trail", "mountain", "outdoor", "nature", "adventure"],
        ]

        for i in range(n_clusters):
            # Create a centroid for this cluster
            centroid = rng.randn(dim).astype(np.float32)
            centroid = centroid / np.linalg.norm(centroid)
            # Add points around centroid with small noise
            for j in range(n_per_cluster):
                noise = rng.randn(dim).astype(np.float32) * 0.1
                emb = centroid + noise
                emb = emb / np.linalg.norm(emb)
                embeddings.append(emb)
                # Pick random words from topic
                words = rng.choice(cluster_topics[i], size=3, replace=True)
                texts.append(f"Let's talk about {' and '.join(words)} today")

        return np.array(embeddings, dtype=np.float32), texts

    def test_discovers_clusters(self):
        embeddings, texts = self._make_cluster_embeddings(n_per_cluster=30, n_clusters=3)
        discovery = TopicDiscovery()
        result = discovery.discover_topics(
            contact_id="test",
            embeddings=embeddings,
            texts=texts,
            min_cluster_size=10,
            min_samples=5,
        )

        assert isinstance(result, TopicResult)
        # Should find at least 2 of the 3 clusters (HDBSCAN may merge close ones)
        assert len(result.topics) >= 2
        # Each topic should have keywords
        for topic in result.topics:
            assert isinstance(topic, TopicCluster)
            assert len(topic.keywords) > 0
            assert topic.message_count >= 10
            assert topic.centroid.shape == (384,)
            assert len(topic.sample_texts) <= 3

    def test_too_few_messages_returns_empty(self):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(5, 384).astype(np.float32)
        texts = ["hello"] * 5

        discovery = TopicDiscovery()
        result = discovery.discover_topics(
            contact_id="test",
            embeddings=embeddings,
            texts=texts,
            min_cluster_size=10,
        )
        assert result.topics == []
        assert result.noise_count == 5

    def test_topics_sorted_by_message_count(self):
        embeddings, texts = self._make_cluster_embeddings(n_per_cluster=30, n_clusters=3)
        discovery = TopicDiscovery()
        result = discovery.discover_topics(
            contact_id="test",
            embeddings=embeddings,
            texts=texts,
            min_cluster_size=10,
            min_samples=5,
        )

        if len(result.topics) >= 2:
            # Verify sorted descending by message_count
            counts = [t.message_count for t in result.topics]
            assert counts == sorted(counts, reverse=True)


class TestExtractKeywords:
    def test_keywords_from_cluster(self):
        cluster_texts = [
            "sushi restaurant downtown",
            "best sushi place for dinner",
            "sushi and ramen tonight",
        ]
        # Add many non-sushi texts to boost sushi's IDF
        all_texts = cluster_texts + [
            "office meeting tomorrow",
            "project deadline friday",
            "hiking trail weekend",
            "camping nature adventure",
            "morning coffee routine",
            "gym workout session",
            "reading books evening",
        ]

        discovery = TopicDiscovery()
        keywords = discovery._extract_keywords(cluster_texts, all_texts, top_k=5)

        assert len(keywords) >= 1
        # "sushi" should be a top keyword (high TF in cluster, appears only in cluster)
        assert "sushi" in keywords
