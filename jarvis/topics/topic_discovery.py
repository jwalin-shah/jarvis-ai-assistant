"""Topic discovery via HDBSCAN clustering on message embeddings.

Finds natural topic clusters in conversations with each contact.
Used by ContactProfileBuilder._discover_topics() to populate
ContactProfile.top_topics, which feeds into the prompt's <style> section.

Usage:
    from jarvis.topics.topic_discovery import TopicDiscovery

    discovery = TopicDiscovery()
    result = discovery.discover_topics(
        contact_id="chat123",
        embeddings=embeddings,  # (n, 384)
        texts=texts,
    )
    for topic in result.topics:
        print(topic.keywords, topic.message_count)
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# TF-IDF stopwords (common English words to exclude from topic keywords)
_STOPWORDS = frozenset(
    {
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "they",
        "them",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "can",
        "may",
        "might",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "and",
        "but",
        "or",
        "not",
        "no",
        "so",
        "if",
        "that",
        "this",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
        "just",
        "like",
        "get",
        "got",
        "go",
        "going",
        "know",
        "think",
        "want",
        "yeah",
        "yes",
        "ok",
        "okay",
        "oh",
        "lol",
        "haha",
        "im",
        "dont",
        "its",
        "thats",
        "too",
        "about",
        "up",
        "out",
        "one",
        "all",
        "been",
        "more",
        "some",
        "than",
        "then",
        "also",
        "back",
        "there",
        "here",
        "now",
        "well",
        "still",
        "really",
        "right",
        "good",
        "much",
        "very",
        "said",
    }
)


@dataclass
class TopicCluster:
    """A discovered topic cluster from message embeddings."""

    cluster_id: int
    keywords: list[str]  # Top 5 representative words (TF-IDF weighted)
    message_count: int
    centroid: NDArray[np.float32]  # (384,) mean embedding
    sample_texts: list[str] = field(default_factory=list)  # 3 representative messages


@dataclass
class TopicResult:
    """Result of topic discovery for a contact."""

    topics: list[TopicCluster]
    noise_count: int  # Messages not assigned to any cluster


class TopicDiscovery:
    """Discover conversation topics via HDBSCAN clustering."""

    def discover_topics(
        self,
        contact_id: str,
        embeddings: NDArray[np.float32],
        texts: list[str],
        min_cluster_size: int = 10,
        min_samples: int = 5,
    ) -> TopicResult:
        """Cluster message embeddings and extract topic keywords.

        Args:
            contact_id: Contact ID (for logging).
            embeddings: Message embeddings, shape (n_messages, 384).
            texts: Corresponding message texts.
            min_cluster_size: HDBSCAN minimum cluster size.
            min_samples: HDBSCAN minimum samples for core points.

        Returns:
            TopicResult with discovered topics sorted by message count.
        """
        from sklearn.cluster import HDBSCAN

        n = len(embeddings)
        if n < min_cluster_size:
            return TopicResult(topics=[], noise_count=n)

        # HDBSCAN clustering (no preset k, finds natural clusters)
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            n_jobs=1,  # 8GB RAM constraint
        )
        labels = clusterer.fit_predict(embeddings)

        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        noise_count = int(np.sum(labels == -1))

        # Pre-compute IDF (document frequency per word) ONCE across all texts,
        # rather than re-tokenizing all_texts inside each cluster's keyword extraction.
        precomputed_idf = _compute_doc_frequencies(texts)

        topics: list[TopicCluster] = []
        for label in sorted(unique_labels):
            mask = labels == label
            cluster_indices = np.where(mask)[0]
            cluster_embeddings = embeddings[mask]
            cluster_texts = [texts[i] for i in cluster_indices]

            # Centroid = mean embedding of cluster members
            centroid = cluster_embeddings.mean(axis=0).astype(np.float32)

            # Extract keywords via simple TF-IDF-like scoring
            keywords = self._extract_keywords(
                cluster_texts,
                texts,
                precomputed_idf=precomputed_idf,
            )

            # Sample 3 representative texts (closest to centroid)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_indices = np.argsort(distances)[:3]
            sample_texts = [cluster_texts[i] for i in closest_indices]

            topics.append(
                TopicCluster(
                    cluster_id=int(label),
                    keywords=keywords,
                    message_count=len(cluster_indices),
                    centroid=centroid,
                    sample_texts=sample_texts,
                )
            )

        # Sort by message count descending
        topics.sort(key=lambda t: t.message_count, reverse=True)

        logger.info(
            "Topic discovery for %s: %d topics, %d noise (of %d messages)",
            contact_id[:16],
            len(topics),
            noise_count,
            n,
        )
        return TopicResult(topics=topics, noise_count=noise_count)

    @staticmethod
    def _extract_keywords(
        cluster_texts: list[str],
        all_texts: list[str],
        top_k: int = 5,
        precomputed_idf: dict[str, int] | None = None,
    ) -> list[str]:
        """Extract representative keywords for a cluster using TF-IDF-like scoring.

        Term frequency is computed within the cluster, and inverse document
        frequency is computed across all messages.

        Args:
            cluster_texts: Texts in this cluster.
            all_texts: All texts across all clusters (for IDF, used only if
                precomputed_idf is None).
            top_k: Number of keywords to return.
            precomputed_idf: Pre-computed document frequency counts per word.
                If provided, skips the expensive all_texts tokenization.

        Returns:
            List of top keywords.
        """
        import math

        # Tokenize cluster texts
        cluster_word_count: Counter[str] = Counter()
        for text in cluster_texts:
            words = _tokenize(text)
            cluster_word_count.update(words)

        if not cluster_word_count:
            return []

        # Use precomputed doc frequencies if available, otherwise compute
        if precomputed_idf is not None:
            doc_count = precomputed_idf
        else:
            doc_count = _compute_doc_frequencies(all_texts)

        n_docs = len(all_texts)
        n_cluster = len(cluster_texts)

        # TF-IDF scoring
        scores: dict[str, float] = {}
        for word, tf in cluster_word_count.items():
            df = doc_count.get(word, 1)
            idf = math.log(n_docs / df)
            # Normalize TF by cluster size
            scores[word] = (tf / n_cluster) * idf

        # Return top-k by score
        sorted_words = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
        return sorted_words[:top_k]


def _compute_doc_frequencies(texts: list[str]) -> dict[str, int]:
    """Count how many documents each word appears in.

    Args:
        texts: All texts to compute document frequencies over.

    Returns:
        Dict of word -> number of documents containing that word.
    """
    doc_count: Counter[str] = Counter()
    for text in texts:
        unique_words = set(_tokenize(text))
        doc_count.update(unique_words)
    return dict(doc_count)


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer: lowercase, alpha-only, skip stopwords."""
    import re

    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if len(w) >= 3 and w not in _STOPWORDS]
