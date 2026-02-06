"""Classifier Mixins - Shared functionality for classifier classes.

Provides composable mixins that encapsulate common patterns:
- EmbedderMixin: Lazy-loaded embedder access
- CentroidMixin: Centroid-based verification and classification

Usage:
    class MyClassifier(EmbedderMixin, CentroidMixin):
        def __init__(self, model_path: Path):
            self._model_path = model_path
            # Mixin state is initialized lazily

        def classify(self, text: str):
            embedding = self.embedder.encode([text], normalize=True)[0]
            label, score = self._find_nearest_centroid(embedding)
            return label
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from jarvis.embedding_adapter import CachedEmbedder

logger = logging.getLogger(__name__)

LabelT = TypeVar("LabelT")


class EmbedderMixin:
    """Mixin providing lazy-loaded embedder access.

    Provides a cached embedder property that loads the embedder on first access.
    The embedder is shared across all instances via the singleton in embedding_adapter.

    Usage:
        class MyClassifier(EmbedderMixin):
            def encode(self, text: str) -> np.ndarray:
                return self.embedder.encode([text], normalize=True)[0]
    """

    _embedder: CachedEmbedder | None = None

    @property
    def embedder(self) -> CachedEmbedder:
        """Get the embedder, loading it lazily on first access.

        Returns:
            The shared CachedEmbedder instance.
        """
        if self._embedder is None:
            from jarvis.embedding_adapter import get_embedder

            self._embedder = get_embedder()
        return self._embedder


class CentroidMixin:
    """Mixin for centroid-based classification and verification.

    Provides methods for loading centroids and finding the nearest centroid.
    Used for verifying predictions or as a standalone classifier.

    Centroids are mean embeddings for each class, stored as:
    - centroids.npz: NumPy archive with label keys and centroid arrays (no pickle)

    Usage:
        class MyClassifier(CentroidMixin):
            def __init__(self, model_path: Path):
                self._model_path = model_path

            def verify(self, embedding: np.ndarray, predicted: str):
                nearest, similarity = self._find_nearest_centroid(embedding)
                return nearest == predicted
    """

    _centroids: dict[str, np.ndarray] | None = None
    _centroids_loaded: bool = False
    _model_path: Path

    # Thresholds for centroid verification (can be overridden by subclass)
    CENTROID_VERIFY_THRESHOLD: float = 0.4  # Minimum similarity to accept
    CENTROID_MARGIN: float = 0.15  # Margin for override

    def _load_centroids(self) -> bool:
        """Load centroids from file.

        Expects self._model_path/centroids.npz to contain numpy arrays
        for each label (saved with np.savez).

        Returns:
            True if centroids loaded successfully.
        """
        if self._centroids_loaded:
            return self._centroids is not None

        if not hasattr(self, "_model_path"):
            raise AttributeError(
                f"{type(self).__name__} must set _model_path before using CentroidMixin"
            )

        centroids_path = self._model_path / "centroids.npz"
        if not centroids_path.exists():
            logger.debug("Centroids not found at %s", centroids_path)
            self._centroids_loaded = True
            return False

        try:
            data = np.load(centroids_path, allow_pickle=False)
            self._centroids = {key: data[key] for key in data.files}
            self._centroids_loaded = True
            logger.info("Loaded centroids for %d classes", len(self._centroids))
            return True

        except Exception as e:
            logger.warning("Failed to load centroids: %s", e)
            self._centroids_loaded = True
            return False

    def _find_nearest_centroid(
        self,
        embedding: np.ndarray,
    ) -> tuple[str | None, float]:
        """Find the nearest centroid to an embedding.

        Args:
            embedding: Normalized embedding vector.

        Returns:
            Tuple of (nearest_label, similarity).
            Returns (None, 0.0) if no centroids loaded.
        """
        if self._centroids is None:
            return None, 0.0

        best_label = None
        best_similarity = -1.0

        for label, centroid in self._centroids.items():
            # Cosine similarity (vectors are normalized)
            similarity = float(np.dot(embedding, centroid))
            if similarity > best_similarity:
                best_similarity = similarity
                best_label = label

        return best_label, best_similarity

    def _verify_with_centroids(
        self,
        embedding: np.ndarray,
        predicted_label: str,
    ) -> tuple[str, float, bool]:
        """Verify a prediction using centroid distance.

        If the embedding is closer to another centroid by a significant margin,
        override the prediction. This catches edge cases where the primary
        classifier is confident but semantically wrong.

        Args:
            embedding: Normalized embedding vector.
            predicted_label: The predicted label to verify.

        Returns:
            Tuple of (final_label, confidence, was_verified).
            was_verified=True if prediction was confirmed, False if overridden.
        """
        if self._centroids is None:
            return predicted_label, 0.0, True

        # Compute similarity to all centroids
        similarities: dict[str, float] = {}
        for label, centroid in self._centroids.items():
            similarities[label] = float(np.dot(embedding, centroid))

        predicted_sim = similarities.get(predicted_label, 0.0)
        best_label = max(similarities, key=lambda k: similarities[k])
        best_sim = similarities[best_label]

        # Decision logic:
        # 1. If predicted class has high similarity -> confirm
        # 2. If another class is significantly closer -> override
        # 3. Otherwise -> confirm prediction

        if predicted_sim >= self.CENTROID_VERIFY_THRESHOLD:
            return predicted_label, predicted_sim, True

        if best_sim - predicted_sim > self.CENTROID_MARGIN:
            logger.debug(
                "Centroid override: %s -> %s (sim: %.2f vs %.2f)",
                predicted_label,
                best_label,
                best_sim,
                predicted_sim,
            )
            return best_label, best_sim, False

        # Default: trust original prediction
        return predicted_label, predicted_sim, True

    @property
    def centroids_available(self) -> bool:
        """Check if centroids are loaded and available.

        Returns:
            True if centroids are ready for use.
        """
        return self._centroids_loaded and self._centroids is not None


__all__ = [
    "EmbedderMixin",
    "CentroidMixin",
]
