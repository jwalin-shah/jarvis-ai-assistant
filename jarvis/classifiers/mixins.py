"""Classifier Mixins - Shared functionality for classifier classes.

Provides composable mixins that encapsulate common patterns:
- EmbedderMixin: Lazy-loaded embedder access
- SVMModelMixin: SVM model loading and prediction
- CentroidMixin: Centroid-based verification and classification

Usage:
    class MyClassifier(EmbedderMixin, SVMModelMixin):
        def __init__(self, model_path: Path):
            self._model_path = model_path
            # Mixin state is initialized lazily

        def classify(self, text: str):
            embedding = self.embedder.encode([text], normalize=True)[0]
            return self._predict_svm(embedding)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from jarvis.embedding_adapter import UnifiedEmbedder

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

    _embedder: UnifiedEmbedder | None = None

    @property
    def embedder(self) -> UnifiedEmbedder:
        """Get the embedder, loading it lazily on first access.

        Returns:
            The shared UnifiedEmbedder instance.
        """
        if self._embedder is None:
            from jarvis.embedding_adapter import get_embedder

            self._embedder = get_embedder()
        return self._embedder


class SVMModelMixin:
    """Mixin for loading and using SVM classifiers.

    Provides methods for loading SVM models from disk and making predictions.
    Expects subclass to set self._model_path to the model directory.

    The model directory should contain:
    - svm.pkl: Pickled sklearn SVM classifier
    - config.json: Configuration with "labels" list

    Usage:
        class MyClassifier(SVMModelMixin):
            def __init__(self, model_path: Path):
                self._model_path = model_path
                self._load_svm()

            def classify(self, embedding: np.ndarray):
                label, confidence = self._predict_svm(embedding)
                return label
    """

    _svm: Any = None
    _svm_labels: list[str] | None = None
    _svm_loaded: bool = False
    _model_path: Path

    def _load_svm(self) -> bool:
        """Load the SVM model and labels from disk.

        Expects self._model_path to be set to the model directory containing:
        - svm.pkl: Pickled SVM classifier
        - config.json: Configuration with "labels" key

        Returns:
            True if model loaded successfully, False otherwise.
        """
        svm_path = self._model_path / "svm.pkl"
        config_path = self._model_path / "config.json"

        if not svm_path.exists() or not config_path.exists():
            logger.debug("SVM model not found at %s", self._model_path)
            self._svm_loaded = True  # Mark as attempted
            return False

        try:
            with open(svm_path, "rb") as f:
                self._svm = pickle.load(f)
            with open(config_path) as f:
                config = json.load(f)
                # Support both "labels" and "classes" keys
                self._svm_labels = config.get("labels") or config.get("classes") or []

            self._svm_loaded = True
            logger.info("Loaded SVM classifier from %s", self._model_path)
            return True

        except Exception as e:
            logger.warning("Failed to load SVM model: %s", e)
            self._svm = None
            self._svm_labels = None
            self._svm_loaded = True
            return False

    def _predict_svm(
        self,
        embedding: np.ndarray,
    ) -> tuple[str | None, float]:
        """Predict using the SVM classifier.

        Args:
            embedding: Normalized embedding vector.

        Returns:
            Tuple of (predicted_label, confidence).
            Returns (None, 0.0) if SVM not available.
        """
        if not self._svm_loaded or self._svm is None or not self._svm_labels:
            return None, 0.0

        try:
            # Reshape for sklearn
            embedding_2d = embedding.reshape(1, -1).astype(np.float32)

            # Get prediction and probability
            probs = self._svm.predict_proba(embedding_2d)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            label = self._svm_labels[pred_idx]
            return label, confidence

        except Exception as e:
            logger.warning("SVM prediction failed: %s", e)
            return None, 0.0

    @property
    def svm_available(self) -> bool:
        """Check if the SVM model is loaded and available.

        Returns:
            True if SVM is ready for predictions.
        """
        return self._svm_loaded and self._svm is not None


class CentroidMixin:
    """Mixin for centroid-based classification and verification.

    Provides methods for loading centroids and finding the nearest centroid.
    Used for verifying SVM predictions or as a standalone classifier.

    Centroids are mean embeddings for each class, stored as:
    - centroids.npy: Numpy file with dict {label: centroid_array}

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

        Expects self._model_path/centroids.npy to contain a dict
        mapping label strings to numpy arrays.

        Returns:
            True if centroids loaded successfully.
        """
        if self._centroids_loaded:
            return self._centroids is not None

        centroids_path = self._model_path / "centroids.npy"
        if not centroids_path.exists():
            logger.debug("Centroids not found at %s", centroids_path)
            self._centroids_loaded = True
            return False

        try:
            data = np.load(centroids_path, allow_pickle=True).item()
            self._centroids = {label: np.array(centroid) for label, centroid in data.items()}
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
    "SVMModelMixin",
    "CentroidMixin",
]
