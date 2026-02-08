"""Classifier utilities - Shared components for classifier implementations.

This package provides reusable building blocks for classifiers:

- SingletonFactory: Thread-safe singleton pattern
- LazyInitializer: Thread-safe lazy initialization for expensive computations
- EmbedderMixin: Lazy-loaded embedder access
- CentroidMixin: Centroid-based verification
- StructuralPatternMatcher: Ordered pattern matching
- PatternMatcherByLabel: Pattern matching grouped by label

Usage:
    from jarvis.classifiers import SingletonFactory, EmbedderMixin, LazyInitializer

    class MyClassifier(EmbedderMixin):
        def __init__(self):
            self._centroids = LazyInitializer(self._compute_centroids, name="centroids")

        def _compute_centroids(self) -> dict[str, np.ndarray]:
            # Expensive computation
            ...

        def classify(self, text: str):
            embedding = self.embedder.encode([text], normalize=True)[0]
            centroids = self._centroids.get()  # Lazily computed
            ...

    _factory = SingletonFactory(MyClassifier)

    def get_classifier() -> MyClassifier:
        return _factory.get()
"""

from jarvis.classifiers.category_classifier import CategoryResult, classify_category
from jarvis.classifiers.factory import SingletonFactory
from jarvis.classifiers.lazy import LazyInitializer
from jarvis.classifiers.mixins import CentroidMixin, EmbedderMixin
from jarvis.classifiers.patterns import PatternMatcherByLabel, StructuralPatternMatcher

__all__ = [
    # Factory
    "SingletonFactory",
    # Lazy initialization
    "LazyInitializer",
    # Mixins
    "EmbedderMixin",
    "CentroidMixin",
    # Pattern matching
    "StructuralPatternMatcher",
    "PatternMatcherByLabel",
    # Category classifier
    "CategoryResult",
    "classify_category",
]
