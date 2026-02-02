"""Classifier utilities - Shared components for classifier implementations.

This package provides reusable building blocks for classifiers:

- SingletonFactory: Thread-safe singleton pattern
- EmbedderMixin: Lazy-loaded embedder access
- SVMModelMixin: SVM model loading and prediction
- CentroidMixin: Centroid-based verification
- StructuralPatternMatcher: Ordered pattern matching
- PatternMatcherByLabel: Pattern matching grouped by label

Usage:
    from jarvis.classifiers import SingletonFactory, EmbedderMixin

    class MyClassifier(EmbedderMixin):
        def classify(self, text: str):
            embedding = self.embedder.encode([text], normalize=True)[0]
            ...

    _factory = SingletonFactory(MyClassifier)

    def get_classifier() -> MyClassifier:
        return _factory.get()
"""

from jarvis.classifiers.factory import SingletonFactory
from jarvis.classifiers.mixins import CentroidMixin, EmbedderMixin, SVMModelMixin
from jarvis.classifiers.patterns import PatternMatcherByLabel, StructuralPatternMatcher

__all__ = [
    # Factory
    "SingletonFactory",
    # Mixins
    "EmbedderMixin",
    "SVMModelMixin",
    "CentroidMixin",
    # Pattern matching
    "StructuralPatternMatcher",
    "PatternMatcherByLabel",
]
