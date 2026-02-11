"""Classifier utilities - Shared components for classifier implementations.

This package provides reusable building blocks for classifiers:

- SingletonFactory: Thread-safe singleton pattern
- EmbedderMixin: Lazy-loaded embedder access
"""

from jarvis.classifiers.category_classifier import CategoryResult, classify_category
from jarvis.classifiers.factory import SingletonFactory
from jarvis.classifiers.mixins import EmbedderMixin

__all__ = [
    # Factory
    "SingletonFactory",
    # Mixins
    "EmbedderMixin",
    # Category classifier
    "CategoryResult",
    "classify_category",
]
