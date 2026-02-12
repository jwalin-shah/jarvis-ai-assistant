"""Classifier Mixins - Shared functionality for classifier classes.

Provides composable mixins that encapsulate common patterns:
- EmbedderMixin: Lazy-loaded embedder access
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.embedding_adapter import CachedEmbedder

logger = logging.getLogger(__name__)


class EmbedderMixin:
    """Mixin providing lazy-loaded embedder access.

    Provides a cached embedder property that loads the embedder on first access.
    The embedder is shared across all instances via the singleton in embedding_adapter.

    Usage:
        class MyClassifier(EmbedderMixin):
            def encode(self, text: str) -> np.ndarray:
                return self.embedder.encode([text], normalize=True)[0]
    """

    @property
    def embedder(self) -> CachedEmbedder:
        """Get the embedder, loading it lazily on first access.

        Returns:
            The shared CachedEmbedder instance.
        """
        # Use instance __dict__ directly to avoid class-variable sharing across instances.
        # A class-level _embedder would be shared by all subclasses and instances,
        # which is incorrect if different instances need independent lifecycle.
        embedder = self.__dict__.get("_embedder")
        if embedder is None:
            from jarvis.embedding_adapter import get_embedder

            embedder = get_embedder()
            self.__dict__["_embedder"] = embedder
        return embedder


__all__ = [
    "EmbedderMixin",
]
