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


__all__ = [
    "EmbedderMixin",
]
