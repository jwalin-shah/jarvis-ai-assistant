"""Coreference Resolver - Resolve pronouns to their referents using FastCoref.

Coreference resolution replaces pronouns (he, she, it, they) with the entities
they refer to. This improves embedding quality for topic segmentation by making
the semantic content explicit.

Example:
    "Jake said he would come to the party"
    -> "Jake said Jake would come to the party"

The FastCoref model is optional and loaded lazily. If not installed, the
resolver gracefully degrades to returning original text.

Installation:
    uv add fastcoref  # or: pip install fastcoref

Usage:
    from jarvis.nlp.coref_resolver import get_coref_resolver

    resolver = get_coref_resolver()
    if resolver and resolver.is_available():
        resolved = resolver.resolve("Jake said he would come")
        # "Jake said Jake would come"
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class CorefResolver:
    """Coreference resolver using FastCoref (f-coref model).

    Wraps FastCoref for pronoun resolution. Loads model lazily on first use.
    Thread-safe for concurrent resolve() calls.

    Memory usage: ~500MB for the f-coref model.
    """

    def __init__(self, model_name: str = "biu-nlp/f-coref") -> None:
        """Initialize resolver.

        Args:
            model_name: FastCoref model name from HuggingFace.
        """
        self.model_name = model_name
        self._model: Any = None
        self._lock = threading.Lock()
        self._init_attempted = False
        self._available = False

    def _detect_device(self) -> str:
        """Detect the best available device for the model.

        Returns:
            Device string: "mps", "cuda", or "cpu".
        """
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
        except (ImportError, AttributeError):
            pass
        return "cpu"

    def _initialize(self) -> None:
        """Initialize the FastCoref model.

        Called lazily on first use.
        """
        if self._init_attempted:
            return

        with self._lock:
            if self._init_attempted:
                return

            self._init_attempted = True

            try:
                from fastcoref import FCoref

                # Auto-detect device (MPS on Apple Silicon, CUDA on Nvidia, CPU fallback)
                device = self._detect_device()
                self._model = FCoref(device=device)
                self._available = True
                logger.info("FastCoref model loaded: %s on %s", self.model_name, device)

            except ImportError:
                logger.info(
                    "FastCoref not installed. Coreference resolution disabled. "
                    "Install with: uv add fastcoref"
                )
                self._available = False

            except Exception as e:
                logger.warning("Failed to load FastCoref model: %s", e)
                self._available = False

    def is_available(self) -> bool:
        """Check if coreference resolution is available.

        Returns:
            True if FastCoref is installed and model loaded successfully.
        """
        self._initialize()
        return self._available

    def resolve(self, text: str) -> str:
        """Resolve coreferences in a single text.

        Replaces pronouns with their referents.

        Args:
            text: Text to resolve coreferences in.

        Returns:
            Text with pronouns replaced by referents. Returns original text
            if resolver is unavailable or resolution fails.
        """
        if not text:
            return text

        self._initialize()

        if not self._available or self._model is None:
            return text

        try:
            preds = self._model.predict(texts=[text])
            # Validate predictions array and first element
            if preds and len(preds) > 0 and preds[0] is not None:
                try:
                    resolved: str = preds[0].get_resolved_text()
                    return resolved if resolved else text
                except (AttributeError, TypeError) as e:
                    logger.warning("Invalid prediction object: %s", e)
                    return text
            return text

        except Exception as e:
            logger.warning("Coreference resolution failed: %s", e)
            return text

    def resolve_batch(self, texts: list[str]) -> list[str]:
        """Resolve coreferences in multiple texts.

        More efficient than calling resolve() in a loop.

        Args:
            texts: List of texts to resolve.

        Returns:
            List of resolved texts. Original text returned for any failures.
        """
        if not texts:
            return []

        self._initialize()

        if not self._available or self._model is None:
            return list(texts)

        # Filter out empty texts
        non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            return list(texts)

        try:
            preds = self._model.predict(texts=non_empty_texts)

            # Validate predictions array length matches input
            if not preds or len(preds) != len(non_empty_texts):
                logger.warning(
                    "Prediction count mismatch: expected %d, got %d",
                    len(non_empty_texts),
                    len(preds) if preds else 0,
                )
                return list(texts)

            # Build result list with resolved texts in correct positions
            result = list(texts)
            for idx, pred in zip(non_empty_indices, preds):
                if pred is not None:
                    try:
                        resolved = pred.get_resolved_text()
                        if resolved:
                            result[idx] = resolved
                    except (AttributeError, TypeError) as e:
                        logger.warning("Invalid prediction object at index %d: %s", idx, e)

            return result

        except Exception as e:
            logger.warning("Batch coreference resolution failed: %s", e)
            return list(texts)

    def unload(self) -> None:
        """Unload the model to free memory."""
        with self._lock:
            self._model = None
            self._init_attempted = False
            self._available = False


# =============================================================================
# Singleton
# =============================================================================

_resolver: CorefResolver | None = None
_resolver_lock = threading.Lock()


def get_coref_resolver() -> CorefResolver | None:
    """Get the singleton CorefResolver instance.

    Returns None if FastCoref is not installed.

    Returns:
        CorefResolver instance, or None if unavailable.
    """
    global _resolver

    if _resolver is None:
        with _resolver_lock:
            if _resolver is None:
                resolver = CorefResolver()
                # Don't return if not available
                if resolver.is_available():
                    _resolver = resolver
                else:
                    return None

    return _resolver


def reset_coref_resolver() -> None:
    """Reset the singleton resolver for testing."""
    global _resolver

    with _resolver_lock:
        if _resolver is not None:
            _resolver.unload()
        _resolver = None


__all__ = [
    "CorefResolver",
    "get_coref_resolver",
    "reset_coref_resolver",
]
