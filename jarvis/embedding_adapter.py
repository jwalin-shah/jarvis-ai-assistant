"""Unified Embedding Adapter - THE CANONICAL INTERFACE for embeddings.

This is the SINGLE SOURCE OF TRUTH for embedding computation in JARVIS.
All modules that need to compute embeddings should import from here.

Provides a unified interface using in-process MLX BERT inference for
GPU-accelerated embeddings on Apple Silicon (~30MB RAM overhead).

Architecture (2-layer embedding stack):
    1. jarvis/embeddings.py       - Embedding STORAGE (SQLite-backed message search)
           Uses get_embedder() from this module
    2. jarvis/embedding_adapter.py - UNIFIED INTERFACE (this file, use this!)
           Wraps InProcessEmbedder from models/bert_embedder.py

Usage:
    from jarvis.embedding_adapter import get_embedder

    embedder = get_embedder()
    print(f"Using model: {embedder.model_name}")

    # Encode texts
    embeddings = embedder.encode(["Hello, world!", "How are you?"])

    # Single text
    embedding = embedder.encode("Hello, world!")  # Returns (1, 384) array

    # Caching is enabled by default via get_embedder()
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Embedding dimension is 384 for all supported models
EMBEDDING_DIM = 384

# Batch size for MLX embedding requests
MLX_BATCH_SIZE = 100

# =============================================================================
# Model Registry
# =============================================================================

# Maps config model_name → (HuggingFace model ID, MLX model name)
# All models output 384 dimensions for index compatibility
EMBEDDING_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    # Default: bge-small (12 layers, ~100-150ms, MTEB ~62)
    "bge-small": ("BAAI/bge-small-en-v1.5", "bge-small"),
    # TaylorAI gte-tiny (6 layers, ~50-70ms, MTEB ~57) - good balance of speed/quality
    "gte-tiny": ("TaylorAI/gte-tiny", "gte-tiny"),
    # MiniLM-L6 (6 layers, ~50-70ms, MTEB ~56) - most popular fast model
    "minilm-l6": ("sentence-transformers/all-MiniLM-L6-v2", "minilm-l6"),
    # bge-micro (3 layers, ~30-40ms, MTEB ~54) - fastest, lowest quality
    "bge-micro": ("TaylorAI/bge-micro-v2", "bge-micro"),
}

# Legacy constant for backwards compatibility (do not use directly)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_model_info(model_name: str | None = None) -> tuple[str, str]:
    """Get HuggingFace model ID and MLX model name for a config model name.

    Args:
        model_name: Config model name (e.g., "bge-small", "gte-tiny").
            If None, reads from config.

    Returns:
        Tuple of (HuggingFace model ID, MLX model name).

    Raises:
        ValueError: If model_name is not in the registry.
    """
    if model_name is None:
        from jarvis.config import get_config

        model_name = get_config().embedding.model_name

    if model_name not in EMBEDDING_MODEL_REGISTRY:
        valid_models = ", ".join(EMBEDDING_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown embedding model '{model_name}'. Valid options: {valid_models}")

    return EMBEDDING_MODEL_REGISTRY[model_name]


def get_configured_model_name() -> str:
    """Get the currently configured embedding model name.

    Returns:
        The model name from config (e.g., "bge-small", "gte-tiny").
    """
    from jarvis.config import get_config

    return get_config().embedding.model_name


# =============================================================================
# MLX Embedder
# =============================================================================


class MLXEmbedder:
    """MLX-only embedder using in-process BERT inference.

    Runs GPU-accelerated embeddings directly in-process via MLX,
    eliminating the need for an external microservice.

    Thread Safety:
        This class is thread-safe for concurrent encode() calls.

    Example:
        >>> embedder = MLXEmbedder()
        >>> embeddings = embedder.encode(["Hello", "World"])
        >>> print(f"Model: {embedder.model_name}, Shape: {embeddings.shape}")
        Model: bge-small, Shape: (2, 384)
    """

    def __init__(self) -> None:
        """Initialize the MLX embedder."""
        self._mlx_embedder: Any = None
        self._model_name: str | None = None
        self._lock = threading.Lock()
        self._init_attempted = False

    def _initialize(self) -> None:
        """Initialize the in-process MLX embedding backend.

        Uses the model configured in config.embedding.model_name.
        Model loading is lazy - happens on first encode() call.

        Raises:
            RuntimeError: If MLX is not available.
        """
        if self._init_attempted:
            return

        with self._lock:
            if self._init_attempted:
                return

            self._init_attempted = True

            # Get configured model info
            _, mlx_model_name = get_model_info()
            self._model_name = get_configured_model_name()

            try:
                from models.bert_embedder import get_in_process_embedder

                self._mlx_embedder = get_in_process_embedder(model_name=mlx_model_name)
                logger.info(
                    "Initialized in-process MLX embedder with model: %s",
                    mlx_model_name,
                )
            except Exception as e:
                logger.error("Failed to initialize in-process MLX embedder: %s", e)
                raise RuntimeError(f"Could not initialize in-process MLX embedder: {e}") from e

    def is_available(self) -> bool:
        """Check if the in-process MLX embedder is available.

        Returns:
            True if MLX is importable and embedder is initialized.
        """
        try:
            self._initialize()
            return self._mlx_embedder is not None
        except RuntimeError:
            return False

    @property
    def backend(self) -> str:
        """Get the backend type (always 'mlx')."""
        return "mlx"

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension (384 for all supported models)."""
        return EMBEDDING_DIM

    @property
    def model_name(self) -> str:
        """Get the configured embedding model name."""
        if self._model_name is None:
            return get_configured_model_name()
        return self._model_name

    def encode(
        self,
        texts: list[str] | str,
        normalize: bool = True,
        convert_to_numpy: bool = True,  # Ignored, always numpy (API compat)
        normalize_embeddings: bool | None = None,  # Alias for normalize (API compat)
    ) -> NDArray[np.float32]:
        """Encode texts into embedding vectors.

        Args:
            texts: Single text string or list of texts to encode.
            normalize: If True (default), L2-normalize the embeddings.
                      Normalized embeddings allow cosine similarity via dot product.

        Returns:
            NumPy array of shape (n_texts, embedding_dim).
            For single string input, still returns shape (1, 384).

        Raises:
            RuntimeError: If MLX embedder is not available.

        Example:
            >>> embedder = MLXEmbedder()
            >>> vecs = embedder.encode(["Hello", "World"])
            >>> similarity = np.dot(vecs[0], vecs[1])  # Cosine similarity
        """
        self._initialize()

        # Handle normalize_embeddings alias (API compatibility)
        if normalize_embeddings is not None:
            normalize = normalize_embeddings

        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, EMBEDDING_DIM)

        # Use MLX service with batching for large requests
        if len(texts) <= MLX_BATCH_SIZE:
            return self._mlx_embedder.encode(texts, normalize=normalize)

        # For large embedding jobs (>10k texts), use np.memmap to stream to disk
        # instead of accumulating in RAM. Prevents OOM on 8GB system.
        # 100k embeddings x 384 dims x 4 bytes = 150MB (ok)
        # 500k embeddings x 384 dims x 4 bytes = 750MB (risky without memmap)
        memmap_threshold = 10000
        use_memmap = len(texts) > memmap_threshold
        memmap_array: np.ndarray | None = None
        memmap_offset = 0
        memmap_path: str | None = None

        if use_memmap:
            import tempfile

            _tmp = tempfile.NamedTemporaryFile(suffix=".mmap", prefix="jarvis_embed_", delete=False)
            memmap_path = _tmp.name
            _tmp.close()
            memmap_array = np.memmap(
                memmap_path,
                dtype=np.float32,
                mode="w+",
                shape=(len(texts), EMBEDDING_DIM),
            )
            logger.info(
                "Using memmap for %d texts (%d MB on disk)",
                len(texts),
                len(texts) * EMBEDDING_DIM * 4 // (1024 * 1024),
            )

        # Process in batches
        all_embeddings: list[np.ndarray] = []
        try:
            for i in range(0, len(texts), MLX_BATCH_SIZE):
                batch = texts[i : i + MLX_BATCH_SIZE]
                batch_embeddings = self._mlx_embedder.encode(batch, normalize=normalize)

                if use_memmap and memmap_array is not None:
                    batch_embeddings = batch_embeddings.astype(np.float32)
                    batch_len = len(batch_embeddings)
                    memmap_array[memmap_offset : memmap_offset + batch_len] = batch_embeddings
                    memmap_offset += batch_len
                else:
                    all_embeddings.append(batch_embeddings)

                # Log progress for large batches
                if len(texts) > 1000 and (i + MLX_BATCH_SIZE) % 1000 == 0:
                    logger.debug(
                        "Encoded %d/%d texts (%.1f%%)",
                        min(i + MLX_BATCH_SIZE, len(texts)),
                        len(texts),
                        100 * min(i + MLX_BATCH_SIZE, len(texts)) / len(texts),
                    )

            if use_memmap and memmap_array is not None:
                # Flush memmap and return as regular array, then clean up temp file
                memmap_array.flush()
                result = np.array(memmap_array[:memmap_offset])
                del memmap_array
                if memmap_path:
                    import os

                    try:
                        os.unlink(memmap_path)
                    except OSError:
                        pass
                return result
            return np.vstack(all_embeddings)
        finally:
            # Ensure temp file cleanup even on exception
            if memmap_path:
                import os

                try:
                    del memmap_array
                except NameError:
                    pass
                try:
                    os.unlink(memmap_path)
                except OSError:
                    pass

    def unload(self) -> None:
        """Unload the in-process embedding model to free GPU memory."""
        with self._lock:
            if self._mlx_embedder is not None:
                self._mlx_embedder.unload()
            self._init_attempted = False


class CachedEmbedder:
    """Per-request embedding cache wrapper.

    Wraps an MLXEmbedder to cache embedding results, avoiding recomputation
    of the same texts within a request. Exposes the same API as MLXEmbedder.
    """

    def __init__(self, base_embedder: MLXEmbedder, maxsize: int = 1000) -> None:
        self.base = base_embedder
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._computations = 0
        self._cache_hits = 0
        self._cache_misses = 0

    # Proxy properties/methods from MLXEmbedder for API compatibility
    @property
    def backend(self) -> str:
        """Get the backend type (always 'mlx')."""
        return self.base.backend

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.base.embedding_dim

    @property
    def model_name(self) -> str:
        """Get the configured embedding model name."""
        return self.base.model_name

    def is_available(self) -> bool:
        """Check if the MLX embedding service is available."""
        return self.base.is_available()

    def unload(self) -> None:
        """Unload the model and clear cache."""
        with self._lock:
            self._cache.clear()
        self.base.unload()

    def _make_key(self, text: str) -> str:
        return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()

    def _get(self, key: str) -> np.ndarray | None:
        with self._lock:
            value = self._cache.get(key)
            if value is None:
                self._cache_misses += 1
                return None
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return value

    def _set(self, key: str, value: np.ndarray) -> None:
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    @property
    def embedding_computations(self) -> int:
        return self._computations

    @property
    def cache_hit(self) -> bool:
        return self._cache_hits > 0

    def encode(
        self,
        texts: list[str] | str,
        normalize: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool | None = None,
    ) -> NDArray[np.float32]:
        if normalize_embeddings is not None:
            normalize = normalize_embeddings

        if isinstance(texts, str):
            key = self._make_key(texts)
            cached = self._get(key)
            if cached is not None:
                return cached

            result = self.base.encode(
                [texts],
                normalize=normalize,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
            )
            vector = result.reshape(1, -1)
            self._set(key, vector)
            self._computations += 1
            return vector

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, EMBEDDING_DIM)

        cached_vectors: list[np.ndarray | None] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_indices: list[int] = []
        missing_keys: list[str] = []

        for i, text in enumerate(texts):
            key = self._make_key(text)
            cached = self._get(key)
            if cached is not None:
                cached_vectors[i] = cached
            else:
                missing_texts.append(text)
                missing_indices.append(i)
                missing_keys.append(key)

        if missing_texts:
            embeddings = self.base.encode(
                missing_texts,
                normalize=normalize,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
            )
            for idx, emb, cache_key in zip(missing_indices, embeddings, missing_keys):
                vector = np.asarray(emb, dtype=np.float32).reshape(1, -1)
                cached_vectors[idx] = vector
                self._set(cache_key, vector)
            self._computations += len(missing_texts)

        if any(v is None for v in cached_vectors):
            raise RuntimeError("Failed to compute embeddings for all inputs")

        return np.vstack(cached_vectors)


# =============================================================================
# Singleton Access
# =============================================================================

_embedder: MLXEmbedder | None = None
_cached_embedder: CachedEmbedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> CachedEmbedder:
    """Get or create the singleton CachedEmbedder instance.

    Returns a CachedEmbedder wrapping an MLXEmbedder, providing automatic
    caching of embedding results to avoid recomputing the same texts.

    Returns:
        The shared CachedEmbedder instance.

    Raises:
        RuntimeError: If MLX embedder is not available.

    Example:
        >>> embedder = get_embedder()
        >>> embeddings = embedder.encode(["Hello, world!"])
    """
    global _embedder, _cached_embedder

    if _cached_embedder is None:
        with _embedder_lock:
            if _cached_embedder is None:
                if _embedder is None:
                    _embedder = MLXEmbedder()
                _cached_embedder = CachedEmbedder(_embedder)

    return _cached_embedder


def reset_embedder() -> None:
    """Reset the singleton embedders.

    Unloads the model and clears the singletons.
    A new instance will be created on the next get_embedder() call.
    """
    global _embedder, _cached_embedder

    with _embedder_lock:
        if _cached_embedder is not None:
            _cached_embedder.unload()
            _cached_embedder = None
        if _embedder is not None:
            _embedder = None


def is_embedder_available() -> bool:
    """Check if the MLX embedding service is available.

    Returns:
        True if embeddings can be computed.
    """
    try:
        embedder = get_embedder()
        return embedder.is_available()
    except RuntimeError:
        return False


# Legacy alias for backwards compatibility
UnifiedEmbedder = MLXEmbedder

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "EMBEDDING_MODEL",  # Legacy, prefer get_model_info()
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL_REGISTRY",
    # Model info functions
    "get_model_info",
    "get_configured_model_name",
    # Class
    "MLXEmbedder",
    "UnifiedEmbedder",  # Legacy alias
    "CachedEmbedder",
    # Singleton functions
    "get_embedder",
    "reset_embedder",
    "is_embedder_available",
]
