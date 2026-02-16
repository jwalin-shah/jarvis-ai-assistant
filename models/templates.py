"""Template Matcher for semantic template matching.

Bypasses model generation for common request patterns using
semantic similarity with sentence embeddings.

Performance optimizations:
- Pre-normalized pattern embeddings for O(1) cosine similarity
- LRU cache for query embeddings (repeated queries)
- Batch encoding for initial setup

Analytics:
- Tracks template hit/miss rates
- Records queries that miss templates for optimization
- Monitors cache efficiency

Custom Templates:
- User-defined templates stored in ~/.jarvis/custom_templates.json
- Support for trigger phrases, category tags, and group size constraints
- Import/export functionality for sharing template packs
"""

from __future__ import annotations

import gc
import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

from jarvis.core.exceptions import ErrorCode, JarvisError
from jarvis.metrics import get_template_analytics
from models import custom_template_store as _custom_template_store_mod
from models.bert_embedder import get_embedder, reset_embedder

CustomTemplate = _custom_template_store_mod.CustomTemplate
CustomTemplateStore = _custom_template_store_mod.CustomTemplateStore
get_custom_template_store = _custom_template_store_mod.get_custom_template_store
reset_custom_template_store = _custom_template_store_mod.reset_custom_template_store

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedder objects that can encode text to embeddings.

    This allows TemplateMatcher to accept either UnifiedEmbedder or CachedEmbedder
    (or any other compatible embedder) for cache cohesion across the request pipeline.
    """

    def encode(
        self,
        texts: list[str] | str,
        normalize: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool | None = None,
    ) -> NDArray[np.float32]:
        """Encode texts to embeddings."""
        ...


logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


class EmbeddingCache(Generic[K, V]):
    """LRU cache for query embeddings.

    Thread-safe implementation with bounded size.
    """

    def __init__(self, maxsize: int = 500) -> None:
        """Initialize the cache.

        Args:
            maxsize: Maximum number of embeddings to cache
        """
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: K, track_analytics: bool = True) -> V | None:
        """Get embedding from cache.

        Args:
            key: Cache key to look up
            track_analytics: Whether to record this access in template analytics
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                if track_analytics:
                    get_template_analytics().record_cache_access(hit=True)
                return self._cache[key]
            self._misses += 1
            if track_analytics:
                get_template_analytics().record_cache_access(hit=False)
            return None

    def set(self, key: K, value: V) -> None:
        """Store embedding in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate."""
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


class SentenceModelError(JarvisError):
    """Raised when sentence transformer model cannot be loaded."""

    default_message = "Sentence transformer model cannot be loaded"
    default_code = ErrorCode.MDL_LOAD_FAILED


def _get_sentence_model() -> Any:
    """Get the embedder for template matching.

    Returns the unified embedder instance. For backward compatibility,
    this function name is preserved but now delegates to the unified adapter.

    Returns:
        The UnifiedEmbedder instance (not SentenceTransformer directly)

    Raises:
        SentenceModelError: If no embedding backend is available
    """
    try:
        embedder = get_embedder()
        if not embedder.is_available():
            raise SentenceModelError("No embedding backend available")
        return embedder
    except Exception as e:
        logger.exception("Failed to initialize embedding backend")
        msg = f"Failed to initialize embedding backend: {e}"
        raise SentenceModelError(msg) from e


def unload_sentence_model() -> None:
    """Unload the embedding model to free memory.

    Call this when template matching is no longer needed and you want
    to reclaim memory for other operations (e.g., loading the MLX model).
    """
    logger.info("Unloading embedding model")
    reset_embedder()
    gc.collect()


def is_sentence_model_loaded() -> bool:
    """Check if an embedding model is currently available.

    Returns:
        True if an embedding backend is available, False otherwise
    """
    try:
        embedder = get_embedder()
        return embedder.backend != "none"
    except Exception:
        return False


@dataclass
class ResponseTemplate:
    """A template for common response patterns.

    Attributes:
        name: Unique identifier for the template
        patterns: Example prompts that match this template
        response: The response to return
        is_group_template: Whether this is a group chat specific template
        min_group_size: Minimum group size for this template (None = any size)
        max_group_size: Maximum group size for this template (None = any size)
    """

    name: str
    patterns: list[str]  # Example prompts that match this template
    response: str  # The response to return
    is_group_template: bool = False
    min_group_size: int | None = None  # None means no minimum
    max_group_size: int | None = None  # None means no maximum


@dataclass
class TemplateMatch:
    """Result of template matching."""

    template: ResponseTemplate
    similarity: float
    matched_pattern: str


# =============================================================================
# Custom Template Support
# =============================================================================


def _get_minimal_fallback_templates() -> list[ResponseTemplate]:
    """Minimal templates for development when WS3 not available."""
    from models.template_defaults import get_minimal_fallback_templates

    return [
        template
        for template in get_minimal_fallback_templates()
        if isinstance(template, ResponseTemplate)
    ]


def _load_templates() -> list[ResponseTemplate]:
    """Load response templates.

    Returns the built-in template set for template matching.
    """
    return _get_minimal_fallback_templates()


class TemplateMatcher:
    """Semantic template matcher using sentence embeddings.

    Computes cosine similarity between input prompt and template patterns.
    Returns best matching template if similarity exceeds threshold.

    Performance optimizations:
    - Pre-normalized pattern embeddings (computed once at init)
    - LRU cache for query embeddings (avoids re-encoding repeated queries)
    - Optimized dot product for cosine similarity (O(n) instead of O(n*d))
    """

    # NOTE: These thresholds can be customized via ~/.jarvis/config.json
    # See jarvis/config.py for configuration options
    SIMILARITY_THRESHOLD = 0.7
    QUERY_CACHE_SIZE = 500

    def __init__(self, templates: list[ResponseTemplate] | None = None) -> None:
        """Initialize the template matcher.

        Args:
            templates: List of templates to use. Loads defaults if not provided.
        """
        self.templates = templates or _load_templates()
        self._pattern_embeddings: np.ndarray | None = None
        self._pattern_norms: np.ndarray | None = None  # Pre-computed norms
        self._pattern_to_template: list[tuple[str, ResponseTemplate]] = []
        self._embeddings_lock = threading.Lock()
        self._query_cache: EmbeddingCache[str, np.ndarray] = EmbeddingCache(
            maxsize=self.QUERY_CACHE_SIZE
        )

    def _ensure_embeddings(self) -> None:
        """Compute and cache embeddings for all template patterns.

        Uses double-check locking for thread-safe lazy initialization.
        Embeddings are normalized by the embedder for direct cosine similarity.
        """
        # Fast path: embeddings already computed
        if self._pattern_embeddings is not None:
            return

        # Slow path: acquire lock and double-check
        with self._embeddings_lock:
            # Double-check after acquiring lock
            if self._pattern_embeddings is not None:
                return

            embedder = _get_sentence_model()

            # Collect all patterns with their templates
            all_patterns = []
            pattern_to_template: list[tuple[str, ResponseTemplate]] = []
            for template in self.templates:
                for pattern in template.patterns:
                    all_patterns.append(pattern)
                    pattern_to_template.append((pattern, template))

            embeddings = None
            norms = None
            try:
                # Compute embeddings in batch (normalized by default)
                embeddings = embedder.encode(all_patterns, normalize=True)

                # Pre-compute norms for faster cosine similarity
                # Since embeddings are L2-normalized, all norms should be 1.0
                norms = np.ones((embeddings.shape[0], 1), dtype=np.float32)

                # Assign atomically
                self._pattern_to_template = pattern_to_template
                self._pattern_norms = norms.flatten()
                self._pattern_embeddings = embeddings
                logger.info("Computed embeddings for %d patterns", len(all_patterns))
            except Exception as e:
                # Clean up partial results to avoid memory leak
                embeddings = None
                norms = None
                logger.error("Failed to compute pattern embeddings: %s", e)
                raise

    def _get_query_embedding(self, query: str, embedder: Embedder | None = None) -> np.ndarray:
        """Get embedding for a query, using cache if available.

        Args:
            query: Query string to encode
            embedder: Optional embedder override. If provided, uses this embedder
                directly (which may have its own caching, e.g., CachedEmbedder).
                If None, uses internal cache with default embedder.

        Returns:
            Query embedding as numpy array (normalized)

        Raises:
            TypeError: If embedder doesn't implement the Embedder protocol
        """
        # SECURITY: Validate embedder implements the required protocol before calling encode()
        if embedder is not None:
            if not isinstance(embedder, Embedder):
                raise TypeError(
                    f"embedder must implement Embedder protocol, got {type(embedder).__name__}"
                )
            embedding_result = embedder.encode([query], normalize=True)[0]
            return np.asarray(embedding_result)

        # Create cache key from query hash
        cache_key = hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()

        # Check cache first
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached

        # Encode and cache (normalized by default)
        default_embedder = _get_sentence_model()
        embedding_result = default_embedder.encode([query], normalize=True)[0]
        # Cast to ndarray to satisfy mypy (encode returns Any)
        embedding: np.ndarray = np.asarray(embedding_result)
        self._query_cache.set(cache_key, embedding)
        return embedding

    def match(
        self,
        query: str,
        track_analytics: bool = True,
        embedder: Embedder | None = None,
    ) -> TemplateMatch | None:
        """Find best matching template for a query.

        Args:
            query: Input prompt to match against templates
            track_analytics: Whether to record this query in template analytics
            embedder: Optional embedder override for cache cohesion. If provided,
                uses this embedder (e.g., CachedEmbedder from router) instead of
                internal cache. This allows sharing cached embeddings across the
                request pipeline.

        Returns:
            TemplateMatch if similarity >= threshold, None otherwise.
            Returns None if sentence model fails to load (falls back to model generation).

        Raises:
            TypeError: If embedder doesn't implement the Embedder protocol
        """
        # SECURITY: Validate embedder type early before any processing
        if embedder is not None and not isinstance(embedder, Embedder):
            raise TypeError(
                f"embedder must implement Embedder protocol, got {type(embedder).__name__}"
            )

        analytics = get_template_analytics() if track_analytics else None

        try:
            self._ensure_embeddings()

            # Type guard: _ensure_embeddings guarantees this is not None
            pattern_embeddings = self._pattern_embeddings
            if pattern_embeddings is None:
                return None

            # Compute norms on-the-fly if not pre-computed (for backward compat with tests)
            pattern_norms = self._pattern_norms
            if pattern_norms is None:
                pattern_norms = np.linalg.norm(pattern_embeddings, axis=1)
                pattern_norms = np.where(pattern_norms == 0, 1, pattern_norms)

            # Get query embedding (uses external embedder if provided for cache cohesion)
            query_embedding = self._get_query_embedding(query, embedder=embedder)
            query_norm = np.linalg.norm(query_embedding)

            if query_norm == 0:
                return None

            # Compute similarities in batch (optimized dot product)
            # similarities.shape = (n_patterns,)
            similarities = np.dot(pattern_embeddings, query_embedding) / (
                pattern_norms * query_norm
            )

            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = float(similarities[best_idx])

            if best_similarity >= self.SIMILARITY_THRESHOLD:
                matched_pattern, template = self._pattern_to_template[best_idx]

                # Update template usage if matched
                if template.name.startswith("custom_"):
                    store = get_custom_template_store()
                    store.increment_usage(template.name.replace("custom_", ""))

                # Record hit in analytics
                if analytics:
                    analytics.record_hit(template.name, best_similarity)

                logger.debug(
                    "Template match: %s (similarity: %.3f)",
                    template.name,
                    best_similarity,
                )
                return TemplateMatch(
                    template=template,
                    similarity=best_similarity,
                    matched_pattern=matched_pattern,
                )

            # Record miss in analytics
            if analytics:
                analytics.record_miss(query, best_similarity)

            return None

        except SentenceModelError:
            logger.warning("Template matching unavailable, falling back to model generation")
            return None

    def _template_matches_group_size(
        self, template: ResponseTemplate, group_size: int | None
    ) -> bool:
        """Check if a template is appropriate for a given group size.

        Args:
            template: The template to check
            group_size: Number of participants in the chat (None means unknown)

        Returns:
            True if template is appropriate, False otherwise
        """
        # If group_size is None, only non-group templates are appropriate
        if group_size is None:
            return not template.is_group_template

        # If no constraints, it's appropriate
        if template.min_group_size is None and template.max_group_size is None:
            # If specifically marked as group template, needs group_size >= 3
            if template.is_group_template:
                return group_size >= 3
            return True

        # Check min constraint
        if template.min_group_size is not None and group_size < template.min_group_size:
            return False

        # Check max constraint
        if template.max_group_size is not None and group_size > template.max_group_size:
            return False

        return True

    def match_with_context(
        self,
        query: str,
        group_size: int | None = None,
        track_analytics: bool = True,
        embedder: Embedder | None = None,
    ) -> TemplateMatch | None:
        """Find best matching template considering conversation context.

        Filters templates based on group size and other context before matching.

        Args:
            query: Input prompt to match
            group_size: Number of participants in the chat
            track_analytics: Whether to record in analytics
            embedder: Optional embedder override for cache cohesion. If provided,
                uses this embedder (e.g., CachedEmbedder from router) instead of
                internal cache.

        Returns:
            TemplateMatch or None
        """
        # If no group size provided, use standard match
        if group_size is None:
            return self.match(query, track_analytics, embedder=embedder)

        try:
            self._ensure_embeddings()
            pattern_embeddings = self._pattern_embeddings
            if pattern_embeddings is None:
                return None

            # Compute norms on-the-fly if not pre-computed
            pattern_norms = self._pattern_norms
            if pattern_norms is None:
                pattern_norms = np.linalg.norm(pattern_embeddings, axis=1)
                pattern_norms = np.where(pattern_norms == 0, 1, pattern_norms)

            # Get query embedding (uses external embedder if provided for cache cohesion)
            query_embedding = self._get_query_embedding(query, embedder=embedder)
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return None

            # Score ALL templates first
            similarities = np.dot(pattern_embeddings, query_embedding) / (
                pattern_norms * query_norm
            )

            # Now find the best match that also satisfies group size constraints
            best_match = None
            best_similarity = -1.0

            for i, (matched_pattern, template) in enumerate(self._pattern_to_template):
                similarity = float(similarities[i])

                # Must meet base threshold
                if similarity < self.SIMILARITY_THRESHOLD:
                    continue

                # Check group size constraints
                if not self._template_matches_group_size(template, group_size):
                    continue

                # Boost similarity for specific group templates if we're in a group
                effective_similarity = similarity
                if group_size >= 3 and template.is_group_template:
                    # Give preference to group-specific templates in group chats
                    effective_similarity = min(1.0, similarity + 0.05)

                if effective_similarity > best_similarity:
                    best_similarity = effective_similarity
                    best_match = TemplateMatch(
                        template=template,
                        similarity=similarity,  # Store actual similarity, not boosted
                        matched_pattern=matched_pattern,
                    )

            if best_match is not None:
                # Record hit in analytics
                if track_analytics:
                    get_template_analytics().record_hit(
                        best_match.template.name, best_match.similarity
                    )

                logger.debug(
                    "Template match with context: %s (similarity: %.3f, group_size: %s)",
                    best_match.template.name,
                    best_match.similarity,
                    group_size,
                )

            return best_match

        except SentenceModelError:
            logger.warning("Template matching unavailable, falling back to model generation")
            return None

    def get_group_templates(self) -> list[ResponseTemplate]:
        """Get all group-specific templates."""
        return [t for t in self.templates if t.is_group_template]

    def get_templates_for_group_size(self, group_size: int) -> list[ResponseTemplate]:
        """Get templates appropriate for a specific group size."""
        return [t for t in self.templates if self._template_matches_group_size(t, group_size)]

    def clear_cache(self) -> None:
        """Clear cached embeddings."""
        self._pattern_embeddings = None
        self._pattern_norms = None
        self._pattern_to_template = []
        self._query_cache.clear()
        logger.debug("Template matcher cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get query cache statistics."""
        return self._query_cache.stats()
