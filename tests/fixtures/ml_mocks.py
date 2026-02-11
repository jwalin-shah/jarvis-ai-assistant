"""Deterministic mocks for ML components.

These mocks provide predictable behavior for tests, eliminating flakiness
from model loading, randomness, or timing.

Usage:
    from tests.fixtures.ml_mocks import DeterministicEmbedder, DeterministicGenerator

    embedder = DeterministicEmbedder()
    embedding = embedder.encode("hello")  # Always same result
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import numpy as np


@dataclass(frozen=True)
class MockEmbeddingConfig:
    """Configuration for deterministic embeddings.

    Attributes:
        dim: Embedding dimension
        seed: Base random seed
        normalize: Whether to L2-normalize embeddings
    """

    dim: int = 384
    seed: int = 42
    normalize: bool = True


class DeterministicEmbedder:
    """Deterministic embedder that produces consistent vectors for text.

    Uses hash-based generation so same text always produces same embedding.
    This eliminates flakiness from random initialization or model versions.

    Example:
        >>> embedder = DeterministicEmbedder()
        >>> emb1 = embedder.encode("hello")
        >>> emb2 = embedder.encode("hello")
        >>> np.allclose(emb1, emb2)
        True
    """

    def __init__(self, config: MockEmbeddingConfig | None = None):
        self.config = config or MockEmbeddingConfig()
        self._cache: dict[str, np.ndarray] = {}

    def encode(
        self,
        texts: list[str] | str,
        normalize: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate deterministic embeddings.

        Args:
            texts: Single text or list of texts to encode
            normalize: Whether to L2-normalize (overrides config if provided)
            **kwargs: Ignored (for API compatibility)

        Returns:
            Array of embeddings with shape (n_texts, dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            if text not in self._cache:
                self._cache[text] = self._generate_embedding(text)
            embeddings.append(self._cache[text])

        result = np.array(embeddings, dtype=np.float32)

        if self.config.normalize and normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid div by zero
            result = result / norms

        return result

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate deterministic vector from text hash.

        Uses MD5 hash of text to seed random number generator,
        ensuring identical texts produce identical embeddings.
        """
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        rng = np.random.RandomState(hash_val % 2**32)

        # Generate base embedding
        embedding = rng.randn(self.config.dim).astype(np.float32)

        # Add some semantic structure:
        # Similar texts (by hash) get similar base directions
        # This is a simplified semantic model

        return embedding

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.config.dim

    def is_available(self) -> bool:
        """Always available."""
        return True

    def __repr__(self) -> str:
        return f"DeterministicEmbedder(dim={self.config.dim})"


@dataclass(frozen=True)
class MockGenerationConfig:
    """Configuration for deterministic generation.

    Attributes:
        default_response: Default response for unmatched prompts
        latency_ms: Fixed latency for all generations
        tokens_per_response: Fixed token count
    """

    default_response: str = "This is a mock response."
    latency_ms: float = 50.0
    tokens_per_response: int = 10


class DeterministicGenerator:
    """Deterministic generator for testing.

    Produces consistent responses based on prompt content,
    with configurable latency and token counts.

    Example:
        >>> gen = DeterministicGenerator()
        >>> response = gen.generate("Hello?")
        >>> "question" in response.text.lower() or "Hello" in response.text
        True
    """

    # Pattern-based response mapping
    RESPONSE_PATTERNS: list[tuple[set[str], str]] = [
        ({"?", "what", "how", "why", "when", "where"}, "That's an interesting question."),
        ({"hello", "hi", "hey", "greetings"}, "Hello! How can I help you today?"),
        ({"bye", "goodbye", "see you", "later"}, "Goodbye! Have a great day."),
        ({"thanks", "thank you", "appreciate"}, "You're welcome!"),
        ({"yes", "yeah", "sure", "ok"}, "Great! Let me know if you need anything else."),
        ({"no", "nope", "nah"}, "Okay, no problem."),
    ]

    def __init__(self, config: MockGenerationConfig | None = None):
        self.config = config or MockGenerationConfig()
        self._response_cache: dict[str, str] = {}

    def generate(self, request: Any) -> MagicMock:
        """Generate deterministic response.

        Args:
            request: Generation request with 'prompt' attribute or string

        Returns:
            Mock response object with .text, .tokens_used, etc.
        """
        # Extract prompt from request
        if isinstance(request, str):
            prompt = request
        else:
            prompt = getattr(request, "prompt", str(request))

        # Cache responses for same prompt
        if prompt not in self._response_cache:
            self._response_cache[prompt] = self._craft_response(prompt)

        response_text = self._response_cache[prompt]

        # Create mock response object matching GenerationResponse interface
        mock_response = MagicMock()
        mock_response.text = response_text
        mock_response.tokens_used = self.config.tokens_per_response
        mock_response.generation_time_ms = self.config.latency_ms
        mock_response.finish_reason = "stop"
        mock_response.model_name = "mock-deterministic"
        mock_response.used_template = False
        mock_response.template_name = None

        return mock_response

    def _craft_response(self, prompt: str) -> str:
        """Craft a context-appropriate mock response."""
        prompt_lower = prompt.lower()
        prompt_words = set(prompt_lower.split())

        # Match against patterns
        for keywords, response in self.RESPONSE_PATTERNS:
            if prompt_words & keywords:  # Intersection
                return response

        # Check for question mark
        if "?" in prompt:
            return "That's an interesting question."

        return self.config.default_response

    def is_loaded(self) -> bool:
        """Always loaded."""
        return True

    def unload(self) -> None:
        """No-op unload."""
        pass

    def __repr__(self) -> str:
        return f"DeterministicGenerator(latency={self.config.latency_ms}ms)"


@dataclass
class ClassificationResult:
    """Simple dataclass for classification results.

    Mirrors the structure of real classification results.
    """

    category: str
    confidence: float
    method: str

    def __repr__(self) -> str:
        return f"ClassificationResult({self.category}, conf={self.confidence:.2f})"


class DeterministicClassifier:
    """Deterministic classifier that returns predictable categories.

    Uses rule-based matching for consistent, fast classification
    without loading ML models.

    Example:
        >>> clf = DeterministicClassifier()
        >>> result = clf.classify("ok")
        >>> result.category
        'acknowledge'
        >>> result.confidence
        0.95
    """

    # (matcher_fn, category, confidence, method)
    RULES: list[tuple[Callable[[str], bool], str, float, str]] = [
        # Exact match acknowledgments
        (lambda t: t.lower() in {"ok", "okay", "k", "kk"}, "acknowledge", 0.95, "exact"),
        (lambda t: t.lower() in {"got it", "gotcha", "roger"}, "acknowledge", 0.92, "exact"),
        (lambda t: t.lower() in {"sounds good", "sounds great"}, "acknowledge", 0.90, "exact"),
        # Questions
        (lambda t: "?" in t, "question", 0.85, "pattern"),
        (
            lambda t: t.lower().startswith(("what", "how", "why", "when", "where")),
            "question",
            0.88,
            "pattern",
        ),
        # Emotions
        (
            lambda t: any(
                w in t.lower() for w in ["love", "loved", "hate", "happy", "sad", "angry"]
            ),
            "emotion",
            0.80,
            "keyword",
        ),
        # Requests
        (
            lambda t: any(w in t.lower() for w in ["can you", "please", "send", "help me"]),
            "request",
            0.75,
            "keyword",
        ),
        # Closings
        (
            lambda t: any(w in t.lower() for w in ["bye", "goodbye", "talk later", "see you"]),
            "closing",
            0.90,
            "keyword",
        ),
    ]

    def classify(self, text: str, **kwargs: Any) -> ClassificationResult:
        """Classify text using deterministic rules.

        Args:
            text: Text to classify
            **kwargs: Ignored (for API compatibility)

        Returns:
            ClassificationResult with category, confidence, method
        """
        for matcher, category, confidence, method in self.RULES:
            if matcher(text):
                return ClassificationResult(category, confidence, method)

        # Default fallback
        return ClassificationResult("statement", 0.60, "fallback")

    def classify_with_scores(self, text: str, **kwargs: Any) -> list[tuple[str, float]]:
        """Return sorted list of (category, confidence) for all categories.

        Args:
            text: Text to classify

        Returns:
            List of (category, confidence) tuples, sorted by confidence desc
        """
        result = self.classify(text)

        # Build score distribution
        scores = []
        for cat in ["acknowledge", "question", "emotion", "request", "closing", "statement"]:
            if cat == result.category:
                scores.append((cat, result.confidence))
            else:
                # Lower score for non-matching categories
                scores.append((cat, max(0.1, result.confidence - 0.3)))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    def reset(self) -> None:
        """No-op reset (for API compatibility)."""
        pass


# Type alias for import convenience
MockEmbedder = DeterministicEmbedder
MockGenerator = DeterministicGenerator
MockClassifier = DeterministicClassifier
