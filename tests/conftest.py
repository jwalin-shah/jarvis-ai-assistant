"""Pytest configuration for JARVIS tests.

Handles mocking of platform-specific dependencies (MLX) to allow tests
to run on non-macOS platforms.
"""

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


def _mock_mlx_modules():
    """Mock MLX modules before they are imported by the tests.

    MLX is only available on macOS Apple Silicon. These mocks allow
    basic tests to run on other platforms (Linux, Intel Mac, Windows).
    Note: Tests that require sentence_transformers will be skipped.
    """
    # Create mock for top-level mlx package (required for mlx.core to work)
    mock_mlx = types.ModuleType("mlx")
    mock_mlx.__path__ = []  # Make it a package

    # Create mock for mlx.core
    mock_mx = MagicMock()
    mock_mx.metal = MagicMock()
    mock_mx.metal.clear_cache = MagicMock()

    mock_mlx_lm = MagicMock()
    mock_mlx_lm.load = MagicMock(return_value=(MagicMock(), MagicMock()))
    mock_mlx_lm.generate = MagicMock(return_value="Generated text")

    mock_sample_utils = MagicMock()
    mock_sample_utils.make_sampler = MagicMock(return_value=MagicMock())

    # Install mocks in sys.modules before any imports
    # Need to mock 'mlx' top-level for 'mlx.core' to be importable
    sys.modules["mlx"] = mock_mlx
    sys.modules["mlx.core"] = mock_mx
    sys.modules["mlx_lm"] = mock_mlx_lm
    sys.modules["mlx_lm.sample_utils"] = mock_sample_utils


# Check if MLX is actually usable (not just installed)
# The package may be installed but fail to load on non-macOS/non-Apple-Silicon
try:
    import mlx.core  # noqa: F401

    MLX_AVAILABLE = True
except (ImportError, OSError):
    MLX_AVAILABLE = False

# Mock MLX modules if not usable - MUST happen before any imports
if not MLX_AVAILABLE:
    _mock_mlx_modules()

# Fix stub torch namespace package: if torch is a namespace stub (no real
# __init__.py), scipy's array_api_compat crashes accessing torch.Tensor.
# Preemptively ensure the stub has a Tensor class if torch resolves to a
# namespace package.
try:
    import torch as _torch_check

    if not hasattr(_torch_check, "Tensor"):
        _torch_check.Tensor = type("Tensor", (), {})
except ImportError:
    pass

# Check if sentence_transformers can load properly (AFTER mocking MLX)
# Import is deferred to avoid torch circular import issues during pytest collection
SENTENCE_TRANSFORMERS_AVAILABLE = False


def _check_sentence_transformers():
    """Lazy check for sentence_transformers availability."""
    global SENTENCE_TRANSFORMERS_AVAILABLE
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401

        SENTENCE_TRANSFORMERS_AVAILABLE = True
    except (ImportError, ValueError, AttributeError, TypeError):
        # AttributeError: torch._C or torch.fx not available (broken torch install)
        # TypeError: packaging version parse error
        SENTENCE_TRANSFORMERS_AVAILABLE = False
    return SENTENCE_TRANSFORMERS_AVAILABLE


# Defer the check to first actual use rather than import time
# This avoids torch circular import issues during pytest collection
try:
    # Quick check without full import - just see if the package exists
    import importlib.util

    SENTENCE_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# Pytest marker for tests that require sentence_transformers
requires_sentence_transformers = pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence_transformers not available (requires macOS Apple Silicon with MLX)",
)


# =============================================================================
# Mock Embedder for Tests
# =============================================================================


class MockEmbedder:
    """Mock embedder for tests that don't need actual MLX service."""

    def __init__(self):
        self._model_name = "bge-small"

    @property
    def backend(self) -> str:
        return "mock"

    @property
    def embedding_dim(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        return True

    def encode(
        self,
        texts: list | str,
        normalize: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool | None = None,
    ) -> np.ndarray:
        """Generate deterministic mock embeddings based on text hash."""
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, 384)

        embeddings = []
        for text in texts:
            # Deterministic embedding based on text hash
            np.random.seed(hash(text) % 2**32)
            emb = np.random.randn(384).astype(np.float32)
            if normalize:
                emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        return np.array(embeddings, dtype=np.float32)

    def unload(self) -> None:
        pass


@pytest.fixture
def mock_embedder():
    """Provide a mock embedder for tests."""
    return MockEmbedder()


# Check if real MLX embedding service is available
def _check_mlx_service_available():
    """Check if MLX embedding service is running."""
    try:
        from jarvis.embedding_adapter import get_embedder

        embedder = get_embedder()
        return embedder.is_available() and embedder.backend == "mlx"
    except Exception:
        return False


# Lazy evaluation - only check once
_MLX_SERVICE_AVAILABLE = None


def is_mlx_service_available():
    """Check if MLX service is available (cached)."""
    global _MLX_SERVICE_AVAILABLE
    if _MLX_SERVICE_AVAILABLE is None:
        _MLX_SERVICE_AVAILABLE = _check_mlx_service_available()
    return _MLX_SERVICE_AVAILABLE


# Marker for tests that require real embeddings (not mocks)
requires_real_embeddings = pytest.mark.skipif(
    not is_mlx_service_available(),
    reason="Test requires real MLX embedding service (not mocks)",
)


@pytest.fixture(autouse=True)
def auto_mock_embedder(monkeypatch, request):
    """Auto-mock get_embedder() for all tests unless MLX service is running.

    This allows tests to run without the MLX service while still testing
    the business logic. Tests marked with @pytest.mark.requires_real_embeddings
    will be skipped if the service isn't available.
    """
    # Skip mocking if real service is available
    if is_mlx_service_available():
        return

    mock = MockEmbedder()

    def mock_get_embedder():
        return mock

    def mock_is_available():
        return True

    # Patch at module level
    monkeypatch.setattr("jarvis.embedding_adapter.get_embedder", mock_get_embedder)
    monkeypatch.setattr("jarvis.embedding_adapter.is_embedder_available", mock_is_available)

    # Also patch where it might be imported directly
    try:
        monkeypatch.setattr("jarvis.router.get_embedder", mock_get_embedder)
    except AttributeError:
        pass

    try:
        monkeypatch.setattr("jarvis.semantic_search.get_embedder", mock_get_embedder)
    except AttributeError:
        pass
