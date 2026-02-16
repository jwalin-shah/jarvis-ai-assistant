"""In-process BERT embedder using MLX - ~30MB RAM overhead.

Implements BERT-style embeddings directly in MLX without requiring an external
microservice. Uses only mlx.core/mlx.nn (for inference) and tokenizers (for
fast tokenization, no transformers dependency).

Thread Safety:
    Uses MLXModelLoader._mlx_load_lock to serialize GPU access with the LLM
    loader, preventing concurrent Metal GPU operations on 8GB systems.

Usage:
    from models.bert_embedder import get_in_process_embedder

    embedder = get_in_process_embedder()
    embeddings = embedder.encode(["Hello, world!"])
    print(embeddings.shape)  # (1, 384)
"""
# mypy: ignore-errors

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import struct
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tokenizers import Tokenizer

from jarvis.core.exceptions import ErrorCode, JarvisError
from models.memory_config import gpu_context
from models.utils import HF_CACHE, find_model_snapshot, map_hf_bert_key

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# =============================================================================
# Embedding Constants
# =============================================================================

# Embedding dimension is 384 for all supported models
MLX_EMBEDDING_DIM = 384

# Default model ID
DEFAULT_MLX_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Batch size for MLX embedding requests
MLX_BATCH_SIZE = 100

# =============================================================================
# Embedding Exceptions
# =============================================================================


class MLXEmbeddingError(JarvisError):
    """Raised when MLX embedding operations fail."""

    default_message = "MLX embedding operation failed"
    default_code = ErrorCode.MDL_LOAD_FAILED


class MLXServiceNotAvailableError(MLXEmbeddingError):
    """Raised when MLX embedding is not available."""

    default_message = "MLX embedding is not available"


class MLXModelLoadError(MLXEmbeddingError):
    """Raised when MLX embedding model fails to load."""

    default_message = "Failed to load MLX embedding model"


# For backwards compatibility
MLXModelNotAvailableError = MLXServiceNotAvailableError


def is_mlx_available() -> bool:
    """Check if in-process MLX embeddings are available.

    Returns:
        True if MLX can be imported (always True on Apple Silicon).
    """
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


# Model registry: name -> (hf_repo, pooling_mode)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    # BGE models (BAAI) - 384d small, 768d base, 1024d large
    "bge-small": ("BAAI--bge-small-en-v1.5", "cls"),
    "bge-base": ("BAAI--bge-base-en-v1.5", "cls"),
    "bge-large": ("BAAI--bge-large-en-v1.5", "cls"),
    # Arctic models (Snowflake) - 384d xs/s, 768d m, 1024d l
    "arctic-xs": ("Snowflake--snowflake-arctic-embed-xs", "cls"),
    "arctic-s": ("Snowflake--snowflake-arctic-embed-s", "cls"),
    "arctic-m": ("Snowflake--snowflake-arctic-embed-m", "cls"),
    "arctic-l": ("Snowflake--snowflake-arctic-embed-l", "cls"),
    # Fast models from old adapter registry
    "gte-tiny": ("TaylorAI--gte-tiny", "cls"),
    "minilm-l6": ("sentence-transformers--all-MiniLM-L6-v2", "cls"),
    "bge-micro": ("TaylorAI--bge-micro-v2", "cls"),
}

# Maps config name to registry name
EMBEDDING_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "bge-small": ("BAAI/bge-small-en-v1.5", "bge-small"),
    "gte-tiny": ("TaylorAI/gte-tiny", "gte-tiny"),
    "minilm-l6": ("sentence-transformers/all-MiniLM-L6-v2", "minilm-l6"),
    "bge-micro": ("TaylorAI/bge-micro-v2", "bge-micro"),
}


# =============================================================================
# Minimal BERT Implementation in MLX
# =============================================================================


class BertEmbeddings(nn.Module):
    """BERT embeddings: word + position + token_type."""

    def __init__(
        self, vocab_size: int, hidden_size: int, max_position: int, type_vocab_size: int = 2
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def __call__(self, input_ids: mx.array, token_type_ids: mx.array = None) -> mx.array:
        seq_len = input_ids.shape[1]
        position_ids = mx.arange(seq_len)

        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = mx.zeros_like(input_ids)
        type_emb = self.token_type_embeddings(token_type_ids)

        embeddings = word_emb + pos_emb + type_emb
        return self.LayerNorm(embeddings)


class BertSelfAttention(nn.Module):
    """Multi-head self attention."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array = None) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        q = (
            self.query(hidden_states)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.key(hidden_states)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.value(hidden_states)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = mx.softmax(scores, axis=-1)
        output = (weights @ v).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return output


class BertLayer(nn.Module):
    """Single BERT transformer layer."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int):
        super().__init__()
        self.attention = BertSelfAttention(hidden_size, num_heads)
        self.attention_output_dense = nn.Linear(hidden_size, hidden_size)
        self.attention_output_LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array = None) -> mx.array:
        # Self-attention
        attn_output = self.attention(hidden_states, attention_mask)
        attn_output = self.attention_output_dense(attn_output)
        hidden_states = self.attention_output_LayerNorm(hidden_states + attn_output)

        # FFN
        intermediate = nn.gelu(self.intermediate_dense(hidden_states))
        output = self.output_dense(intermediate)
        hidden_states = self.output_LayerNorm(hidden_states + output)

        return hidden_states


class BertEncoder(nn.Module):
    """Stack of BERT layers."""

    def __init__(self, num_layers: int, hidden_size: int, intermediate_size: int, num_heads: int):
        super().__init__()
        self.layers = [
            BertLayer(hidden_size, intermediate_size, num_heads) for _ in range(num_layers)
        ]

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array = None) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertModel(nn.Module):
    """Minimal BERT model for embeddings."""

    def __init__(self, config: dict, add_pooler: bool = True):
        super().__init__()
        self.embeddings = BertEmbeddings(
            config["vocab_size"],
            config["hidden_size"],
            config["max_position_embeddings"],
            config.get("type_vocab_size", 2),
        )
        self.encoder = BertEncoder(
            config["num_hidden_layers"],
            config["hidden_size"],
            config["intermediate_size"],
            config["num_attention_heads"],
        )
        # Pooler is optional (some models like Arctic don't have it)
        if add_pooler:
            self.pooler_dense = nn.Linear(config["hidden_size"], config["hidden_size"])

    def __call__(
        self, input_ids: mx.array, attention_mask: mx.array = None, token_type_ids: mx.array = None
    ) -> mx.array:
        hidden_states = self.embeddings(input_ids, token_type_ids)

        # Create attention mask for self-attention
        if attention_mask is not None:
            # Convert [B, L] mask to [B, 1, 1, L] for broadcasting
            extended_mask = attention_mask[:, None, None, :]
            extended_mask = (1.0 - extended_mask) * -1e9
        else:
            extended_mask = None

        hidden_states = self.encoder(hidden_states, extended_mask)
        return hidden_states


# =============================================================================
# Weight Loading (convert HF names to our names)
# =============================================================================


def load_bert_weights(model: BertModel, weights_path: Path, has_pooler: bool = True) -> None:
    """Load weights from HuggingFace safetensors into our model."""
    hf_weights = mx.load(str(weights_path))

    # Rename keys in-place to avoid building a second ~500MB dict (peak 1.5GB -> ~1GB)
    for hf_name in list(hf_weights.keys()):
        weight = hf_weights.pop(hf_name)

        stripped = hf_name.replace("bert.", "")
        mapped = map_hf_bert_key(stripped)
        if mapped is None:
            continue
        if "pooler" in hf_name and not has_pooler:
            continue

        hf_weights[mapped] = weight

    model.load_weights(list(hf_weights.items()))
    del hf_weights


# =============================================================================
# In-Process Embedder
# =============================================================================


def _check_has_pooler(weights_path: Path) -> bool:
    """Check if model has pooler by reading safetensors header (not full weights)."""
    with open(weights_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        data = f.read(header_size)
        if len(data) != header_size:
            raise ValueError(
                f"Truncated safetensors header in {weights_path}: "
                f"expected {header_size} bytes, got {len(data)}"
            )
        header_json = data.decode("utf-8")
    return "pooler" in header_json


class InProcessEmbedder:
    """In-process MLX BERT embedder with CLS pooling.

    Thread-safe:
    - MLXModelLoader._mlx_load_lock serializes both tokenization and GPU operations,
      preventing race conditions where another thread changes tokenizer padding state
      between tokenization and the forward pass.
    """

    def __init__(self, model_name: str = "bge-small") -> None:
        self._default_model = model_name
        self.model: BertModel | None = None
        self.tokenizer: Tokenizer | None = None
        self.model_name: str | None = None
        self.config: dict | None = None

    def load_model(self, model_name: str) -> None:
        """Load a model by name. Thread-safe via shared GPU lock."""
        if self.model_name == model_name:
            return

        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
            )

        with gpu_context():
            # Double-check after acquiring lock
            if self.model_name == model_name:
                return

            # Unload previous model first to free memory
            if self.model is not None:
                logger.info("Unloading %s before loading %s", self.model_name, model_name)
                self._unload_unlocked()

            hf_repo, _pooling = MODEL_REGISTRY[model_name]
            model_dir = HF_CACHE / f"models--{hf_repo}"

            if not model_dir.exists():
                raise FileNotFoundError(
                    f"Model not found in cache: {model_dir}. "
                    f"Download it first with: huggingface-cli download {hf_repo}"
                )

            snapshot = find_model_snapshot(model_dir)

            # Load config
            config_path = snapshot / "config.json"
            with open(config_path) as f:
                self.config = json.load(f)

            # Load tokenizer
            tokenizer_path = snapshot / "tokenizer.json"
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            self.tokenizer.enable_truncation(max_length=512)

            # Check pooler from safetensors header (no full weight load)
            weights_path = snapshot / "model.safetensors"
            has_pooler = _check_has_pooler(weights_path)

            hidden = self.config["hidden_size"]
            layers = self.config["num_hidden_layers"]
            logger.info(
                "Loading %s (hidden=%d, layers=%d, pooler=%s)",
                model_name,
                hidden,
                layers,
                has_pooler,
            )
            start = time.time()

            self.model = BertModel(self.config, add_pooler=has_pooler)
            load_bert_weights(self.model, weights_path, has_pooler=has_pooler)

            mx.eval(self.model.parameters())

            gc.collect()

            self.model_name = model_name
            logger.info("Model loaded in %.2fs", time.time() - start)

    def _unload_unlocked(self) -> None:
        """Unload model without acquiring the GPU lock (caller holds it)."""
        prev_model = self.model_name
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.config = None
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        if prev_model:
            logger.info("Unloaded %s", prev_model)

    def unload(self) -> None:
        """Unload model to free memory. Thread-safe."""
        with gpu_context():
            self._unload_unlocked()

    def encode(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 64,
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        """Encode texts to embeddings with length-sorted batching.

        Args:
            texts: List of texts to encode.
            normalize: Whether to L2-normalize embeddings.
            batch_size: Batch size for processing.
            dtype: Output dtype (float32 or float16). Use float16 to save 50% memory.

        Returns:
            NumPy array of shape (n_texts, embedding_dim).
        """
        if self.model is None:
            self.load_model(self._default_model)

        if not texts:
            return np.array([], dtype=dtype)

        # Single critical section: tokenization + GPU forward pass.
        # The GPU lock protects both the tokenizer state (padding config)
        # and the Metal GPU operations, preventing race conditions where
        # another thread changes padding between tokenization and forward pass.
        with gpu_context():
            # Tokenize once WITHOUT padding to get actual lengths for sorting.
            # We manually pad per-batch below so each batch only pads to its
            # own max length (not the global max), saving memory.
            self.tokenizer.no_padding()
            encodings = self.tokenizer.encode_batch(texts)
            lengths = [len(e.ids) for e in encodings]

            # Sort by length (longest first) to minimize padding waste
            sorted_indices = np.argsort(-np.array(lengths))
            reverse_indices = np.empty_like(sorted_indices)
            reverse_indices[sorted_indices] = np.arange(len(sorted_indices))

            output_embeddings = []

            for batch_start in range(0, len(texts), batch_size):
                batch_end = min(batch_start + batch_size, len(texts))
                batch_indices = sorted_indices[batch_start:batch_end]

                # Reuse pre-computed encodings, manually pad to batch max
                batch_encodings = [encodings[i] for i in batch_indices]
                max_len = max(len(e.ids) for e in batch_encodings)

                input_ids = np.array(
                    [e.ids + [0] * (max_len - len(e.ids)) for e in batch_encodings],
                    dtype=np.int32,
                )
                attention_mask = np.array(
                    [
                        e.attention_mask + [0] * (max_len - len(e.attention_mask))
                        for e in batch_encodings
                    ],
                    dtype=np.int32,
                )

                input_ids_mx = mx.array(input_ids)
                attention_mask_mx = mx.array(attention_mask)

                hidden_states = self.model(input_ids_mx, attention_mask_mx)

                # CLS pooling
                batch_emb = hidden_states[:, 0, :]

                if normalize:
                    norms = mx.maximum(mx.linalg.norm(batch_emb, axis=1, keepdims=True), 1e-9)
                    batch_emb = batch_emb / norms

                mx.eval(batch_emb)

                # Convert to numpy and free MLX arrays immediately
                batch_emb_np = np.array(batch_emb)
                # Convert to requested dtype (float16 saves 50% memory)
                if dtype != np.float32:
                    batch_emb_np = batch_emb_np.astype(dtype)
                del input_ids_mx, attention_mask_mx, hidden_states, batch_emb

                output_embeddings.append(batch_emb_np)

        all_embeddings = np.vstack(output_embeddings)
        return all_embeddings[reverse_indices]

    @property
    def is_loaded(self) -> bool:
        """Whether a model is currently loaded."""
        return self.model is not None

    def is_available(self) -> bool:
        """Check if the in-process embedder can run (MLX importable)."""
        return True


# =============================================================================
# Singleton
# =============================================================================

_in_process_embedder: InProcessEmbedder | None = None
_in_process_lock = threading.Lock()


def get_in_process_embedder(model_name: str = "bge-small") -> InProcessEmbedder:
    """Get or create the singleton InProcessEmbedder.

    Args:
        model_name: Default model to use when encode() is called without prior load.

    Returns:
        The shared InProcessEmbedder instance.
    """
    global _in_process_embedder

    if _in_process_embedder is not None:
        return _in_process_embedder

    with _in_process_lock:
        if _in_process_embedder is None:
            _in_process_embedder = InProcessEmbedder(model_name=model_name)
        return _in_process_embedder


def reset_in_process_embedder() -> None:
    """Reset the singleton. Unloads model and clears instance."""
    global _in_process_embedder

    with _in_process_lock:
        if _in_process_embedder is not None:
            _in_process_embedder.unload()
        _in_process_embedder = None


def get_model_info(model_name: str | None = None) -> tuple[str, str]:
    """Get HuggingFace model ID and MLX model name for a config model name."""
    if model_name is None:
        from jarvis.config import get_config

        model_name = get_config().embedding.model_name

    if model_name not in EMBEDDING_MODEL_REGISTRY:
        valid_models = ", ".join(EMBEDDING_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown embedding model '{model_name}'. Valid options: {valid_models}")

    return EMBEDDING_MODEL_REGISTRY[model_name]


def get_configured_model_name() -> str:
    """Get the currently configured embedding model name."""
    from jarvis.config import get_config

    return get_config().embedding.model_name


class MLXEmbedder:
    """High-level MLX embedder adapter."""

    def __init__(self) -> None:
        self._mlx_embedder: Any = None
        self._model_name: str | None = None
        self._lock = threading.Lock()
        self._init_attempted = False

    def _initialize(self) -> None:
        if self._init_attempted:
            return
        with self._lock:
            if self._init_attempted:
                return
            self._init_attempted = True
            _, mlx_model_name = get_model_info()
            self._model_name = get_configured_model_name()
            try:
                self._mlx_embedder = get_in_process_embedder(model_name=mlx_model_name)
            except Exception as e:
                logger.error("Failed to initialize MLX embedder: %s", e)
                raise RuntimeError(f"Could not initialize MLX embedder: {e}") from e

    def is_available(self) -> bool:
        try:
            self._initialize()
            return self._mlx_embedder is not None
        except RuntimeError:
            return False

    @property
    def backend(self) -> str:
        return "mlx"

    @property
    def embedding_dim(self) -> int:
        return MLX_EMBEDDING_DIM

    @property
    def model_name(self) -> str:
        if self._model_name is None:
            return get_configured_model_name()
        return self._model_name

    def encode(
        self,
        texts: list[str] | str,
        normalize: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool | None = None,
        dtype: np.dtype = np.float32,
    ) -> NDArray[np.float32]:
        self._initialize()
        if normalize_embeddings is not None:
            normalize = normalize_embeddings
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.array([], dtype=dtype).reshape(0, MLX_EMBEDDING_DIM)

        if len(texts) <= MLX_BATCH_SIZE:
            return self._mlx_embedder.encode(texts, normalize=normalize, dtype=dtype)

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), MLX_BATCH_SIZE):
            batch = texts[i : i + MLX_BATCH_SIZE]
            all_embeddings.append(
                self._mlx_embedder.encode(batch, normalize=normalize, dtype=dtype)
            )
        return np.vstack(all_embeddings)

    def unload(self) -> None:
        with self._lock:
            if self._mlx_embedder is not None:
                self._mlx_embedder.unload()
            self._init_attempted = False


class CachedEmbedder:
    """Per-request embedding cache wrapper."""

    def __init__(self, base_embedder: MLXEmbedder, maxsize: int = 1000) -> None:
        self.base = base_embedder
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._computations = 0
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def backend(self) -> str:
        return self.base.backend

    @property
    def embedding_dim(self) -> int:
        return self.base.embedding_dim

    @property
    def model_name(self) -> str:
        return self.base.model_name

    def is_available(self) -> bool:
        return self.base.is_available()

    def unload(self) -> None:
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
        dtype: np.dtype = np.float32,
    ) -> NDArray[np.float32]:
        if normalize_embeddings is not None:
            normalize = normalize_embeddings
        if isinstance(texts, str):
            key = self._make_key(texts)
            cached = self._get(key)
            if cached is not None:
                return cached
            result = self.base.encode([texts], normalize=normalize, dtype=dtype)
            vector = result.reshape(1, -1)
            self._set(key, vector)
            self._computations += 1
            return vector
        if not texts:
            return np.array([], dtype=dtype).reshape(0, MLX_EMBEDDING_DIM)

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
            embeddings = self.base.encode(missing_texts, normalize=normalize, dtype=dtype)
            for idx, emb, cache_key in zip(missing_indices, embeddings, missing_keys):
                vector = np.asarray(emb, dtype=dtype).reshape(1, -1)
                cached_vectors[idx] = vector
                self._set(cache_key, vector)
            self._computations += len(missing_texts)
        return np.vstack(cached_vectors)

    def embed_batch(self, texts: list[str]) -> NDArray[np.float32]:
        """Alias for encode for common internal API compatibility."""
        return self.encode(texts)


_embedder: MLXEmbedder | None = None
_cached_embedder: CachedEmbedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> CachedEmbedder:
    global _embedder, _cached_embedder
    if _cached_embedder is None:
        with _embedder_lock:
            if _cached_embedder is None:
                if _embedder is None:
                    _embedder = MLXEmbedder()
                _cached_embedder = CachedEmbedder(_embedder)
    return _cached_embedder


def reset_embedder() -> None:
    global _embedder, _cached_embedder
    with _embedder_lock:
        if _cached_embedder is not None:
            _cached_embedder.unload()
            _cached_embedder = None
        if _embedder is not None:
            _embedder = None


def is_embedder_available() -> bool:
    try:
        embedder = get_embedder()
        return embedder.is_available()
    except RuntimeError:
        return False
