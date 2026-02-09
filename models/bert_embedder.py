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

from __future__ import annotations

import gc
import json
import logging
import math
import struct
import threading
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

HF_CACHE = Path.home() / ".cache/huggingface/hub"

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
    import psutil
    proc = psutil.Process()

    mem_before = proc.memory_info()
    print(f"[BERT] BEFORE mx.load: RSS={mem_before.rss/1024/1024:.1f} MB, VMS={mem_before.vms/1024/1024:.1f} MB")

    hf_weights = mx.load(str(weights_path))

    mem_after = proc.memory_info()
    print(f"[BERT] AFTER mx.load: RSS={mem_after.rss/1024/1024:.1f} MB (+{(mem_after.rss-mem_before.rss)/1024/1024:.1f}), VMS={mem_after.vms/1024/1024:.1f} MB (+{(mem_after.vms-mem_before.vms)/1024/1024:.1f})")

    new_weights = {}

    for hf_name, weight in hf_weights.items():
        if "position_ids" in hf_name:
            continue

        if "pooler" in hf_name and not has_pooler:
            continue

        name = hf_name.replace("bert.", "")

        # Encoder layers
        if "encoder.layer." in name:
            name = name.replace("encoder.layer.", "encoder.layers.")
            name = name.replace(".attention.self.query", ".attention.query")
            name = name.replace(".attention.self.key", ".attention.key")
            name = name.replace(".attention.self.value", ".attention.value")
            name = name.replace(".attention.output.dense", ".attention_output_dense")
            name = name.replace(".attention.output.LayerNorm", ".attention_output_LayerNorm")
            name = name.replace(".intermediate.dense", ".intermediate_dense")
            name = name.replace(".output.dense", ".output_dense")
            name = name.replace(".output.LayerNorm", ".output_LayerNorm")

        # Pooler
        name = name.replace("pooler.dense", "pooler_dense")

        new_weights[name] = weight

    mem_before_load = proc.memory_info()
    print(f"[BERT] BEFORE model.load_weights: RSS={mem_before_load.rss/1024/1024:.1f} MB, VMS={mem_before_load.vms/1024/1024:.1f} MB")

    model.load_weights(list(new_weights.items()))

    mem_after_load = proc.memory_info()
    print(f"[BERT] AFTER model.load_weights: RSS={mem_after_load.rss/1024/1024:.1f} MB (+{(mem_after_load.rss-mem_before_load.rss)/1024/1024:.1f}), VMS={mem_after_load.vms/1024/1024:.1f} MB (+{(mem_after_load.vms-mem_before_load.vms)/1024/1024:.1f})")

    # FREE MEMORY: Delete weight dicts immediately (they're copied into model params)
    del hf_weights, new_weights

    mem_after_del = proc.memory_info()
    print(f"[BERT] AFTER deleting weight dicts: RSS={mem_after_del.rss/1024/1024:.1f} MB, VMS={mem_after_del.vms/1024/1024:.1f} MB")


# =============================================================================
# In-Process Embedder
# =============================================================================


def _check_has_pooler(weights_path: Path) -> bool:
    """Check if model has pooler by reading safetensors header (not full weights)."""
    with open(weights_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
    return "pooler" in header_json


class InProcessEmbedder:
    """In-process MLX BERT embedder with CLS pooling.

    Thread-safe:
    - _encode_lock serializes encode() calls (Rust tokenizer is not thread-safe)
    - MLXModelLoader._mlx_load_lock serializes model load/unload with the LLM loader
    """

    def __init__(self, model_name: str = "bge-small") -> None:
        self._default_model = model_name
        self._encode_lock = threading.Lock()
        self.model: BertModel | None = None
        self.tokenizer: Tokenizer | None = None
        self.model_name: str | None = None
        self.config: dict | None = None

    def _get_gpu_lock(self) -> threading.Lock:
        """Get the shared MLX GPU lock from MLXModelLoader."""
        from models.loader import MLXModelLoader

        return MLXModelLoader._mlx_load_lock

    def load_model(self, model_name: str) -> None:
        """Load a model by name. Thread-safe via shared GPU lock."""
        if self.model_name == model_name:
            return

        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
            )

        with self._get_gpu_lock():
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
                    "Download it first with: huggingface-cli download {hf_repo}"
                )

            # Find snapshot
            snapshots_dir = model_dir / "snapshots"
            snapshot = next(snapshots_dir.iterdir())

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
                model_name, hidden, layers, has_pooler,
            )
            start = time.time()

            import psutil
            proc = psutil.Process()

            # CRITICAL: Set MLX memory limit BEFORE model creation to prevent 5GB VMS spike on 8GB systems
            # Without this, MLX allocates 4-5GB virtual memory during batch encoding,
            # triggering swap even though RSS stays low
            mx.set_memory_limit(1 * 1024 * 1024 * 1024)  # 1 GB
            mx.set_cache_limit(512 * 1024 * 1024)  # 512 MB cache

            self.model = BertModel(self.config, add_pooler=has_pooler)
            load_bert_weights(self.model, weights_path, has_pooler=has_pooler)

            mem_before_eval = proc.memory_info()
            print(f"[BERT] BEFORE mx.eval: RSS={mem_before_eval.rss/1024/1024:.1f} MB, VMS={mem_before_eval.vms/1024/1024:.1f} MB")

            mx.eval(self.model.parameters())

            mem_after_eval = proc.memory_info()
            print(f"[BERT] AFTER mx.eval: RSS={mem_after_eval.rss/1024/1024:.1f} MB (+{(mem_after_eval.rss-mem_before_eval.rss)/1024/1024:.1f}), VMS={mem_after_eval.vms/1024/1024:.1f} MB (+{(mem_after_eval.vms-mem_before_eval.vms)/1024/1024:.1f})")

            # Clear cache after loading to free temp GPU buffers
            gc.collect()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()

            mem_after_clear = proc.memory_info()
            print(f"[BERT] AFTER mx.clear_cache: RSS={mem_after_clear.rss/1024/1024:.1f} MB, VMS={mem_after_clear.vms/1024/1024:.1f} MB")

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
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
        if prev_model:
            logger.info("Unloaded %s", prev_model)

    def unload(self) -> None:
        """Unload model to free memory. Thread-safe."""
        with self._get_gpu_lock():
            self._unload_unlocked()

    def encode(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Encode texts to embeddings with length-sorted batching.

        Args:
            texts: List of texts to encode.
            normalize: Whether to L2-normalize embeddings.
            batch_size: Batch size for processing.

        Returns:
            NumPy array of shape (n_texts, embedding_dim).
        """
        if self.model is None:
            self.load_model(self._default_model)

        if not texts:
            return np.array([], dtype=np.float32)

        # Serialize encode calls: tokenizer mutates internal state (padding on/off)
        # and the Rust tokenizer is not thread-safe for concurrent access.
        with self._encode_lock:
            # Tokenize without padding to get lengths
            try:
                self.tokenizer.no_padding()
                encodings = self.tokenizer.encode_batch(texts)
                lengths = [len(e.ids) for e in encodings]
            finally:
                # Re-enable padding for batch processing (always restore state)
                self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

            # Sort by length (longest first) to minimize padding waste
            sorted_indices = np.argsort([-length for length in lengths])
            reverse_indices = np.argsort(sorted_indices)

            output_embeddings = []

            for batch_start in range(0, len(texts), batch_size):
                batch_end = min(batch_start + batch_size, len(texts))
                batch_indices = sorted_indices[batch_start:batch_end]
                batch_texts = [texts[i] for i in batch_indices]

                batch_encodings = self.tokenizer.encode_batch(batch_texts)
                input_ids = np.array([e.ids for e in batch_encodings], dtype=np.int32)
                attention_mask = np.array(
                    [e.attention_mask for e in batch_encodings], dtype=np.int32
                )

                input_ids_mx = mx.array(input_ids)
                attention_mask_mx = mx.array(attention_mask)

                # Hold the shared GPU lock during forward pass + eval to
                # prevent concurrent Metal GPU access with LLM model loads.
                with self._get_gpu_lock():
                    hidden_states = self.model(input_ids_mx, attention_mask_mx)

                    # CLS pooling
                    batch_emb = hidden_states[:, 0, :]

                    if normalize:
                        norms = mx.maximum(
                            mx.linalg.norm(batch_emb, axis=1, keepdims=True), 1e-9
                        )
                        batch_emb = batch_emb / norms

                    mx.eval(batch_emb)

                    # Convert to numpy and free MLX arrays immediately
                    batch_emb_np = np.array(batch_emb)
                    del input_ids_mx, attention_mask_mx, hidden_states, batch_emb

                output_embeddings.append(batch_emb_np)

                # Clear MLX cache every 100 batches to prevent accumulation
                # Do this OUTSIDE the GPU lock to reduce lock contention
                if (batch_end // batch_size) % 100 == 0:
                    if hasattr(mx, "clear_cache"):
                        mx.clear_cache()

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
