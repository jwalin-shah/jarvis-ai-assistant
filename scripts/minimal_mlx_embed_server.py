#!/usr/bin/env python3
"""Minimal MLX Embedding Server - ~30MB RAM overhead instead of ~350MB.

Implements BERT-style embeddings using only:
- mlx.core / mlx.nn (for inference)
- tokenizers (for fast tokenization, no transformers)
- Standard library (asyncio, json, etc.)

Usage:
    uv run python scripts/minimal_mlx_embed_server.py

Protocol: JSON-RPC 2.0 over Unix socket (same as original)
Socket: /tmp/jarvis-embed-minimal.sock
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import math
import os
import signal
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tokenizers import Tokenizer

# Limit MLX memory pool to 1GB to avoid massive pre-allocation on 8GB systems
if hasattr(mx, "set_memory_limit"):
    mx.set_memory_limit(1024 * 1024 * 1024)  # 1GB
elif hasattr(mx.metal, "set_memory_limit"):
    mx.metal.set_memory_limit(1024 * 1024 * 1024)  # 1GB (deprecated)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("minimal-mlx-embed")

SOCKET_PATH = Path(os.environ.get("MLX_EMBED_SOCKET", "/tmp/jarvis-embed-minimal.sock"))
HF_CACHE = Path.home() / ".cache/huggingface/hub"

# Sequence length buckets for efficient batching (matches mlx-embedding-models)
# Texts are rounded up to nearest bucket, then grouped together
SEQ_LENS = list(range(16, 128, 16)) + list(range(128, 512, 32)) + [512]
# = [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512]

# Model registry: name -> (hf_repo, pooling_mode)
# All models use CLS pooling for fast MLX inference
MODEL_REGISTRY = {
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
        B, L, _ = hidden_states.shape

        q = (
            self.query(hidden_states)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.key(hidden_states)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.value(hidden_states)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = mx.softmax(scores, axis=-1)
        output = (weights @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
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

    # Build mapping from HF names to our names
    new_weights = {}

    for hf_name, weight in hf_weights.items():
        # Skip buffers that aren't model parameters
        if "position_ids" in hf_name:
            continue

        # Skip pooler if model doesn't have it
        if "pooler" in hf_name and not has_pooler:
            continue

        # Remove "bert." prefix if present
        name = hf_name.replace("bert.", "")

        # Embeddings
        name = name.replace("embeddings.word_embeddings", "embeddings.word_embeddings")
        name = name.replace("embeddings.position_embeddings", "embeddings.position_embeddings")
        name = name.replace("embeddings.token_type_embeddings", "embeddings.token_type_embeddings")
        name = name.replace("embeddings.LayerNorm", "embeddings.LayerNorm")

        # Encoder layers
        if "encoder.layer." in name:
            # attention.self.query -> layers.X.attention.query
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

    model.load_weights(list(new_weights.items()))


# =============================================================================
# Embedder Class
# =============================================================================


class MinimalEmbedder:
    """Minimal MLX embedder with CLS pooling."""

    def __init__(self):
        self.model: BertModel | None = None
        self.tokenizer: Tokenizer | None = None
        self.model_name: str | None = None
        self.config: dict | None = None

    def load_model(self, model_name: str) -> None:
        """Load a model by name."""
        if self.model_name == model_name:
            return

        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
            )

        # Unload previous model first to free memory
        if self.model is not None:
            logger.info(f"Unloading {self.model_name} before loading {model_name}")
            self.unload()

        hf_repo, _pooling = MODEL_REGISTRY[model_name]
        model_dir = HF_CACHE / f"models--{hf_repo}"

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model not found in cache: {model_dir}. Run original server once to download."
            )

        # Find snapshot
        snapshots_dir = model_dir / "snapshots"
        snapshot = next(snapshots_dir.iterdir())  # Get first snapshot

        # Load config
        config_path = snapshot / "config.json"
        with open(config_path) as f:
            self.config = json.load(f)

        # Load tokenizer with padding/truncation enabled
        tokenizer_path = snapshot / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=512)

        # Check if model has pooler by reading safetensors header (no full load)
        weights_path = snapshot / "model.safetensors"
        has_pooler = self._check_has_pooler(weights_path)

        # Create and load model (single load, not double)
        logger.info(
            f"Loading {model_name} (hidden={self.config['hidden_size']}, layers={self.config['num_hidden_layers']}, pooler={has_pooler})"
        )
        start = time.time()

        self.model = BertModel(self.config, add_pooler=has_pooler)
        load_bert_weights(self.model, weights_path, has_pooler=has_pooler)

        # Eval and compile
        mx.eval(self.model.parameters())

        self.model_name = model_name
        logger.info(f"Model loaded in {time.time() - start:.2f}s")

    @staticmethod
    def _check_has_pooler(weights_path: Path) -> bool:
        """Check if model has pooler by reading safetensors header (not full weights)."""
        import struct

        with open(weights_path, "rb") as f:
            # Safetensors format: 8-byte header size (little endian), then JSON header
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
        return "pooler" in header_json

    def unload(self) -> None:
        """Unload model to free memory."""
        prev_model = self.model_name
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.config = None
        gc.collect()
        # Clear MLX cache (use new API, fall back to deprecated if needed)
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
        if prev_model:
            logger.info(f"Unloaded {prev_model}")

    def encode(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts to embeddings with automatic length-sorted batching.

        Matches mlx-embedding-models optimization:
        1. Tokenize all texts
        2. Sort by token length to minimize padding waste
        3. Process in batches (sorted texts have similar lengths)
        4. Restore original order

        Args:
            texts: List of texts to encode.
            normalize: Whether to L2-normalize embeddings.
            batch_size: Batch size for processing.
            show_progress: Whether to print progress.
        """
        if self.model is None:
            self.load_model("bge-small")

        if not texts:
            return np.array([])

        # Tokenize without padding to get lengths
        self.tokenizer.no_padding()
        encodings = self.tokenizer.encode_batch(texts)
        lengths = [len(e.ids) for e in encodings]

        # Re-enable padding for batch processing
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        # Sort by length (longest first, like mlx-embedding-models)
        sorted_indices = np.argsort([-l for l in lengths])
        reverse_indices = np.argsort(sorted_indices)

        # Allocate output
        self.config["hidden_size"]
        output_embeddings = []

        # Process in batches
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_indices = sorted_indices[batch_start:batch_end]
            batch_texts = [texts[i] for i in batch_indices]

            # Tokenize batch (will pad to longest in batch)
            batch_encodings = self.tokenizer.encode_batch(batch_texts)
            input_ids = np.array([e.ids for e in batch_encodings], dtype=np.int32)
            attention_mask = np.array([e.attention_mask for e in batch_encodings], dtype=np.int32)

            # Forward pass
            input_ids_mx = mx.array(input_ids)
            attention_mask_mx = mx.array(attention_mask)
            hidden_states = self.model(input_ids_mx, attention_mask_mx)

            # CLS pooling
            batch_emb = hidden_states[:, 0, :]

            # Normalize
            if normalize:
                norms = mx.maximum(mx.linalg.norm(batch_emb, axis=1, keepdims=True), 1e-9)
                batch_emb = batch_emb / norms

            mx.eval(batch_emb)
            output_embeddings.append(np.array(batch_emb))

            if show_progress:
                print(f"  {batch_end}/{len(texts)} ({100 * batch_end / len(texts):.0f}%)")

        # Concatenate and restore original order
        all_embeddings = np.vstack(output_embeddings)
        return all_embeddings[reverse_indices]

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


# =============================================================================
# JSON-RPC Server
# =============================================================================


class MinimalEmbedServer:
    """Async Unix socket server."""

    def __init__(self, embedder: MinimalEmbedder):
        self._embedder = embedder
        self._server = None
        self._running = False
        self._start_time = time.time()

    async def start(self) -> None:
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()

        self._server = await asyncio.start_unix_server(self._handle_client, path=str(SOCKET_PATH))
        os.chmod(SOCKET_PATH, 0o600)
        self._running = True

        logger.info(f"Minimal MLX Embed Server listening on {SOCKET_PATH}")
        logger.info("RAM overhead: ~30MB (vs ~350MB for mlx_embedding_models)")

        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        self._embedder.unload()
        logger.info("Server stopped")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while self._running:
                line = await asyncio.wait_for(reader.readline(), timeout=300)
                if not line:
                    break
                response = await self._process(line.decode())
                writer.write(response.encode() + b"\n")
                await writer.drain()
        except TimeoutError:
            pass
        except Exception as e:
            logger.debug(f"Client error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _process(self, message: str) -> str:
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            return json.dumps(
                {"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e)}, "id": None}
            )

        method = data.get("method")
        params = data.get("params", {})
        req_id = data.get("id")

        handlers = {
            "health": self._health,
            "embed": self._embed,
            "unload": self._unload,
            "list_models": self._list_models,
            "ping": lambda p: {"status": "ok"},
        }

        handler = handlers.get(method)
        if not handler:
            return json.dumps(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                    "id": req_id,
                }
            )

        try:
            result = await asyncio.to_thread(handler, params)
            return json.dumps({"jsonrpc": "2.0", "result": result, "id": req_id})
        except Exception as e:
            return json.dumps(
                {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}, "id": req_id}
            )

    def _health(self, params: dict) -> dict:
        return {
            "status": "healthy",
            "model_loaded": self._embedder.is_loaded,
            "model_name": self._embedder.model_name,
            "uptime_seconds": time.time() - self._start_time,
            "backend": "minimal-mlx",
        }

    def _embed(self, params: dict) -> dict:
        texts = params.get("texts", [])
        model = params.get("model")
        normalize = params.get("normalize", True)
        binary = params.get("binary", False)  # Return base64-encoded binary

        if not texts:
            raise ValueError("No texts provided")

        if model:
            self._embedder.load_model(model)

        embeddings = self._embedder.encode(texts, normalize)

        if binary:
            # Return as base64-encoded float32 bytes (much smaller than JSON)
            import base64

            emb_bytes = embeddings.astype(np.float32).tobytes()
            return {
                "embeddings_b64": base64.b64encode(emb_bytes).decode("ascii"),
                "model": self._embedder.model_name,
                "dimension": embeddings.shape[1],
                "count": len(embeddings),
                "dtype": "float32",
            }
        else:
            return {
                "embeddings": embeddings.tolist(),
                "model": self._embedder.model_name,
                "dimension": embeddings.shape[1],
                "count": len(embeddings),
            }

    def _unload(self, params: dict) -> dict:
        self._embedder.unload()
        return {"status": "unloaded"}

    def _list_models(self, params: dict) -> dict:
        return {
            "models": [
                {"name": name, "pooling": pooling} for name, (_, pooling) in MODEL_REGISTRY.items()
            ],
            "current": self._embedder.model_name,
        }


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    embedder = MinimalEmbedder()
    server = MinimalEmbedServer(embedder)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))

    try:
        await server.start()
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
