"""Pure MLX cross-encoder for sequence-pair classification.

Reuses BERT infrastructure from bert_embedder.py to score (query, document) pairs
for reranking. ~50MB incremental memory cost (model weights + tokenizer).

Target model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params, 6 layers, 384 hidden)

Thread Safety:
    Uses MLXModelLoader._mlx_load_lock to serialize GPU access, same pattern
    as InProcessEmbedder in bert_embedder.py.

Usage:
    from models.cross_encoder import get_cross_encoder

    ce = get_cross_encoder()
    scores = ce.predict([("query", "doc1"), ("query", "doc2")])
    print(scores)  # array([0.87, 0.12])
"""

from __future__ import annotations

import gc
import json
import logging
import threading
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tokenizers import Tokenizer

from models.bert_embedder import BertModel

logger = logging.getLogger(__name__)

HF_CACHE = Path.home() / ".cache/huggingface/hub"

# Cross-encoder registry: name -> (hf_repo, num_labels, activation)
CROSS_ENCODER_REGISTRY: dict[str, tuple[str, int, str]] = {
    "ms-marco-MiniLM-L-6-v2": (
        "cross-encoder--ms-marco-MiniLM-L-6-v2",
        1,
        "sigmoid",
    ),
}


class BertForSequenceClassification(nn.Module):
    """BERT with a linear classification head on [CLS] token."""

    def __init__(self, config: dict, num_labels: int = 1) -> None:
        super().__init__()
        self.bert = BertModel(config, add_pooler=True)
        self.classifier = nn.Linear(config["hidden_size"], num_labels)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        token_type_ids: mx.array | None = None,
    ) -> mx.array:
        """Forward pass returning raw logits.

        Returns:
            Logits of shape (batch_size, num_labels).
        """
        hidden_states = self.bert(input_ids, attention_mask, token_type_ids)
        # [CLS] token is at position 0
        cls_output = hidden_states[:, 0, :]
        # Apply pooler (tanh activation) if available
        if hasattr(self.bert, "pooler_dense"):
            cls_output = mx.tanh(self.bert.pooler_dense(cls_output))
        logits = self.classifier(cls_output)
        return logits


def load_cross_encoder_weights(
    model: BertForSequenceClassification, weights_path: Path
) -> None:
    """Load HuggingFace cross-encoder weights into our model.

    Reuses load_bert_weights for the BERT backbone, then loads the classifier head.
    """
    # Load all weights
    hf_weights = mx.load(str(weights_path))

    # Separate classifier weights from BERT weights
    classifier_weights: dict[str, mx.array] = {}
    bert_weights: dict[str, mx.array] = {}

    for name, weight in hf_weights.items():
        if name.startswith("classifier."):
            # Map classifier.weight -> classifier.weight (already matches)
            classifier_weights[name] = weight
        else:
            bert_weights[name] = weight

    # Save bert weights to a temp mapping and load via the existing function
    # We need to build the name mapping manually since load_bert_weights
    # expects a file path. Instead, replicate the mapping logic.
    new_weights: dict[str, mx.array] = {}

    for hf_name, weight in bert_weights.items():
        if "position_ids" in hf_name:
            continue

        name = hf_name.replace("bert.", "bert.")  # keep bert. prefix for our model

        # Map HF names to our model structure (prefixed with "bert.")
        name_inner = hf_name.replace("bert.", "")

        # Encoder layers
        if "encoder.layer." in name_inner:
            name_inner = name_inner.replace("encoder.layer.", "encoder.layers.")
            name_inner = name_inner.replace(".attention.self.query", ".attention.query")
            name_inner = name_inner.replace(".attention.self.key", ".attention.key")
            name_inner = name_inner.replace(".attention.self.value", ".attention.value")
            name_inner = name_inner.replace(
                ".attention.output.dense", ".attention_output_dense"
            )
            name_inner = name_inner.replace(
                ".attention.output.LayerNorm", ".attention_output_LayerNorm"
            )
            name_inner = name_inner.replace(".intermediate.dense", ".intermediate_dense")
            name_inner = name_inner.replace(".output.dense", ".output_dense")
            name_inner = name_inner.replace(".output.LayerNorm", ".output_LayerNorm")

        # Pooler
        name_inner = name_inner.replace("pooler.dense", "pooler_dense")

        new_weights[f"bert.{name_inner}"] = weight

    # Add classifier weights
    for name, weight in classifier_weights.items():
        new_weights[name] = weight

    model.load_weights(list(new_weights.items()))


class InProcessCrossEncoder:
    """In-process MLX cross-encoder for scoring (query, document) pairs.

    Thread-safe:
    - _encode_lock serializes tokenizer calls (Rust tokenizer is not thread-safe)
    - MLXModelLoader._mlx_load_lock serializes GPU operations
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2") -> None:
        self._default_model = model_name
        self._encode_lock = threading.Lock()
        self.model: BertForSequenceClassification | None = None
        self.tokenizer: Tokenizer | None = None
        self.model_name: str | None = None
        self.config: dict | None = None
        self._num_labels: int = 1
        self._activation: str = "sigmoid"

    def _get_gpu_lock(self) -> threading.Lock:
        """Get the shared MLX GPU lock from MLXModelLoader."""
        from models.loader import MLXModelLoader

        return MLXModelLoader._mlx_load_lock

    def load_model(self, model_name: str | None = None) -> None:
        """Load a cross-encoder model. Thread-safe via shared GPU lock."""
        model_name = model_name or self._default_model

        if self.model_name == model_name:
            return

        if model_name not in CROSS_ENCODER_REGISTRY:
            raise ValueError(
                f"Unknown cross-encoder: {model_name}. "
                f"Available: {list(CROSS_ENCODER_REGISTRY.keys())}"
            )

        with self._get_gpu_lock():
            if self.model_name == model_name:
                return

            if self.model is not None:
                logger.info("Unloading %s before loading %s", self.model_name, model_name)
                self._unload_unlocked()

            hf_repo, num_labels, activation = CROSS_ENCODER_REGISTRY[model_name]
            self._num_labels = num_labels
            self._activation = activation

            model_dir = HF_CACHE / f"models--{hf_repo}"
            if not model_dir.exists():
                self._download_model(hf_repo.replace("--", "/"))
                if not model_dir.exists():
                    raise FileNotFoundError(
                        f"Model not found in cache: {model_dir}. "
                        f"Download with: huggingface-cli download {hf_repo.replace('--', '/')}"
                    )

            # Find snapshot
            snapshots_dir = model_dir / "snapshots"
            snapshot = next(snapshots_dir.iterdir())

            # Load config
            with open(snapshot / "config.json") as f:
                self.config = json.load(f)

            # Load tokenizer
            self.tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
            self.tokenizer.enable_truncation(max_length=512)
            self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

            hidden = self.config["hidden_size"]
            layers = self.config["num_hidden_layers"]
            logger.info(
                "Loading cross-encoder %s (hidden=%d, layers=%d, labels=%d)",
                model_name, hidden, layers, num_labels,
            )
            start = time.time()

            self.model = BertForSequenceClassification(self.config, num_labels)
            weights_path = snapshot / "model.safetensors"
            load_cross_encoder_weights(self.model, weights_path)
            mx.eval(self.model.parameters())

            self.model_name = model_name
            logger.info("Cross-encoder loaded in %.2fs", time.time() - start)

    def _download_model(self, repo_id: str) -> None:
        """Download model from HuggingFace Hub."""
        try:
            from huggingface_hub import snapshot_download

            logger.info("Downloading cross-encoder model: %s", repo_id)
            snapshot_download(
                repo_id,
                allow_patterns=["config.json", "tokenizer.json", "model.safetensors"],
            )
        except ImportError:
            logger.warning("huggingface_hub not installed, cannot auto-download")
        except Exception as e:
            logger.warning("Failed to download %s: %s", repo_id, e)

    def _unload_unlocked(self) -> None:
        """Unload model without acquiring GPU lock (caller holds it)."""
        prev = self.model_name
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.config = None
        gc.collect()
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
        if prev:
            logger.info("Unloaded cross-encoder %s", prev)

    def unload(self) -> None:
        """Unload model to free memory. Thread-safe."""
        with self._get_gpu_lock():
            self._unload_unlocked()

    def predict(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Score (query, document) pairs.

        Args:
            pairs: List of (query, document) text pairs.
            batch_size: Batch size for processing.

        Returns:
            NumPy array of relevance scores, shape (n_pairs,).
        """
        if self.model is None:
            self.load_model()

        if not pairs:
            return np.array([], dtype=np.float32)

        with self._encode_lock:
            # Tokenize pairs - encode(text_a, text_b) produces correct
            # token_type_ids for sentence pair classification
            self.tokenizer.no_padding()
            encodings = [
                self.tokenizer.encode(query, doc) for query, doc in pairs
            ]
            lengths = [len(e.ids) for e in encodings]
            self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

            # Sort by length for efficient batching
            sorted_indices = np.argsort([-l for l in lengths])
            all_scores: list[np.ndarray] = []

            for batch_start in range(0, len(pairs), batch_size):
                batch_end = min(batch_start + batch_size, len(pairs))
                batch_indices = sorted_indices[batch_start:batch_end]

                # Re-encode batch with padding to uniform length
                batch_encodings = [
                    self.tokenizer.encode(pairs[i][0], pairs[i][1])
                    for i in batch_indices
                ]
                # Pad all sequences in this batch to the same length
                max_len = max(len(e.ids) for e in batch_encodings)
                input_ids = np.array(
                    [e.ids + [0] * (max_len - len(e.ids)) for e in batch_encodings],
                    dtype=np.int32,
                )
                attention_mask = np.array(
                    [e.attention_mask + [0] * (max_len - len(e.attention_mask))
                     for e in batch_encodings],
                    dtype=np.int32,
                )
                token_type_ids = np.array(
                    [e.type_ids + [0] * (max_len - len(e.type_ids))
                     for e in batch_encodings],
                    dtype=np.int32,
                )

                input_ids_mx = mx.array(input_ids)
                attention_mask_mx = mx.array(attention_mask)
                token_type_ids_mx = mx.array(token_type_ids)

                with self._get_gpu_lock():
                    logits = self.model(
                        input_ids_mx, attention_mask_mx, token_type_ids_mx
                    )

                    if self._activation == "sigmoid":
                        scores = mx.sigmoid(logits)
                    else:
                        scores = logits

                    # Squeeze num_labels dim if single label
                    if self._num_labels == 1:
                        scores = scores.squeeze(-1)

                    mx.eval(scores)

                all_scores.append(np.array(scores))

            # Concatenate and unsort
            all_scores_arr = np.concatenate(all_scores)
            reverse_indices = np.argsort(sorted_indices)
            return all_scores_arr[reverse_indices]

    @property
    def is_loaded(self) -> bool:
        """Whether a model is currently loaded."""
        return self.model is not None


# =============================================================================
# Singleton
# =============================================================================

_cross_encoder: InProcessCrossEncoder | None = None
_cross_encoder_lock = threading.Lock()


def get_cross_encoder(
    model_name: str = "ms-marco-MiniLM-L-6-v2",
) -> InProcessCrossEncoder:
    """Get or create the singleton InProcessCrossEncoder."""
    global _cross_encoder

    if _cross_encoder is not None:
        return _cross_encoder

    with _cross_encoder_lock:
        if _cross_encoder is None:
            _cross_encoder = InProcessCrossEncoder(model_name=model_name)
        return _cross_encoder


def reset_cross_encoder() -> None:
    """Reset the singleton. Unloads model and clears instance."""
    global _cross_encoder

    with _cross_encoder_lock:
        if _cross_encoder is not None:
            _cross_encoder.unload()
        _cross_encoder = None
