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
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tokenizers import Tokenizer

from models.bert_embedder import BertModel
from models.memory_config import gpu_context
from models.utils import HF_CACHE, find_model_snapshot, map_hf_bert_key

logger = logging.getLogger(__name__)

# Cross-encoder registry: name -> (hf_repo, num_labels, activation)
CROSS_ENCODER_REGISTRY: dict[str, tuple[str, int, str]] = {
    "ms-marco-MiniLM-L-6-v2": (
        "cross-encoder--ms-marco-MiniLM-L-6-v2",
        1,
        "sigmoid",
    ),
}


class BertForSequenceClassification(nn.Module):  # type: ignore[name-defined,misc]
    """BERT with a linear classification head on [CLS] token."""

    def __init__(self, config: dict[str, Any], num_labels: int = 1) -> None:
        super().__init__()
        self.bert = BertModel(config, add_pooler=True)
        self.classifier = nn.Linear(config["hidden_size"], num_labels)  # type: ignore[attr-defined]

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
        hidden_states = self.bert(
            input_ids,
            attention_mask if attention_mask is not None else mx.ones_like(input_ids),
            token_type_ids if token_type_ids is not None else mx.zeros_like(input_ids),
        )
        # [CLS] token is at position 0
        cls_output = hidden_states[:, 0, :]
        # Apply pooler (tanh activation) if available
        if hasattr(self.bert, "pooler_dense"):
            cls_output = mx.tanh(self.bert.pooler_dense(cls_output))
        logits: mx.array = self.classifier(cls_output)
        return logits


def load_cross_encoder_weights(model: BertForSequenceClassification, weights_path: Path) -> None:
    """Load HuggingFace cross-encoder weights into our model.

    Reuses load_bert_weights for the BERT backbone, then loads the classifier head.
    """
    # Load all weights
    hf_weights = mx.load(str(weights_path))

    # Separate classifier weights from BERT weights
    classifier_weights: dict[str, mx.array] = {}
    bert_weights: dict[str, mx.array] = {}

    # mx.load returns dict[str, array] for safetensors files
    hf_weights_dict: dict[str, mx.array]
    if isinstance(hf_weights, dict):
        hf_weights_dict = hf_weights
    else:
        hf_weights_dict = {}
    for name, weight in hf_weights_dict.items():
        if name.startswith("classifier."):
            # Map classifier.weight -> classifier.weight (already matches)
            classifier_weights[name] = weight
        else:
            bert_weights[name] = weight

    new_weights: dict[str, mx.array] = {}
    for hf_name, weight in bert_weights.items():
        stripped = hf_name.replace("bert.", "")
        mapped = map_hf_bert_key(stripped)
        if mapped is None:
            continue
        new_weights[f"bert.{mapped}"] = weight

    # Add classifier weights
    for name, weight in classifier_weights.items():
        new_weights[name] = weight

    model.load_weights(list(new_weights.items()))


class InProcessCrossEncoder:
    """In-process MLX cross-encoder for scoring (query, document) pairs.

    Thread-safe:
    - MLXModelLoader._mlx_load_lock serializes both tokenization and GPU operations,
      preventing race conditions where another thread changes tokenizer padding state
      between tokenization and the forward pass.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2") -> None:
        self._default_model = model_name
        self.model: BertForSequenceClassification | None = None
        self.tokenizer: Tokenizer | None = None
        self.model_name: str | None = None
        self.config: dict[str, Any] | None = None
        self._num_labels: int = 1
        self._activation: str = "sigmoid"

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

        with gpu_context():
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

            snapshot = find_model_snapshot(model_dir)

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
                model_name,
                hidden,
                layers,
                num_labels,
            )
            start = time.time()

            self.model = BertForSequenceClassification(self.config, num_labels)
            weights_path = snapshot / "model.safetensors"
            load_cross_encoder_weights(self.model, weights_path)
            mx.eval(self.model.parameters())

            self.model_name = model_name
            logger.info("Cross-encoder loaded in %.2fs", time.time() - start)

    def _download_model(self, repo_id: str) -> None:
        """Download model from HuggingFace Hub.

        Raises:
            OSError: If download fails due to network or disk issues.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            logger.warning("huggingface_hub not installed, cannot auto-download")
            return

        logger.info("Downloading cross-encoder model: %s", repo_id)
        try:
            snapshot_download(
                repo_id,
                allow_patterns=["config.json", "tokenizer.json", "model.safetensors"],
            )
        except OSError as e:
            logger.error("Failed to download model %s: %s", repo_id, e)
            raise

    def _unload_unlocked(self) -> None:
        """Unload model without acquiring GPU lock (caller holds it)."""
        prev = self.model_name
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.config = None
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        if prev:
            logger.info("Unloaded cross-encoder %s", prev)

    def unload(self) -> None:
        """Unload model to free memory. Thread-safe."""
        with gpu_context():
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

        # Single critical section: tokenization + GPU forward pass.
        # The GPU lock protects both the tokenizer state (padding config)
        # and the Metal GPU operations, preventing race conditions where
        # another thread changes padding between tokenization and forward pass.
        with gpu_context():
            # Tokenize WITHOUT padding so each batch pads only to its own
            # max length (not global max), saving memory.
            assert self.tokenizer is not None
            self.tokenizer.no_padding()
            encodings = self.tokenizer.encode_batch([(query, doc) for query, doc in pairs])

            # Sort by descending length for efficient batching
            sorted_indices = np.argsort([-len(e.ids) for e in encodings])
            all_scores: list[np.ndarray] = []

            for batch_start in range(0, len(pairs), batch_size):
                batch_end = min(batch_start + batch_size, len(pairs))
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
                token_type_ids = np.array(
                    [e.type_ids + [0] * (max_len - len(e.type_ids)) for e in batch_encodings],
                    dtype=np.int32,
                )

                input_ids_mx = mx.array(input_ids)
                attention_mask_mx = mx.array(attention_mask)
                token_type_ids_mx = mx.array(token_type_ids)

                assert self.model is not None
                logits = self.model(input_ids_mx, attention_mask_mx, token_type_ids_mx)

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
            all_scores_arr: np.ndarray = np.concatenate(all_scores)
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


# =============================================================================
# Reranker - thin layer over cross-encoder for retrieval reranking
# =============================================================================


class CrossEncoderReranker:
    """Reranks retrieval candidates using a cross-encoder.

    Lazy-loads the cross-encoder on first call to avoid startup cost
    when reranking is disabled.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._cross_encoder: InProcessCrossEncoder | None = None

    def _get_cross_encoder(self) -> InProcessCrossEncoder:
        """Lazy-load the cross-encoder singleton."""
        if self._cross_encoder is None:
            self._cross_encoder = get_cross_encoder(self._model_name)
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        text_key: str = "trigger_text",
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Rerank candidates by cross-encoder relevance to query.

        Args:
            query: The query text to score against.
            candidates: List of candidate dicts from vec_search.
            text_key: Key in candidate dicts containing the text to score.
            top_k: Number of top candidates to return.

        Returns:
            Top-k candidates sorted by rerank_score (descending),
            each augmented with a 'rerank_score' field.
        """
        if not candidates:
            return []

        if len(candidates) <= 1:
            for c in candidates:
                c["rerank_score"] = 1.0
            return candidates

        # Build (query, doc) pairs
        pairs = []
        valid_indices: set[int] = set()
        for i, cand in enumerate(candidates):
            text = cand.get(text_key, "")
            if text:
                pairs.append((query, text))
                valid_indices.add(i)

        if not pairs:
            return candidates[:top_k]

        try:
            ce = self._get_cross_encoder()
            scores = ce.predict(pairs)
        except (FileNotFoundError, OSError) as e:
            logger.warning("Cross-encoder unavailable, skipping reranking: %s", e)
            return candidates[:top_k]

        # Assign scores back to candidates
        scored = []
        score_idx = 0
        for i, cand in enumerate(candidates):
            if i in valid_indices:
                cand["rerank_score"] = float(scores[score_idx])
                score_idx += 1
            else:
                cand["rerank_score"] = 0.0
            scored.append(cand)

        # Sort by rerank_score descending
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored[:top_k]


_reranker: CrossEncoderReranker | None = None
_reranker_lock = threading.Lock()


def get_reranker(model_name: str = "ms-marco-MiniLM-L-6-v2") -> CrossEncoderReranker:
    """Get or create the singleton CrossEncoderReranker."""
    global _reranker

    if _reranker is not None:
        return _reranker

    with _reranker_lock:
        if _reranker is None:
            _reranker = CrossEncoderReranker(model_name=model_name)
        return _reranker


def reset_reranker() -> None:
    """Reset the singleton for testing."""
    global _reranker

    with _reranker_lock:
        _reranker = None
