"""Pure MLX NLI cross-encoder for entailment verification.

Uses cross-encoder/nli-deberta-v3-xsmall (22M params, 87.77% MNLI accuracy)
implemented in pure MLX with disentangled attention.

Thread Safety:
    Uses MLXModelLoader._mlx_load_lock to serialize GPU access.
    Same pattern as InProcessCrossEncoder in cross_encoder.py.

Usage:
    from models.nli_cross_encoder import get_nli_cross_encoder

    nli = get_nli_cross_encoder()
    scores = nli.predict_entailment("I moved to Austin", "The person lives in Austin")
    print(scores)  # {"contradiction": 0.05, "entailment": 0.90, "neutral": 0.05}
"""

from __future__ import annotations

import gc
import json
import logging
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from models.deberta import DebertaForSequenceClassification, convert_hf_weights

logger = logging.getLogger(__name__)

HF_CACHE = Path.home() / ".cache/huggingface/hub"

# Label order from DeBERTa NLI config.json
NLI_LABELS = ["contradiction", "entailment", "neutral"]

# Default model
DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


class NLICrossEncoder:
    """In-process MLX NLI cross-encoder for entailment scoring.

    Thread-safe:
    - _encode_lock serializes tokenizer calls (Rust tokenizer is not thread-safe)
    - MLXModelLoader._mlx_load_lock serializes GPU operations
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._encode_lock = threading.Lock()
        self.model: DebertaForSequenceClassification | None = None
        self.tokenizer: Tokenizer | None = None
        self.config: dict | None = None
        self._loaded = False

    def _get_gpu_lock(self) -> threading.Lock:
        """Get the shared MLX GPU lock."""
        from models.loader import MLXModelLoader

        return MLXModelLoader._mlx_load_lock

    def load_model(self) -> None:
        """Load the NLI model. Thread-safe via GPU lock."""
        if self._loaded:
            return

        with self._get_gpu_lock():
            # Set MLX memory limits to prevent memory spikes on 8GB systems
            from models.memory_config import apply_embedder_limits

            apply_embedder_limits()

            if self._loaded:
                return

            hf_repo = self._model_name.replace("/", "--")
            model_dir = HF_CACHE / f"models--{hf_repo}"

            if not model_dir.exists():
                self._download_model()
                if not model_dir.exists():
                    raise FileNotFoundError(
                        f"Model not found: {model_dir}. "
                        f"Download with: huggingface-cli download {self._model_name}"
                    )

            snapshot = self._find_snapshot(model_dir)

            # Load config
            with open(snapshot / "config.json") as f:
                self.config = json.load(f)

            # Load tokenizer
            tokenizer_path = snapshot / "tokenizer.json"
            if not tokenizer_path.exists():
                raise FileNotFoundError(
                    f"tokenizer.json not found in {snapshot}. "
                    "DeBERTa-v3 requires the fast tokenizer file."
                )
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self.tokenizer.enable_truncation(max_length=512)

            # Build model
            num_labels = self.config.get("num_labels", 3)
            logger.info(
                "Loading NLI model %s (hidden=%d, layers=%d, labels=%d)",
                self._model_name,
                self.config["hidden_size"],
                self.config["num_hidden_layers"],
                num_labels,
            )
            start = time.time()

            self.model = DebertaForSequenceClassification(self.config, num_labels)

            # Load weights
            weights_path = snapshot / "model.safetensors"
            if not weights_path.exists():
                raise FileNotFoundError(f"model.safetensors not found in {snapshot}")

            hf_weights = mx.load(str(weights_path))
            converted = convert_hf_weights(hf_weights)
            self.model.load_weights(list(converted.items()))
            mx.eval(self.model.parameters())

            self._loaded = True
            logger.info("NLI model loaded in %.2fs", time.time() - start)

    def _find_snapshot(self, model_dir: Path) -> Path:
        """Find the snapshot directory in HF cache."""
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            raise FileNotFoundError(f"No snapshots in {model_dir}")
        snapshots = list(snapshots_dir.iterdir())
        if not snapshots:
            raise FileNotFoundError(f"Empty snapshots dir: {snapshots_dir}")
        return snapshots[0]

    def _download_model(self) -> None:
        """Download model from HuggingFace Hub."""
        try:
            from huggingface_hub import snapshot_download

            logger.info("Downloading NLI model: %s", self._model_name)
            snapshot_download(
                self._model_name,
                allow_patterns=[
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "model.safetensors",
                    "spm.model",
                ],
            )
        except ImportError:
            logger.warning("huggingface_hub not installed, cannot auto-download")
        except Exception as e:
            logger.warning("Failed to download %s: %s", self._model_name, e)

    def unload(self) -> None:
        """Unload model to free memory."""
        with self._get_gpu_lock():
            self.model = None
            self.tokenizer = None
            self.config = None
            self._loaded = False
            gc.collect()
            if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
            logger.info("Unloaded NLI model")

    def predict_entailment(
        self,
        premise: str,
        hypothesis: str,
    ) -> dict[str, float]:
        """Score a single (premise, hypothesis) pair.

        Returns:
            Dict with "contradiction", "entailment", "neutral" probabilities.
        """
        results = self.predict_batch([(premise, hypothesis)])
        return results[0]

    def predict_batch(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
    ) -> list[dict[str, float]]:
        """Score multiple (premise, hypothesis) pairs.

        Args:
            pairs: List of (premise, hypothesis) text pairs.
            batch_size: Processing batch size.

        Returns:
            List of dicts with NLI label probabilities.
        """
        if not self._loaded:
            self.load_model()

        if not pairs:
            return []

        all_results: list[dict[str, float]] = []

        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start : batch_start + batch_size]

            # Tokenize
            with self._encode_lock:
                self.tokenizer.no_padding()
                encodings = self.tokenizer.encode_batch(
                    [(premise, hypothesis) for premise, hypothesis in batch]
                )

            # Pad to uniform length
            max_len = max(len(e.ids) for e in encodings)
            input_ids = np.array(
                [e.ids + [0] * (max_len - len(e.ids)) for e in encodings],
                dtype=np.int32,
            )
            attention_mask = np.array(
                [e.attention_mask + [0] * (max_len - len(e.attention_mask)) for e in encodings],
                dtype=np.int32,
            )

            input_ids_mx = mx.array(input_ids)
            attention_mask_mx = mx.array(attention_mask)

            # Forward pass under GPU lock
            with self._get_gpu_lock():
                logits = self.model(input_ids_mx, attention_mask_mx)
                probs = mx.softmax(logits, axis=-1)
                mx.eval(probs)

            probs_np = np.array(probs)
            for row in probs_np:
                all_results.append({label: float(row[i]) for i, label in enumerate(NLI_LABELS)})

        return all_results

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        return self._loaded


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_nli_encoder: NLICrossEncoder | None = None
_nli_lock = threading.Lock()


def get_nli_cross_encoder(model_name: str = DEFAULT_MODEL) -> NLICrossEncoder:
    """Get or create the singleton NLI cross-encoder."""
    global _nli_encoder

    if _nli_encoder is not None:
        return _nli_encoder

    with _nli_lock:
        if _nli_encoder is None:
            _nli_encoder = NLICrossEncoder(model_name=model_name)
        return _nli_encoder


def reset_nli_cross_encoder() -> None:
    """Reset the singleton. Unloads model and clears instance."""
    global _nli_encoder

    with _nli_lock:
        if _nli_encoder is not None:
            _nli_encoder.unload()
        _nli_encoder = None
