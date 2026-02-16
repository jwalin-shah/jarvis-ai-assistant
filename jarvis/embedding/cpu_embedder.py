"""Lightweight CPU embedder using ONNX Runtime.

Designed to complement MLX embedder for parallel processing:
- MLX (GPU): Fast, but shares lock with LLM
- ONNX (CPU): Slower, but runs parallel to GPU LLM

Memory-efficient design:
- Lazy loading (only loads when first used)
- Singleton pattern (one instance system-wide)
- Explicit unload() to free RAM
- ~150MB RAM when loaded (vs 250MB for sentence-transformers)
"""

from __future__ import annotations

import gc
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ONNX Runtime is optional - graceful fallback
try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime not installed. CPU embedder unavailable.")


# Default model - same as MLX embedder for consistency
DEFAULT_CPU_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


class CPUEmbedder:
    """Singleton CPU embedder with lazy loading.

    Use this when GPU is busy with LLM generation to achieve parallelism.

    Memory profile:
    - Unloaded: ~0MB
    - Loaded: ~150MB (model + session overhead)

    Example:
        embedder = CPUEmbedder.get_instance()
        embeddings = embedder.encode(["Hello world"])
        embedder.unload()  # Free RAM when done
    """

    _instance: CPUEmbedder | None = None
    _lock = threading.Lock()
    _init_lock = threading.Lock()

    def __new__(cls) -> CPUEmbedder:
        """Singleton pattern - one embedder system-wide."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._session: ort.InferenceSession | None = None
                    cls._instance._tokenizer = None
                    cls._instance._model_name: str | None = None
                    cls._instance._loaded = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> CPUEmbedder:
        """Get singleton instance."""
        return cls()

    @classmethod
    def is_available(cls) -> bool:
        """Check if ONNX runtime is available."""
        return ONNX_AVAILABLE

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def load(self, model_name: str = DEFAULT_CPU_MODEL) -> bool:
        """Load ONNX model. Thread-safe.

        Args:
            model_name: HuggingFace model name (must have ONNX export)

        Returns:
            True if loaded successfully
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime not installed")

        if self._loaded:
            return True

        with self._init_lock:
            if self._loaded:
                return True

            try:
                # Try to find or download ONNX model
                model_path = self._get_onnx_model_path(model_name)

                if not model_path.exists():
                    logger.error(f"ONNX model not found: {model_path}")
                    logger.info("Download with: huggingface-cli download {model_name}")
                    return False

                # CPU-only session with optimizations
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                sess_options.intra_op_num_threads = 4  # Use 4 CPU cores
                sess_options.inter_op_num_threads = 2

                self._session = ort.InferenceSession(
                    str(model_path),
                    sess_options,
                    providers=["CPUExecutionProvider"],
                )

                # Load tokenizer (using HF tokenizers for consistency)
                from tokenizers import Tokenizer

                tokenizer_path = model_path.parent / "tokenizer.json"
                if tokenizer_path.exists():
                    self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
                    self._tokenizer.enable_truncation(max_length=512)
                else:
                    # Fallback: use MLX embedder's tokenizer pattern
                    logger.warning("tokenizer.json not found, using default")
                    self._tokenizer = None

                self._model_name = model_name
                self._loaded = True
                logger.info(f"CPU embedder loaded: {model_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to load CPU embedder: {e}")
                return False

    def _get_onnx_model_path(self, model_name: str) -> Path:
        """Get path to ONNX model file.

        Tries several locations:
        1. models/bge-small-onnx/model.onnx (pre-downloaded)
        2. models/{model_name}/model.onnx
        3. HF cache with ONNX export
        """
        # 1. Pre-downloaded model location
        predownloaded = Path("models") / "bge-small-onnx" / "model.onnx"
        if predownloaded.exists():
            return predownloaded

        # 2. Local models directory with model name
        local_path = Path("models") / model_name.replace("/", "--") / "model.onnx"
        if local_path.exists():
            return local_path

        # 3. Try HF cache
        try:
            from huggingface_hub import snapshot_download

            cache_dir = snapshot_download(
                repo_id=model_name,
                local_files_only=True,
            )
            cache_path = Path(cache_dir) / "model.onnx"
            if cache_path.exists():
                return cache_path
        except Exception:
            pass

        return local_path

    def unload(self) -> None:
        """Unload model to free memory. Thread-safe."""
        with self._init_lock:
            if not self._loaded:
                return

            prev_model = self._model_name
            self._session = None
            self._tokenizer = None
            self._model_name = None
            self._loaded = False
            gc.collect()
            logger.info(f"Unloaded CPU embedder: {prev_model}")

    def encode(
        self,
        texts: list[str] | str,
        normalize: bool = True,
        batch_size: int = 32,
        dtype: np.dtype = np.float32,
    ) -> NDArray[np.float32]:
        """Encode texts to embeddings.

        Args:
            texts: Single text or list of texts
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for processing
            dtype: Output dtype

        Returns:
            Embeddings array of shape (n_texts, 384)
        """
        if not self._loaded:
            if not self.load():
                raise RuntimeError("Failed to load CPU embedder")

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([], dtype=dtype).reshape(0, EMBEDDING_DIM)

        # Tokenize all texts
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            encodings = self._tokenizer.encode_batch(batch_texts)

            max_len = max(len(e.ids) for e in encodings)
            input_ids = np.array(
                [e.ids + [0] * (max_len - len(e.ids)) for e in encodings],
                dtype=np.int64,
            )
            attention_mask = np.array(
                [e.attention_mask + [0] * (max_len - len(e.attention_mask))
                 for e in encodings],
                dtype=np.int64,
            )
            # token_type_ids is required by some ONNX models
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

            # ONNX inference
            outputs = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )

            # CLS pooling (first output, first token)
            # Output shape: (batch, seq_len, hidden_dim)
            last_hidden = outputs[0]
            cls_embeddings = last_hidden[:, 0, :]  # Take [CLS] token

            if normalize:
                norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                cls_embeddings = cls_embeddings / norms

            all_embeddings.append(cls_embeddings.astype(dtype))

        return np.vstack(all_embeddings)


def get_cpu_embedder() -> CPUEmbedder | None:
    """Get CPU embedder instance if available.

    Returns:
        CPUEmbedder instance, or None if ONNX not available
    """
    if not CPUEmbedder.is_available():
        return None
    return CPUEmbedder.get_instance()


def is_cpu_embedder_available() -> bool:
    """Check if CPU embedder is available."""
    return CPUEmbedder.is_available()
