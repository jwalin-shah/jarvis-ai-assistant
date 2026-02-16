"""High-level model lifecycle and resource manager.

Orchestrates the loading and unloading of different model types (LLM, Embedder, NLI)
to ensure they fit within the system's RAM constraints (optimized for 8GB).

Automatically handles serialization of GPU access via shared locks.
"""

from __future__ import annotations

import logging
import threading
from typing import Literal

logger = logging.getLogger(__name__)

ModelType = Literal["llm", "embedder", "nli"]


class ModelManager:
    """Manages system-wide model resources and memory limits."""

    _instance: ModelManager | None = None
    _lock = threading.Lock()
    _initialized: bool

    def __new__(cls) -> ModelManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._active_type: ModelType | None = None
        self._initialized = True

    def prepare_for(self, model_type: ModelType) -> None:
        """Prepare system resources for a specific model type.

        Unloads conflicting models to ensure enough RAM is available.
        """
        if self._active_type == model_type:
            return

        logger.info(f"Preparing system for model type: {model_type}")

        # Unload based on requested type
        if model_type == "llm":
            # LLM needs most memory, unload everything else
            self._unload_embedder()
            self._unload_nli()
        elif model_type == "embedder":
            # Embedder can coexist with NLI but not LLM
            self._unload_llm()
        elif model_type == "nli":
            # NLI can coexist with Embedder but not LLM
            self._unload_llm()

        self._active_type = model_type

    def _unload_llm(self) -> None:
        try:
            from models.loader import reset_model

            reset_model()
        except ImportError:
            pass
        try:
            from jarvis.contacts.instruction_extractor import reset_instruction_extractor

            reset_instruction_extractor()
        except ImportError:
            pass

    def _unload_embedder(self) -> None:
        try:
            from jarvis.embedding_adapter import reset_embedder

            reset_embedder()
        except ImportError:
            pass

    def _unload_nli(self) -> None:
        try:
            from models.nli_cross_encoder import reset_nli_cross_encoder

            reset_nli_cross_encoder()
        except ImportError:
            pass

    def unload_all(self) -> None:
        """Unload all models from memory."""
        logger.info("Unloading all models")
        self._unload_llm()
        self._unload_embedder()
        self._unload_nli()
        self._active_type = None


def get_model_manager() -> ModelManager:
    """Get the singleton ModelManager instance."""
    return ModelManager()
