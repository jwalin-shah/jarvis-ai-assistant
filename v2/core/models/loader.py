"""MLX model loader for JARVIS v2.

Handles lazy loading and generation with MLX models.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass

from .registry import DEFAULT_MODEL, MODELS, ModelSpec, get_model_spec

logger = logging.getLogger(__name__)

# Singleton instance
_model_loader: ModelLoader | None = None
_loader_lock = threading.Lock()


@dataclass
class GenerationResult:
    """Result from text generation."""

    text: str
    tokens_generated: int
    generation_time_ms: float
    model_id: str
    formatted_prompt: str = ""  # The actual prompt sent to the model (with chat template)


class ModelLoader:
    """Lazy-loading MLX model wrapper."""

    def __init__(self, model_id: str = DEFAULT_MODEL):
        """Initialize loader.

        Args:
            model_id: Model to load (from registry)
        """
        self.model_id = model_id
        self.spec = get_model_spec(model_id)
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._model is not None

    @property
    def current_model(self) -> str:
        """Get current model ID."""
        return self.model_id

    def _ensure_loaded(self) -> None:
        """Load model if not already loaded (thread-safe)."""
        if self._model is not None:
            return

        with self._load_lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return

            logger.info(f"Loading model: {self.spec.display_name}")
            start = time.time()

            try:
                from mlx_lm import load

                self._model, self._tokenizer = load(self.spec.path)
                elapsed = time.time() - start
                logger.info(f"Model loaded in {elapsed:.1f}s")

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise RuntimeError(f"Failed to load model {self.model_id}: {e}") from e

    def preload(self) -> None:
        """Explicitly load the model (for eager initialization).

        Call this at application startup to avoid cold-start latency
        on the first generation request.
        """
        self._ensure_loaded()

    def unload(self) -> None:
        """Unload model to free memory."""
        with self._load_lock:
            if self._model is not None:
                logger.info(f"Unloading model: {self.spec.display_name}")
                self._model = None
                self._tokenizer = None
                gc.collect()

                # Clear MLX cache if available
                try:
                    import mlx.core as mx
                    mx.metal.clear_cache()
                except Exception:
                    pass

    def switch_model(self, model_id: str) -> None:
        """Switch to a different model.

        Args:
            model_id: New model to load
        """
        if model_id == self.model_id and self._model is not None:
            return

        self.unload()
        self.model_id = model_id
        self.spec = get_model_spec(model_id)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.1,
        stop: list[str] | None = None,
        use_chat_template: bool = True,
    ) -> GenerationResult:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            stop: Stop sequences
            use_chat_template: If True, wrap prompt in chat template (for Qwen3)

        Returns:
            GenerationResult with generated text
        """
        load_start = time.time()
        was_loaded = self._model is not None
        self._ensure_loaded()
        load_time = (time.time() - load_start) * 1000
        if not was_loaded:
            logger.info(f"Model loaded on-demand in {load_time:.0f}ms")

        start = time.time()

        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            # Create sampler
            # LFM2.5 recommends: temp=0.1, top_p=0.1, top_k=50, min_p=0.15
            # For text replies, use slightly higher temp for variety
            sampler = make_sampler(
                temp=temperature,
                top_p=0.1,
                top_k=50,
                min_p=0.15,
            )

            # Apply chat template for Qwen3 (non-thinking mode)
            final_prompt = prompt
            if use_chat_template and self._tokenizer.chat_template:
                try:
                    messages = [{"role": "user", "content": prompt}]
                    final_prompt = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,  # Non-thinking mode for fast replies
                    )
                except Exception as e:
                    logger.debug(f"Chat template failed, using raw prompt: {e}")
                    final_prompt = prompt

            # Generate
            logger.debug(f"Starting generation with {max_tokens} max tokens, temp={temperature}")
            output = generate(
                model=self._model,
                tokenizer=self._tokenizer,
                prompt=final_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            )

            # Handle stop sequences manually if needed
            if stop:
                for seq in stop:
                    if seq in output:
                        output = output.split(seq)[0]

            elapsed = (time.time() - start) * 1000

            # Estimate tokens (rough)
            tokens = len(output.split())
            logger.info(f"LLM generated ~{tokens} tokens in {elapsed:.0f}ms ({tokens/(elapsed/1000):.1f} tok/s)")

            return GenerationResult(
                text=output.strip(),
                tokens_generated=tokens,
                generation_time_ms=elapsed,
                model_id=self.model_id,
                formatted_prompt=final_prompt,
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}") from e


def get_model_loader(model_id: str | None = None) -> ModelLoader:
    """Get singleton model loader.

    Args:
        model_id: Model to use (uses default if not specified)

    Returns:
        ModelLoader instance
    """
    global _model_loader

    if _model_loader is None:
        with _loader_lock:
            if _model_loader is None:
                _model_loader = ModelLoader(model_id or DEFAULT_MODEL)
    elif model_id and model_id != _model_loader.model_id:
        _model_loader.switch_model(model_id)

    return _model_loader


def reset_model_loader() -> None:
    """Reset the model loader singleton."""
    global _model_loader
    with _loader_lock:
        if _model_loader is not None:
            _model_loader.unload()
            _model_loader = None
