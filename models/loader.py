"""MLX Model Loader with memory-safe lifecycle management.

Handles model loading/unloading with thread-safety, memory tracking,
and double-check locking patterns for lazy initialization.

Supports model selection via the registry system:
    from models.loader import MLXModelLoader, ModelConfig

    # Use default model from registry
    loader = MLXModelLoader()

    # Use specific model from registry
    config = ModelConfig(model_id="qwen-3b")
    loader = MLXModelLoader(config)

    # Use custom model path (not in registry)
    config = ModelConfig(model_path="custom/model-path")
    loader = MLXModelLoader(config)
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import psutil

from jarvis.core.exceptions import (
    ErrorCode,
    ModelGenerationError,
    ModelLoadError,
    model_not_found,
    model_out_of_memory,
)
from models.memory_config import apply_llm_limits
from models.registry import DEFAULT_MODEL_ID, MODEL_REGISTRY, ModelSpec, get_model_spec

logger = logging.getLogger(__name__)

# Constants
BYTES_PER_MB = 1024 * 1024


def _get_default_generation_timeout() -> float:
    """Get default generation timeout from config.

    Returns:
        Timeout in seconds from config, or 60.0 as fallback.
    """
    try:
        from jarvis.config import get_config

        return get_config().model.generation_timeout_seconds
    except Exception:
        return 60.0  # Fallback default


@dataclass
class ModelConfig:
    """Configuration for MLX model loading.

    Can be initialized with either model_id (from registry) or model_path.
    If both are provided, model_id takes precedence. If neither is provided,
    uses the default model from the registry.

    Attributes:
        model_id: Model identifier from registry (e.g., "qwen-1.5b").
        model_path: Direct HuggingFace path (overridden by model_id if set).
        estimated_memory_mb: Estimated memory usage (auto-set from registry if model_id used).
        memory_buffer_multiplier: Safety buffer for memory checks.
        default_max_tokens: Default max tokens for generation.
        default_temperature: Default sampling temperature.
        generation_timeout_seconds: Timeout for generation in seconds (None = no timeout).
            If not explicitly set, uses the value from config.model.generation_timeout_seconds.
    """

    model_id: str | None = None
    model_path: str = ""
    estimated_memory_mb: float = 800
    memory_buffer_multiplier: float = 1.3  # 30% buffer: MLX GPU spikes can be 20-30%
    default_max_tokens: int = 100
    default_temperature: float = 0.7
    generation_timeout_seconds: float | None = field(
        default_factory=_get_default_generation_timeout
    )
    kv_cache_bits: int | None = None
    _resolved_spec: ModelSpec | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Resolve model_id to model_path and estimated_memory_mb."""
        # ABSOLUTE GATE: Directly read the file to bypass any potential caching
        if not self.model_id and not self.model_path:
            try:
                import json
                from pathlib import Path

                raw_cfg_path = Path.home() / ".jarvis" / "config.json"
                if raw_cfg_path.exists():
                    with open(raw_cfg_path) as f:
                        raw_data = json.load(f)
                        # The config file has a "model" nested object
                        model_data = raw_data.get("model", {})
                        direct_model_id = model_data.get("model_id")
                        if direct_model_id:
                            print(f"DEBUG: DIRECT DISK LOAD OF MODEL_ID: {direct_model_id}")
                            self.model_id = direct_model_id
            except Exception as e:
                print(f"DEBUG: CONFIG DISK LOAD FAILED: {e}")

        if self.model_id:
            spec = get_model_spec(self.model_id)
            if spec:
                self._resolved_spec = spec
                self.model_path = spec.path
                self.estimated_memory_mb = spec.estimated_memory_mb
            else:
                logger.warning("Unknown model_id '%s', using as-is", self.model_id)
                self.model_path = self.model_id
        elif not self.model_path:
            # Registry should always have DEFAULT_MODEL_ID, but handle gracefully
            spec = get_model_spec(DEFAULT_MODEL_ID)
            if spec:
                self._resolved_spec = spec
                self.model_id = spec.id
                self.model_path = spec.path
                self.estimated_memory_mb = spec.estimated_memory_mb
            else:
                logger.warning("DEFAULT_MODEL_ID '%s' not in registry", DEFAULT_MODEL_ID)
                default_spec = MODEL_REGISTRY[DEFAULT_MODEL_ID]
                self.model_path = default_spec.path

        if self.kv_cache_bits is None:
            try:
                from jarvis.config import get_config

                self.kv_cache_bits = get_config().model.kv_cache_bits
            except Exception:
                self.kv_cache_bits = 16  # Default to full precision if config fails

    @property
    def is_local(self) -> bool:
        """Check if model_path is a local directory."""
        return os.path.isdir(self.model_path)

    @property
    def exists(self) -> bool:
        """Check if model exists locally (either as a directory or in HF cache)."""
        if self.is_local:
            return True
        if self.model_id:
            from models.registry import is_model_available

            return is_model_available(self.model_id)
        return False

    @property
    def display_name(self) -> str:
        """Return a human-readable name for the model."""
        if self._resolved_spec:
            return self._resolved_spec.display_name
        return self.model_path.split("/")[-1]

    @property
    def spec(self) -> ModelSpec | None:
        """Return the resolved ModelSpec if available."""
        return self._resolved_spec


@dataclass
class GenerationResult:
    """Result from synchronous generation."""

    text: str
    tokens_generated: int
    generation_time_ms: float
    tokens_per_second: float = 0.0
    draft_accepted_tokens: int = 0
    speculative_enabled: bool = False

    @property
    def acceptance_rate(self) -> float:
        """Return draft token acceptance rate (0.0 if speculative not used)."""
        if not self.speculative_enabled or self.tokens_generated == 0:
            return 0.0
        return self.draft_accepted_tokens / self.tokens_generated


@dataclass
class StreamToken:
    """A single token from streaming generation."""

    token: str
    token_index: int
    is_final: bool = False
    ttft_ms: float | None = None
    from_draft: bool = False


class NegativeConstraintLogitsProcessor:
    """Logits processor that penalizes specified token sequences (phrases).

    Used to reduce 'AI-isms' like 'As an AI language model' or 'I hope this helps'.
    """

    def __init__(
        self,
        tokenizer: Any,
        phrases: list[str],
        penalty: float = 5.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.penalty = penalty
        # Pre-tokenize phrases into token IDs
        self.phrase_ids = [tokenizer.encode(p, add_special_tokens=False) for p in phrases]
        # Filter out empty or single-token phrases (handled better by logit bias)
        self.phrase_ids = [p for p in self.phrase_ids if len(p) > 1]
        # For single tokens, we can just use a set for O(1) lookup
        self.single_token_penalties: dict[int, float] = {}
        for p in phrases:
            ids = tokenizer.encode(p, add_special_tokens=False)
            if len(ids) == 1:
                self.single_token_penalties[ids[0]] = penalty

    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        """Apply penalties to logits based on input_ids history."""
        # 1. Apply single token penalties
        if self.single_token_penalties:
            for token_id, penalty in self.single_token_penalties.items():
                logits[:, token_id] -= penalty

        # 2. Apply multi-token sequence penalties
        # input_ids shape can be (L,) or (1, L) depending on the generator step
        from typing import cast

        if input_ids.ndim == 1:
            input_list = cast(list[int], input_ids.tolist())
        else:
            input_list = cast(list[int], input_ids[0].tolist())

        for phrase in self.phrase_ids:
            # Check if current input_ids end matches phrase prefix
            for i in range(1, len(phrase)):
                prefix: list[int] = phrase[:i]
                if input_list[-i:] == prefix:
                    # Penalize the NEXT token in the phrase
                    next_token = phrase[i]
                    logits[:, next_token] -= self.penalty

        return logits


class MLXModelLoader:
    """MLX model lifecycle manager with thread-safe loading and memory tracking.

    Implements lazy loading with double-check locking, memory pressure checks,
    and complete unloading including Metal GPU cache cleanup.

    Supports eager loading via preload() for faster first-request latency.

    IMPORTANT: _mlx_load_lock is a class-level lock shared by ALL instances.
    This serializes ALL Metal GPU operations (load, generate, encode) to prevent
    concurrent access which crashes with assertion failures ("A command encoder
    is already encoding to this command buffer") or malloc errors. Multiple
    instances may exist (e.g. get_model() singleton vs MLXGenerator's internal
    loader), and all must serialize GPU ops through this shared lock.
    """

    _mlx_load_lock = threading.Lock()

    @classmethod
    def gpu_lock(cls) -> threading.Lock:
        """Public accessor for the GPU serialization lock."""
        return cls._mlx_load_lock

    @staticmethod
    def _configure_environment() -> None:
        """Configure environment variables for offline model loading."""
        # Disable HuggingFace hub network checks after initial download
        # This prevents slow version checks on every model load/generate call
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize the loader.

        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        self._configure_environment()
        self.config = config or ModelConfig()
        # MLX model returned by mlx_lm.load() - typed as Any due to external library complexity
        self._model: Any = None
        # Tokenizer returned by mlx_lm.load() - typed as Any due to external library complexity
        self._tokenizer: Any = None
        self._loaded_at: float | None = None
        self._preload_thread: threading.Thread | None = None
        self._preload_error: Exception | None = None
        # KV cache for static prompt prefix reuse (MLX cache structure, opaque)
        self._prompt_cache: Any | None = None
        self._cache_prefix_len: int = 0
        # Pre-encoded prefix tokens to avoid redundant tokenization
        self._cache_prefix_tokens: mx.array | None = None
        self.last_load_time_ms: float | None = None
        # Speculative decoding draft model
        self._draft_model: Any = None  # MLX model
        self._draft_config: ModelConfig | None = None

    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self._model is not None and self._tokenizer is not None

    def is_preloading(self) -> bool:
        """Check if model is currently being preloaded in background."""
        return self._preload_thread is not None and self._preload_thread.is_alive()

    def preload(self, wait: bool = False, timeout: float | None = None) -> None:
        """Start loading the model in a background thread.

        This allows the model to be loaded during app startup, avoiding the
        2-5 second delay on the first request.

        Args:
            wait: If True, blocks until loading completes.
            timeout: Maximum seconds to wait if wait=True (None = no timeout).

        Raises:
            ModelLoadError: If wait=True and loading fails.
        """
        if self.is_loaded():
            logger.debug("Model already loaded, skipping preload")
            return

        if self.is_preloading():
            logger.debug("Model already preloading")
            if wait:
                self._wait_for_preload(timeout)
            return

        def _preload_worker() -> None:
            """Background worker to load the model."""
            try:
                self.load()
                logger.info("Background preload completed for %s", self.config.display_name)
            except Exception as e:
                self._preload_error = e
                logger.error("Background preload failed: %s", e)

        self._preload_error = None
        self._preload_thread = threading.Thread(
            target=_preload_worker,
            name="model-preload",
            daemon=True,
        )
        self._preload_thread.start()
        logger.info("Started background preload for %s", self.config.display_name)

        if wait:
            self._wait_for_preload(timeout)

    def _wait_for_preload(self, timeout: float | None = None) -> None:
        """Wait for preload to complete, raising any error that occurred.

        Args:
            timeout: Maximum seconds to wait (None = no timeout).

        Raises:
            ModelLoadError: If preload failed.
        """
        if self._preload_thread is not None:
            self._preload_thread.join(timeout=timeout)
            if self._preload_thread.is_alive():
                logger.warning("Preload still in progress after timeout")
            elif self._preload_error is not None:
                raise self._preload_error

    def wait_for_ready(self, timeout: float | None = None) -> bool:
        """Wait for model to be ready (loaded or preload complete).

        Args:
            timeout: Maximum seconds to wait (None = no timeout).

        Returns:
            True if model is loaded and ready.

        Raises:
            ModelLoadError: If preload failed.
        """
        if self.is_loaded():
            return True

        if self.is_preloading():
            self._wait_for_preload(timeout)
            return self.is_loaded()

        # Not loaded and not preloading - load synchronously
        self.load()
        return self.is_loaded()

    def _can_load_model(self) -> tuple[bool, int, int]:
        """Check if sufficient memory is available for loading.

        Requires available memory to exceed estimated model size
        multiplied by buffer factor to avoid memory pressure.

        Returns:
            Tuple of (can_load, available_mb, required_mb).
        """
        mem = psutil.virtual_memory()
        available_mb = int(mem.available / BYTES_PER_MB)
        required_mb = int(self.config.estimated_memory_mb * self.config.memory_buffer_multiplier)

        if available_mb < required_mb:
            logger.warning(
                "Insufficient memory for model load: %dMB available, %dMB required",
                available_mb,
                required_mb,
            )
            return False, available_mb, required_mb
        return True, available_mb, required_mb

    def load(self) -> bool:
        """Load model into memory with thread-safe double-check locking.

        Returns:
            True if model loaded successfully or was already loaded.

        Raises:
            ModelLoadError: If loading fails due to memory, missing files, or other issues.
        """
        # Fast path: already loaded
        if self.is_loaded():
            return True

        # Check memory before acquiring lock
        can_load, available_mb, required_mb = self._can_load_model()
        if not can_load:
            raise model_out_of_memory(
                self.config.display_name,
                available_mb=available_mb,
                required_mb=required_mb,
            )

        with MLXModelLoader._mlx_load_lock:
            # Double-check after acquiring lock
            if self.is_loaded():
                return True

            try:
                from mlx_lm import load

                logger.info(
                    "Loading model: %s (%s)",
                    self.config.display_name,
                    self.config.model_path,
                )
                start_time = time.perf_counter()

                # Set MLX memory limits before loading (critical on 8GB systems)
                apply_llm_limits()

                # Check if model exists locally before trying to load it
                # This prevents noisy HF Hub tracebacks when offline or model is missing.
                if not self.config.exists:
                    logger.error(
                        "Model '%s' not found locally. Run `jarvis setup` to download.",
                        self.config.display_name,
                    )
                    raise model_not_found(self.config.model_path)

                # mlx_lm.load returns (model, tokenizer) tuple
                load_kwargs = {}
                if self.config.kv_cache_bits:
                    # MLX used to support 'kv_bits' in load() in older versions.
                    # Recent versions (>=0.20.0) moved this to generate() or removed it.
                    # We check the signature to remain compatible with various versions.
                    import inspect

                    sig = inspect.signature(load)
                    if "kv_bits" in sig.parameters:
                        load_kwargs["kv_bits"] = self.config.kv_cache_bits
                    elif "cache_bits" in sig.parameters:
                        load_kwargs["cache_bits"] = self.config.kv_cache_bits

                if load_kwargs:
                    logger.debug("Loading model with kwargs: %s", list(load_kwargs.keys()))

                try:
                    result = load(self.config.model_path, **load_kwargs)
                except TypeError as e:
                    if "unexpected keyword argument" in str(e) and load_kwargs:
                        logger.warning("Retry loading model without kwargs due to TypeError: %s", e)
                        result = load(self.config.model_path)
                    else:
                        raise

                self._model = result[0]
                self._tokenizer = result[1]
                self._loaded_at = time.perf_counter()

                load_time = (self._loaded_at - start_time) * 1000
                self.last_load_time_ms = load_time
                logger.info("Model loaded in %.0fms", load_time)
                return True

            except FileNotFoundError as e:
                logger.error(
                    "Model not found: %s. Run `huggingface-cli download %s` first.",
                    self.config.model_path,
                    self.config.model_path,
                )
                self.unload()
                raise model_not_found(self.config.model_path) from e
            except MemoryError as e:
                logger.error("Out of memory loading model '%s'", self.config.display_name)
                self.unload()
                raise model_out_of_memory(self.config.display_name) from e
            except Exception as e:
                # Cleaner error message for various loading failures (network, permissions, etc)
                error_msg = str(e).split("\n")[0]
                logger.error("Failed to load model '%s': %s", self.config.display_name, error_msg)
                self.unload()
                if "OfflineModeIsEnabled" in str(e) or "LocalEntryNotFoundError" in str(e):
                    raise ModelLoadError(
                        f"Model not found locally and system is offline: {self.config.model_path}",
                        model_name=self.config.display_name,
                        code=ErrorCode.MDL_NOT_FOUND,
                    ) from e
                raise ModelLoadError(
                    f"Failed to load model: {error_msg}",
                    model_name=self.config.display_name,
                    model_path=self.config.model_path,
                    cause=e,
                ) from e

    def unload(self) -> None:
        """Unload model and free all memory including GPU cache.

        Call order: Set references to None -> Clear Metal cache -> Force GC
        """
        logger.info("Unloading model")

        # Clear references
        self._model = None
        self._tokenizer = None
        self._loaded_at = None
        # Explicitly delete prompt cache before setting to None
        if self._prompt_cache is not None:
            del self._prompt_cache
        self._prompt_cache = None
        self._cache_prefix_len = 0
        self._cache_prefix_tokens = None
        # Explicitly delete draft model before setting to None
        if self._draft_model is not None:
            del self._draft_model
        self._draft_model = None
        self._draft_config = None

        # Clear Metal GPU memory (using current API, not deprecated mx.metal.clear_cache)
        # This is critical for 8GB RAM systems to avoid GPU memory accumulation
        # Use non-blocking acquire to avoid deadlock when called from load() error paths
        # (which already hold _mlx_load_lock)
        try:
            acquired = MLXModelLoader._mlx_load_lock.acquire(blocking=False)
            try:
                mx.clear_cache()
            finally:
                if acquired:
                    MLXModelLoader._mlx_load_lock.release()
        except Exception:
            logger.debug("Cache clear not available")

        # Force garbage collection
        gc.collect()
        logger.info("Model unloaded")

    @property
    def has_draft_model(self) -> bool:
        """Check if a draft model is loaded for speculative decoding."""
        return self._draft_model is not None

    def load_draft_model(self, draft_model_id: str) -> bool:
        """Load a draft model for speculative decoding.

        Validates tokenizer compatibility (vocab_size match) with the target model.

        Args:
            draft_model_id: Model ID from registry (e.g., "lfm-0.3b").

        Returns:
            True if draft model loaded successfully.
        """
        if self._draft_model is not None:
            logger.debug("Draft model already loaded")
            return True

        draft_config = ModelConfig(model_id=draft_model_id)

        # Check if draft model is available locally before trying to load it
        # This prevents noisy HF Hub tracebacks when offline or model is missing.
        if not draft_config.exists:
            logger.warning(
                "Draft model '%s' not found locally. Skipping speculative decoding. "
                "Run `jarvis setup` to download recommended models.",
                draft_model_id,
            )
            return False

        with MLXModelLoader._mlx_load_lock:
            try:
                from mlx_lm import load

                logger.info(
                    "Loading draft model: %s (%s)",
                    draft_config.display_name,
                    draft_config.model_path,
                )
                start = time.perf_counter()
                draft_result = load(draft_config.model_path)
                draft_model = draft_result[0]
                draft_tokenizer = draft_result[1]

                # Validate tokenizer compatibility
                if self._tokenizer is not None:
                    target_vocab = self._tokenizer.vocab_size
                    draft_vocab = draft_tokenizer.vocab_size
                    if target_vocab != draft_vocab:
                        logger.warning(
                            "Tokenizer mismatch: target vocab=%d, draft vocab=%d. "
                            "Skipping speculative decoding.",
                            target_vocab,
                            draft_vocab,
                        )
                        del draft_model, draft_tokenizer
                        return False

                self._draft_model = draft_model
                self._draft_config = draft_config
                load_ms = (time.perf_counter() - start) * 1000
                logger.info("Draft model loaded in %.0fms", load_ms)
                return True

            except Exception as e:
                # Capture and log only the error message to avoid flooding logs with tracebacks
                # for what is an optional performance feature.
                error_msg = str(e).split("\n")[0]
                logger.warning(
                    "Failed to load draft model %s: %s. Speculative decoding disabled.",
                    draft_model_id,
                    error_msg,
                )
                return False

    def unload_draft_model(self) -> None:
        """Unload the draft model to free memory."""
        if self._draft_model is None:
            return
        logger.info("Unloading draft model")
        self._draft_model = None
        self._draft_config = None

        def prefill_prompt_cache(self, prefix_text: str) -> None:
            """Pre-compute KV cache for a static prompt prefix.



            The cached KV state can be reused across generation calls that share

            the same prefix, saving ~50-100ms of prefill per call for a ~130 token

            system prompt on a 1.2B model.



            Must be called after load() and while holding _mlx_load_lock (or from

            a context where no concurrent GPU ops are possible).



            Args:

                prefix_text: The static prompt text to cache (e.g. system prompt).

            """

            if not self.is_loaded():
                logger.warning("Cannot prefill cache: model not loaded")

                return


        try:
            # Tokenize the prefix once and store for reuse (avoids redundant
            # re-encoding if callers need the token IDs later)
            tokens = mx.array(self._tokenizer.encode(prefix_text))

            cache_kwargs = {}
            if self.config.kv_cache_bits:
                # Check if make_prompt_cache supports kv_bits (some MLX versions don't)
                import inspect

                sig = inspect.signature(make_prompt_cache)
                if "kv_bits" in sig.parameters:
                    cache_kwargs["kv_bits"] = self.config.kv_cache_bits

            try:
                self._prompt_cache = make_prompt_cache(self._model, **cache_kwargs)
            except TypeError as e:
                if "unexpected keyword argument" in str(e) and cache_kwargs:
                    logger.warning("Retry make_prompt_cache without kwargs due to TypeError: %s", e)
                    self._prompt_cache = make_prompt_cache(self._model)
                else:
                    raise

            self._cache_prefix_len = tokens.shape[0]
            self._cache_prefix_tokens = tokens

            # Prefill: run forward pass with max_tokens=0 to populate cache.
            # generate_step receives pre-tokenized mx.array directly, no
            # re-encoding occurs here.
            from mlx_lm.generate import generate_step

            # Use the same kv_bits for prefilling to match generation
            gen_step_kwargs = {}
            if self.config.kv_cache_bits and self.config.kv_cache_bits != 16:
                gen_step_kwargs["kv_bits"] = self.config.kv_cache_bits

            for _ in generate_step(
                self._cache_prefix_tokens,
                self._model,
                max_tokens=0,
                prompt_cache=self._prompt_cache,
                **gen_step_kwargs,
            ):
                pass
            mx.eval([c.state for c in self._prompt_cache if hasattr(c, "state")])

            logger.info("Prefilled prompt cache with %d prefix tokens", self._cache_prefix_len)
        except Exception:
            logger.exception("Failed to prefill prompt cache")
            self._prompt_cache = None
            self._cache_prefix_len = 0
            self._cache_prefix_tokens = None

    def trim_prompt_cache(self) -> None:
        """Reset the prompt cache back to the prefilled prefix state.

        After each generation, the cache contains KV pairs for prefix + suffix.
        This trims it back to just the prefix so it can be reused for the next
        generation with a different suffix.
        """
        if self._prompt_cache is None or self._cache_prefix_len == 0:
            return

        try:
            for c in self._prompt_cache:
                if hasattr(c, "reuse"):
                    # KVCache/RotatingKVCache support reuse(num_to_keep, offset)
                    c.reuse(self._cache_prefix_len, 0)
        except Exception:
            logger.debug("Could not trim prompt cache, will recreate on next use")
            self._prompt_cache = None
            self._cache_prefix_len = 0
            self._cache_prefix_tokens = None

    @property
    def has_prompt_cache(self) -> bool:
        """Check if a prefilled prompt cache is available."""
        return self._prompt_cache is not None and self._cache_prefix_len > 0

    def get_memory_usage_mb(self) -> float:
        """Return estimated memory usage of loaded model.

        Returns actual process memory delta if loaded, 0 otherwise.
        """
        if not self.is_loaded():
            return 0.0

        # Use estimated size since MLX doesn't expose exact GPU memory
        return self.config.estimated_memory_mb

    def _prepare_generation_params(
        self,
        prompt: str,
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        min_p: float | None,
        top_k: int | None,
        repetition_penalty: float | None,
        pre_formatted: bool,
        negative_constraints: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, int, Any, list[Any]]:
        """Resolve defaults and prepare shared generation parameters.

        Returns:
            Tuple of (formatted_prompt, max_tokens, sampler, logits_processors).
        """
        from mlx_lm.sample_utils import make_repetition_penalty, make_sampler

        max_tokens = max_tokens or self.config.default_max_tokens
        temperature = temperature if temperature is not None else self.config.default_temperature
        top_p = top_p if top_p is not None else 0.1
        top_k = top_k if top_k is not None else 50
        repetition_penalty = repetition_penalty if repetition_penalty is not None else 1.05

        default_negative_constraints = [
            "as an ai",
            "i'm an ai",
            "i am an ai",
            "as a language model",
            "i hope this helps",
            "certainly!",
            "of course!",
            "happy to help",
            "feel free to",
            "let me know if",
        ]

        if pre_formatted:
            formatted_prompt = prompt
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            formatted_prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        min_p = min_p if min_p is not None else 0.0
        sampler = make_sampler(temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k)

        logits_processors: list[Any] = []
        if repetition_penalty > 1.0:
            logits_processors.append(make_repetition_penalty(repetition_penalty))

        # Add negative constraints to reduce AI-sounding output
        constraints = negative_constraints or default_negative_constraints
        if constraints:
            logits_processors.append(
                NegativeConstraintLogitsProcessor(self._tokenizer, constraints)
            )

        return formatted_prompt, max_tokens, sampler, logits_processors

    def _process_generation_result(
        self,
        response: str,
        formatted_prompt: str,
        stop_sequences: list[str] | None,
        start_time: float,
        draft_accepted: int,
        using_speculative: bool,
    ) -> GenerationResult:
        """Post-process generation output into a GenerationResult."""
        generation_time = (time.perf_counter() - start_time) * 1000
        original_raw = response

        # Strip the prompt ONLY if MLX returned it (some versions/wrappers do)
        if response.startswith(formatted_prompt):
            response = response[len(formatted_prompt) :].strip()

        # Apply stop sequences
        if stop_sequences:
            for stop_seq in stop_sequences:
                if not stop_seq:
                    continue
                if stop_seq in response:
                    response = response[: response.index(stop_seq)]
                elif stop_seq.strip() in response:
                    trimmed_seq = stop_seq.strip()
                    response = response[: response.index(trimmed_seq)]

        # Extra safety: strip any common trailing tags if they leak through
        trailing_tags = ["<|im_end|>", "<|endoftext|>", "</reply>", "</summary>"]
        for tag in trailing_tags:
            if tag in response:
                response = response[: response.index(tag)]

        response = response.strip()

        # If post-processing killed a non-empty response, fall back to original
        if not response and original_raw.strip():
            logger.warning("Post-processing emptied a non-empty response, using raw output")
            response = original_raw.strip()

        # Handle common prefixes
        prefixes = ["Reply: ", "Result: ", "JARVIS: ", "Me: "]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix) :].strip()
                break

        # If we stripped everything, keep original for visibility in debug/bakeoff
        if not response and original_raw.strip():
            logger.warning(
                "Stripping resulted in empty string, reverting to original: '%s'",
                original_raw.strip(),
            )
            response = original_raw.strip()

        if response.endswith(("</", "<")):
            response = response.rsplit("<", 1)[0].strip()

        response = response.strip()

        # Count tokens using tokenizer for accuracy
        try:
            tokens_generated = len(self._tokenizer.encode(response))
        except Exception:
            tokens_generated = len(response.split())

        generation_time_s = generation_time / 1000
        tps = tokens_generated / generation_time_s if generation_time_s > 0 else 0.0

        return GenerationResult(
            text=response,
            tokens_generated=tokens_generated,
            generation_time_ms=generation_time,
            tokens_per_second=round(tps, 1),
            draft_accepted_tokens=draft_accepted,
            speculative_enabled=using_speculative,
        )

    def generate_sync(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        stop_sequences: list[str] | None = None,
        timeout_seconds: float | None = None,
        prompt_cache: list[Any] | None = None,
        num_draft_tokens: int | None = None,
        pre_formatted: bool = False,
        negative_constraints: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        if not self.is_loaded():
            raise ModelLoadError("Model not loaded")

        formatted_prompt, max_tokens, sampler, logits_processors = self._prepare_generation_params(
            prompt,
            max_tokens,
            temperature,
            top_p,
            min_p,
            top_k,
            repetition_penalty,
            pre_formatted,
            negative_constraints,
            system_prompt,
        )

        # NOTE: SYSTEM_PREFIX is already baked into RAG_REPLY_PROMPT.template,
        # so we do NOT prepend it here. Doing so would duplicate it.

        try:
            from mlx_lm import generate

            start_time = time.perf_counter()
            with MLXModelLoader._mlx_load_lock:
                apply_llm_limits()
                # Direct MLX call
                response = generate(
                    model=self._model,
                    tokenizer=self._tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    verbose=False,
                )

            return self._process_generation_result(
                response, formatted_prompt, stop_sequences, start_time, 0, False
            )

        except Exception as e:
            logger.exception("Generation failed")
            raise ModelGenerationError(f"Generation failed: {e}", prompt=prompt) from e

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        stop_sequences: list[str] | None = None,
        stop_event: threading.Event | None = None,
        prompt_cache: list[Any] | None = None,
        num_draft_tokens: int | None = None,
        pre_formatted: bool = False,
        system_prompt: str | None = None,
    ) -> Any:
        """Generate text with true streaming output (yields tokens as generated).

        Uses mlx_lm.stream_generate for real token-by-token streaming.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (LFM optimal: 0.3)
            top_p: Nucleus sampling threshold (LFM optimal: 0.1)
            min_p: Minimum probability threshold (LFM optimal: 0.15)
            top_k: Top-k sampling limit (LFM optimal: 50)
            repetition_penalty: Penalty for repeated tokens (LFM optimal: 1.05)
            stop_sequences: Strings that stop generation
            stop_event: If set, generation stops early (e.g. on client disconnect)
            system_prompt: Optional system prompt to include in chat template.

        Yields:
            StreamToken objects with individual tokens as they're generated

        Raises:
            ModelGenerationError: If model is not loaded, prompt is invalid,
                or generation fails.
        """
        if not prompt or not prompt.strip():
            raise ModelGenerationError(
                "Prompt cannot be empty",
                model_name=self.config.display_name,
                code=ErrorCode.MDL_INVALID_REQUEST,
            )

        if not self.is_loaded():
            raise ModelGenerationError(
                "Model not loaded. Call load() first.",
                model_name=self.config.display_name,
                code=ErrorCode.MDL_LOAD_FAILED,
            )

        try:
            formatted_prompt, max_tokens, sampler, logits_processors = (
                self._prepare_generation_params(
                    prompt,
                    max_tokens,
                    temperature,
                    top_p,
                    min_p,
                    top_k,
                    repetition_penalty,
                    pre_formatted,
                    system_prompt=system_prompt,
                )
            )

            logger.debug("Prompt sent to model (%d chars)", len(formatted_prompt))

            # Use real streaming with mlx_lm.stream_generate
            # Hold the shared GPU lock for the entire generation to prevent
            # concurrent Metal access with embedding encode() or other callers.
            accumulated_text = ""
            token_index = 0
            stream_start = time.perf_counter()

            stream_kwargs: dict[str, Any] = {}
            if prompt_cache is not None:
                stream_kwargs["prompt_cache"] = prompt_cache
            if self.config.kv_cache_bits and self.config.kv_cache_bits != 16:
                stream_kwargs["kv_bits"] = self.config.kv_cache_bits
            if self._draft_model is not None:
                stream_kwargs["draft_model"] = self._draft_model
                stream_kwargs["num_draft_tokens"] = num_draft_tokens or 3

            from mlx_lm import stream_generate

            with MLXModelLoader._mlx_load_lock:
                for response in stream_generate(
                    model=self._model,
                    tokenizer=self._tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    **stream_kwargs,
                ):
                    # Check if caller requested cancellation (e.g. client disconnected)
                    if stop_event is not None and stop_event.is_set():
                        logger.debug("Streaming generation cancelled via stop_event")
                        break

                    # response.text contains just the current token's text (not accumulated)
                    new_text = response.text
                    accumulated_text += new_text

                    # Check for stop sequences
                    should_stop = False
                    if stop_sequences and new_text:
                        for stop_seq in stop_sequences:
                            if stop_seq in accumulated_text:
                                # Trim to stop sequence
                                stop_idx = accumulated_text.index(stop_seq)
                                # Calculate how much of new_text to keep
                                text_before_new = accumulated_text[: -len(new_text)]
                                stop_idx_in_new = stop_idx - len(text_before_new)
                                if stop_idx_in_new >= 0:
                                    new_text = new_text[:stop_idx_in_new]
                                should_stop = True
                                break

                    if new_text:
                        is_final = response.finish_reason is not None or should_stop
                        # Track time to first token
                        ttft = None
                        if token_index == 0:
                            ttft = (time.perf_counter() - stream_start) * 1000
                        yield StreamToken(
                            token=new_text,
                            token_index=token_index,
                            is_final=is_final,
                            ttft_ms=ttft,
                            from_draft=getattr(response, "from_draft", False),
                        )
                        token_index += 1

                    if should_stop:
                        break

                    # Yield on finish
                    if response.finish_reason is not None:
                        break

        except ModelGenerationError:
            raise
        except Exception as e:
            logger.exception("Streaming generation failed")
            raise ModelGenerationError(
                f"Streaming generation failed: {e}",
                prompt=prompt,
                model_name=self.config.display_name,
                cause=e,
            ) from e

    def switch_model(self, model_id: str) -> bool:
        """Switch to a different model.

        Unloads the current model (if any) and updates configuration to use
        the new model. The new model will be loaded on the next generation request.

        Args:
            model_id: Model identifier from registry (e.g., "qwen-3b").

        Returns:
            True if configuration was updated successfully, False if model_id unknown.
        """
        spec = get_model_spec(model_id)
        if spec is None:
            logger.error("Unknown model_id: %s", model_id)
            return False

        # Unload current model if loaded
        if self.is_loaded():
            logger.info("Switching from %s to %s", self.config.display_name, spec.display_name)
            self.unload()

        # Update configuration
        self.config = ModelConfig(model_id=model_id)
        logger.info("Model configuration updated to %s", spec.display_name)

        # Eagerly load the new model and prefill prompt cache to avoid
        # full prefill cost on first generation after switch
        try:
            if self.load():
                # Prefill cache while holding the GPU lock to ensure thread-safety
                with MLXModelLoader._mlx_load_lock:
                    # Import system prefix lazily to avoid circular imports
                    from jarvis.prompts import SYSTEM_PREFIX

                    self.prefill_prompt_cache(SYSTEM_PREFIX)
        except Exception as e:
            # Best-effort: log warning but don't fail the switch
            # The model will still work, just without the cache optimization
            logger.warning("Failed to prefill prompt cache after model switch: %s", e)

        return True

    def get_current_model_info(self) -> dict[str, Any]:
        """Return information about the current model configuration.

        Returns:
            Dictionary with model information including id, display_name,
            loaded status, and memory usage.
        """
        return {
            "id": self.config.model_id,
            "path": self.config.model_path,
            "display_name": self.config.display_name,
            "loaded": self.is_loaded(),
            "memory_usage_mb": self.get_memory_usage_mb(),
            "quality_tier": getattr(self.config.spec, "quality_tier", None),
        }


# Singleton model loader for convenience
_model_loader: MLXModelLoader | None = None
_model_loader_lock = threading.Lock()


def get_model() -> MLXModelLoader:
    """Get or create singleton model loader instance.

    Thread-safe using double-check locking pattern.
    The model is loaded lazily on first generation call.

    Returns:
        The shared MLXModelLoader instance
    """
    global _model_loader

    if _model_loader is None:
        with _model_loader_lock:
            if _model_loader is None:
                _model_loader = MLXModelLoader()
    return _model_loader


def reset_model() -> None:
    """Unload the singleton model loader so memory is free for another LLM (e.g. extractor).

    Keeps the singleton reference so callers (e.g. generator) holding that loader
    can call load() again without creating a second loader instance.
    """
    global _model_loader
    with _model_loader_lock:
        if _model_loader is not None:
            _model_loader.unload()
