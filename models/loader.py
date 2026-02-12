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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any

# Disable HuggingFace hub network checks after initial download
# This prevents slow version checks on every model load/generate call
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx
import psutil
from mlx_lm import generate, load, stream_generate

from models.memory_config import apply_llm_limits
from mlx_lm.sample_utils import make_repetition_penalty, make_sampler

from jarvis.errors import (
    ErrorCode,
    ModelGenerationError,
    ModelLoadError,
    model_generation_timeout,
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
    _resolved_spec: ModelSpec | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Resolve model_id to model_path and estimated_memory_mb."""
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
            # Neither model_id nor model_path specified, use default
            spec = get_model_spec(DEFAULT_MODEL_ID)
            if spec:
                self._resolved_spec = spec
                self.model_id = spec.id
                self.model_path = spec.path
                self.estimated_memory_mb = spec.estimated_memory_mb
            else:
                # Registry should always have DEFAULT_MODEL_ID, but handle gracefully
                logger.warning("DEFAULT_MODEL_ID '%s' not in registry", DEFAULT_MODEL_ID)
                default_spec = MODEL_REGISTRY[DEFAULT_MODEL_ID]
                self.model_path = default_spec.path

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

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize the loader.

        Args:
            config: Model configuration. Uses defaults if not provided.
        """
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
                logger.info(
                    "Loading model: %s (%s)",
                    self.config.display_name,
                    self.config.model_path,
                )
                start_time = time.perf_counter()

                # Set MLX memory limits before loading (critical on 8GB systems)
                apply_llm_limits()

                # mlx_lm.load returns (model, tokenizer) tuple
                result = load(self.config.model_path)
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
                logger.error("Out of memory loading model. Free up memory or use a smaller model.")
                self.unload()
                raise model_out_of_memory(self.config.display_name) from e
            except OSError as e:
                # Covers network errors, disk errors, permission issues
                logger.error("OS error loading model: %s", e)
                self.unload()
                raise ModelLoadError(
                    f"OS error loading model: {e}",
                    model_name=self.config.display_name,
                    model_path=self.config.model_path,
                    cause=e,
                ) from e
            except Exception as e:
                logger.exception("Failed to load model")
                self.unload()
                raise ModelLoadError(
                    f"Failed to load model: {e}",
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

        with MLXModelLoader._mlx_load_lock:
            try:
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

            except Exception:
                logger.exception("Failed to load draft model %s", draft_model_id)
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

        from mlx_lm.models.cache import make_prompt_cache

        try:
            # Tokenize the prefix once and store for reuse (avoids redundant
            # re-encoding if callers need the token IDs later)
            tokens = mx.array(self._tokenizer.encode(prefix_text))
            self._prompt_cache = make_prompt_cache(self._model)
            self._cache_prefix_len = tokens.shape[0]
            self._cache_prefix_tokens = tokens

            # Prefill: run forward pass with max_tokens=0 to populate cache.
            # generate_step receives pre-tokenized mx.array directly, no
            # re-encoding occurs here.
            from mlx_lm.generate import generate_step

            for _ in generate_step(
                self._cache_prefix_tokens,
                self._model,
                max_tokens=0,
                prompt_cache=self._prompt_cache,
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
    ) -> tuple[str, int, Any, list[Any] | None]:
        """Resolve defaults and prepare shared generation parameters.

        Returns:
            Tuple of (formatted_prompt, max_tokens, sampler, logits_processors).
        """
        max_tokens = max_tokens or self.config.default_max_tokens
        temperature = temperature if temperature is not None else self.config.default_temperature
        top_p = top_p if top_p is not None else 0.1
        top_k = top_k if top_k is not None else 50
        repetition_penalty = repetition_penalty if repetition_penalty is not None else 1.05

        if pre_formatted:
            formatted_prompt = prompt
        else:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        min_p = min_p if min_p is not None else 0.0
        sampler = make_sampler(temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k)

        logits_processors = None
        if repetition_penalty > 1.0:
            logits_processors = [make_repetition_penalty(repetition_penalty)]

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

        # Strip the prompt from response
        if response.startswith(formatted_prompt):
            response = response[len(formatted_prompt) :].strip()

        # Apply stop sequences
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in response:
                    response = response[: response.index(stop_seq)]

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
    ) -> GenerationResult:
        """Generate text synchronously.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (LFM optimal: 0.3)
            top_p: Nucleus sampling threshold (LFM optimal: 0.1)
            min_p: Minimum probability threshold (LFM optimal: 0.15)
            top_k: Top-k sampling limit (LFM optimal: 50)
            repetition_penalty: Penalty for repeated tokens (LFM optimal: 1.05)
            stop_sequences: Strings that stop generation
            timeout_seconds: Timeout for generation (overrides config if provided).
                WARNING: Timeout cannot stop running MLX computation. If timeout
                is reached, the generation thread may continue running in background
                until completion. This is a limitation of MLX's synchronous API.
            prompt_cache: Optional pre-computed KV cache from prefill_prompt_cache().
                If provided, passed through to mlx_lm.generate() for prefix reuse.

        Returns:
            GenerationResult with text, token count, and timing

        Raises:
            ModelGenerationError: If model is not loaded, prompt is invalid,
                generation fails, or timeout is exceeded.

        Note:
            Due to MLX's synchronous generation API, timeouts are best-effort only.
            The generation thread cannot be forcefully stopped and may continue
            consuming GPU resources until the model completes or hits max_tokens.
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

        # Use provided timeout, fall back to config, then None (no timeout)
        effective_timeout: float | None
        if timeout_seconds is not None:
            effective_timeout = timeout_seconds
        else:
            effective_timeout = self.config.generation_timeout_seconds

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
                )
            )

            # Generate with timeout handling
            start_time = time.perf_counter()

            # Track speculative decoding stats
            draft_accepted = 0
            using_speculative = self._draft_model is not None

            # Cancellation flag: set on timeout so streaming generation exits early
            cancel_event = threading.Event()

            def _do_generate() -> tuple[str, int]:
                """Inner function for generation to run in executor.

                Holds the shared GPU lock to prevent concurrent Metal access
                with embedding encode() or other generation calls.

                Returns:
                    Tuple of (response_text, draft_accepted_count).
                """
                kwargs: dict[str, Any] = {}
                if prompt_cache is not None:
                    kwargs["prompt_cache"] = prompt_cache

                with MLXModelLoader._mlx_load_lock:
                    # Restore LLM memory limits in case embedder lowered them
                    apply_llm_limits()

                    if self._draft_model is not None:
                        # Speculative decoding: use stream_generate with draft_model
                        # to track per-token acceptance
                        text = ""
                        total = 0
                        draft_tokens_verified = 0
                        for resp in stream_generate(
                            model=self._model,
                            tokenizer=self._tokenizer,
                            prompt=formatted_prompt,
                            max_tokens=max_tokens,
                            sampler=sampler,
                            logits_processors=logits_processors,
                            draft_model=self._draft_model,
                            num_draft_tokens=num_draft_tokens or 3,
                            **kwargs,
                        ):
                            # Check cancellation flag each token to exit early on timeout
                            if cancel_event.is_set():
                                logger.info("Generation cancelled via timeout after %d tokens", total)
                                break
                            text = resp.text
                            total += 1
                            if getattr(resp, "from_draft", False):
                                draft_tokens_verified += 1
                        return text, draft_tokens_verified
                    else:
                        # Standard generation (mlx_lm.generate is a single blocking
                        # call with no per-token hook, so cancellation is not possible)
                        result = generate(
                            model=self._model,
                            tokenizer=self._tokenizer,
                            prompt=formatted_prompt,
                            max_tokens=max_tokens,
                            sampler=sampler,
                            logits_processors=logits_processors,
                            verbose=False,
                            **kwargs,
                        )
                        return result, 0

            if effective_timeout is not None and effective_timeout > 0:
                # Use ThreadPoolExecutor for timeout handling
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_do_generate)
                    try:
                        response, draft_accepted = future.result(timeout=effective_timeout)
                    except FuturesTimeoutError:
                        # Signal the generation thread to stop (works for streaming path)
                        cancel_event.set()
                        future.cancel()
                        logger.warning(
                            "Generation timed out after %.1f seconds for model %s. "
                            "Cancellation signal sent to generation thread.",
                            effective_timeout,
                            self.config.display_name,
                        )
                        raise model_generation_timeout(
                            self.config.display_name,
                            effective_timeout,
                            prompt=prompt,
                        )
            else:
                # No timeout - run directly
                response, draft_accepted = _do_generate()

            return self._process_generation_result(
                response,
                formatted_prompt,
                stop_sequences,
                start_time,
                draft_accepted,
                using_speculative,
            )

        except ModelGenerationError:
            # Re-raise JARVIS errors as-is
            raise
        except Exception as e:
            logger.exception("Generation failed")
            raise ModelGenerationError(
                f"Generation failed: {e}",
                prompt=prompt,
                model_name=self.config.display_name,
                cause=e,
            ) from e

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
                )
            )

            # Use real streaming with mlx_lm.stream_generate
            # Hold the shared GPU lock for the entire generation to prevent
            # concurrent Metal access with embedding encode() or other callers.
            accumulated_text = ""
            token_index = 0
            stream_start = time.perf_counter()

            stream_kwargs: dict[str, Any] = {}
            if prompt_cache is not None:
                stream_kwargs["prompt_cache"] = prompt_cache
            if self._draft_model is not None:
                stream_kwargs["draft_model"] = self._draft_model
                stream_kwargs["num_draft_tokens"] = num_draft_tokens or 3

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

                    # response.text contains the full text so far
                    # Extract just the new part
                    new_text = response.text[len(accumulated_text) :]
                    accumulated_text = response.text

                    # Check for stop sequences
                    should_stop = False
                    if stop_sequences and new_text:
                        for stop_seq in stop_sequences:
                            if stop_seq in accumulated_text:
                                # Trim to stop sequence
                                stop_idx = accumulated_text.index(stop_seq)
                                new_text = accumulated_text[
                                    len(accumulated_text) - len(new_text) : stop_idx
                                ]
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
    """Reset the singleton model loader and unload any loaded model.

    Use this to:
    - Clear state between tests
    - Switch to a different model configuration
    - Force complete reinitialization
    """
    global _model_loader
    with _model_loader_lock:
        if _model_loader is not None:
            _model_loader.unload()
        _model_loader = None
