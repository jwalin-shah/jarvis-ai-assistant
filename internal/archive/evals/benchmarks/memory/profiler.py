"""Memory profiler implementation for MLX models.  # noqa: E501
  # noqa: E501
Workstream 1: Memory Profiler  # noqa: E501
  # noqa: E501
Implements the MemoryProfiler protocol from contracts/memory.py.  # noqa: E501
Measures RSS, virtual memory, and Metal GPU memory during model loading.  # noqa: E501
"""  # noqa: E501
  # noqa: E501
import gc  # noqa: E501
import logging  # noqa: E501
import time  # noqa: E501
from datetime import UTC, datetime  # noqa: E402  # noqa: E501
from typing import Any  # noqa: E402  # noqa: E501

# noqa: E501
import psutil  # noqa: E501

# noqa: E501
from jarvis.contracts.memory import MemoryProfile  # noqa: E402  # noqa: E501

  # noqa: E501
logger = logging.getLogger(__name__)  # noqa: E501
  # noqa: E501
# MLX is only available on Apple Silicon  # noqa: E501
# Use lazy imports to allow testing on other platforms  # noqa: E501
_mlx_available = False  # noqa: E501
_mx: Any = None  # noqa: E501
_load: Any = None  # noqa: E501
  # noqa: E501
try:  # noqa: E501
    import mlx.core as _mx_module  # noqa: E501
    from mlx_lm import load as _load_module  # noqa: E501
  # noqa: E501
    _mx = _mx_module  # noqa: E501
    _load = _load_module  # noqa: E501
    _mlx_available = True  # noqa: E501
except ImportError:  # noqa: E501
    logger.debug("MLX not available - memory profiler will use fallback mode")  # noqa: E501
  # noqa: E501
# Constants  # noqa: E501
BYTES_PER_MB = 1024 * 1024  # noqa: E501
  # noqa: E501
  # noqa: E501
def _extract_model_info(model_path: str) -> tuple[str, str]:  # noqa: E501
    """Extract model name and quantization from path.  # noqa: E501
  # noqa: E501
    Args:  # noqa: E501
        model_path: HuggingFace model path like "mlx-community/Qwen2.5-0.5B-Instruct-4bit"  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        Tuple of (model_name, quantization)  # noqa: E501
    """  # noqa: E501
    # Extract the model identifier from the path  # noqa: E501
    parts = model_path.split("/")  # noqa: E501
    model_id = parts[-1] if parts else model_path  # noqa: E501
  # noqa: E501
    # Try to extract quantization from common patterns  # noqa: E501
    quantization = "unknown"  # noqa: E501
    for quant in ["4bit", "8bit", "fp16", "bf16", "fp32"]:  # noqa: E501
        if quant in model_id.lower():  # noqa: E501
            quantization = quant  # noqa: E501
            break  # noqa: E501
  # noqa: E501
    return model_id, quantization  # noqa: E501
  # noqa: E501
  # noqa: E501
def _get_metal_memory_mb() -> float:  # noqa: E501
    """Get current Metal GPU memory usage in MB.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        Metal memory in MB, or 0.0 if unavailable  # noqa: E501
    """  # noqa: E501
    if not _mlx_available or _mx is None:  # noqa: E501
        return 0.0  # noqa: E501
    try:  # noqa: E501
        # Get Metal memory stats  # noqa: E501
        # Note: MLX exposes peak_memory and active_memory  # noqa: E501
        peak_memory = _mx.metal.get_peak_memory()  # noqa: E501
        return float(peak_memory) / BYTES_PER_MB  # noqa: E501
    except (AttributeError, RuntimeError):  # noqa: E501
        logger.debug("Metal memory stats not available")  # noqa: E501
        return 0.0  # noqa: E501
  # noqa: E501
  # noqa: E501
def _get_process_memory() -> tuple[float, float]:  # noqa: E501
    """Get current process memory usage.  # noqa: E501
  # noqa: E501
    Returns:  # noqa: E501
        Tuple of (rss_mb, virtual_mb)  # noqa: E501
    """  # noqa: E501
    process = psutil.Process()  # noqa: E501
    mem_info = process.memory_info()  # noqa: E501
    rss_mb = mem_info.rss / BYTES_PER_MB  # noqa: E501
    vms_mb = mem_info.vms / BYTES_PER_MB  # noqa: E501
    return rss_mb, vms_mb  # noqa: E501
  # noqa: E501
  # noqa: E501
def _unload_model() -> None:  # noqa: E501
    """Unload model and free all memory including GPU cache.  # noqa: E501
  # noqa: E501
    Following the pattern from models/loader.py.  # noqa: E501
    """  # noqa: E501
    # Clear Metal GPU memory  # noqa: E501
    if _mlx_available and _mx is not None:  # noqa: E501
        try:  # noqa: E501
            _mx.metal.clear_cache()  # noqa: E501
        except (AttributeError, RuntimeError):  # noqa: E501
            logger.debug("Metal cache clear not available")  # noqa: E501
  # noqa: E501
    # Force garbage collection  # noqa: E501
    gc.collect()  # noqa: E501
  # noqa: E501
  # noqa: E501
class MLXMemoryProfiler:  # noqa: E501
    """Memory profiler for MLX models.  # noqa: E501
  # noqa: E501
    Implements the MemoryProfiler protocol from contracts/memory.py.  # noqa: E501
  # noqa: E501
    Measures actual memory usage by loading the model, capturing metrics,  # noqa: E501
    and then unloading to stay within memory budget.  # noqa: E501
    """  # noqa: E501
  # noqa: E501
    def __init__(self) -> None:  # noqa: E501
        """Initialize the profiler."""  # noqa: E501
        self._baseline_rss: float = 0.0  # noqa: E501
        self._baseline_vms: float = 0.0  # noqa: E501
  # noqa: E501
    def _capture_baseline(self) -> None:  # noqa: E501
        """Capture baseline memory before loading model."""  # noqa: E501
        _unload_model()  # Ensure clean state  # noqa: E501
        self._baseline_rss, self._baseline_vms = _get_process_memory()  # noqa: E501
        logger.info(  # noqa: E501
            "Baseline memory - RSS: %.1fMB, VMS: %.1fMB",  # noqa: E501
            self._baseline_rss,  # noqa: E501
            self._baseline_vms,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
    def profile_model(self, model_path: str, context_length: int) -> MemoryProfile:  # noqa: E501
        """Profile a model's memory usage.  # noqa: E501
  # noqa: E501
        Loads the model, measures memory, then unloads to free memory.  # noqa: E501
        CRITICAL: Always unloads model after profiling for 8GB safety.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            model_path: HuggingFace model path or local path  # noqa: E501
            context_length: Context window size (affects KV cache allocation)  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            MemoryProfile with measured memory usage  # noqa: E501
        """  # noqa: E501
        logger.info("Profiling model: %s at context_length: %d", model_path, context_length)  # noqa: E501
  # noqa: E501
        # Extract model info  # noqa: E501
        model_name, quantization = _extract_model_info(model_path)  # noqa: E501
  # noqa: E501
        # Capture baseline memory  # noqa: E501
        self._capture_baseline()  # noqa: E501
  # noqa: E501
        # Reset peak memory counter before loading  # noqa: E501
        if _mlx_available and _mx is not None:  # noqa: E501
            try:  # noqa: E501
                _mx.metal.reset_peak_memory()  # noqa: E501
            except (AttributeError, RuntimeError):  # noqa: E501
                logger.debug("Metal reset_peak_memory not available")  # noqa: E501
  # noqa: E501
        if not _mlx_available or _load is None:  # noqa: E501
            msg = "MLX is not available. Memory profiler requires Apple Silicon with MLX installed."  # noqa: E501
            raise RuntimeError(msg)  # noqa: E501
  # noqa: E501
        # Load the model and measure time  # noqa: E501
        start_time = time.perf_counter()  # noqa: E501
        try:  # noqa: E501
            model, tokenizer = _load(model_path)  # noqa: E501
            load_time = time.perf_counter() - start_time  # noqa: E501
  # noqa: E501
            # Force evaluation to ensure model is fully loaded  # noqa: E501
            _mx.eval(model.parameters())  # noqa: E501
  # noqa: E501
            # Capture memory after load  # noqa: E501
            rss_mb, vms_mb = _get_process_memory()  # noqa: E501
            metal_mb = _get_metal_memory_mb()  # noqa: E501
  # noqa: E501
            # Calculate delta from baseline  # noqa: E501
            rss_delta = rss_mb - self._baseline_rss  # noqa: E501
            vms_delta = vms_mb - self._baseline_vms  # noqa: E501
  # noqa: E501
            logger.info(  # noqa: E501
                "Model loaded in %.2fs - RSS: %.1fMB (+%.1fMB), Metal: %.1fMB",  # noqa: E501
                load_time,  # noqa: E501
                rss_mb,  # noqa: E501
                rss_delta,  # noqa: E501
                metal_mb,  # noqa: E501
            )  # noqa: E501
  # noqa: E501
            profile = MemoryProfile(  # noqa: E501
                model_name=model_name,  # noqa: E501
                quantization=quantization,  # noqa: E501
                context_length=context_length,  # noqa: E501
                rss_mb=rss_delta,  # Report delta, not absolute  # noqa: E501
                virtual_mb=vms_delta,  # noqa: E501
                metal_mb=metal_mb,  # noqa: E501
                load_time_seconds=load_time,  # noqa: E501
                timestamp=datetime.now(UTC).isoformat(),  # noqa: E501
            )  # noqa: E501
  # noqa: E501
        except FileNotFoundError:  # noqa: E501
            logger.error("Model not found: %s", model_path)  # noqa: E501
            raise  # noqa: E501
        except Exception:  # noqa: E501
            logger.exception("Error profiling model: %s", model_path)  # noqa: E501
            raise  # noqa: E501
        finally:  # noqa: E501
            # CRITICAL: Always unload model to stay within memory budget  # noqa: E501
            logger.info("Unloading model for memory safety")  # noqa: E501
            model = None  # noqa: F841  # noqa: E501
            tokenizer = None  # noqa: F841  # noqa: E501
            _unload_model()  # noqa: E501
  # noqa: E501
        return profile  # noqa: E501
  # noqa: E501
    def profile_with_generation(  # noqa: E501
        self,  # noqa: E501
        model_path: str,  # noqa: E501
        context_length: int,  # noqa: E501
        prompt: str = "Hello",  # noqa: E501
        max_tokens: int = 10,  # noqa: E501
    ) -> MemoryProfile:  # noqa: E501
        """Profile memory during model loading and generation.  # noqa: E501
  # noqa: E501
        This measures peak memory including KV cache allocation during inference.  # noqa: E501
  # noqa: E501
        Args:  # noqa: E501
            model_path: HuggingFace model path or local path  # noqa: E501
            context_length: Context window size  # noqa: E501
            prompt: Test prompt for generation  # noqa: E501
            max_tokens: Maximum tokens to generate  # noqa: E501
  # noqa: E501
        Returns:  # noqa: E501
            MemoryProfile with measured memory usage including generation  # noqa: E501
        """  # noqa: E501
        logger.info(  # noqa: E501
            "Profiling model with generation: %s at context_length: %d",  # noqa: E501
            model_path,  # noqa: E501
            context_length,  # noqa: E501
        )  # noqa: E501
  # noqa: E501
        # Extract model info  # noqa: E501
        model_name, quantization = _extract_model_info(model_path)  # noqa: E501
  # noqa: E501
        # Capture baseline memory  # noqa: E501
        self._capture_baseline()  # noqa: E501
  # noqa: E501
        # Reset peak memory counter  # noqa: E501
        if _mlx_available and _mx is not None:  # noqa: E501
            try:  # noqa: E501
                _mx.metal.reset_peak_memory()  # noqa: E501
            except (AttributeError, RuntimeError):  # noqa: E501
                pass  # noqa: E501
  # noqa: E501
        if not _mlx_available or _load is None:  # noqa: E501
            msg = "MLX is not available. Memory profiler requires Apple Silicon with MLX installed."  # noqa: E501
            raise RuntimeError(msg)  # noqa: E501
  # noqa: E501
        start_time = time.perf_counter()  # noqa: E501
        try:  # noqa: E501
            model, tokenizer = _load(model_path)  # noqa: E501
            load_time = time.perf_counter() - start_time  # noqa: E501
  # noqa: E501
            # Run a generation to allocate KV cache  # noqa: E501
            from mlx_lm import generate  # noqa: E501
  # noqa: E501
            _ = generate(  # noqa: E501
                model=model,  # noqa: E501
                tokenizer=tokenizer,  # noqa: E501
                prompt=prompt,  # noqa: E501
                max_tokens=max_tokens,  # noqa: E501
                verbose=False,  # noqa: E501
            )  # noqa: E501
  # noqa: E501
            # Capture peak memory after generation  # noqa: E501
            rss_mb, vms_mb = _get_process_memory()  # noqa: E501
            metal_mb = _get_metal_memory_mb()  # noqa: E501
  # noqa: E501
            rss_delta = rss_mb - self._baseline_rss  # noqa: E501
            vms_delta = vms_mb - self._baseline_vms  # noqa: E501
  # noqa: E501
            profile = MemoryProfile(  # noqa: E501
                model_name=model_name,  # noqa: E501
                quantization=quantization,  # noqa: E501
                context_length=context_length,  # noqa: E501
                rss_mb=rss_delta,  # noqa: E501
                virtual_mb=vms_delta,  # noqa: E501
                metal_mb=metal_mb,  # noqa: E501
                load_time_seconds=load_time,  # noqa: E501
                timestamp=datetime.now(UTC).isoformat(),  # noqa: E501
            )  # noqa: E501
  # noqa: E501
        finally:  # noqa: E501
            # CRITICAL: Always unload  # noqa: E501
            model = None  # noqa: F841  # noqa: E501
            tokenizer = None  # noqa: F841  # noqa: E501
            _unload_model()  # noqa: E501
  # noqa: E501
        return profile  # noqa: E501
