"""Health check API endpoints.

Provides system health status including memory, model, and permission state.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import psutil
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse

from api.schemas import HealthResponse, ModelStatusResponse, PreloadResponse
from integrations.imessage import ChatDBReader

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

router = APIRouter(tags=["health"])

# Constants
BYTES_PER_MB = 1024 * 1024


def _get_process_memory() -> tuple[float, float]:
    """Get JARVIS process memory usage.

    Returns:
        Tuple of (rss_mb, vms_mb) - actual RAM usage and virtual memory allocation
    """
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / BYTES_PER_MB
        vms_mb = mem_info.vms / BYTES_PER_MB
        return rss_mb, vms_mb
    except Exception:
        return 0.0, 0.0


def _check_imessage_access() -> bool:
    """Check if iMessage database is accessible."""
    try:
        reader = ChatDBReader()
        result = reader.check_access()
        reader.close()
        return result
    except Exception:
        return False


def _get_memory_mode(available_gb: float) -> str:
    """Determine memory mode based on available memory."""
    if available_gb >= 4.0:
        return "FULL"
    elif available_gb >= 2.0:
        return "LITE"
    else:
        return "MINIMAL"


def _check_model_loaded() -> bool:
    """Check if the MLX model is currently loaded."""
    try:
        from models import get_generator

        generator = get_generator()
        # Check if internal model is loaded (accessing private attribute for status check)
        return generator._model is not None  # type: ignore[attr-defined]
    except Exception:
        return False


def _get_model_status() -> ModelStatusResponse:
    """Get detailed model loading status."""
    try:
        from models import get_generator

        generator = get_generator()
        # Access the internal loader's status
        loader = generator._loader
        status = loader.get_loading_status()

        return ModelStatusResponse(
            state=status.state,
            progress=status.progress if status.state == "loading" else None,
            message=status.message or None,
            memory_usage_mb=loader.get_memory_usage_mb() if status.state == "loaded" else None,
            load_time_seconds=status.load_time_seconds,
            error=status.error,
        )
    except Exception as e:
        return ModelStatusResponse(
            state="error",
            progress=None,
            message=f"Failed to get status: {e}",
            error=str(e),
        )


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Get comprehensive system health status.

    Returns information about:
    - iMessage database access
    - System memory usage (total system)
    - JARVIS process memory usage (what this app is using)
    - Memory controller mode
    - Model loading state
    - Overall system health
    """
    # System memory stats
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    used_gb = memory.used / (1024**3)

    # JARVIS process memory
    jarvis_rss_mb, jarvis_vms_mb = _get_process_memory()

    # Check various components
    imessage_access = _check_imessage_access()
    memory_mode = _get_memory_mode(available_gb)
    model_loaded = _check_model_loaded()

    # Determine overall status
    details: dict[str, str] = {}

    if not imessage_access:
        details["imessage"] = "Full Disk Access required"

    if available_gb < 2.0:
        details["memory"] = f"Low memory: {available_gb:.1f}GB available"

    # Determine overall health status
    if not imessage_access:
        status = "unhealthy"
    elif available_gb < 2.0:
        status = "degraded"
    else:
        status = "healthy"

    return HealthResponse(
        status=status,
        imessage_access=imessage_access,
        memory_available_gb=round(available_gb, 2),
        memory_used_gb=round(used_gb, 2),
        memory_mode=memory_mode,
        model_loaded=model_loaded,
        permissions_ok=imessage_access,
        details=details if details else None,
        jarvis_rss_mb=round(jarvis_rss_mb, 1),
        jarvis_vms_mb=round(jarvis_vms_mb, 1),
    )


@router.get("/")
def root() -> dict[str, str]:
    """Root endpoint - simple health ping."""
    return {"status": "ok", "service": "jarvis-api"}


@router.get("/model-status", response_model=ModelStatusResponse)
def get_model_status() -> ModelStatusResponse:
    """Get current model loading status.

    Returns detailed information about the model's loading state,
    progress, memory usage, and any errors.
    """
    return _get_model_status()


@router.get("/model-status/stream")
async def stream_model_status() -> StreamingResponse:
    """Stream model status updates via Server-Sent Events.

    Useful for displaying real-time loading progress in the UI.
    Automatically closes when model reaches "loaded" or "error" state.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        max_iterations = 120  # 60 seconds max (120 * 0.5s)
        iteration = 0

        while iteration < max_iterations:
            status = _get_model_status()
            yield f"data: {status.model_dump_json()}\n\n"

            # Stop streaming when loading is complete or errored
            if status.state in ("loaded", "error", "unloaded"):
                # Send one final update and close
                break

            await asyncio.sleep(0.5)
            iteration += 1

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
    )


def _preload_model_sync() -> tuple[bool, str]:
    """Synchronously preload the model (runs in background)."""
    try:
        from models import get_generator

        generator = get_generator()
        success = generator.load()
        if success:
            return True, "Model loaded successfully"
        return False, "Failed to load model"
    except Exception as e:
        return False, f"Error loading model: {e}"


@router.post("/model-preload", response_model=PreloadResponse)
async def preload_model(background_tasks: BackgroundTasks) -> PreloadResponse:
    """Trigger model preloading.

    Initiates model loading in the background and returns immediately.
    Use /model-status or /model-status/stream to monitor progress.
    """
    status = _get_model_status()

    # Already loaded
    if status.state == "loaded":
        return PreloadResponse(
            success=True,
            message="Model already loaded",
            state="loaded",
        )

    # Already loading
    if status.state == "loading":
        return PreloadResponse(
            success=True,
            message="Model loading in progress",
            state="loading",
        )

    # Start loading in background
    background_tasks.add_task(_preload_model_sync)

    return PreloadResponse(
        success=True,
        message="Model preload initiated",
        state="loading",
    )


@router.post("/model-unload", response_model=PreloadResponse)
def unload_model() -> PreloadResponse:
    """Unload the model to free memory.

    Useful when the model is not needed and memory should be reclaimed.
    """
    try:
        from models import unload_generator

        unload_generator()
        return PreloadResponse(
            success=True,
            message="Model unloaded successfully",
            state="unloaded",
        )
    except Exception as e:
        return PreloadResponse(
            success=False,
            message=f"Failed to unload model: {e}",
            state="error",
        )
