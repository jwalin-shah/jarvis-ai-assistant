"""Health check API endpoints.

Provides system health status including memory, model, and permission state.
"""

import os

import psutil
from fastapi import APIRouter

from api.schemas import HealthResponse
from integrations.imessage import ChatDBReader

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
