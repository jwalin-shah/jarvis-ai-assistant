"""Health and readiness API endpoints."""

from __future__ import annotations

import logging
import os
from typing import Any

import psutil
from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from api.ratelimit import RATE_LIMIT_READ, limiter
from api.schemas import HealthResponse, ModelInfo
from jarvis.metrics import get_health_cache, get_model_info_cache

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)

BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024**3


def _get_process_memory() -> tuple[float, float]:
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / BYTES_PER_MB, mem_info.vms / BYTES_PER_MB
    except Exception as e:
        logger.error(f"Failed to fetch process memory: {e}")
        return 0.0, 0.0


def _check_imessage_access() -> bool:
    try:
        from integrations.imessage import ChatDBReader

        reader = ChatDBReader()
        result = reader.check_access()
        reader.close()
        return result
    except Exception as e:
        logger.error(f"iMessage access check failed: {e}")
        return False


def _get_memory_mode(available_gb: float) -> str:
    if available_gb >= 4.0:
        return "FULL"
    if available_gb >= 2.0:
        return "LITE"
    return "MINIMAL"


def _check_model_loaded() -> bool:
    try:
        from models import get_generator

        generator = get_generator()
        return generator.is_loaded()
    except Exception:
        return False


def _get_model_info() -> ModelInfo | None:
    cache = get_model_info_cache()
    found, cached = cache.get("model_info")
    if found:
        return cached  # type: ignore[no-any-return]

    try:
        from models import get_generator

        generator = get_generator()
        if generator is None:
            return None

        loader = getattr(generator, "_loader", None)
        if loader is None:
            return None

        info = loader.get_current_model_info()
        if info is None:
            return None

        result = ModelInfo(
            id=info.get("id"),
            display_name=info.get("display_name", "Unknown"),
            loaded=info.get("loaded", False),
            memory_usage_mb=info.get("memory_usage_mb", 0.0),
            quality_tier=info.get("quality_tier"),
        )
        cache.set("model_info", result)
        return result
    except (ImportError, AttributeError, KeyError, TypeError):
        return None
    except Exception:
        return None


def _get_recommended_model(total_ram_gb: float) -> str | None:
    try:
        from models import get_recommended_model

        spec = get_recommended_model(total_ram_gb)
        return spec.id
    except Exception:
        return None


@router.get("/health", response_model=HealthResponse, response_model_exclude_unset=True)
@limiter.limit(RATE_LIMIT_READ)
async def get_health(request: Request) -> HealthResponse:
    cache = get_health_cache()
    found, cached = cache.get("health_status")
    if found:
        return cached  # type: ignore[no-any-return]

    memory = psutil.virtual_memory()
    available_gb = memory.available / BYTES_PER_GB
    used_gb = memory.used / BYTES_PER_GB
    total_gb = memory.total / BYTES_PER_GB

    jarvis_rss_mb, jarvis_vms_mb = _get_process_memory()
    imessage_access = await run_in_threadpool(_check_imessage_access)
    memory_mode = _get_memory_mode(available_gb)
    model_loaded = _check_model_loaded()
    model_info = _get_model_info()
    recommended_model = _get_recommended_model(total_gb)

    details: dict[str, str] = {}
    if not imessage_access:
        details["imessage"] = "Full Disk Access required"
    if available_gb < 2.0:
        details["memory"] = f"Low memory: {available_gb:.1f}GB available"

    if not imessage_access:
        status = "unhealthy"
    elif available_gb < 2.0:
        status = "degraded"
    else:
        status = "healthy"

    result = HealthResponse(
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
        model=model_info,
        recommended_model=recommended_model,
        system_ram_gb=round(total_gb, 2),
    )
    cache.set("health_status", result)
    return result


@router.get("/", response_model_exclude_unset=True)
@limiter.limit(RATE_LIMIT_READ)
async def root(request: Request) -> dict[str, str]:
    return {"status": "ok", "service": "jarvis-api"}


@router.get("/health/diagnostic", response_model_exclude_unset=True)
@limiter.limit(RATE_LIMIT_READ)
async def get_diagnostic(request: Request) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    issues: list[str] = []

    checks["memory"] = {
        "status": "ok",
        "available_gb": round(psutil.virtual_memory().available / BYTES_PER_GB, 2),
    }

    try:
        from models import get_generator

        gen = get_generator()
        checks["model"] = {"status": "ok", "loaded": gen.is_loaded()}
    except Exception as e:
        checks["model"] = {"status": "error", "message": str(e)}
        issues.append(f"Model check failed: {e}")

    try:
        from integrations.imessage import ChatDBReader

        reader = ChatDBReader()
        _ = len(reader.get_conversations(limit=1))
        reader.close()
        checks["database"] = {"status": "ok", "accessible": True}
    except Exception as e:
        checks["database"] = {"status": "error", "message": str(e)}
        issues.append(f"Database access failed: {e}")

    status = "healthy" if not issues else "degraded" if len(issues) < 3 else "unhealthy"
    return {"status": status, "checks": checks, "issues": issues}
