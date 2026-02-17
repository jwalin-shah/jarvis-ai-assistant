from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from jarvis.handlers.base import BaseHandler, rpc_handler

if TYPE_CHECKING:
    pass

# Cache health status for 5 seconds to avoid expensive lookups
_HEALTH_CACHE_TTL = 5.0
_cached_health: dict[str, Any] | None = None
_cache_timestamp: float = 0.0


class HealthHandler(BaseHandler):
    """Handler for health-related RPC methods."""

    def register(self) -> None:
        """Register health-related RPC methods."""
        self.server.register("ping", self._ping)
        self.server.register("get_health", self._get_health)

    @rpc_handler("Ping failed")
    async def _ping(self) -> dict[str, Any]:
        """Simple health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "models_ready": self.server.models_ready,
        }

    @rpc_handler("Failed to fetch full health status")
    async def _get_health(self) -> dict[str, Any]:
        """Comprehensive health check endpoint with caching.

        Returns:
            Dict with system status, memory, permissions, and model info.
        """
        global _cached_health, _cache_timestamp

        # Return cached result if still valid
        now = time.time()
        if _cached_health is not None and (now - _cache_timestamp) < _HEALTH_CACHE_TTL:
            return _cached_health

        import psutil
        from starlette.concurrency import run_in_threadpool

        from api.routers.health_readiness import (
            _check_imessage_access,
            _check_model_loaded,
            _get_memory_mode,
            _get_model_info,
            _get_process_memory,
            _get_recommended_model,
        )

        bytes_per_gb = 1024**3

        memory = psutil.virtual_memory()
        available_gb = memory.available / bytes_per_gb
        used_gb = memory.used / bytes_per_gb
        total_gb = memory.total / bytes_per_gb

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

        status = "healthy"
        if not imessage_access:
            status = "unhealthy"
        elif available_gb < 2.0:
            status = "degraded"

        # Construct response matching HealthResponse schema
        result = {
            "status": status,
            "imessage_access": imessage_access,
            "memory_available_gb": round(available_gb, 2),
            "memory_used_gb": round(used_gb, 2),
            "memory_mode": memory_mode,
            "model_loaded": model_loaded,
            "permissions_ok": imessage_access,
            "details": details if details else None,
            "jarvis_rss_mb": round(jarvis_rss_mb, 1),
            "jarvis_vms_mb": round(jarvis_vms_mb, 1),
            "recommended_model": recommended_model,
            "system_ram_gb": round(total_gb, 2),
        }

        # Add model info if available
        if model_info:
            result["model"] = {
                "id": model_info.id,
                "display_name": model_info.display_name,
                "loaded": model_info.loaded,
                "memory_usage_mb": model_info.memory_usage_mb,
                "quality_tier": model_info.quality_tier,
            }

        # Cache the result
        _cached_health = result
        _cache_timestamp = now

        return result
