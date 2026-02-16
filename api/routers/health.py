"""Health router composition module.

Keeps backward-compatible import path (`api.routers.health.router`) while
splitting readiness and debug concerns into dedicated modules.
"""

from fastapi import APIRouter

from api.routers.health_debug import router as debug_router
from api.routers.health_readiness import router as readiness_router

router = APIRouter()
router.include_router(readiness_router)
router.include_router(debug_router)

__all__ = ["router"]
