"""Insights API endpoints.

Provides conversation analytics, statistics, and digest generation.
This module consolidates insights, stats, and digest routers.
"""

# Re-export all endpoints from the original routers
from api.routers.digest import router as digest_router
from api.routers.stats import router as stats_router

# Export the routers for use in main.py
# Note: insights_router was removed to fix circular import
router = None  # Placeholder - individual routers are used directly

__all__ = [
    "stats_router",
    "digest_router",
]
