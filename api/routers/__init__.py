"""API routers for JARVIS endpoints.

Routers are imported lazily within the app factory pattern (api.main.create_app)
to reduce import-time coupling and improve testability.

Individual routers can be imported directly from their modules:
    from api.routers.health import router as health_router
"""

__all__ = [
    "analytics",
    "attachments",
    "batch",
    "calendar",
    "contacts",
    "conversations",
    "custom_templates",
    "drafts",
    "experiments",
    "export",
    "feedback",
    "graph",
    "health",
    "metrics",
    "priority",
    "search",
    "settings",
    "stats",
    "suggestions",
    "tasks",
    "threads",
    "websocket",
]
