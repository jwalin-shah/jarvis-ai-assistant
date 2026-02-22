"""FastAPI application for JARVIS desktop frontend.

Provides REST API for the Tauri desktop app to access iMessage data,
generate AI-powered replies, manage settings, and monitor system health.

Usage:
    uvicorn api.main:app --reload --port 8742

Documentation:
    - Swagger UI: http://localhost:8742/docs
    - ReDoc: http://localhost:8742/redoc
    - OpenAPI JSON: http://localhost:8742/openapi.json
"""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.utils import get_openapi
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.errors import register_exception_handlers
from api.ratelimit import limiter, rate_limit_exceeded_handler
from jarvis.metrics import get_latency_histogram, get_request_counter

# API metadata for OpenAPI documentation
API_TITLE = "JARVIS API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
# JARVIS - Intelligent iMessage Assistant API

JARVIS is a local-first AI assistant for macOS that provides intelligent iMessage
management using MLX-based language models. All processing runs entirely on Apple
Silicon with no cloud data transmission.

## Features

- **iMessage Access**: Read conversations, messages, attachments, and reactions
- **AI-Powered Replies**: Generate contextual reply suggestions using local LLM
- **Conversation Summaries**: Get AI-generated summaries of long conversations
- **Quick Suggestions**: Fast pattern-based reply suggestions (no model required)
- **Settings Management**: Configure models, generation parameters, and behavior
- **Health Monitoring**: System health status, memory usage, and permissions

## Authentication

This API is designed for local use by the JARVIS desktop application.
No authentication is required as the API only binds to localhost.

## Privacy

All data processing happens locally on your Mac. No conversation data,
messages, or personal information is ever sent to external servers.
"""


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
    """Lifecycle event handler for the FastAPI application."""
    import asyncio

    from jarvis.interfaces.desktop.server import JarvisSocketServer
    from jarvis.model_warmer import get_model_warmer
    from jarvis.tasks.worker import start_worker, stop_worker

    socket_server = JarvisSocketServer(
        enable_watcher=True,
        preload_models=False,
        enable_prefetch=True,
    )

    socket_task = asyncio.create_task(socket_server.serve_forever())
    app_instance.state.socket_server = socket_server

    get_model_warmer().start()
    start_worker()

    yield

    stop_worker()
    get_model_warmer().stop()

    await socket_server.stop()
    socket_task.cancel()
    try:
        await socket_task
    except asyncio.CancelledError:
        pass


API_TAGS_METADATA = [
    {
        "name": "health",
        "description": "System health monitoring and status checks.",
    },
    {
        "name": "conversations",
        "description": "iMessage conversation and message management.",
    },
    {
        "name": "search",
        "description": "Semantic search using AI-powered embeddings.",
    },
    {
        "name": "drafts",
        "description": "AI-powered draft generation using the local MLX language model.",
    },
    {
        "name": "contacts",
        "description": "Contact management and relationship profiling.",
    },
    {
        "name": "settings",
        "description": "Application configuration management.",
    },
    {
        "name": "websocket",
        "description": "Real-time WebSocket communication and streaming.",
    },
    {
        "name": "attachments",
        "description": "iMessage attachment management.",
    },
    {
        "name": "calendars",
        "description": "Calendar integration.",
    },
    {
        "name": "tasks",
        "description": "Background task management.",
    },
    {
        "name": "suggestions",
        "description": "Fast pattern-based reply suggestions.",
    },
]

API_CONTACT = {
    "name": "JARVIS Support",
    "url": "https://github.com/jwalinshah/jarvis-ai-assistant",
}

API_LICENSE = {
    "name": "MIT License",
    "url": "https://opensource.org/licenses/MIT",
}


def _create_openapi_generator(app_instance: FastAPI) -> callable:  # type: ignore[valid-type]
    """Create a custom OpenAPI schema generator for the given app instance."""

    def custom_openapi() -> dict[str, Any]:
        if app_instance.openapi_schema:
            return app_instance.openapi_schema

        openapi_schema = get_openapi(
            title=API_TITLE,
            version=API_VERSION,
            description=API_DESCRIPTION,
            routes=app_instance.routes,
            tags=API_TAGS_METADATA,
        )

        openapi_schema["info"]["contact"] = API_CONTACT
        openapi_schema["info"]["license"] = API_LICENSE
        openapi_schema["servers"] = [
            {"url": "http://localhost:8742", "description": "Local development server"},
        ]

        app_instance.openapi_schema = openapi_schema
        return app_instance.openapi_schema

    return custom_openapi


def _configure_middleware(app_instance: FastAPI) -> None:
    """Configure middleware for the FastAPI application."""
    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "tauri://localhost",
            "http://localhost:5173",
            "http://localhost:1420",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:1420",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app_instance.add_middleware(GZipMiddleware, minimum_size=500)
    app_instance.add_middleware(SlowAPIMiddleware)

    @app_instance.middleware("http")
    async def metrics_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        start_time = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        endpoint = request.url.path
        method = request.method

        if not endpoint.startswith("/metrics"):
            counter = get_request_counter()
            histogram = get_latency_histogram()
            counter.increment(endpoint, method)
            histogram.observe(f"{method} {endpoint}", duration)

        response.headers["X-Response-Time"] = f"{duration:.4f}s"
        return response


def _register_routers(app_instance: FastAPI) -> None:
    """Register API routers.

    Core routers are always loaded. Non-essential routers (analytics,
    experiments, graph, etc.) are commented out to reduce startup time
    and attack surface. Uncomment as needed.
    """
    # --- Core routers (always loaded) ---
    from api.routers.attachments import router as attachments_router
    from api.routers.calendar import router as calendar_router
    from api.routers.contacts import router as contacts_router
    from api.routers.conversations import router as conversations_router
    from api.routers.drafts import router as drafts_router
    from api.routers.health import router as health_router
    from api.routers.search import router as search_router
    from api.routers.settings import router as settings_router
    from api.routers.suggestions import router as suggestions_router
    from api.routers.tasks import router as tasks_router
    from api.routers.websocket import router as websocket_router

    app_instance.include_router(health_router)
    app_instance.include_router(conversations_router)
    app_instance.include_router(drafts_router)
    app_instance.include_router(search_router)
    app_instance.include_router(contacts_router)
    app_instance.include_router(settings_router)
    app_instance.include_router(websocket_router)
    app_instance.include_router(attachments_router)
    app_instance.include_router(calendar_router)
    app_instance.include_router(suggestions_router)
    app_instance.include_router(tasks_router)

    # --- Non-essential routers (uncomment as needed) ---
    # from api.routers.analytics import router as analytics_router
    # from api.routers.batch import router as batch_router
    # from api.routers.custom_templates import router as custom_templates_router
    # from api.routers.experiments import router as experiments_router
    # from api.routers.export import router as export_router
    # from api.routers.feedback import router as feedback_router
    # from api.routers.graph import router as graph_router
    # from api.routers.metrics import router as metrics_router
    # from api.routers.priority import router as priority_router
    # from api.routers.stats import router as stats_router
    # from api.routers.threads import router as threads_router
    # app_instance.include_router(analytics_router)
    # app_instance.include_router(batch_router)
    # app_instance.include_router(custom_templates_router)
    # app_instance.include_router(experiments_router)
    # app_instance.include_router(export_router)
    # app_instance.include_router(feedback_router)
    # app_instance.include_router(graph_router)
    # app_instance.include_router(metrics_router)
    # app_instance.include_router(priority_router)
    # app_instance.include_router(stats_router)
    # app_instance.include_router(threads_router)


def create_app() -> FastAPI:
    """Application factory for creating configured FastAPI instances."""
    app_instance = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=API_TAGS_METADATA,
        contact=API_CONTACT,
        license_info=API_LICENSE,
        lifespan=lifespan,
    )

    app_instance.state.limiter = limiter
    app_instance.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore[arg-type]
    app_instance.openapi = _create_openapi_generator(app_instance)  # type: ignore[method-assign]
    _configure_middleware(app_instance)
    _register_routers(app_instance)
    register_exception_handlers(app_instance)

    return app_instance


app = create_app()
