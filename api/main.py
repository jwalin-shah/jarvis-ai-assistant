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

- **📱 iMessage Access**: Read conversations, messages, attachments, and reactions
- **🤖 AI-Powered Replies**: Generate contextual reply suggestions using local LLM
- **📝 Conversation Summaries**: Get AI-generated summaries of long conversations
- **⚡ Quick Suggestions**: Fast pattern-based reply suggestions (no model required)
- **⚙️ Settings Management**: Configure models, generation parameters, and behavior
- **🏥 Health Monitoring**: System health status, memory usage, and permissions

## Authentication

This API is designed for local use by the JARVIS desktop application.
No authentication is required as the API only binds to localhost.

## Rate Limiting

Rate limiting is applied to protect system resources:
- **Read endpoints** (GET, search): 60 requests per minute
- **Write endpoints** (POST, PUT): 30 requests per minute
- **Generation endpoints** (AI-powered): 10 requests per minute

Exceeding these limits returns HTTP 429 with a `Retry-After` header.

## Error Handling

All errors return a JSON response with the following structure:
```json
{
    "error": "Brief error message",
    "detail": "Detailed explanation",
    "code": "MACHINE_READABLE_CODE"
}
```

## Privacy

All data processing happens locally on your Mac. No conversation data,
messages, or personal information is ever sent to external servers.

## Getting Started

1. Ensure Full Disk Access is granted for iMessage database access
2. Check system health: `GET /health`
3. List conversations: `GET /conversations`
4. Generate reply suggestions: `POST /drafts/reply`
"""


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
    """Lifecycle event handler for the FastAPI application."""
    # Start model warmer
    from jarvis.model_warmer import get_model_warmer

    get_model_warmer().start()

    yield

    # Stop model warmer
    get_model_warmer().stop()


API_TAGS_METADATA = [
    {
        "name": "health",
        "description": "System health monitoring and status checks. "
        "Use these endpoints to verify the API is running and check system resources.",
    },
    {
        "name": "conversations",
        "description": "iMessage conversation and message management. "
        "List conversations, retrieve messages, search, and send messages.",
    },
    {
        "name": "search",
        "description": "Semantic search using AI-powered embeddings. "
        "Find messages by meaning rather than exact keyword matching.",
    },
    {
        "name": "drafts",
        "description": "AI-powered draft generation using the local MLX language model. "
        "Generate reply suggestions and conversation summaries.",
    },
    {
        "name": "embeddings",
        "description": "Semantic search and relationship profiling using message embeddings. "
        "Index conversations, search by meaning, and analyze communication patterns.",
    },
    {
        "name": "suggestions",
        "description": "Fast pattern-based reply suggestions. "
        "Lightweight alternative to AI drafts for common response scenarios.",
    },
    {
        "name": "settings",
        "description": "Application configuration management. "
        "Manage model selection, generation parameters, and behavior preferences.",
    },
    {
        "name": "custom-templates",
        "description": "User-defined template management. "
        "Create, edit, test, and organize custom response templates. "
        "Supports import/export for sharing template packs.",
    },
    {
        "name": "topics",
        "description": "Automatic topic detection for conversations. "
        "Analyzes message content to identify common themes like scheduling and questions.",
    },
    {
        "name": "stats",
        "description": "Conversation statistics and analytics. "
        "Get insights on messaging patterns, word frequency, activity by hour/day, "
        "response times, emoji usage, and attachment breakdowns.",
    },
    {
        "name": "insights",
        "description": "Advanced conversation insights and relationship analytics. "
        "Provides sentiment analysis over time, response time patterns, message frequency "
        "trends, and relationship health scores.",
    },
    {
        "name": "analytics",
        "description": "Comprehensive conversation and template analytics. "
        "Dashboard metrics, time-series data, and template matching optimization.",
    },
    {
        "name": "websocket",
        "description": "Real-time WebSocket communication. "
        "Stream model generation responses, receive health updates, and manage live connections.",
    },
    {
        "name": "tasks",
        "description": "Background task management. "
        "Create, monitor, and manage background tasks for batch operations.",
    },
    {
        "name": "batch",
        "description": "Batch operations for bulk processing. "
        "Export, summarize, and generate replies for multiple conversations at once.",
    },
    {
        "name": "priority",
        "description": "Smart priority inbox for message importance scoring. "
        "Get messages sorted by urgency and detect questions and action items.",
    },
    {
        "name": "calendars",
        "description": "Calendar integration. "
        "Detect events in messages and create calendar entries via macOS Calendar.",
    },
    {
        "name": "threads",
        "description": "Conversation threading for organizing messages into logical groups. "
        "Detect threads based on time gaps, topic shifts, and explicit reply references.",
    },
    {
        "name": "attachments",
        "description": "iMessage attachment management. "
        "List, download, and analyze file attachments from conversations.",
    },
    {
        "name": "feedback",
        "description": "User feedback tracking and response evaluation. "
        "Record feedback on suggestions, track acceptance rates, and get improvement insights.",
    },
    {
        "name": "experiments",
        "description": "A/B testing infrastructure for prompt experimentation. "
        "Create experiments, assign variants to contacts, record outcomes, and analyze results.",
    },
    {
        "name": "relationships",
        "description": "Relationship learning and communication profiling. "
        "Build and manage profiles that capture communication patterns with each contact "
        "for personalized reply generation.",
    },
    {
        "name": "debug",
        "description": "Debug and tracing endpoints for development observability. "
        "Request traces, generation logs, and system status for troubleshooting.",
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
        """Generate custom OpenAPI schema with enhanced documentation."""
        if app_instance.openapi_schema:
            return app_instance.openapi_schema

        openapi_schema = get_openapi(
            title=API_TITLE,
            version=API_VERSION,
            description=API_DESCRIPTION,
            routes=app_instance.routes,
            tags=API_TAGS_METADATA,
        )

        # Add contact and license info
        openapi_schema["info"]["contact"] = API_CONTACT
        openapi_schema["info"]["license"] = API_LICENSE

        # Add server information
        openapi_schema["servers"] = [
            {
                "url": "http://localhost:8742",
                "description": "Local development server",
            },
            {
                "url": "http://127.0.0.1:8742",
                "description": "Local development server (alternative)",
            },
        ]

        # Add external documentation
        openapi_schema["externalDocs"] = {
            "description": "Full documentation and user guide",
            "url": "https://github.com/jwalinshah/jarvis-ai-assistant/docs",
        }

        app_instance.openapi_schema = openapi_schema
        return app_instance.openapi_schema

    return custom_openapi


def _configure_middleware(app_instance: FastAPI) -> None:
    """Configure middleware for the FastAPI application."""
    # Configure CORS for Tauri and development
    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "tauri://localhost",  # Tauri production
            "http://localhost:5173",  # Vite dev server
            "http://localhost:1420",  # Tauri dev server
            "http://127.0.0.1:5173",
            "http://127.0.0.1:1420",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Enable GZip compression for responses (50-75% bandwidth savings for large JSON responses)
    app_instance.add_middleware(GZipMiddleware, minimum_size=500)

    # Rate limiting middleware (applies default_limits to all routes)
    app_instance.add_middleware(SlowAPIMiddleware)

    # Request timing middleware
    @app_instance.middleware("http")
    async def metrics_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Middleware to track request timing and counts."""
        start_time = time.perf_counter()

        # Process the request
        response = await call_next(request)

        # Record metrics
        duration = time.perf_counter() - start_time
        endpoint = request.url.path
        method = request.method

        # Skip metrics endpoints to avoid infinite recursion in monitoring
        if not endpoint.startswith("/metrics"):
            counter = get_request_counter()
            histogram = get_latency_histogram()

            counter.increment(endpoint, method)
            histogram.observe(f"{method} {endpoint}", duration)

        # Add timing header
        response.headers["X-Response-Time"] = f"{duration:.4f}s"

        return response


def _register_routers(app_instance: FastAPI) -> None:
    """Register all API routers explicitly.

    Imports routers on-demand to avoid eager loading at module import time.
    This improves testability and reduces import-time coupling.
    """
    # Import routers locally to avoid module-level coupling
    from api.routers.analytics import router as analytics_router
    from api.routers.attachments import router as attachments_router
    from api.routers.batch import router as batch_router
    from api.routers.calendar import router as calendar_router
    from api.routers.contacts import router as contacts_router
    from api.routers.conversations import router as conversations_router
    from api.routers.custom_templates import router as custom_templates_router
    from api.routers.drafts import router as drafts_router

    # from api.routers.embeddings import router as embeddings_router  # Missing
    from api.routers.experiments import router as experiments_router
    from api.routers.export import router as export_router
    from api.routers.feedback import router as feedback_router
    from api.routers.graph import router as graph_router
    from api.routers.health import router as health_router
    from api.routers.metrics import router as metrics_router
    from api.routers.priority import router as priority_router

    # from api.routers.relationships import router as relationships_router  # Missing
    from api.routers.search import router as search_router
    from api.routers.settings import router as settings_router
    from api.routers.stats import router as stats_router
    from api.routers.suggestions import router as suggestions_router
    from api.routers.tasks import router as tasks_router
    from api.routers.threads import router as threads_router

    # from api.routers.topics import router as topics_router  # Missing
    from api.routers.websocket import router as websocket_router

    # Register routers in logical order
    app_instance.include_router(health_router)
    app_instance.include_router(attachments_router)
    app_instance.include_router(calendar_router)
    app_instance.include_router(contacts_router)
    app_instance.include_router(conversations_router)
    app_instance.include_router(custom_templates_router)
    app_instance.include_router(drafts_router)
    # app_instance.include_router(embeddings_router)  # Missing
    app_instance.include_router(export_router)
    app_instance.include_router(search_router)
    app_instance.include_router(suggestions_router)
    app_instance.include_router(settings_router)
    app_instance.include_router(stats_router)
    app_instance.include_router(metrics_router)
    app_instance.include_router(threads_router)
    # app_instance.include_router(topics_router)  # Missing
    app_instance.include_router(websocket_router)
    app_instance.include_router(tasks_router)
    app_instance.include_router(batch_router)
    app_instance.include_router(priority_router)
    app_instance.include_router(feedback_router)
    app_instance.include_router(experiments_router)
    # app_instance.include_router(relationships_router)  # Missing
    app_instance.include_router(analytics_router)
    app_instance.include_router(graph_router)


def create_app() -> FastAPI:
    """Application factory for creating configured FastAPI instances.

    This factory pattern allows:
    - Explicit router registration without import-time coupling
    - Easy creation of test apps with subset of routers
    - Better testability and modularity

    Returns:
        Configured FastAPI application instance
    """
    # Create FastAPI app
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

    # Configure rate limiting
    app_instance.state.limiter = limiter
    app_instance.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # Use custom OpenAPI schema
    app_instance.openapi = _create_openapi_generator(app_instance)  # type: ignore[method-assign]

    # Configure middleware
    _configure_middleware(app_instance)

    # Register all routers
    _register_routers(app_instance)

    # Register JARVIS exception handlers for standardized error responses
    register_exception_handlers(app_instance)

    return app_instance


# Create the default app instance for backward compatibility
# This allows existing code to import `app` from `api.main`
app = create_app()
