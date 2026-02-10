"""API routers for JARVIS endpoints."""

from .analytics import router as analytics_router
from .attachments import router as attachments_router
from .batch import router as batch_router
from .calendar import router as calendar_router
from .contacts import router as contacts_router
from .conversations import router as conversations_router
from .custom_templates import router as custom_templates_router
from .debug import router as debug_router
from .drafts import router as drafts_router
from .embeddings import router as embeddings_router
from .experiments import router as experiments_router
from .export import router as export_router
from .feedback import router as feedback_router
from .graph import router as graph_router
from .health import router as health_router
from .metrics import router as metrics_router
from .pdf_export import router as pdf_export_router
from .priority import router as priority_router
from .relationships import router as relationships_router
from .scheduler import router as scheduler_router
from .search import router as search_router
from .settings import router as settings_router
from .stats import router as stats_router
from .suggestions import router as suggestions_router
from .tags import router as tags_router
from .tasks import router as tasks_router
from .template_analytics import router as template_analytics_router
from .threads import router as threads_router
from .topics import router as topics_router
from .websocket import router as websocket_router

__all__ = [
    "analytics_router",
    "attachments_router",
    "batch_router",
    "calendar_router",
    "contacts_router",
    "conversations_router",
    "custom_templates_router",
    "debug_router",
    "drafts_router",
    "embeddings_router",
    "experiments_router",
    "export_router",
    "feedback_router",
    "graph_router",
    "health_router",
    "metrics_router",
    "pdf_export_router",
    "priority_router",
    "relationships_router",
    "scheduler_router",
    "search_router",
    "settings_router",
    "stats_router",
    "suggestions_router",
    "tags_router",
    "tasks_router",
    "template_analytics_router",
    "threads_router",
    "topics_router",
    "websocket_router",
]
