"""Arize Phoenix Local Observability Configuration.

Starts a local Phoenix server and configures OpenTelemetry tracing
for local-first AI observability.
"""

import logging
import phoenix as px
from phoenix.otel import register
from opentelemetry import trace

logger = logging.getLogger(__name__)

def initialize_phoenix():
    """Start Phoenix server and register the tracer provider."""
    try:
        # 1. Start the local Phoenix server (default localhost:6006)
        session = px.launch_app()
        logger.info(f"Arize Phoenix dashboard: {session.url}")
        
        # 2. Register Phoenix as the OpenTelemetry collector
        # This configures the global trace provider to send spans to Phoenix
        register()
        
        return session
    except Exception as e:
        logger.error(f"Failed to initialize Arize Phoenix: {e}")
        return None

def get_tracer(name: str):
    """Get a tracer instance for manual instrumentation."""
    return trace.get_tracer(name)
