"""OpenTelemetry tracing instrumentation for JARVIS.

Provides distributed tracing for key operations:
- Reply generation pipeline
- Search operations
- Classification
- LLM generation

Usage:
    from jarvis.observability.tracing import get_tracer, traced
    
    @traced("my_operation")
    def my_function():
        ...
    
    # Or manual tracing:
    tracer = get_tracer()
    with tracer.start_as_current_span("operation") as span:
        span.set_attribute("key", "value")
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)

# Global tracer instance
_tracer: trace.Tracer | None = None
_tracer_initialized = False

F = TypeVar("F", bound=Callable[..., Any])


def _init_tracer() -> trace.Tracer | None:
    """Initialize the OpenTelemetry tracer.
    
    Returns:
        Tracer instance or None if disabled.
    """
    global _tracer, _tracer_initialized
    
    if _tracer_initialized:
        return _tracer
    
    _tracer_initialized = True
    
    # Check if tracing is disabled via env var
    if os.getenv("JARVIS_DISABLE_TRACING", "").lower() in ("1", "true", "yes"):
        logger.debug("Tracing disabled via JARVIS_DISABLE_TRACING")
        return None
    
    try:
        # Create resource identifying this service
        resource = Resource.create(
            {
                "service.name": "jarvis-assistant",
                "service.version": "1.0.0",
            }
        )
        
        # Create provider
        provider = TracerProvider(resource=resource)
        
        # Add console exporter for development/debugging
        # In production, you'd use OTLP exporter to send to Jaeger/Tempo/etc
        if os.getenv("JARVIS_TRACE_CONSOLE", "").lower() in ("1", "true", "yes"):
            processor = BatchSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)
            logger.info("OpenTelemetry tracing initialized with console exporter")
        
        # Set as global provider
        trace.set_tracer_provider(provider)
        
        _tracer = trace.get_tracer("jarvis")
        return _tracer
        
    except Exception as e:
        logger.warning(f"Failed to initialize OpenTelemetry tracer: {e}")
        return None


def get_tracer() -> trace.Tracer | None:
    """Get the global tracer instance.
    
    Returns:
        Tracer instance or None if tracing is disabled.
    """
    global _tracer
    if _tracer is None and not _tracer_initialized:
        _tracer = _init_tracer()
    return _tracer


def traced(
    operation_name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Decorator to trace a function.
    
    Args:
        operation_name: Name of the operation (defaults to function name).
        attributes: Static attributes to add to the span.
    
    Example:
        @traced("generate_reply", attributes={"component": "reply_service"})
        def generate_reply(...):
            ...
    """
    def decorator(func: F) -> F:
        nonlocal operation_name
        if operation_name is None:
            operation_name = func.__name__
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            if tracer is None:
                return func(*args, **kwargs)
            
            with tracer.start_as_current_span(operation_name) as span:
                # Add static attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                # Add function arguments as attributes (safely)
                try:
                    for i, arg in enumerate(args):
                        if isinstance(arg, (str, int, float, bool)):
                            span.set_attribute(f"arg.{i}", arg)
                    for key, value in kwargs.items():
                        if isinstance(value, (str, int, float, bool)):
                            span.set_attribute(f"kwarg.{key}", value)
                except Exception:
                    pass  # Don't fail if attribute setting fails
                
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper  # type: ignore[return-value]
    
    return decorator


class SpanContext:
    """Context manager for manual span creation.
    
    Example:
        with SpanContext("my_operation", {"key": "value"}) as span:
            # Do work
            span.set_attribute("result", "success")
    """
    
    def __init__(
        self,
        operation_name: str,
        attributes: dict[str, Any] | None = None,
    ):
        self.operation_name = operation_name
        self.attributes = attributes or {}
        self.span = None
        self._tracer = get_tracer()
    
    def __enter__(self) -> trace.Span:
        if self._tracer is None:
            # Return a no-op span
            return trace.NonRecordingSpan(trace.INVALID_SPAN_CONTEXT)
        
        self.span = self._tracer.start_span(self.operation_name)
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
        return self.span
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.span is not None:
            if exc_val is not None:
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
            else:
                self.span.set_status(Status(StatusCode.OK))
            self.span.end()


def add_span_attribute(key: str, value: Any) -> None:
    """Add an attribute to the current span.
    
    Args:
        key: Attribute name.
        value: Attribute value (str, int, float, bool).
    """
    current_span = trace.get_current_span()
    if current_span is not None:
        try:
            current_span.set_attribute(key, value)
        except Exception:
            pass
