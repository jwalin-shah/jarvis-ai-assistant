"""Debug/Trace API endpoints for development observability.

Provides real-time tracing of request flows through the system,
including per-step latencies, memory usage, and input/output data.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any

import psutil
from fastapi import APIRouter
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from api.ratelimit import RATE_LIMIT_READ, limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["debug"])

# Constants
BYTES_PER_MB = 1024 * 1024
MAX_TRACES = 100  # Keep last N traces in memory


@dataclass
class TraceStep:
    """A single step in a request trace."""

    name: str
    start_time: float
    end_time: float | None = None
    input_summary: str = ""
    output_summary: str = ""
    memory_before_mb: float = 0
    memory_after_mb: float = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0
        return (self.end_time - self.start_time) * 1000

    @property
    def memory_delta_mb(self) -> float:
        return self.memory_after_mb - self.memory_before_mb


@dataclass
class RequestTrace:
    """Complete trace for a single request."""

    trace_id: str
    timestamp: datetime
    endpoint: str
    steps: list[TraceStep] = field(default_factory=list)
    total_duration_ms: float = 0
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp.isoformat(),
            "endpoint": self.endpoint,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "success": self.success,
            "error": self.error,
            "steps": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 2),
                    "input": s.input_summary[:200] if s.input_summary else None,
                    "output": s.output_summary[:200] if s.output_summary else None,
                    "memory_before_mb": round(s.memory_before_mb, 1),
                    "memory_after_mb": round(s.memory_after_mb, 1),
                    "memory_delta_mb": round(s.memory_delta_mb, 1),
                    "metadata": s.metadata,
                }
                for s in self.steps
            ],
        }


class TraceStore:
    """Thread-safe storage for request traces."""

    def __init__(self, max_traces: int = MAX_TRACES):
        self._traces: deque[RequestTrace] = deque(maxlen=max_traces)
        self._lock = Lock()
        self._current_trace: RequestTrace | None = None

    def start_trace(self, trace_id: str, endpoint: str) -> RequestTrace:
        """Start a new trace."""
        trace = RequestTrace(
            trace_id=trace_id,
            timestamp=datetime.now(UTC),
            endpoint=endpoint,
        )
        with self._lock:
            self._current_trace = trace
        return trace

    def add_step(
        self,
        name: str,
        input_summary: str = "",
        output_summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TraceStep:
        """Add a step to the current trace."""
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / BYTES_PER_MB

        step = TraceStep(
            name=name,
            start_time=time.perf_counter(),
            input_summary=input_summary,
            memory_before_mb=mem_before,
            metadata=metadata or {},
        )

        with self._lock:
            if self._current_trace:
                self._current_trace.steps.append(step)

        return step

    def end_step(self, step: TraceStep, output_summary: str = "") -> None:
        """End a trace step."""
        step.end_time = time.perf_counter()
        step.output_summary = output_summary

        process = psutil.Process(os.getpid())
        step.memory_after_mb = process.memory_info().rss / BYTES_PER_MB

        # Log to terminal for immediate visibility
        logger.info(
            "TRACE [%s] %.1fms | mem: %.1f→%.1f MB (%+.1f) | in: %s | out: %s",
            step.name,
            step.duration_ms,
            step.memory_before_mb,
            step.memory_after_mb,
            step.memory_delta_mb,
            step.input_summary[:50] if step.input_summary else "-",
            step.output_summary[:50] if step.output_summary else "-",
        )

    def end_trace(self, success: bool = True, error: str | None = None) -> None:
        """Finalize the current trace."""
        with self._lock:
            if self._current_trace:
                self._current_trace.success = success
                self._current_trace.error = error
                if self._current_trace.steps:
                    first = self._current_trace.steps[0].start_time
                    last = self._current_trace.steps[-1].end_time or time.perf_counter()
                    self._current_trace.total_duration_ms = (last - first) * 1000
                self._traces.append(self._current_trace)
                self._current_trace = None

    def get_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent traces."""
        with self._lock:
            traces = list(self._traces)[-limit:]
        return [t.to_dict() for t in reversed(traces)]

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        with self._lock:
            traces = list(self._traces)

        if not traces:
            return {"total_traces": 0, "avg_duration_ms": 0, "success_rate": 0}

        durations = [t.total_duration_ms for t in traces]
        successes = sum(1 for t in traces if t.success)

        return {
            "total_traces": len(traces),
            "avg_duration_ms": round(sum(durations) / len(durations), 2),
            "min_duration_ms": round(min(durations), 2),
            "max_duration_ms": round(max(durations), 2),
            "success_rate": round(successes / len(traces) * 100, 1),
        }


# Global trace store
_trace_store: TraceStore | None = None


def get_trace_store() -> TraceStore:
    """Get the global trace store (singleton)."""
    global _trace_store
    if _trace_store is None:
        _trace_store = TraceStore()
    return _trace_store


# Response models
class TraceStepResponse(BaseModel):
    name: str
    duration_ms: float
    input: str | None
    output: str | None
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    metadata: dict[str, Any]


class TraceResponse(BaseModel):
    trace_id: str
    timestamp: str
    endpoint: str
    total_duration_ms: float
    success: bool
    error: str | None
    steps: list[TraceStepResponse]


class TraceSummaryResponse(BaseModel):
    total_traces: int
    avg_duration_ms: float
    min_duration_ms: float = 0
    max_duration_ms: float = 0
    success_rate: float


class SystemStatusResponse(BaseModel):
    memory_rss_mb: float = Field(..., description="Current RSS memory usage")
    memory_vms_mb: float = Field(..., description="Current VMS memory usage")
    memory_percent: float = Field(..., description="Memory usage as % of system")
    cpu_percent: float = Field(..., description="CPU usage %")
    model_loaded: bool = Field(..., description="Whether LLM is loaded")
    embedding_service: bool = Field(..., description="Whether embedding service is available")


class GenerationLogResponse(BaseModel):
    id: int
    chat_id: str | None
    contact_id: str | None
    incoming_text: str | None
    classification_json: str | None
    rag_context_json: str | None
    final_prompt: str | None
    response_text: str | None
    confidence: float | None
    metadata_json: str | None
    created_at: datetime


@router.get(
    "/traces",
    response_model=list[TraceResponse],
    summary="Get recent request traces",
)
@limiter.limit(RATE_LIMIT_READ)
async def get_traces(
    request: Request,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Get the last N request traces with full step breakdowns."""
    store = get_trace_store()
    return await run_in_threadpool(store.get_traces, limit)


@router.get(
    "/generation-logs",
    response_model=list[GenerationLogResponse],
    summary="Get persistent generation logs",
)
@limiter.limit(RATE_LIMIT_READ)
async def get_generation_logs(
    request: Request,
    limit: int = 20,
    chat_id: str | None = None,
) -> list[dict[str, Any]]:
    """Get recent generation logs from the database for full traceability.

    These logs include the original input, RAG context, final prompt, and response.
    """
    from jarvis.db import get_db

    db = get_db()
    return await run_in_threadpool(db.get_recent_reply_logs, limit, chat_id)


@router.get(
    "/traces/summary",
    response_model=TraceSummaryResponse,
    summary="Get trace statistics",
)
@limiter.limit(RATE_LIMIT_READ)
async def get_trace_summary(request: Request) -> dict[str, Any]:
    """Get summary statistics for recent traces."""
    store = get_trace_store()
    return await run_in_threadpool(store.get_summary)


@router.get(
    "/status",
    response_model=SystemStatusResponse,
    summary="Get current system status",
)
@limiter.limit(RATE_LIMIT_READ)
async def get_system_status(request: Request) -> dict[str, Any]:
    """Get current system resource usage and component status."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    # Check model status
    model_loaded = False
    try:
        from models import get_generator

        gen = get_generator()
        model_loaded = gen.is_loaded()
    except Exception:
        pass

    # Check embedding service
    embedding_available = False
    try:
        from models.bert_embedder import is_mlx_available

        embedding_available = is_mlx_available()
    except Exception:
        pass

    return {
        "memory_rss_mb": round(mem_info.rss / BYTES_PER_MB, 1),
        "memory_vms_mb": round(mem_info.vms / BYTES_PER_MB, 1),
        "memory_percent": round(process.memory_percent(), 1),
        "cpu_percent": round(process.cpu_percent(), 1),
        "model_loaded": model_loaded,
        "embedding_service": embedding_available,
    }


@router.delete(
    "/traces",
    summary="Clear all traces",
)
async def clear_traces(request: Request) -> dict[str, str]:
    """Clear all stored traces."""
    global _trace_store
    _trace_store = TraceStore()
    return {"status": "cleared"}
