"""Tracing service for debug endpoints.

Provides request trace data models, in-memory storage, and response schemas.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any

import psutil
from pydantic import BaseModel, Field

MAX_TRACES = 100
BYTES_PER_MB = 1024 * 1024


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
        self._active_traces: dict[str, RequestTrace] = {}

    def start_trace(self, trace_id: str, endpoint: str) -> RequestTrace:
        trace = RequestTrace(
            trace_id=trace_id,
            timestamp=datetime.now(UTC),
            endpoint=endpoint,
        )
        with self._lock:
            self._active_traces[trace_id] = trace
        return trace

    def add_step(
        self,
        name: str,
        trace_id: str | None = None,
        input_summary: str = "",
        output_summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TraceStep:
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
            target_trace: RequestTrace | None = None
            if trace_id is not None:
                target_trace = self._active_traces.get(trace_id)
            elif len(self._active_traces) == 1:
                target_trace = next(iter(self._active_traces.values()))

            if target_trace is not None:
                target_trace.steps.append(step)

        return step

    def end_step(self, step: TraceStep, output_summary: str = "") -> None:
        step.end_time = time.perf_counter()
        step.output_summary = output_summary

        process = psutil.Process(os.getpid())
        step.memory_after_mb = process.memory_info().rss / BYTES_PER_MB

    def end_trace(
        self,
        trace_id: str | bool | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        # Backward compatibility: historical signature was end_trace(success, error).
        if isinstance(trace_id, bool):
            success = trace_id
            trace_id = None

        with self._lock:
            target_trace: RequestTrace | None = None
            if trace_id is not None:
                target_trace = self._active_traces.pop(trace_id, None)
            elif len(self._active_traces) == 1:
                only_trace_id = next(iter(self._active_traces))
                target_trace = self._active_traces.pop(only_trace_id, None)

            if target_trace is None:
                return

            target_trace.success = success
            target_trace.error = error
            if target_trace.steps:
                first = target_trace.steps[0].start_time
                last = target_trace.steps[-1].end_time or time.perf_counter()
                target_trace.total_duration_ms = (last - first) * 1000
            self._traces.append(target_trace)

    def get_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            traces = list(self._traces)[-limit:]
        return [t.to_dict() for t in reversed(traces)]

    def get_summary(self) -> dict[str, Any]:
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


_trace_store: TraceStore | None = None


def get_trace_store() -> TraceStore:
    global _trace_store
    if _trace_store is None:
        _trace_store = TraceStore()
    return _trace_store


def clear_trace_store() -> None:
    global _trace_store
    _trace_store = TraceStore()
