"""Debug/trace endpoints related to health and observability."""

from __future__ import annotations

import os
from typing import Any

import psutil
from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from api.ratelimit import RATE_LIMIT_READ, limiter
from api.services.tracing import (
    SystemStatusResponse,
    TraceResponse,
    TraceSummaryResponse,
    clear_trace_store,
    get_trace_store,
)

router = APIRouter(tags=["debug"])
BYTES_PER_MB = 1024 * 1024


@router.get(
    "/debug/traces",
    response_model=list[TraceResponse],
    summary="Get recent request traces",
)
@limiter.limit(RATE_LIMIT_READ)
async def get_traces(request: Request, limit: int = 20) -> list[dict[str, Any]]:
    store = get_trace_store()
    return await run_in_threadpool(store.get_traces, limit)


@router.get(
    "/debug/traces/summary",
    response_model=TraceSummaryResponse,
    summary="Get trace statistics",
)
@limiter.limit(RATE_LIMIT_READ)
async def get_trace_summary(request: Request) -> dict[str, Any]:
    store = get_trace_store()
    return await run_in_threadpool(store.get_summary)


@router.get(
    "/debug/status",
    response_model=SystemStatusResponse,
    summary="Get current system status",
)
@limiter.limit(RATE_LIMIT_READ)
async def get_system_status(request: Request) -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    model_loaded = False
    try:
        from models import get_generator

        gen = get_generator()
        model_loaded = gen.is_loaded()
    except Exception:
        pass

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


@router.delete("/debug/traces", summary="Clear all traces")
async def clear_traces(request: Request) -> dict[str, str]:
    clear_trace_store()
    return {"status": "cleared"}
