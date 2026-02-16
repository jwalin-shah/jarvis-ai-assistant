from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from jarvis.observability.metrics_router import (
    RoutingMetrics,
    get_routing_metrics_store,
    hash_query,
)

if TYPE_CHECKING:
    from jarvis.embedding_adapter import CachedEmbedder

logger = logging.getLogger(__name__)


def record_routing_metrics(
    incoming: str,
    decision: str,
    similarity_score: float,
    latency_ms: dict[str, float],
    cached_embedder: CachedEmbedder,
    vec_candidates: int,
    model_loaded: bool,
) -> None:
    """Record detailed routing metrics."""
    try:
        metrics = RoutingMetrics(
            timestamp=time.time(),
            query_hash=hash_query(incoming),
            latency_ms=latency_ms,
            embedding_computations=cached_embedder.embedding_computations,
            vec_candidates=vec_candidates,
            routing_decision=decision,
            similarity_score=similarity_score,
            cache_hit=cached_embedder.cache_hit,
            model_loaded=model_loaded,
            generation_time_ms=latency_ms.get("generation", 0.0),
            tokens_per_second=0.0,
        )
        get_routing_metrics_store().record(metrics)
    except Exception as e:
        logger.debug("Metrics write failed: %s", e)


def record_rpc_latency(method: str, elapsed_ms: float) -> None:
    """Record RPC call latency to the global latency tracker."""
    from jarvis.utils.latency_tracker import OPERATION_BUDGETS, LatencyRecord, get_tracker

    op = f"rpc.{method}"
    budget = OPERATION_BUDGETS.get(op)
    threshold = budget[1] if budget and budget[1] > 0 else None
    exceeded = threshold is not None and elapsed_ms > threshold
    get_tracker()._records.append(
        LatencyRecord(
            operation=op,
            elapsed_ms=elapsed_ms,
            timestamp=time.time(),
            threshold_ms=threshold,
            exceeded=exceeded,
        )
    )
    if exceeded:
        logger.warning(
            f"[RPC Budget] {method} took {elapsed_ms:.1f}ms (budget: {threshold}ms)"
        )
