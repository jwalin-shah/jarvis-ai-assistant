"""Generation utilities for JARVIS.

Provides confidence scoring, logging, and metrics for generation tasks.
"""

from jarvis.core.generation.confidence import compute_confidence, compute_example_diversity
from jarvis.core.generation.logging import log_custom_generation, persist_reply_log
from jarvis.core.generation.metrics import record_routing_metrics, record_rpc_latency

__all__ = [
    "compute_confidence",
    "compute_example_diversity",
    "log_custom_generation",
    "persist_reply_log",
    "record_rpc_latency",
    "record_routing_metrics",
]
