"""Embedding backends: MLX (GPU) and CPU (ONNX).

Exports:
    - CPUEmbedder: Lightweight CPU embedder (ONNX)
    - get_cpu_embedder: Get CPU embedder instance
    - is_cpu_embedder_available: Check if CPU embedder is available
    - get_embedder_for_context: Router for selecting backend
    - get_parallel_embedder: Get CPU embedder for parallel processing
"""

from __future__ import annotations

from jarvis.embedding.cpu_embedder import (
    CPUEmbedder,
    get_cpu_embedder,
    is_cpu_embedder_available,
)
from jarvis.embedding.router import get_embedder_for_context, get_parallel_embedder

__all__ = [
    "CPUEmbedder",
    "get_cpu_embedder",
    "is_cpu_embedder_available",
    "get_embedder_for_context",
    "get_parallel_embedder",
]
