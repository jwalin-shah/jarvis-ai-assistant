"""Contract interfaces for JARVIS v1.

This module exports all Protocol interfaces that enable parallel workstream development.
All implementations should code against these contracts, not concrete implementations.
"""

from contracts.memory import (
    MemoryProfile,
    MemoryMode,
    MemoryState,
    MemoryProfiler,
    MemoryController,
)
from contracts.hallucination import (
    HHEMResult,
    HHEMBenchmarkResult,
    HallucinationEvaluator,
)
from contracts.coverage import (
    TemplateMatch,
    CoverageResult,
    CoverageAnalyzer,
)
from contracts.latency import (
    Scenario,
    LatencyResult,
    LatencyBenchmarkResult,
    LatencyBenchmarker,
)
from contracts.health import (
    FeatureState,
    Permission,
    PermissionStatus,
    SchemaInfo,
    DegradationPolicy,
    DegradationController,
    PermissionMonitor,
    SchemaDetector,
)
from contracts.models import (
    GenerationRequest,
    GenerationResponse,
    Generator,
)
from contracts.gmail import (
    Email,
    EmailSearchResult,
    GmailClient,
)
from contracts.imessage import (
    Message,
    Conversation,
    iMessageReader,
)

__all__ = [
    # Memory (WS1, WS5)
    "MemoryProfile",
    "MemoryMode",
    "MemoryState",
    "MemoryProfiler",
    "MemoryController",
    # Hallucination (WS2)
    "HHEMResult",
    "HHEMBenchmarkResult",
    "HallucinationEvaluator",
    # Coverage (WS3)
    "TemplateMatch",
    "CoverageResult",
    "CoverageAnalyzer",
    # Latency (WS4)
    "Scenario",
    "LatencyResult",
    "LatencyBenchmarkResult",
    "LatencyBenchmarker",
    # Health (WS6, WS7)
    "FeatureState",
    "Permission",
    "PermissionStatus",
    "SchemaInfo",
    "DegradationPolicy",
    "DegradationController",
    "PermissionMonitor",
    "SchemaDetector",
    # Models (WS8)
    "GenerationRequest",
    "GenerationResponse",
    "Generator",
    # Gmail (WS9)
    "Email",
    "EmailSearchResult",
    "GmailClient",
    # iMessage (WS10)
    "Message",
    "Conversation",
    "iMessageReader",
]
