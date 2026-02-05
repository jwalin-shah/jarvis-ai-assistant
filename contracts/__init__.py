"""Contract interfaces for JARVIS v1.

This module exports all Protocol interfaces that enable parallel workstream development.
All implementations should code against these contracts, not concrete implementations.
"""

from __future__ import annotations

from contracts.calendar import (
    Calendar,
    CalendarEvent,
    CalendarReader,
    CalendarWriter,
    CreateEventResult,
    DetectedEvent,
    EventDetector,
)
from contracts.hallucination import (
    HallucinationEvaluator,
    HHEMBenchmarkResult,
    HHEMResult,
)
from contracts.health import (
    DegradationController,
    DegradationPolicy,
    FeatureState,
    Permission,
    PermissionMonitor,
    PermissionStatus,
    SchemaDetector,
    SchemaInfo,
)
from contracts.imessage import (
    Attachment,
    AttachmentSummary,
    Conversation,
    Message,
    Reaction,
    iMessageReader,
)
from contracts.latency import (
    LatencyBenchmarker,
    LatencyBenchmarkResult,
    LatencyResult,
    Scenario,
)
from contracts.memory import (
    MemoryController,
    MemoryMode,
    MemoryProfile,
    MemoryProfiler,
    MemoryState,
)
from contracts.models import (
    GenerationRequest,
    GenerationResponse,
    Generator,
)

__all__ = [
    # Calendar
    "Calendar",
    "CalendarEvent",
    "CalendarReader",
    "CalendarWriter",
    "CreateEventResult",
    "DetectedEvent",
    "EventDetector",
    # Hallucination (WS2)
    "HallucinationEvaluator",
    "HHEMBenchmarkResult",
    "HHEMResult",
    # Health (WS6, WS7)
    "DegradationController",
    "DegradationPolicy",
    "FeatureState",
    "Permission",
    "PermissionMonitor",
    "PermissionStatus",
    "SchemaDetector",
    "SchemaInfo",
    # iMessage (WS10)
    "Attachment",
    "AttachmentSummary",
    "Conversation",
    "Message",
    "Reaction",
    "iMessageReader",
    # Latency (WS4)
    "LatencyBenchmarker",
    "LatencyBenchmarkResult",
    "LatencyResult",
    "Scenario",
    # Memory (WS1, WS5)
    "MemoryController",
    "MemoryMode",
    "MemoryProfile",
    "MemoryProfiler",
    "MemoryState",
    # Models (WS8)
    "GenerationRequest",
    "GenerationResponse",
    "Generator",
]
