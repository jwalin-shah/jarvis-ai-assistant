"""Pipeline observability and metrics tracking.

Provides centralized monitoring for the data ingestion and extraction pipeline,
tracking throughput, success rates, and resource usage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""

    name: str
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None
    items_processed: int = 0
    success_count: int = 0
    failure_count: int = 0
    rejection_count: int = 0
    token_count: int = 0  # For LLM stages

    @property
    def duration(self) -> float:
        if self.end_time is None:
            return time.monotonic() - self.start_time
        return self.end_time - self.start_time

    @property
    def throughput(self) -> float:
        dur = self.duration
        return self.items_processed / dur if dur > 0 else 0.0


class PipelineMonitor:
    """Tracks and reports on the overall pipeline performance."""

    def __init__(self) -> None:
        self.stages: dict[str, StageMetrics] = {}
        self.overall_start = time.monotonic()

    def start_stage(self, name: str) -> StageMetrics:
        """Start tracking a new stage."""
        stage = StageMetrics(name=name)
        self.stages[name] = stage
        logger.info(f"Pipeline Stage Started: {name}")
        return stage

    def end_stage(self, name: str) -> None:
        """Complete tracking for a stage."""
        if name in self.stages:
            stage = self.stages[name]
            stage.end_time = time.monotonic()
            logger.info(
                f"Pipeline Stage Finished: {name} | "
                f"Processed: {stage.items_processed} | "
                f"Success: {stage.success_count} | "
                f"Failed: {stage.failure_count} | "
                f"Duration: {stage.duration:.2f}s | "
                f"Throughput: {stage.throughput:.2f} items/s"
            )

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all tracked metrics."""
        total_duration = time.monotonic() - self.overall_start
        summary: dict[str, Any] = {
            "total_duration_s": round(total_duration, 2),
            "stages": {},
        }
        for name, stage in self.stages.items():
            summary["stages"][name] = {
                "duration_s": round(stage.duration, 2),
                "processed": stage.items_processed,
                "success": stage.success_count,
                "failed": stage.failure_count,
                "rejected": stage.rejection_count,
                "throughput": round(stage.throughput, 2),
            }
        return summary
