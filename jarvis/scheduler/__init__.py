"""Scheduler module for draft scheduling and smart timing.

Provides a scheduling system for automated message sending with features like:
- Priority-based send queue
- Smart timing based on contact history
- Quiet hours support
- Time zone awareness
- Retry with exponential backoff

Usage:
    from jarvis.scheduler import get_scheduler, ScheduledItem, Priority

    scheduler = get_scheduler()
    item = scheduler.schedule_draft(
        draft_id="abc123",
        contact_id=1,
        send_at=datetime.now() + timedelta(hours=2),
        priority=Priority.NORMAL,
    )
"""

from jarvis.scheduler.models import (
    Priority,
    ScheduledItem,
    ScheduledStatus,
    SendResult,
    TimingSuggestion,
)
from jarvis.scheduler.queue import SchedulerQueue, get_scheduler_queue
from jarvis.scheduler.scheduler import (
    DraftScheduler,
    get_scheduler,
    start_scheduler,
    stop_scheduler,
)
from jarvis.scheduler.timing import (
    TimingAnalyzer,
    get_timing_analyzer,
    suggest_send_time,
)

__all__ = [
    # Models
    "Priority",
    "ScheduledItem",
    "ScheduledStatus",
    "SendResult",
    "TimingSuggestion",
    # Queue
    "SchedulerQueue",
    "get_scheduler_queue",
    # Scheduler
    "DraftScheduler",
    "get_scheduler",
    "start_scheduler",
    "stop_scheduler",
    # Timing
    "TimingAnalyzer",
    "get_timing_analyzer",
    "suggest_send_time",
]
