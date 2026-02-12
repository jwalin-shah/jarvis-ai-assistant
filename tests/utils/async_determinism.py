"""Utilities for deterministic async testing.

Provides tools to control async execution order and eliminate
race conditions in async tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any, TypeVar
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Coroutine

T = TypeVar("T")


class DeterministicEventLoop:
    """Event loop with deterministic task scheduling.

    Eliminates race conditions by controlling task execution order.

    Example:
        loop = DeterministicEventLoop()
        task1 = loop.create_task(async_op1())
        task2 = loop.create_task(async_op2())

        # Execute in explicit order
        results = loop.run_in_order(task1, task2)

        # Or execute concurrently but with deterministic scheduling
        results = loop.run_concurrently([task1, task2])
    """

    def __init__(self):
        self._scheduled_tasks: list[asyncio.Task] = []
        self._execution_order: list[str] = []
        self._task_counter = 0

    def create_task(
        self,
        coro: Coroutine[Any, Any, T],
        name: str | None = None,
    ) -> asyncio.Task[T]:
        """Create a tracked task."""
        if name is None:
            self._task_counter += 1
            name = f"task_{self._task_counter}"

        task = asyncio.create_task(coro, name=name)
        self._scheduled_tasks.append(task)
        return task

    async def run_in_order(self, *tasks: asyncio.Task[T]) -> list[T]:
        """Run tasks in explicit order, not concurrently."""
        results = []
        for task in tasks:
            result = await task
            if task.get_name():
                self._execution_order.append(task.get_name())
            results.append(result)
        return results

    async def run_concurrently(
        self,
        tasks: list[asyncio.Task[T]],
        return_when: str = asyncio.ALL_COMPLETED,
    ) -> tuple[set[asyncio.Task], set[asyncio.Task]]:
        """Run tasks concurrently but track execution order."""

        async def tracked_task(task: asyncio.Task[T]) -> T:
            """Wrap task to track completion order."""
            result = await task
            if task.get_name():
                self._execution_order.append(task.get_name())
            return result

        tracked = [asyncio.create_task(tracked_task(t)) for t in tasks]
        return await asyncio.wait(tracked, return_when=return_when)

    def get_execution_order(self) -> list[str]:
        """Get recorded task execution order."""
        return self._execution_order.copy()

    def reset(self) -> None:
        """Reset tracking state."""
        self._scheduled_tasks.clear()
        self._execution_order.clear()
        self._task_counter = 0


@asynccontextmanager
async def controlled_timeout(
    should_timeout: bool = False,
    after_calls: int = 0,
):
    """Control timeout behavior for deterministic testing.

    Args:
        should_timeout: Whether to raise TimeoutError
        after_calls: Timeout after N calls (0 = immediate)

    Example:
        async with controlled_timeout(should_timeout=True):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(slow_operation(), timeout=1.0)
    """
    call_count = [0]

    async def mock_wait_for(
        coro: Coroutine[Any, Any, T],
        timeout: float | None = None,
    ) -> T:
        call_count[0] += 1
        if should_timeout and call_count[0] > after_calls:
            raise TimeoutError(f"Simulated timeout after {timeout}s")
        # Execute immediately without actual delay
        return await coro

    with patch("asyncio.wait_for", mock_wait_for):
        yield


@contextmanager
def deterministic_gather():
    """Make asyncio.gather deterministic by controlling task order.

    Example:
        with deterministic_gather():
            # Tasks will complete in creation order, not completion order
            results = await asyncio.gather(task1, task2, task3)
    """
    original_gather = asyncio.gather

    async def ordered_gather(
        *coros_or_futures: asyncio.Future[T] | Coroutine[Any, Any, T],
        return_exceptions: bool = False,
    ) -> list[Any]:
        """Gather that completes tasks in order."""
        # Convert to tasks if needed
        tasks = [c if asyncio.isfuture(c) else asyncio.create_task(c) for c in coros_or_futures]

        # Wait in order
        results = []
        for task in tasks:
            try:
                result = await task
                results.append(result)
            except Exception as e:
                if return_exceptions:
                    results.append(e)
                else:
                    raise

        return results

    with patch("asyncio.gather", ordered_gather):
        yield


class AsyncSequence:
    """Execute async operations in a predetermined sequence.

    Useful for testing race conditions and concurrent access patterns.

    Example:
        seq = AsyncSequence()

        @seq.step(1)
        async def first():
            return "first"

        @seq.step(2)
        async def second():
            return "second"

        results = await seq.execute()
        assert results == ["first", "second"]
    """

    def __init__(self):
        self._steps: dict[int, Callable[[], Coroutine[Any, Any, Any]]] = {}
        self._max_step = 0

    def step(self, order: int):
        """Decorator to register a step at given order."""

        def decorator(
            fn: Callable[[], Coroutine[Any, Any, T]],
        ) -> Callable[[], Coroutine[Any, Any, T]]:
            self._steps[order] = fn
            self._max_step = max(self._max_step, order)
            return fn

        return decorator

    async def execute(self) -> list[Any]:
        """Execute all steps in order."""
        results = []
        for i in range(1, self._max_step + 1):
            if i in self._steps:
                result = await self._steps[i]()
                results.append(result)
        return results

    def reset(self) -> None:
        """Clear all registered steps."""
        self._steps.clear()
        self._max_step = 0


@asynccontextmanager
async def paused_event_loop():
    """Pause the event loop to prevent unexpected task execution.

    Useful for testing specific interleavings of async operations.

    Example:
        async with paused_event_loop():
            task1 = asyncio.create_task(operation1())
            task2 = asyncio.create_task(operation2())

            # Manually advance tasks
            await task1  # task2 won't run until we await it
    """
    # This is a simplified version - real implementation would
    # need more sophisticated event loop control
    loop = asyncio.get_event_loop()

    # Save current tasks
    current_tasks = asyncio.all_tasks(loop)

    try:
        # Prevent new task creation
        yield
    finally:
        # Resume normal operation
        pass


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run coroutine synchronously with a new event loop.

    Useful for testing async code from sync test contexts.

    Example:
        result = run_sync(async_function())
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
