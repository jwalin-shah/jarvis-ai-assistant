"""Background worker for processing tasks.

Provides a threaded worker that continuously processes tasks from the queue.
Supports batch operations like export, summarization, and reply generation.

Usage:
    from jarvis.tasks import get_worker, start_worker, stop_worker

    # Start the background worker
    start_worker()

    # Stop the worker
    stop_worker()
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from jarvis.contracts.imessage import Message
from jarvis.tasks.models import Task, TaskResult, TaskType
from jarvis.tasks.queue import TaskQueue, get_task_queue

logger = logging.getLogger(__name__)

# Type alias for task handlers
TaskHandler = Callable[[Task, Callable[[int, int, str], None]], TaskResult]


class TaskWorker:
    """Background worker for processing tasks.

    Runs in a separate thread and continuously processes tasks from the queue.
    Supports registering custom handlers for different task types.

    Attributes:
        poll_interval: Seconds between queue polls when idle.
        max_concurrent: Maximum concurrent tasks (currently 1 for simplicity).
    """

    def __init__(
        self,
        queue: TaskQueue | None = None,
        poll_interval: float = 1.0,
    ) -> None:
        """Initialize the worker.

        Args:
            queue: Task queue to process. Defaults to singleton queue.
            poll_interval: Seconds between queue polls.
        """
        self._queue = queue or get_task_queue()
        self._poll_interval = poll_interval
        self._running = False
        self._paused = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._handlers: dict[TaskType, TaskHandler] = {}

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register handlers for built-in task types."""
        self._handlers[TaskType.BATCH_EXPORT] = self._handle_batch_export
        self._handlers[TaskType.BATCH_SUMMARIZE] = self._handle_batch_summarize
        self._handlers[TaskType.BATCH_GENERATE_REPLIES] = self._handle_batch_generate_replies
        self._handlers[TaskType.SINGLE_EXPORT] = self._handle_single_export
        self._handlers[TaskType.SINGLE_SUMMARIZE] = self._handle_single_summarize
        self._handlers[TaskType.SINGLE_GENERATE_REPLY] = self._handle_single_generate_reply
        self._handlers[TaskType.FACT_EXTRACTION] = self._handle_fact_extraction

    def register_handler(self, task_type: TaskType, handler: TaskHandler) -> None:
        """Register a custom handler for a task type.

        Args:
            task_type: The task type to handle.
            handler: Function that processes the task.
        """
        self._handlers[task_type] = handler

    def start(self) -> None:
        """Start the background worker thread."""
        if self._running:
            logger.warning("Worker is already running")
            return

        self._running = True
        self._paused = False
        self._stop_event.clear()
        self._pause_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="TaskWorker")
        self._thread.start()
        logger.info("Task worker started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background worker.

        Args:
            timeout: Seconds to wait for graceful shutdown.
        """
        if not self._running:
            return

        self._running = False
        self._stop_event.set()
        self._pause_event.set()  # Unblock if paused

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Worker thread did not stop gracefully")

        self._thread = None
        logger.info("Task worker stopped")

    def pause(self) -> None:
        """Pause the worker (stops processing new tasks)."""
        if not self._paused:
            logger.info("Pausing task worker")
            self._paused = True
            self._pause_event.clear()

    def resume(self) -> None:
        """Resume the worker."""
        if self._paused:
            logger.info("Resuming task worker")
            self._paused = False
            self._pause_event.set()

    @property
    def is_running(self) -> bool:
        """Check if the worker is running."""
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self) -> bool:
        """Check if the worker is paused."""
        return self._paused

    def _run(self) -> None:
        """Main worker loop."""
        logger.debug("Worker loop started")

        while not self._stop_event.is_set():
            try:
                # Handle pause
                if self._paused:
                    logger.debug("Worker is paused, waiting...")
                    self._pause_event.wait(timeout=1.0)
                    continue

                # Get next pending task
                task = self._queue.get_next_pending()

                if task is None:
                    # No tasks, wait before polling again
                    self._stop_event.wait(self._poll_interval)
                    continue

                # Process the task
                self._process_task(task)

            except Exception as e:
                logger.exception(f"Error in worker loop: {e}")
                # Brief pause to avoid tight error loops
                self._stop_event.wait(1.0)

        logger.debug("Worker loop exited")

    def _process_task(self, task: Task) -> None:
        """Process a single task.

        Args:
            task: The task to process.
        """
        logger.info(f"Processing task {task.id} ({task.task_type.value})")

        # Get handler for this task type
        handler = self._handlers.get(task.task_type)
        if handler is None:
            logger.error(f"No handler for task type: {task.task_type.value}")
            task.fail(f"No handler for task type: {task.task_type.value}")
            self._queue.update(task)
            return

        # Mark task as running
        task.start()
        self._queue.update(task)

        # Create progress callback
        def update_progress(current: int, total: int, message: str = "") -> None:
            task.update_progress(current, total, message)
            self._queue.update(task)

        # Execute handler
        try:
            result = handler(task, update_progress)
            task.complete(result)
            logger.info(f"Task {task.id} completed successfully")

        except Exception as e:
            logger.exception(f"Task {task.id} failed: {e}")
            task.fail(str(e))

            # Schedule retry if possible
            if task.can_retry():
                task.retry()
                logger.info(f"Task {task.id} scheduled for retry (attempt {task.retry_count})")

        # Update final state
        self._queue.update(task)

    # Default task handlers

    def _handle_batch_export(
        self,
        task: Task,
        update_progress: Callable[[int, int, str], None],
    ) -> TaskResult:
        """Handle batch export task."""
        from integrations.imessage import ChatDBReader
        from jarvis.config import validate_path
        from jarvis.export import ExportFormat, export_messages

        params = task.params
        chat_ids = params.get("chat_ids", [])
        format_str = params.get("format", "json")
        export_format = ExportFormat(format_str)
        output_dir = params.get("output_dir")

        if output_dir:
            try:
                from pathlib import Path

                # Security fix: Validate output_dir to prevent arbitrary file writes
                # This prevents path traversal (e.g. "../") and ensures the path is valid
                output_dir_path = validate_path(output_dir, "export output directory")

                # Ensure output_dir is within user home directory
                home = Path.home().resolve()
                try:
                    output_dir_path.relative_to(home)
                except ValueError:
                    # Allow /tmp in debug/test modes if needed, but for now strict
                    # Check if we are in a test environment (optional, but good for local dev)
                    # For security, we strictly enforce home directory.
                    raise ValueError(
                        f"Export directory must be within user home ({home}): {output_dir_path}"
                    )

                # Create directory if it doesn't exist
                output_dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return TaskResult(success=False, error=f"Invalid output directory: {e}")

        if not chat_ids:
            return TaskResult(success=False, error="No chat_ids provided")

        results: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        total = len(chat_ids)

        with ChatDBReader() as reader:
            conversations = reader.get_conversations(limit=500)
            conv_map = {c.chat_id: c for c in conversations}

            for i, chat_id in enumerate(chat_ids):
                update_progress(i, total, f"Exporting {chat_id}")

                try:
                    conv = conv_map.get(chat_id)
                    if conv is None:
                        failed.append({"chat_id": chat_id, "error": "Not found"})
                        continue

                    messages = reader.get_messages(chat_id=chat_id, limit=1000)
                    if not messages:
                        failed.append({"chat_id": chat_id, "error": "No messages"})
                        continue

                    exported_data = export_messages(
                        messages=messages,
                        format=export_format,
                        conversation=conv,
                    )

                    result_item: dict[str, Any] = {
                        "chat_id": chat_id,
                        "message_count": len(messages),
                        "format": format_str,
                    }

                    # Save to file if output_dir specified
                    if output_dir:
                        from jarvis.export import get_export_filename

                        filename = get_export_filename(
                            format=export_format,
                            prefix="conversation",
                            chat_id=chat_id,
                        )
                        # We use the already validated output_dir_path
                        output_path = output_dir_path / filename
                        output_path.write_text(exported_data, encoding="utf-8")
                        result_item["output_file"] = str(output_path)
                    else:
                        result_item["data"] = exported_data

                    results.append(result_item)

                except Exception as e:
                    logger.warning(f"Failed to export {chat_id}: {e}")
                    failed.append({"chat_id": chat_id, "error": str(e)})

        update_progress(total, total, "Export complete")

        return TaskResult(
            success=len(failed) == 0,
            data={"exports": results},
            items_processed=len(results),
            items_failed=len(failed),
            partial_results=failed if failed else [],
        )

    def _handle_batch_summarize(
        self,
        task: Task,
        update_progress: Callable[[int, int, str], None],
    ) -> TaskResult:
        """Handle batch summarize task."""
        from integrations.imessage import ChatDBReader
        from jarvis.context import ContextFetcher
        from jarvis.contracts.models import GenerationRequest
        from jarvis.prompts import SUMMARY_EXAMPLES, build_summary_prompt
        from models import get_generator

        params = task.params
        chat_ids = params.get("chat_ids", [])
        num_messages = params.get("num_messages", 50)

        if not chat_ids:
            return TaskResult(success=False, error="No chat_ids provided")

        results: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        total = len(chat_ids)

        generator = get_generator()

        with ChatDBReader() as reader:
            fetcher = ContextFetcher(reader)
            conversations = reader.get_conversations(limit=500)
            conv_map = {c.chat_id: c for c in conversations}

            for i, chat_id in enumerate(chat_ids):
                conv = conv_map.get(chat_id)
                display_name = conv.display_name if conv else chat_id
                update_progress(i, total, f"Summarizing {display_name}")

                try:
                    context = fetcher.get_summary_context(chat_id, num_messages=num_messages)

                    if len(context.messages) < 3:
                        failed.append({"chat_id": chat_id, "error": "Not enough messages"})
                        continue

                    formatted_prompt = build_summary_prompt(context=context.formatted_context)

                    request = GenerationRequest(
                        prompt=formatted_prompt,
                        context_documents=[context.formatted_context],
                        few_shot_examples=SUMMARY_EXAMPLES,
                        max_tokens=500,
                        temperature=0.5,
                    )
                    response = generator.generate(request)

                    # Log for traceability
                    from jarvis.reply_service import get_reply_service

                    reply_service = get_reply_service()
                    reply_service.log_custom_generation(
                        chat_id=chat_id,
                        incoming_text=f"Batch summarize {len(context.messages)} messages",
                        final_prompt=formatted_prompt,
                        response_text=response.text,
                        category="batch_summary",
                        metadata={"num_messages": len(context.messages), "task_id": task.id},
                    )

                    start_date = context.date_range[0].strftime("%Y-%m-%d")
                    end_date = context.date_range[1].strftime("%Y-%m-%d")

                    results.append(
                        {
                            "chat_id": chat_id,
                            "display_name": display_name,
                            "summary": response.text,
                            "message_count": len(context.messages),
                            "date_range": {"start": start_date, "end": end_date},
                        }
                    )

                except Exception as e:
                    logger.warning(f"Failed to summarize {chat_id}: {e}")
                    failed.append({"chat_id": chat_id, "error": str(e)})

        update_progress(total, total, "Summarization complete")

        return TaskResult(
            success=len(failed) == 0,
            data={"summaries": results},
            items_processed=len(results),
            items_failed=len(failed),
            partial_results=failed if failed else [],
        )

    def _handle_batch_generate_replies(
        self,
        task: Task,
        update_progress: Callable[[int, int, str], None],
    ) -> TaskResult:
        """Handle batch reply generation task."""
        from integrations.imessage import ChatDBReader
        from jarvis.context import ContextFetcher
        from jarvis.contracts.models import GenerationRequest
        from models import get_generator

        params = task.params
        chat_ids = params.get("chat_ids", [])
        instruction = params.get("instruction")
        num_suggestions = params.get("num_suggestions", 3)

        if not chat_ids:
            return TaskResult(success=False, error="No chat_ids provided")

        results: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        total = len(chat_ids)

        generator = get_generator()

        with ChatDBReader() as reader:
            fetcher = ContextFetcher(reader)
            conversations = reader.get_conversations(limit=500)
            conv_map = {c.chat_id: c for c in conversations}

            for i, chat_id in enumerate(chat_ids):
                conv = conv_map.get(chat_id)
                display_name = conv.display_name if conv else chat_id
                update_progress(i, total, f"Generating replies for {display_name}")

                try:
                    context = fetcher.get_reply_context(chat_id, num_messages=20)

                    if not context.last_received_message:
                        failed.append(
                            {"chat_id": chat_id, "error": "No recent messages to reply to"}
                        )
                        continue

                    last_msg = context.last_received_message

                    # Resolve contact name
                    contact_name = display_name or "them"

                    from jarvis.prompts.rag import build_rag_reply_prompt

                    formatted_prompt = build_rag_reply_prompt(
                        context=context.formatted_context,
                        last_message=last_msg.text,
                        contact_name=contact_name,
                        similar_exchanges=[],  # Vector search for similar exchanges would go here
                        relationship_profile=context.contact_profile,
                        contact_facts=context.contact_facts,
                        relationship_graph=context.relationship_graph,
                        auto_context=context.auto_context,
                        instruction=instruction,
                    )

                    from jarvis.reply_service import get_reply_service

                    reply_service = get_reply_service()

                    suggestions: list[str] = []
                    for j in range(num_suggestions):
                        request = GenerationRequest(
                            prompt=formatted_prompt,
                            context_documents=[],  # Already in prompt
                            few_shot_examples=[],  # V4 favors zero-shot/RAG anchors
                            max_tokens=150,
                            temperature=0.7 + (j * 0.1),
                        )
                        response = generator.generate(request)
                        suggestions.append(response.text)

                        # Log for traceability
                        reply_service.log_custom_generation(
                            chat_id=chat_id,
                            incoming_text=last_msg.text,
                            final_prompt=formatted_prompt,
                            response_text=response.text,
                            category="batch_reply",
                            metadata={"suggestion_index": j, "task_id": task.id},
                        )

                    results.append(
                        {
                            "chat_id": chat_id,
                            "display_name": display_name,
                            "last_message": last_msg.text[:200],
                            "suggestions": suggestions,
                        }
                    )

                except Exception as e:
                    logger.warning(f"Failed to generate replies for {chat_id}: {e}")
                    failed.append({"chat_id": chat_id, "error": str(e)})

        update_progress(total, total, "Reply generation complete")

        return TaskResult(
            success=len(failed) == 0,
            data={"replies": results},
            items_processed=len(results),
            items_failed=len(failed),
            partial_results=failed if failed else [],
        )

    def _handle_single_export(
        self,
        task: Task,
        update_progress: Callable[[int, int, str], None],
    ) -> TaskResult:
        """Handle single export task."""
        # Wrap single item in batch handler
        params = task.params
        chat_id = params.get("chat_id")
        if not chat_id:
            return TaskResult(success=False, error="No chat_id provided")

        task.params = {**task.params, "chat_ids": [chat_id]}
        return self._handle_batch_export(task, update_progress)

    def _handle_single_summarize(
        self,
        task: Task,
        update_progress: Callable[[int, int, str], None],
    ) -> TaskResult:
        """Handle single summarize task."""
        params = task.params
        chat_id = params.get("chat_id")
        if not chat_id:
            return TaskResult(success=False, error="No chat_id provided")

        task.params = {**task.params, "chat_ids": [chat_id]}
        return self._handle_batch_summarize(task, update_progress)

    def _handle_single_generate_reply(
        self,
        task: Task,
        update_progress: Callable[[int, int, str], None],
    ) -> TaskResult:
        """Handle single reply generation task."""
        params = task.params
        chat_id = params.get("chat_id")
        if not chat_id:
            return TaskResult(success=False, error="No chat_id provided")

        task.params = {**task.params, "chat_ids": [chat_id]}
        return self._handle_batch_generate_replies(task, update_progress)

    def _handle_fact_extraction(
        self,
        task: Task,
        update_progress: Callable[[int, int, str], None],
    ) -> TaskResult:
        """Handle background fact extraction task using sliding windows with progress tracking."""
        from integrations.imessage import ChatDBReader
        from jarvis.contacts.fact_storage import save_facts
        from jarvis.contacts.instruction_extractor import get_instruction_extractor
        from jarvis.db import get_db

        params = task.params
        chat_id = params.get("chat_id")
        if not chat_id:
            return TaskResult(success=False, error="No chat_id provided")

        window_size = params.get("window_size", 25)
        overlap = params.get("overlap", 5)
        force_historical = params.get("force_historical", False)  # Re-extract everything

        db = get_db()

        # Get last extracted rowid for this chat
        last_extracted_rowid = None
        with db.connection() as conn:
            row = conn.execute(
                "SELECT last_extracted_rowid FROM contacts WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            if row and row[0]:
                last_extracted_rowid = row[0]

        update_progress(0, 100, f"Fetching messages for {chat_id}")

        with ChatDBReader() as reader:
            if force_historical or last_extracted_rowid is None:
                # Historical pass: get all messages
                messages = reader.get_messages(chat_id, limit=10000)
                messages.reverse()  # oldest first
                mode = "historical"
            else:
                # Incremental: get messages after last_extracted_rowid
                messages = reader.get_messages_after(last_extracted_rowid, chat_id, limit=10000)
                # Already returns oldest first
                mode = "incremental"

            if not messages:
                return TaskResult(
                    success=True,
                    items_processed=0,
                    data={"mode": mode, "fact_count": 0, "messages_processed": 0},
                )

            # Get names for identity anchor
            user_name = reader.get_user_name()
            contact_name = "Contact"
            with db.connection() as conn:
                row = conn.execute(
                    "SELECT display_name FROM contacts WHERE chat_id = ?", (chat_id,)
                ).fetchone()
                if row and row[0]:
                    contact_name = row[0].split()[0]

            # Create extraction windows (not topic segments) with overlap for better NLI.
            extraction_windows = []
            for i in range(0, len(messages), window_size - overlap):
                window = messages[i : i + window_size]
                if len(window) < 5:
                    break

                from dataclasses import dataclass

                @dataclass
                class MockSeg:
                    messages: list[Message]
                    text: str

                seg_lines = []
                if window:
                    curr_sender = user_name if window[0].is_from_me else contact_name
                    curr_msgs = []
                    for m in window:
                        sender = user_name if m.is_from_me else contact_name
                        text = " ".join((m.text or "").splitlines()).strip()
                        if not text:
                            continue
                        if sender == curr_sender:
                            curr_msgs.append(text)
                        else:
                            if curr_msgs:
                                seg_lines.append(f"{curr_sender}: {' '.join(curr_msgs)}")
                            curr_sender = sender
                            curr_msgs = [text]
                    if curr_msgs:
                        seg_lines.append(f"{curr_sender}: {' '.join(curr_msgs)}")

                seg_text = "\n".join(seg_lines)
                extraction_windows.append(MockSeg(messages=window, text=seg_text))

            total_windows = len(extraction_windows)
            if not total_windows:
                return TaskResult(success=True, items_processed=0)

            extractor = get_instruction_extractor(tier="0.7b")
            max_rowid_processed = last_extracted_rowid or 0
            for msg in messages:
                if msg.id and msg.id > max_rowid_processed:
                    max_rowid_processed = msg.id

            update_progress(
                0, total_windows, f"Extracting from {total_windows} extraction windows ({mode})"
            )
            all_facts: list[Any] = []
            for window_idx, extraction_window in enumerate(extraction_windows, start=1):
                window_results = extractor.extract_facts_from_batch(
                    [extraction_window],
                    contact_id=chat_id,
                    contact_name=contact_name,
                    user_name=user_name,
                )
                if window_results:
                    all_facts.extend(window_results[0])
                update_progress(
                    window_idx,
                    total_windows,
                    f"Extracting from window {window_idx}/{total_windows} ({mode})",
                )
            if all_facts:
                save_facts(
                    all_facts,
                    chat_id,
                    log_raw_facts=True,
                    log_chat_id=chat_id,
                    log_stage="task_worker",
                )
            total_extracted = len(all_facts)
            update_progress(total_windows, total_windows, "Extraction complete")

            # Update tracking after successful extraction
            if max_rowid_processed > (last_extracted_rowid or 0):
                with db.connection() as conn:
                    conn.execute(
                        """UPDATE contacts
                           SET last_extracted_rowid = ?, last_extracted_at = CURRENT_TIMESTAMP
                           WHERE chat_id = ?""",
                        (max_rowid_processed, chat_id),
                    )

            return TaskResult(
                success=True,
                items_processed=total_extracted,
                data={
                    "mode": mode,
                    "fact_count": total_extracted,
                    "messages_processed": len(messages),
                    "last_extracted_rowid": max_rowid_processed,
                },
            )


# Module-level singleton
_worker: TaskWorker | None = None
_worker_lock = threading.Lock()


def get_worker() -> TaskWorker:
    """Get the singleton worker instance.

    Returns:
        Shared TaskWorker instance.
    """
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = TaskWorker()
    return _worker


def start_worker() -> None:
    """Start the background task worker."""
    worker = get_worker()
    worker.start()


def stop_worker() -> None:
    """Stop the background task worker."""
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.stop()


def pause_worker() -> None:
    """Pause the background task worker."""
    worker = get_worker()
    worker.pause()


def resume_worker() -> None:
    """Resume the background task worker."""
    worker = get_worker()
    worker.resume()


def reset_worker() -> None:
    """Reset the singleton worker (for testing)."""
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.stop()
        _worker = None


# Export all public symbols
__all__ = [
    "TaskWorker",
    "get_worker",
    "reset_worker",
    "start_worker",
    "stop_worker",
    "pause_worker",
    "resume_worker",
]
