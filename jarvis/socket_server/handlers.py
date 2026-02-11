"""RPC method handler implementations.

Standalone async functions for each JSON-RPC method. The server class
delegates to these functions, passing required dependencies explicitly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from jarvis.socket_server.protocol import INTERNAL_ERROR, INVALID_PARAMS, JsonRpcError
from jarvis.utils.latency_tracker import track_latency

logger = logging.getLogger(__name__)


async def handle_ping(models_ready: bool) -> dict[str, str | bool]:
    """Health check with model readiness status."""
    return {
        "status": "ok",
        "models_ready": models_ready,
    }


async def handle_generate_draft(
    chat_id: str,
    prefetch_manager: Any | None,
    instruction: str | None = None,
    context_messages: int = 20,
    _writer: asyncio.StreamWriter | None = None,
    _request_id: Any = None,
    skip_cache: bool = False,
    send_stream_token: Any = None,
    send_stream_response: Any = None,
) -> dict[str, Any]:
    """Generate draft replies for a conversation.

    Supports streaming: if _writer is provided, streams tokens in real-time.
    Uses prefetch cache for near-instant responses when available.
    """
    try:
        # Check prefetch cache first for instant response (non-streaming only)
        if not skip_cache and _writer is None and prefetch_manager:
            cached_draft = prefetch_manager.get_draft(chat_id)
            if cached_draft and "suggestions" in cached_draft:
                logger.debug(f"Serving prefetched draft for {chat_id}")
                cached_draft["from_cache"] = True
                return cached_draft

        # Get context from iMessage
        from integrations.imessage import ChatDBReader

        with track_latency("socket_get_messages", chat_id=chat_id, limit=context_messages):
            with ChatDBReader() as reader:
                messages = reader.get_messages(chat_id, limit=context_messages)

        if not messages:
            raise JsonRpcError(INVALID_PARAMS, "No messages found in conversation")

        # Build context from messages
        context = []
        participants: set[str] = set()
        for msg in reversed(messages):  # Oldest first
            sender = msg.sender_name or msg.sender
            participants.add(sender)
            prefix = "Me" if msg.is_from_me else sender
            if msg.text:
                context.append(f"{prefix}: {msg.text}")

        # Get the last incoming message to respond to
        last_incoming = None
        for msg in messages:
            if not msg.is_from_me and msg.text:
                last_incoming = msg.text
                break

        if not last_incoming:
            raise JsonRpcError(INVALID_PARAMS, "No message to respond to")

        context_used = {
            "num_messages": len(messages),
            "participants": list(participants),
            "last_message": messages[0].text if messages else None,
        }

        # Check if streaming is requested
        if _writer is not None:
            return await _generate_draft_streaming(
                last_incoming=last_incoming,
                context=context,
                chat_id=chat_id,
                instruction=instruction,
                writer=_writer,
                request_id=_request_id,
                context_used=context_used,
                send_stream_token=send_stream_token,
                send_stream_response=send_stream_response,
            )

        # Non-streaming: use router (run in thread to avoid blocking event loop)
        from jarvis.router import get_reply_router

        router = get_reply_router()
        result = await asyncio.to_thread(
            router.route,
            incoming=last_incoming,
            thread=context[-10:] if context else None,
            chat_id=chat_id,
        )

        response_text = result.get("response", "")
        confidence = 0.8 if result.get("confidence") == "high" else 0.6

        return {
            "suggestions": [{"text": response_text, "confidence": confidence}],
            "context_used": context_used,
        }

    except JsonRpcError:
        raise
    except Exception as e:
        logger.exception("Error generating draft")
        raise JsonRpcError(INTERNAL_ERROR, "Generation failed") from e


async def _generate_draft_streaming(
    last_incoming: str,
    context: list[str],
    chat_id: str,
    instruction: str | None,
    writer: asyncio.StreamWriter,
    request_id: Any,
    context_used: dict[str, Any],
    send_stream_token: Any,
    send_stream_response: Any,
) -> dict[str, Any]:
    """Generate draft with real token streaming through full pipeline."""
    from jarvis.reply_service import get_reply_service

    reply_service = get_reply_service()

    # Phase 1: Build request through full pipeline (sync, run in thread)
    try:
        request, metadata = await asyncio.to_thread(
            reply_service.prepare_streaming_context,
            incoming=last_incoming,
            thread=context[-10:] if context else None,
            chat_id=chat_id,
            instruction=instruction,
        )
    except Exception as e:
        logger.exception("Failed to prepare streaming context")
        raise JsonRpcError(INTERNAL_ERROR, f"Context preparation failed: {e}") from e

    confidence = 0.8 if metadata.get("confidence") == "high" else 0.6

    # Phase 2: Stream tokens from generator
    full_response = ""
    token_count = 0

    try:
        async for token_data in reply_service.generator.generate_stream(request):
            token_text = token_data["token"]
            token_index = token_data["token_index"]
            is_final = token_data["is_final"]

            full_response += token_text
            token_count += 1

            await send_stream_token(
                writer, token_text, token_index, is_final, request_id=request_id
            )
    except Exception as e:
        logger.exception("Streaming generation failed")
        raise JsonRpcError(INTERNAL_ERROR, "Streaming failed") from e

    # Phase 3: Send final response
    result = {
        "suggestions": [{"text": full_response.strip(), "confidence": confidence}],
        "context_used": context_used,
        "streamed": True,
        "tokens_generated": token_count,
    }
    await send_stream_response(writer, request_id, result)
    return result


async def handle_summarize(
    chat_id: str,
    num_messages: int = 50,
    _writer: asyncio.StreamWriter | None = None,
    _request_id: Any = None,
    send_stream_token: Any = None,
    send_stream_response: Any = None,
) -> dict[str, Any]:
    """Summarize a conversation."""
    try:
        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            messages = reader.get_messages(chat_id, limit=num_messages)

        if not messages:
            raise JsonRpcError(INVALID_PARAMS, "No messages found")

        # Build conversation text
        conversation = []
        for msg in reversed(messages):
            sender = msg.sender_name or msg.sender
            prefix = "Me" if msg.is_from_me else sender
            if msg.text:
                conversation.append(f"{prefix}: {msg.text}")

        if not conversation:
            return {
                "summary": "No text messages found in conversation",
                "key_points": [],
                "message_count": len(messages),
            }

        from models.loader import get_model

        model = get_model()
        if not model:
            raise JsonRpcError(INTERNAL_ERROR, "Model not available")

        conversation_text = "\n".join(conversation[-30:])  # Limit for context window
        prompt = f"""Summarize this conversation in 2-3 sentences, then list key points:

{conversation_text}

Summary:"""

        # Check if streaming is requested
        if _writer is not None:
            return await _summarize_streaming(
                model=model,
                prompt=prompt,
                message_count=len(messages),
                writer=_writer,
                request_id=_request_id,
                send_stream_token=send_stream_token,
                send_stream_response=send_stream_response,
            )

        # Non-streaming generation - run in thread to avoid blocking event loop
        def _summarize_sync():
            if not model.is_loaded():
                model.load()
            return model.generate_sync(prompt, max_tokens=300)

        result = await asyncio.to_thread(_summarize_sync)
        response_text = result.text

        # Parse response - simple extraction
        lines = response_text.strip().split("\n")
        summary = lines[0] if lines else "Conversation summary unavailable"

        # Extract bullet points
        key_points = [
            line.lstrip("•-*0123456789.)").strip()
            for line in lines[1:]
            if line.strip() and len(line.strip()) > 5
        ][:5]  # Max 5 key points

        return {
            "summary": summary,
            "key_points": key_points or ["See full conversation for details"],
            "message_count": len(messages),
        }

    except JsonRpcError:
        raise
    except Exception as e:
        logger.exception("Error summarizing conversation")
        raise JsonRpcError(INTERNAL_ERROR, "Summarization failed") from e


async def _summarize_streaming(
    model: Any,
    prompt: str,
    message_count: int,
    writer: asyncio.StreamWriter,
    request_id: Any,
    send_stream_token: Any,
    send_stream_response: Any,
) -> dict[str, Any]:
    """Summarize with real token streaming."""
    import queue
    import threading

    # Ensure model is loaded before streaming
    if not model.is_loaded():
        await asyncio.wait_for(asyncio.to_thread(model.load), timeout=120.0)

    token_queue: queue.Queue[tuple[str, int, bool] | None] = queue.Queue(maxsize=100)
    generation_error: list[Exception] = []
    stop_event = threading.Event()

    def producer():
        try:
            from models.loader import MLXModelLoader

            with MLXModelLoader._mlx_load_lock:
                for stream_token in model.generate_stream(
                    prompt, max_tokens=300, stop_event=stop_event
                ):
                    token_queue.put(
                        (
                            stream_token.token,
                            stream_token.token_index,
                            stream_token.is_final,
                        )
                    )
                    if stop_event.is_set():
                        break
        except Exception as e:
            generation_error.append(e)
        finally:
            token_queue.put(None)  # Always signal completion

    gen_thread = threading.Thread(target=producer, daemon=True)
    gen_thread.start()

    full_response = ""
    token_count = 0

    try:
        while True:
            try:
                item = await asyncio.to_thread(token_queue.get, timeout=30.0)
            except Exception:
                break

            if item is None:
                break

            token_text, token_index, is_final = item
            full_response += token_text
            token_count += 1

            await send_stream_token(
                writer, token_text, token_index, is_final, request_id=request_id
            )

        gen_thread.join(timeout=5.0)

        if generation_error:
            raise generation_error[0]

    except Exception as e:
        logger.exception("Streaming summarization failed")
        raise JsonRpcError(INTERNAL_ERROR, "Streaming failed") from e
    finally:
        stop_event.set()

    # Parse the streamed response
    lines = full_response.strip().split("\n")
    summary = lines[0] if lines else "Conversation summary unavailable"
    key_points = [
        line.lstrip("•-*0123456789.)").strip()
        for line in lines[1:]
        if line.strip() and len(line.strip()) > 5
    ][:5]

    result = {
        "summary": summary,
        "key_points": key_points or ["See full conversation for details"],
        "message_count": message_count,
        "streamed": True,
        "tokens_generated": token_count,
    }
    await send_stream_response(writer, request_id, result)
    return result


async def handle_get_smart_replies(
    last_message: str,
    num_suggestions: int = 3,
) -> dict[str, Any]:
    """Get smart reply suggestions."""
    try:
        from jarvis.router import get_reply_router

        router = get_reply_router()
        result = await asyncio.to_thread(router.route, incoming=last_message)

        response_text = result.get("response", "")

        suggestions = []
        if response_text:
            suggestions.append(
                {
                    "text": response_text,
                    "score": 0.9 if result.get("confidence") == "high" else 0.7,
                }
            )

        return {"suggestions": suggestions}

    except Exception as e:
        logger.exception("Error getting smart replies")
        raise JsonRpcError(INTERNAL_ERROR, "Smart replies failed") from e


async def handle_semantic_search(
    query: str,
    limit: int = 20,
    threshold: float = 0.3,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Semantic search across messages."""
    try:
        from integrations.imessage.reader import ChatDBReader
        from jarvis.search.semantic_search import SearchFilters, get_semantic_searcher

        # Build filter args
        search_filters = SearchFilters()
        if filters:
            if "chat_id" in filters:
                search_filters.chat_id = filters["chat_id"]
            if "sender" in filters:
                search_filters.sender = filters["sender"]

        with ChatDBReader() as reader:
            searcher = get_semantic_searcher(reader)
            searcher.similarity_threshold = threshold
            results = searcher.search(query, filters=search_filters, limit=limit)

        return {
            "results": [
                {
                    "message": {
                        "id": r.message.id,
                        "chat_id": r.message.chat_id,
                        "text": r.message.text,
                        "sender": r.message.sender,
                        "date": r.message.date.isoformat(),
                    },
                    "similarity": r.similarity,
                }
                for r in results
            ],
            "total_results": len(results),
        }

    except Exception as e:
        logger.exception("Error in semantic search")
        raise JsonRpcError(INTERNAL_ERROR, "Search failed") from e


async def handle_batch(
    requests: list[dict[str, Any]],
    methods: dict[str, Any],
) -> dict[str, Any]:
    """Execute multiple RPC calls in a single request."""
    if not requests:
        return {"results": []}

    if len(requests) > 50:
        raise JsonRpcError(INVALID_PARAMS, "Maximum 50 requests per batch")

    async def execute_single(req: dict[str, Any]) -> dict[str, Any]:
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if not method:
            return {
                "id": req_id,
                "error": {"code": -32600, "message": "Missing method"},
            }

        handler = methods.get(method)
        if not handler:
            return {
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        try:
            if isinstance(params, dict):
                params = {
                    k: v
                    for k, v in params.items()
                    if k not in ("stream", "_writer", "_request_id")
                }
                result = await handler(**params)
            elif isinstance(params, list):
                result = await handler(*params)
            else:
                result = await handler()

            return {"id": req_id, "result": result}

        except JsonRpcError as e:
            return {
                "id": req_id,
                "error": {"code": e.code, "message": e.message},
            }
        except Exception:
            logger.exception(f"Batch error for {method}")
            return {
                "id": req_id,
                "error": {"code": -32603, "message": "Internal server error"},
            }

    results = await asyncio.gather(
        *[execute_single(req) for req in requests],
        return_exceptions=False,
    )

    return {"results": list(results)}


async def handle_resolve_contacts(identifiers: list[str]) -> dict[str, str | None]:
    """Resolve a batch of phone numbers/emails to contact names."""
    if not identifiers:
        return {}

    identifiers = identifiers[:500]

    try:
        from integrations.imessage import ChatDBReader

        def _resolve_sync() -> dict[str, str | None]:
            result: dict[str, str | None] = {}
            with ChatDBReader() as reader:
                for identifier in identifiers:
                    result[identifier] = reader._resolve_contact_name(identifier)
            return result

        return await asyncio.to_thread(_resolve_sync)
    except Exception as e:
        logger.warning(f"Contact resolution failed: {e}")
        return {}


async def handle_list_conversations(
    limit: int = 50,
    since: float | None = None,
    before: float | None = None,
) -> dict[str, Any]:
    """List recent conversations via socket (fast, no HTTP overhead)."""
    import time

    start_time = time.time()

    try:
        from datetime import datetime

        from integrations.imessage import ChatDBReader

        since_dt = datetime.fromtimestamp(since) if since else None
        before_dt = datetime.fromtimestamp(before) if before else None

        db_start = time.time()
        with track_latency("socket_list_conversations", limit=limit):
            with ChatDBReader() as reader:
                conversations = reader.get_conversations(
                    limit=limit,
                    since=since_dt,
                    before=before_dt,
                )
        db_elapsed_ms = (time.time() - db_start) * 1000

        serialize_start = time.time()
        result = {
            "conversations": [
                {
                    "chat_id": c.chat_id,
                    "participants": c.participants,
                    "display_name": c.display_name,
                    "last_message_date": c.last_message_date.isoformat(),
                    "message_count": c.message_count,
                    "is_group": c.is_group,
                    "last_message_text": c.last_message_text,
                }
                for c in conversations
            ],
            "total": len(conversations),
        }
        serialize_elapsed_ms = (time.time() - serialize_start) * 1000
        total_elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            f"[list_conversations] returned {len(conversations)} convos in "
            f"{total_elapsed_ms:.1f}ms (db={db_elapsed_ms:.1f}ms, "
            f"serialize={serialize_elapsed_ms:.1f}ms)"
        )
        return result

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.exception(f"Error listing conversations (failed after {elapsed_ms:.1f}ms)")
        raise JsonRpcError(INTERNAL_ERROR, "Failed to list conversations") from e


async def handle_get_routing_metrics(
    since: float | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Get routing metrics for the dashboard."""
    try:
        from jarvis.observability.metrics_router import get_routing_metrics_store

        store = get_routing_metrics_store()
        return await asyncio.to_thread(store.query_metrics, since, limit)
    except Exception as e:
        logger.warning(f"Failed to get routing metrics: {e}")
        return {"recent_requests": [], "summary": {}}


async def handle_prefetch_stats(prefetch_manager: Any | None) -> dict[str, Any]:
    """Get prefetch system statistics."""
    if not prefetch_manager:
        return {"enabled": False, "error": "Prefetch system not enabled"}

    try:
        stats = prefetch_manager.stats()
        stats["enabled"] = True
        return stats
    except Exception as e:
        logger.exception("Error getting prefetch stats")
        raise JsonRpcError(INTERNAL_ERROR, "Stats retrieval failed") from e


async def handle_prefetch_invalidate(
    prefetch_manager: Any | None,
    chat_id: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Invalidate prefetch cache entries."""
    if not prefetch_manager:
        return {"invalidated": 0, "error": "Prefetch system not enabled"}

    try:
        count = prefetch_manager.invalidate(chat_id=chat_id, tags=tags)
        return {"invalidated": count}
    except Exception as e:
        logger.exception("Error invalidating cache")
        raise JsonRpcError(INTERNAL_ERROR, "Invalidation failed") from e


async def handle_prefetch_focus(
    prefetch_manager: Any | None,
    chat_id: str = "",
) -> dict[str, Any]:
    """Signal that user focused on a chat (triggers high-priority prefetch)."""
    if not prefetch_manager:
        return {"status": "disabled"}

    try:
        prefetch_manager.on_focus(chat_id)
        draft = prefetch_manager.get_draft(chat_id)
        if draft:
            return {"status": "ok", "prefetched": True, "draft": draft}
        return {"status": "ok", "prefetched": False}
    except Exception as e:
        logger.debug(f"Prefetch focus error: {e}")
        return {"status": "error", "error": str(e)}


async def handle_prefetch_hover(
    prefetch_manager: Any | None,
    chat_id: str = "",
) -> dict[str, Any]:
    """Signal that user hovered over a chat (triggers low-priority prefetch)."""
    if not prefetch_manager:
        return {"status": "disabled"}

    try:
        prefetch_manager.on_hover(chat_id)
        return {"status": "ok"}
    except Exception as e:
        logger.debug(f"Prefetch hover error: {e}")
        return {"status": "error", "error": str(e)}
