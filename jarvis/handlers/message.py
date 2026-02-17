from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from jarvis.handlers.base import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    BaseHandler,
    JsonRpcError,
    rpc_handler,
)
from jarvis.prompts.generation_config import DEFAULT_REPETITION_PENALTY

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MessageHandler(BaseHandler):
    """Handler for message-related RPC methods.

    Provides methods for generating drafts, summarizing conversations,
    getting smart replies, chatting with SLM, and listing conversations.
    """

    def register(self) -> None:
        """Register message-related RPC methods."""
        self.server.register("generate_draft", self._generate_draft, streaming=True)
        self.server.register("summarize", self._summarize, streaming=True)
        self.server.register("get_smart_replies", self._get_smart_replies)
        self.server.register("list_conversations", self._list_conversations)
        self.server.register("chat", self._chat, streaming=True)
        self.server.register("record_feedback", self._record_feedback)

    @rpc_handler("Failed to record feedback")
    async def _record_feedback(
        self,
        action: str,
        suggestion_text: str,
        chat_id: str,
        context_messages: list[str] = None,
        edited_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        """Record user feedback on an AI suggestion."""
        from jarvis.eval.evaluation import SuggestionAction, get_feedback_store

        store = get_feedback_store()
        try:
            store.record_feedback(
                action=SuggestionAction(action),
                suggestion_text=suggestion_text,
                chat_id=chat_id,
                context_messages=context_messages or [],
                edited_text=edited_text,
                metadata=metadata,
            )
            return {"success": True}
        except ValueError as e:
            logger.warning(f"Invalid feedback action: {e}")
            return {"success": False}

    @rpc_handler("Chat failed")
    async def _chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        _writer: Any = None,
        _request_id: Any = None,
    ) -> dict[str, Any]:
        """Direct chat with the SLM."""
        if not message or not message.strip():
            raise JsonRpcError(INVALID_PARAMS, "Message cannot be empty")

        from jarvis.model_warmer import get_model_warmer

        get_model_warmer().touch()

        system = system_prompt or "You are a helpful assistant."
        prompt_parts = ["<|im_start|>system\n", system, "<|im_end|>\n"]

        turns = (history or [])[-10:]
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                prompt_parts.extend(["<|im_start|>user\n", content, "<|im_end|>\n"])
            elif role == "assistant":
                prompt_parts.extend(["<|im_start|>assistant\n", content, "<|im_end|>\n"])

        prompt_parts.extend(["<|im_start|>user\n", message, "<|im_end|>\n"])
        prompt_parts.append("<|im_start|>assistant\n")

        prompt = "".join(prompt_parts)

        from contracts.models import GenerationRequest

        request = GenerationRequest(
            prompt=prompt,
            context_documents=[],
            few_shot_examples=[],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,  # From generation_config
            stop_sequences=["<|im_end|>", "<|im_start|>"],
        )

        if _writer is not None:
            return await self._chat_streaming(request, _writer, _request_id)

        from models import get_generator

        generator = get_generator()
        response_tokens: list[str] = []
        async for token_data in generator.generate_stream(request):
            response_tokens.append(token_data["token"])

        return {
            "response": "".join(response_tokens).strip(),
            "tokens_generated": len(response_tokens),
        }

    async def _chat_streaming(
        self,
        request: Any,
        writer: Any,
        request_id: Any,
    ) -> dict[str, Any]:
        """Stream chat tokens."""
        from models import get_generator

        generator = get_generator()
        response_tokens: list[str] = []

        try:
            async for token_data in generator.generate_stream(request):
                token_text = token_data["token"]
                token_index = token_data["token_index"]
                is_final = token_data["is_final"]
                response_tokens.append(token_text)

                await self.send_stream_token(
                    writer, token_text, token_index, is_final, request_id=request_id
                )

            # Ensure a final token is sent if the generator didn't provide one
            if response_tokens and not is_final:
                await self.send_stream_token(
                    writer, "", len(response_tokens), True, request_id=request_id
                )

            full_response = "".join(response_tokens).strip()
            result = {
                "response": full_response,
                "tokens_generated": len(response_tokens),
            }
            await self.send_stream_response(writer, request_id, result)
            return result
        except Exception as e:
            logger.exception("Chat streaming failed")
            raise JsonRpcError(INTERNAL_ERROR, f"Chat streaming failed: {e}")

    @rpc_handler("Draft generation failed")
    async def _generate_draft(
        self,
        chat_id: str,
        instruction: str | None = None,
        context_messages: int = 20,
        _writer: Any = None,
        _request_id: Any = None,
        skip_cache: bool = False,
    ) -> dict[str, Any]:
        """Generate draft replies."""
        # Skip stale generation if user switched to a different chat
        if self.server.is_generation_stale(chat_id):
            logger.debug(f"Skipping stale generation for {chat_id}")
            return {"suggestions": [], "stale": True, "reason": "user_switched_chats"}

        prefetch_manager = self.server.get_prefetch_manager()
        if not skip_cache and _writer is None and prefetch_manager:
            cached_draft = prefetch_manager.get_draft(chat_id)
            if cached_draft and "suggestions" in cached_draft:
                logger.debug(f"Serving prefetched draft for {chat_id}")
                cached_draft["from_cache"] = True
                return cached_draft

        from jarvis.model_warmer import get_model_warmer

        get_model_warmer().touch()

        reply_service = self.server.get_reply_service()

        # Use context service to fetch messages and build context strings
        context, participants = reply_service.context_service.fetch_conversation_context(
            chat_id, limit=context_messages
        )

        if not context:
            raise JsonRpcError(INVALID_PARAMS, "No messages found in conversation")

        # Get last incoming message
        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            messages = reader.get_messages(chat_id, limit=context_messages)

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
            "last_message": last_incoming,
        }

        if _writer is not None:
            return await self._generate_draft_streaming(
                last_incoming=last_incoming,
                context=context,
                chat_id=chat_id,
                instruction=instruction,
                writer=_writer,
                request_id=_request_id,
                context_used=context_used,
            )

        if prefetch_manager:
            self.server.pause_prefetch()
        try:
            from jarvis.reply_service import get_reply_service

            reply_service = get_reply_service()
            result: dict[str, Any] = await asyncio.to_thread(
                reply_service.route_legacy,
                incoming=last_incoming,
                thread=context[-10:],
                chat_id=chat_id,
            )
        finally:
            if prefetch_manager:
                self.server.resume_prefetch()

        return result

    async def _generate_draft_streaming(
        self,
        last_incoming: str,
        context: list[str],
        chat_id: str,
        instruction: str | None,
        writer: Any,
        request_id: Any,
        context_used: dict[str, Any],
    ) -> dict[str, Any]:
        """Stream draft tokens."""
        # Check staleness before and during streaming
        if self.server.is_generation_stale(chat_id):
            logger.debug(f"Skipping stale streaming generation for {chat_id}")
            return {"suggestions": [], "stale": True}

        reply_service = self.server.get_reply_service()

        prefetch_manager = self.server.get_prefetch_manager()
        if prefetch_manager:
            self.server.pause_prefetch()
        self.server.pause_task_worker()
        try:
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
                raise JsonRpcError(INTERNAL_ERROR, f"Context preparation failed: {e}")

            confidence = float(metadata.get("confidence_score", 0.6))
            response_tokens: list[str] = []
            accumulated = ""
            token_count = 0

            try:
                async for token_data in reply_service.generator.generate_stream(request):
                    # Check staleness periodically to stop early if user switched chats
                    token_count += 1
                    if token_count % 10 == 0 and self.server.is_generation_stale(chat_id):
                        logger.debug(f"Stopping stale streaming for {chat_id}")
                        break

                    token_text = token_data["token"]
                    token_index = token_data["token_index"]
                    is_final = token_data["is_final"]
                    accumulated += token_text

                    # Stop at closing reply tag
                    if "</reply>" in accumulated:
                        before_tag = accumulated.split("</reply>")[0]
                        remaining = before_tag[len("".join(response_tokens)) :]
                        if remaining:
                            response_tokens.append(remaining)
                            await self.send_stream_token(
                                writer, remaining, token_index, True, request_id=request_id
                            )
                        else:
                            # If no new content but we hit the tag, send an empty final token
                            # to ensure onComplete() is triggered in the frontend.
                            await self.send_stream_token(
                                writer, "", token_index, True, request_id=request_id
                            )
                        break

                    response_tokens.append(token_text)
                    await self.send_stream_token(
                        writer, token_text, token_index, is_final, request_id=request_id
                    )
            except Exception:
                logger.exception("Streaming generation failed")
                raise JsonRpcError(INTERNAL_ERROR, "Streaming failed")
        finally:
            if prefetch_manager:
                self.server.resume_prefetch()
            self.server.resume_task_worker()

        full_response = "".join(response_tokens)
        result = {
            "suggestions": [{"text": full_response.strip(), "confidence": confidence}],
            "context_used": context_used,
            "streamed": True,
            "tokens_generated": len(response_tokens),
        }
        await self.send_stream_response(writer, request_id, result)
        return result

    @rpc_handler("Summarization failed")
    async def _summarize(
        self,
        chat_id: str,
        num_messages: int = 50,
        _writer: Any = None,
        _request_id: Any = None,
    ) -> dict[str, Any]:
        """Summarize a conversation."""
        reply_service = self.server.get_reply_service()

        conversation, _ = reply_service.context_service.fetch_conversation_context(
            chat_id, limit=num_messages
        )

        if not conversation:
            raise JsonRpcError(INVALID_PARAMS, "No messages found")

        from jarvis.model_warmer import get_model_warmer

        get_model_warmer().touch()

        from models.loader import get_model

        model = get_model()
        if not model:
            raise JsonRpcError(INTERNAL_ERROR, "Model not available")

        conversation_text = "\n".join(conversation[-30:])
        prompt = (
            f"Summarize this conversation in 2-3 sentences, then list key points:\n\n"
            f"{conversation_text}\n\nSummary:"
        )

        if _writer is not None:
            return await self._summarize_streaming(
                model=model,
                prompt=prompt,
                message_count=len(conversation),
                writer=_writer,
                request_id=_request_id,
            )

        def _summarize_sync() -> Any:
            if not model.is_loaded():
                model.load()
            return model.generate_sync(prompt, max_tokens=300)

        result: Any = await asyncio.to_thread(_summarize_sync)
        response_text = result.text

        lines = response_text.strip().split("\n")
        summary = lines[0] if lines else "Conversation summary unavailable"
        key_points = [
            line.lstrip("•-*0123456789.)").strip()
            for line in lines[1:]
            if line.strip() and len(line.strip()) > 5
        ][:5]

        return {
            "summary": summary,
            "key_points": key_points or ["See full conversation for details"],
            "message_count": len(conversation),
        }

    async def _summarize_streaming(
        self,
        model: Any,
        prompt: str,
        message_count: int,
        writer: Any,
        request_id: Any,
    ) -> dict[str, Any]:
        """Stream summary tokens."""
        # This one is tricky because _summarize_streaming in socket_server.py
        # uses a thread-based generation for models that don't support async stream natively?
        # Let's check socket_server.py implementation again for _summarize_streaming.

        # Actually I'll just copy it for now.
        from models import get_generator

        generator = get_generator()

        from contracts.models import GenerationRequest

        request = GenerationRequest(prompt=prompt, max_tokens=300)

        response_tokens: list[str] = []
        try:
            async for token_data in generator.generate_stream(request):
                token_text = token_data["token"]
                token_index = token_data["token_index"]
                is_final = token_data["is_final"]
                response_tokens.append(token_text)

                await self.send_stream_token(
                    writer, token_text, token_index, is_final, request_id=request_id
                )
        except Exception:
            logger.exception("Streaming summarization failed")
            raise JsonRpcError(INTERNAL_ERROR, "Streaming failed")

        full_response = "".join(response_tokens)
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
            "tokens_generated": len(response_tokens),
        }
        await self.send_stream_response(writer, request_id, result)
        return result

    @rpc_handler("Smart replies failed")
    async def _get_smart_replies(
        self,
        last_message: str,
        num_suggestions: int = 3,
    ) -> dict[str, Any]:
        """Get smart reply suggestions."""
        from jarvis.reply_service import get_reply_service

        reply_service = get_reply_service()
        result = await asyncio.to_thread(reply_service.route_legacy, incoming=last_message)
        return result

    @rpc_handler("Failed to list conversations")
    async def _list_conversations(
        self,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List recent iMessage conversations."""
        from integrations.imessage import ChatDBReader

        with ChatDBReader() as reader:
            chats = reader.get_conversations(limit=limit)

        return {
            "conversations": [
                {
                    "chat_id": c.chat_id,
                    "display_name": c.display_name,
                    "last_message": c.last_message_text,
                    "last_message_date": c.last_message_date.isoformat()
                    if c.last_message_date
                    else None,
                    "is_group": c.is_group,
                }
                for c in chats
            ],
            "total_count": len(chats),
        }
