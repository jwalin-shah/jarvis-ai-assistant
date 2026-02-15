"""Reply logging operations mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.db.core import JarvisDBBase


class ReplyLogMixin:
    """Mixin providing reply logging operations."""

    def save_reply_log(
        self: JarvisDBBase,
        chat_id: str | None,
        contact_id: str | None,
        incoming_text: str,
        classification_json: str,
        rag_context_json: str,
        final_prompt: str,
        response_text: str,
        confidence: float,
        metadata_json: str,
    ) -> int:
        """Save a full generation log for traceability.

        Args:
            chat_id: iMessage chat identifier.
            contact_id: Associated contact identifier.
            incoming_text: The user's input message.
            classification_json: JSON string of classification results.
            rag_context_json: JSON string of retrieved RAG documents.
            final_prompt: The actual prompt sent to the LLM.
            response_text: The generated response.
            confidence: Numeric confidence score (0-1).
            metadata_json: Additional metadata (latencies, model info, etc.).

        Returns:
            The ID of the newly created log entry.
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO reply_logs (
                    chat_id, contact_id, incoming_text, classification_json,
                    rag_context_json, final_prompt, response_text,
                    confidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    contact_id,
                    incoming_text,
                    classification_json,
                    rag_context_json,
                    final_prompt,
                    response_text,
                    confidence,
                    metadata_json,
                ),
            )
            return cursor.lastrowid

    def get_recent_reply_logs(
        self: JarvisDBBase, limit: int = 20, chat_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve recent reply logs for inspection."""
        with self.connection() as conn:
            query = "SELECT * FROM reply_logs"
            params = []
            if chat_id:
                query += " WHERE chat_id = ?"
                params.append(chat_id)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor]
