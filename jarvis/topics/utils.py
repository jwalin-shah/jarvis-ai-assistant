"""Shared utilities for conversation segmentation.

Provides common logic for message sorting, junk filtering, and
boundary detection helpers to ensure consistency across different
segmenter implementations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.contracts.imessage import Message

logger = logging.getLogger(__name__)


def prepare_messages_for_segmentation(messages: list[Message]) -> tuple[list[Message], list[str]]:
    """Sort and filter messages for segmentation.

    Args:
        messages: Raw list of messages

    Returns:
        Tuple of (filtered_messages, normalized_texts)
    """
    if not messages:
        return [], []

    # Sort by date (oldest first)
    sorted_messages = sorted(messages, key=lambda m: m.date)

    from jarvis.contacts.junk_filters import is_junk_message
    from jarvis.text_normalizer import normalize_text

    filtered_messages: list[Message] = []
    filtered_norm_texts: list[str] = []

    for m in sorted_messages:
        raw_text = m.text or ""
        norm_text = normalize_text(
            raw_text,
            expand_slang=True,
            filter_garbage=True,
            filter_attributed_artifacts=True,
            strip_signatures=True,
        )
        if not norm_text:
            continue
        if is_junk_message(norm_text, m.chat_id):
            continue
        filtered_messages.append(m)
        filtered_norm_texts.append(norm_text)

    return filtered_messages, filtered_norm_texts


def get_embeddings_for_segmentation(
    messages: list[Message],
    norm_texts: list[str],
    pre_fetched_embeddings: dict[int, Any] | None = None,
) -> list[Any]:
    """Get embeddings for a list of messages.

    Args:
        messages: List of filtered messages
        norm_texts: List of normalized message texts
        pre_fetched_embeddings: Optional cached embeddings

    Returns:
        List of embeddings (None for failed/skipped messages)
    """
    from jarvis.embedding_adapter import get_embedder

    embedder = get_embedder()
    n = len(messages)
    embeddings = [None] * n

    # 1. Use pre-fetched if available
    if pre_fetched_embeddings:
        for i, m in enumerate(messages):
            if m.id in pre_fetched_embeddings:
                embeddings[i] = pre_fetched_embeddings[m.id]

    # 2. Batch encode remaining
    to_encode_indices = []
    to_encode_texts = []

    for i, emb in enumerate(embeddings):
        if emb is None:
            text = norm_texts[i]
            if text and len(text.strip()) >= 3:
                to_encode_indices.append(i)
                to_encode_texts.append(text)

    if to_encode_texts:
        try:
            new_embs = embedder.embed_batch(to_encode_texts)
            for idx, emb in zip(to_encode_indices, new_embs):
                embeddings[idx] = emb
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}")
            # Fallback to individual encoding for remaining
            for idx, text in zip(to_encode_indices, to_encode_texts):
                try:
                    embeddings[idx] = embedder.encode(text, normalize=True)
                except Exception:
                    pass

    return embeddings
