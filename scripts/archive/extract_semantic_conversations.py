#!/usr/bin/env python3
"""
Extract conversations using semantic chunking for non-threaded messages.

For messages without explicit threading (89.8% of data), we use semantic
similarity to detect topic boundaries:

1. Group messages by chat_id (each contact/group separately)
2. Embed all messages using MLX embeddings
3. Compute similarity between consecutive messages
4. Similarity drop below threshold = topic boundary
5. Extract (context, my_response) pairs within each semantic chunk

THRESHOLD CALIBRATION (2026-01-31):
- Within-thread (same topic) messages have similarity >= 0.41
- Topic changes typically have similarity 0.35-0.45
- Calibrated threshold: 0.45 (10th percentile of all transitions)
- This means ~10% of message transitions are treated as topic changes

Research: Based on "Unsupervised Dialogue Topic Segmentation with
Utterance-Pair Coherence Scoring" approach (https://arxiv.org/pdf/2305.02747)

Usage:
    uv run python -m scripts.extract_semantic_conversations --chat-limit 10
    uv run python -m scripts.extract_semantic_conversations --sample 5
    uv run python -m scripts.extract_semantic_conversations --export semantic_convos.jsonl
    uv run python -m scripts.extract_semantic_conversations --threshold 0.50  # More boundaries
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class Message:
    """A single message."""
    rowid: int
    text: str
    is_from_me: bool
    date: datetime
    chat_id: int


@dataclass
class SemanticChunk:
    """A semantically coherent chunk of messages."""
    messages: list[Message]
    start_idx: int
    end_idx: int

    @property
    def size(self) -> int:
        return len(self.messages)


@dataclass
class ConversationPair:
    """A (context, my_response) pair from semantic chunking."""
    context_messages: list[Message]
    my_response: Message
    chunk_size: int
    position_in_chunk: int
    chat_id: int
    chat_identifier: str = ""
    is_group: bool = False

    @property
    def context_text(self) -> str:
        lines = []
        for msg in self.context_messages:
            who = "Me" if msg.is_from_me else "Them"
            lines.append(f"{who}: {msg.text}")
        return "\n".join(lines)

    @property
    def immediate_trigger(self) -> str:
        if self.context_messages:
            return self.context_messages[-1].text
        return ""

    def to_dict(self) -> dict:
        return {
            "context": [
                {"text": m.text, "is_from_me": m.is_from_me, "date": m.date.isoformat()}
                for m in self.context_messages
            ],
            "context_formatted": self.context_text,
            "immediate_trigger": self.immediate_trigger,
            "my_response": self.my_response.text,
            "my_response_date": self.my_response.date.isoformat(),
            "chunk_size": self.chunk_size,
            "position_in_chunk": self.position_in_chunk,
            "context_length": len(self.context_messages),
            "chat_identifier": self.chat_identifier,
            "is_group": self.is_group,
            "source": "semantic_chunking",
        }


def get_chat_db_path() -> Path:
    return Path.home() / "Library" / "Messages" / "chat.db"


def apple_to_datetime(ts: int | None) -> datetime:
    if ts is None:
        return datetime.now()
    if ts > 1e15:
        ts = ts / 1e9
    return datetime.fromtimestamp(ts + 978307200)


def get_embedder():
    """Get the MLX embedder (lazy load)."""
    from models.embeddings import get_mlx_embedder
    return get_mlx_embedder()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_semantic_boundaries(
    messages: list[Message],
    embeddings: np.ndarray,
    threshold: float = 0.45,  # Calibrated from threaded conversation data
    min_chunk_size: int = 2,
    max_chunk_size: int = 50,
) -> list[int]:
    """
    Find indices where semantic topic changes occur.

    Returns list of boundary indices (where new topics start).
    """
    if len(messages) < 2:
        return []

    boundaries = [0]  # First message always starts a chunk
    current_chunk_start = 0

    for i in range(1, len(messages)):
        chunk_size = i - current_chunk_start

        # Compute similarity with previous message
        sim = cosine_similarity(embeddings[i-1], embeddings[i])

        # Force boundary if chunk too large
        if chunk_size >= max_chunk_size:
            boundaries.append(i)
            current_chunk_start = i
            continue

        # Check for topic change (similarity drop)
        if sim < threshold and chunk_size >= min_chunk_size:
            boundaries.append(i)
            current_chunk_start = i

    return boundaries


def chunk_messages(
    messages: list[Message],
    embeddings: np.ndarray,
    threshold: float = 0.3,
) -> Iterator[SemanticChunk]:
    """Split messages into semantic chunks."""
    if not messages:
        return

    boundaries = find_semantic_boundaries(messages, embeddings, threshold)
    boundaries.append(len(messages))  # Add end boundary

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        yield SemanticChunk(
            messages=messages[start:end],
            start_idx=start,
            end_idx=end,
        )


def extract_pairs_from_chunk(
    chunk: SemanticChunk, chat_identifier: str, is_group: bool
) -> list[ConversationPair]:
    """Extract (context, my_response) pairs from a semantic chunk."""
    pairs = []

    for pos, msg in enumerate(chunk.messages):
        if msg.is_from_me and pos > 0:
            # Context = all messages before this one in the chunk
            context = chunk.messages[:pos]

            # Only include if there's at least one message from them
            has_their_message = any(not m.is_from_me for m in context)
            if not has_their_message:
                continue

            pairs.append(ConversationPair(
                context_messages=context,
                my_response=msg,
                chunk_size=chunk.size,
                position_in_chunk=pos + 1,
                chat_id=msg.chat_id,
                chat_identifier=chat_identifier,
                is_group=is_group,
            ))

    return pairs


def extract_semantic_conversations(
    db_path: Path,
    threshold: float = 0.45,  # Calibrated: 10th percentile of consecutive similarities
    chat_limit: int | None = None,
    verbose: bool = True,
) -> list[ConversationPair]:
    """
    Extract (context, my_response) pairs using semantic chunking.

    Each chat (contact/group) is processed INDIVIDUALLY:
    1. Get all non-threaded messages for that chat
    2. Embed each message
    3. Find semantic boundaries (similarity drops)
    4. Extract (context, my_response) pairs within each chunk

    Args:
        db_path: Path to chat.db
        threshold: Similarity threshold for topic boundaries (lower = more boundaries)
        chat_limit: Process only N chats (for testing)
        verbose: Print progress
    """
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()

        # Get all chats
        cursor.execute("""
            SELECT c.ROWID as chat_id, c.chat_identifier,
                   COUNT(DISTINCT h.ROWID) as participant_count
            FROM chat c
            LEFT JOIN chat_handle_join chj ON c.ROWID = chj.chat_id
            LEFT JOIN handle h ON chj.handle_id = h.ROWID
            GROUP BY c.ROWID
            ORDER BY c.ROWID
        """)
        chats = cursor.fetchall()

        if chat_limit:
            chats = chats[:chat_limit]

        if verbose:
            print(f"Processing {len(chats)} chats...")

        # Get embedder
        embedder = get_embedder()

        all_pairs: list[ConversationPair] = []
        total_messages = 0
        total_chunks = 0

        for chat_idx, chat in enumerate(chats):
            chat_id = chat["chat_id"]
            chat_identifier = chat["chat_identifier"] or f"chat_{chat_id}"
            is_group = chat["participant_count"] > 1

            # Get non-threaded messages for this chat
            cursor.execute("""
                SELECT m.ROWID, m.text, m.is_from_me, m.date
                FROM message m
                JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                WHERE cmj.chat_id = ?
                  AND m.text IS NOT NULL AND m.text != ''
                  AND (m.thread_originator_guid IS NULL OR m.thread_originator_guid = '')
                  AND m.guid NOT IN (
                      SELECT DISTINCT thread_originator_guid
                      FROM message
                      WHERE thread_originator_guid IS NOT NULL
                  )
                ORDER BY m.date
            """, (chat_id,))

            rows = cursor.fetchall()
            if len(rows) < 2:
                continue

            messages = [
                Message(
                    rowid=row["ROWID"],
                    text=row["text"],
                    is_from_me=bool(row["is_from_me"]),
                    date=apple_to_datetime(row["date"]),
                    chat_id=chat_id,
                )
                for row in rows
            ]

            total_messages += len(messages)

            # Embed messages in batches (service may have size limits)
            texts = [m.text for m in messages]

            # Batch embed to avoid HTTP payload limits
            batch_size = 50
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                # Clean texts (remove null bytes, limit length)
                batch_texts = [t.replace('\x00', '')[:2000] for t in batch_texts]
                try:
                    batch_emb = embedder.encode(batch_texts)
                    all_embeddings.append(batch_emb)
                except Exception as e:
                    # Fall back to one-by-one if batch fails
                    if verbose:
                        print(f"    Batch failed, falling back to single: {e}")
                    for t in batch_texts:
                        try:
                            emb = embedder.encode(t)
                            all_embeddings.append(emb)
                        except Exception:
                            # Skip problematic texts
                            all_embeddings.append(np.zeros((1, 384), dtype=np.float32))

            embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])

            # Chunk by semantic similarity
            for chunk in chunk_messages(messages, embeddings, threshold):
                total_chunks += 1
                pairs = extract_pairs_from_chunk(chunk, chat_identifier, is_group)
                all_pairs.extend(pairs)

            if verbose and (chat_idx + 1) % 10 == 0:
                print(f"  Processed {chat_idx + 1}/{len(chats)} chats, "
                      f"{total_messages:,} messages, {len(all_pairs):,} pairs...")

        if verbose:
            print(
                f"\nDone! {total_messages:,} messages → "
                f"{total_chunks:,} chunks → {len(all_pairs):,} pairs"
            )

        return all_pairs

    finally:
        conn.close()


def print_sample_pairs(pairs: list[ConversationPair], n: int = 5) -> None:
    """Print sample pairs for review."""
    import random

    samples = random.sample(pairs, min(n, len(pairs)))

    print(f"\n{'='*70}")
    print(f"SAMPLE SEMANTIC CONVERSATIONS (showing {len(samples)} of {len(pairs):,})")
    print('='*70)

    for i, pair in enumerate(samples, 1):
        print(f"\n{'─'*70}")
        print(f"CONVERSATION {i} (chunk: {pair.chunk_size}, pos: {pair.position_in_chunk})")
        print(f"Chat: {pair.chat_identifier[:30]}... | Group: {pair.is_group}")
        print(f"{'─'*70}")
        print("\n[CONTEXT]:")
        for msg in pair.context_messages[-5:]:
            who = "  Me" if msg.is_from_me else "Them"
            print(f"  {who}: {msg.text[:80]}")
        if len(pair.context_messages) > 5:
            print(f"  ... ({len(pair.context_messages) - 5} earlier messages)")
        print(f"\n[MY RESPONSE]: {pair.my_response.text[:200]}")


def export_pairs(
    pairs: list[ConversationPair],
    output_path: Path,
    limit: int | None = None,
) -> int:
    """Export pairs to JSONL file."""
    import random

    to_export = pairs.copy()
    random.shuffle(to_export)

    if limit:
        to_export = to_export[:limit]

    with open(output_path, "w") as f:
        for pair in to_export:
            f.write(json.dumps(pair.to_dict()) + "\n")

    return len(to_export)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract conversations using semantic chunking"
    )
    parser.add_argument("--threshold", type=float, default=0.45,
                       help="Similarity threshold for topic boundaries (default: 0.45, calibrated)")
    parser.add_argument("--chat-limit", type=int,
                       help="Process only N chats (for testing)")
    parser.add_argument("--sample", type=int, default=0,
                       help="Show N sample conversations")
    parser.add_argument("--export", type=str,
                       help="Export pairs to JSONL file")
    parser.add_argument("--limit", type=int,
                       help="Limit export to N pairs")
    parser.add_argument("--db-path", type=str,
                       help="Custom chat.db path")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else get_chat_db_path()

    if not db_path.exists():
        print(f"Error: chat.db not found at {db_path}")
        return

    print(f"Extracting semantic conversations from {db_path}...")
    print(f"Similarity threshold: {args.threshold}")
    print("-" * 60)

    pairs = extract_semantic_conversations(
        db_path,
        threshold=args.threshold,
        chat_limit=args.chat_limit,
    )

    # Statistics
    print("\n" + "=" * 60)
    print("EXTRACTION STATISTICS")
    print("=" * 60)
    print(f"Total (context, my_response) pairs: {len(pairs):,}")

    if pairs:
        context_lengths = [len(p.context_messages) for p in pairs]
        print("\nContext length distribution:")
        print(f"  Min: {min(context_lengths)}")
        print(f"  Max: {max(context_lengths)}")
        print(f"  Avg: {sum(context_lengths) / len(context_lengths):.1f}")

        # Group vs direct
        group_count = len([p for p in pairs if p.is_group])
        direct_count = len(pairs) - group_count
        print("\nBy chat type:")
        print(f"  Direct messages: {direct_count:,}")
        print(f"  Group chats:     {group_count:,}")

    if args.sample > 0:
        print_sample_pairs(pairs, args.sample)

    if args.export:
        export_path = Path(args.export)
        count = export_pairs(pairs, export_path, limit=args.limit)
        print(f"\n✅ Exported {count:,} pairs to {export_path}")


if __name__ == "__main__":
    main()
