#!/usr/bin/env python3
"""Preprocess all conversations: chunk and index for RAG.

This script:
1. Ensures MLX embedding service is running (starts if needed)
2. Reads all conversations from iMessage
3. Chunks each conversation by topic
4. Stores chunks in jarvis.db
5. Builds FAISS index from chunks

Usage:
    uv run python scripts/preprocess_chunks.py [--limit N] [--rebuild-index]
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from integrations.imessage import ChatDBReader
from jarvis.chunk_index import ChunkIndexBuilder, ChunkIndexConfig
from jarvis.contacts.contact_profile import ContactProfileBuilder, invalidate_profile_cache, save_profile
from jarvis.db import get_db
from jarvis.topics.topic_chunker import chunk_conversation

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

console = Console()


def parse_datetime(value: Any) -> datetime:
    """Parse datetime from DB value (handles string, datetime, None, or other types)."""
    if value is None:
        return datetime.now()
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        if not value.strip():  # Empty string
            return datetime.now()
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse datetime string '{value}': {e}, using now()")
            return datetime.now()
    # Handle other types (int, float, etc.) - log and use now()
    logger.warning(f"Unexpected datetime type: {type(value).__name__} = {value}, using now()")
    return datetime.now()


def ensure_mlx_service_running() -> bool:
    """Ensure MLX embedding service is running, start if needed.

    Returns:
        True if service is available, False otherwise.
    """
    try:
        from models.embeddings import get_mlx_embedder, start_mlx_service

        embedder = get_mlx_embedder()

        # Check if already running and model loaded
        if embedder.is_available():
            if embedder.is_loaded():
                logger.info("MLX embedding service is running with model loaded")
                return True
            else:
                # Service up but model not loaded, wait for it
                logger.info("MLX service running, waiting for model to load...")
                for _ in range(10):
                    time.sleep(1)
                    if embedder.is_loaded():
                        logger.info("MLX embedding model loaded")
                        return True
                logger.warning("Model not loaded after waiting")
                return True  # Still available, model will load on first request

        # Try to start it
        logger.info("MLX embedding service not running, attempting to start...")
        process = start_mlx_service()

        if process is None:
            logger.warning(
                "Could not start MLX embedding service. "
                "Start services with: uv run python -m jarvis services start"
            )
            return False

        # Wait for model to load (takes ~4-6 seconds)
        logger.info("Waiting for model to load...")
        for _ in range(15):
            time.sleep(1)
            if embedder.is_available() and embedder.is_loaded():
                logger.info("MLX embedding service started and model loaded")
                return True

        if embedder.is_available():
            logger.warning("MLX service started but model not fully loaded")
            return True  # Model will load on first request
        else:
            logger.warning("MLX embedding service not responding")
            return False

    except Exception as e:
        logger.error(f"Error checking/starting MLX service: {e}")
        logger.warning(
            "Please ensure services are running: uv run python -m jarvis services start"
        )
        return False


def preprocess_all_conversations(
    limit: int | None = None,
    rebuild_index: bool = False,
    min_messages: int = 5,
    skip_chunking: bool = False,
    incremental: bool = False,
    index_config: ChunkIndexConfig | None = None,
) -> dict[str, int]:
    """Preprocess all conversations: chunk and store in DB.

    Args:
        limit: Maximum number of conversations to process (None = all).
        rebuild_index: If True, rebuild FAISS index after chunking.
        min_messages: Minimum messages required to chunk a conversation.
        skip_chunking: If True, skip conversation processing and only build index.
        incremental: If True, only process new messages for conversations that already have chunks.

    Returns:
        Stats dictionary with counts.
    """
    # Ensure MLX embedding service is running
    if not ensure_mlx_service_running():
        logger.error("MLX embedding service is required but not available")
        return {"error": "MLX service not available"}

    db = get_db()
    db.init_schema()

    stats = {
        "conversations_processed": 0,
        "conversations_skipped": 0,
        "chunks_created": 0,
        "chunks_existing": 0,
        "chunks_incremental": 0,
        "errors": 0,
    }

    # Skip conversation processing if requested
    if skip_chunking:
        console.print("[yellow]Skipping conversation processing, using existing chunks[/yellow]")
        # Verify chunks exist
        chunk_count = db.count_chunks()
        console.print(f"[green]Found {chunk_count} existing chunks in database[/green]")
        if chunk_count == 0:
            console.print("[red]ERROR: No chunks found in database! Cannot build index without chunks.[/red]")
            return {"error": "No chunks found", "chunks_existing": 0}
    else:
        try:
            with ChatDBReader() as reader:
                conversations = reader.get_conversations(limit=limit or 10000)
                total = len(conversations)
                console.print(f"[green]Found {total} conversations to process[/green]")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TextColumn("({task.completed}/{task.total})"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        "[cyan]Processing conversations...",
                        total=total,
                    )

                    for i, conv in enumerate(conversations, 1):
                        display_name = conv.display_name or conv.chat_id
                        progress.update(
                            task,
                            description=f"[cyan]Processing: {display_name[:40]}...",
                            advance=0,  # We'll advance manually after processing
                        )

                        try:
                            # Check if conversation already has chunks (for incremental processing)
                            last_chunk = db.get_last_chunk_for_chat(conv.chat_id) if incremental else None

                            # Get messages for this conversation
                            all_messages = reader.get_messages(conv.chat_id, limit=10000)

                            # If incremental and we have a last chunk, only process new messages
                            if incremental and last_chunk:
                                last_end_time = parse_datetime(last_chunk.get("end_time"))
                                # Filter to only messages after the last chunk's end_time
                                messages = [msg for msg in all_messages if msg.date > last_end_time]

                                if len(messages) == 0:
                                    # No new messages, skip this conversation
                                    stats["conversations_skipped"] += 1
                                    progress.advance(task)
                                    continue

                                # For incremental chunks, we need to include some context from the last chunk
                                # Get a few messages before the cutoff for context
                                context_messages = [msg for msg in all_messages if msg.date <= last_end_time][-5:]
                                messages = context_messages + messages
                                stats["chunks_incremental"] += 1
                            else:
                                messages = all_messages

                            if len(messages) < min_messages:
                                stats["conversations_skipped"] += 1
                                progress.advance(task)
                                continue

                            # Chunk conversation (or new messages only)
                            chunks = chunk_conversation(
                                messages=messages,
                                chat_id=conv.chat_id,
                                contact_id=conv.chat_id,  # Use chat_id as contact_id for now
                            )

                            # Store chunks in DB
                            for chunk in chunks:
                                # Check if chunk already exists
                                existing = db.get_chunk(chunk.chunk_id)
                                if existing:
                                    stats["chunks_existing"] += 1
                                    continue

                                db.add_chunk(
                                    chunk_id=chunk.chunk_id,
                                    chat_id=chunk.chat_id,
                                    formatted_text=chunk.formatted_text,
                                    text_for_embedding=chunk.text_for_embedding,
                                    contact_id=None,  # TODO: resolve contact_id from chat_id
                                    label=chunk.label,
                                    keywords=list(chunk.keywords),
                                    start_time=chunk.start_time,
                                    end_time=chunk.end_time,
                                    message_count=chunk.message_count,
                                    has_my_response=chunk.has_my_response,
                                    my_message_count=chunk.my_message_count,
                                    their_message_count=chunk.their_message_count,
                                    boundary_reason=chunk.boundary_reason,
                                    last_trigger=chunk.last_trigger,
                                    last_response=chunk.last_response,
                                )
                                stats["chunks_created"] += 1

                            stats["conversations_processed"] += 1
                            progress.advance(task)

                        except Exception as e:
                            logger.error(f"Error processing {conv.chat_id}: {e}", exc_info=True)
                            stats["errors"] += 1
                            progress.advance(task)

        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            stats["errors"] += 1

        console.print(f"[green]✓[/green] Preprocessing complete: {stats}")

    # Build index if requested
    if rebuild_index:
        console.print("\n[cyan]Building FAISS index...[/cyan]")
        try:
            # Load all chunks from DB
            chunk_rows = db.get_all_chunks()
            if not chunk_rows:
                console.print("[yellow]No chunks found to index[/yellow]")
                return stats

            console.print(f"[green]Found {len(chunk_rows)} chunks to index[/green]")

            # Convert DB chunks to TopicChunk-like objects for indexing
            # Create a minimal class that matches TopicChunk interface
            from dataclasses import dataclass
            from datetime import datetime

            @dataclass
            class ChunkForIndexing:
                """Minimal chunk for indexing - uses stored text_for_embedding."""
                chunk_id: str
                chat_id: str
                contact_id: str | None
                text_for_embedding: str  # Stored value from DB
                formatted_text: str
                label: str
                keywords: set[str]
                start_time: datetime
                end_time: datetime
                message_count: int
                has_my_response: bool
                my_message_count: int
                their_message_count: int

            topic_chunks = []
            for row in chunk_rows:
                start_time = parse_datetime(row.get("start_time"))
                end_time = parse_datetime(row.get("end_time"))

                topic_chunks.append(
                    ChunkForIndexing(
                        chunk_id=row["chunk_id"],
                        chat_id=row["chat_id"],
                        contact_id=str(row["contact_id"]) if row.get("contact_id") else None,
                        text_for_embedding=row.get("text_for_embedding", row.get("formatted_text", "")),
                        formatted_text=row.get("formatted_text", ""),
                        label=row.get("label", ""),
                        keywords=set(row.get("keywords", [])),
                        start_time=start_time,
                        end_time=end_time,
                        message_count=int(row.get("message_count", 0)),
                        has_my_response=bool(row.get("has_my_response", False)),
                        my_message_count=int(row.get("my_message_count", 0)),
                        their_message_count=int(row.get("their_message_count", 0)),
                    )
                )

            # Build index with progress
            def progress_callback(stage: str, progress_pct: float, message: str) -> None:
                """Progress callback for index building."""
                console.print(f"[cyan]{stage}[/cyan]: {message} ({progress_pct:.0%})")

            builder = ChunkIndexBuilder(config=index_config)
            result = builder.build_index(topic_chunks, db, progress_callback=progress_callback)
            console.print(f"[green]✓[/green] Index built: {result.get('chunks_indexed', 0)} chunks indexed")

        except Exception as e:
            console.print(f"[red]✗[/red] Error building index: {e}")
            logger.error(f"Error building index: {e}", exc_info=True)
            stats["errors"] += 1

    return stats


def build_contact_profiles(
    limit: int | None = None,
    min_messages: int = 10,
    with_topics: bool = False,
) -> dict[str, int]:
    """Build contact profiles for all conversations.

    Args:
        limit: Maximum conversations to process.
        min_messages: Minimum messages for a conversation to get a profile.
        with_topics: If True, compute embeddings for HDBSCAN topic discovery.
            Requires the MLX embedding service to be running.

    Returns:
        Stats dictionary.
    """
    stats = {
        "profiles_built": 0,
        "profiles_skipped": 0,
        "profile_errors": 0,
    }

    builder = ContactProfileBuilder(min_messages=min_messages)

    # Initialize embedder once if topic discovery requested
    embedder = None
    if with_topics:
        try:
            from jarvis.embedding_adapter import get_embedder

            embedder = get_embedder()
            console.print("[green]Embedder initialized for topic discovery[/green]")
        except Exception as e:
            console.print(
                f"[yellow]Could not initialize embedder, skipping topics: {e}[/yellow]"
            )
            embedder = None

    try:
        with ChatDBReader() as reader:
            conversations = reader.get_conversations(limit=limit or 10000)
            console.print(
                f"[green]Found {len(conversations)} conversations for profile building[/green]"
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Building contact profiles...",
                    total=len(conversations),
                )

                for conv in conversations:
                    display_name = conv.display_name or conv.chat_id
                    progress.update(
                        task,
                        description=f"[cyan]Profile: {display_name[:40]}...",
                    )

                    try:
                        messages = reader.get_messages(conv.chat_id, limit=5000)

                        if len(messages) < min_messages:
                            stats["profiles_skipped"] += 1
                            progress.advance(task)
                            continue

                        # Compute embeddings for topic discovery if requested
                        embeddings = None
                        if embedder is not None:
                            texts = [m.text for m in messages if m.text]
                            if len(texts) >= 30:
                                try:
                                    embeddings = embedder.encode(texts)
                                except Exception as e:
                                    logger.warning(
                                        "Embedding failed for %s: %s",
                                        conv.chat_id, e,
                                    )

                        profile = builder.build_profile(
                            contact_id=conv.chat_id,
                            messages=messages,
                            contact_name=conv.display_name,
                            embeddings=embeddings,
                        )

                        if save_profile(profile):
                            stats["profiles_built"] += 1
                        else:
                            stats["profile_errors"] += 1

                    except Exception as e:
                        logger.error(
                            "Error building profile for %s: %s", conv.chat_id, e
                        )
                        stats["profile_errors"] += 1

                    progress.advance(task)

        # Clear LRU cache so new profiles are picked up
        invalidate_profile_cache()

    except Exception as e:
        logger.error("Fatal error building profiles: %s", e)
        stats["profile_errors"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess conversations: chunk and index")
    parser.add_argument("--limit", type=int, help="Maximum conversations to process")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild FAISS index after chunking")
    parser.add_argument("--min-messages", type=int, default=5, help="Minimum messages to chunk (default: 5)")
    parser.add_argument("--index-only", action="store_true", help="Skip chunking, only rebuild index from existing chunks")
    parser.add_argument("--incremental", action="store_true", help="Only process new messages for conversations that already have chunks")
    parser.add_argument("--index-type", type=str, default=None, choices=["flat", "ivf", "ivfpq_4x", "ivfpq_8x"], help="FAISS index type (default: ivfpq_4x)")
    parser.add_argument(
        "--skip-profiles", action="store_true",
        help="Skip building contact profiles",
    )
    parser.add_argument(
        "--profiles-only", action="store_true",
        help="Only build contact profiles (skip chunking/indexing)",
    )
    parser.add_argument(
        "--with-topics", action="store_true",
        help="Compute embeddings for HDBSCAN topic discovery (requires MLX embedding service)",
    )
    args = parser.parse_args()

    # Profiles-only mode
    if args.profiles_only:
        console.print("[cyan]Building contact profiles only...[/cyan]")
        profile_stats = build_contact_profiles(
            limit=args.limit,
            min_messages=args.min_messages,
            with_topics=args.with_topics,
        )
        console.print(f"\n[green]✓[/green] Profiles built: {profile_stats['profiles_built']}")
        console.print(f"  Skipped: {profile_stats['profiles_skipped']}")
        if profile_stats['profile_errors'] > 0:
            console.print(f"  [red]Errors: {profile_stats['profile_errors']}[/red]")
        return

    # Build index config if custom index type specified
    index_config = None
    if args.index_type:
        index_config = ChunkIndexConfig(index_type=args.index_type)

    stats = preprocess_all_conversations(
        limit=args.limit,
        rebuild_index=args.rebuild_index or args.index_only,
        min_messages=args.min_messages,
        skip_chunking=args.index_only,
        incremental=args.incremental,
        index_config=index_config,
    )

    # Phase 2: Build contact profiles (unless skipped)
    if not args.skip_profiles and not args.index_only:
        console.print("\n[cyan]Phase 2: Building contact profiles...[/cyan]")
        profile_stats = build_contact_profiles(
            limit=args.limit,
            min_messages=args.min_messages,
            with_topics=args.with_topics,
        )
        stats.update(profile_stats)

    # Print summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Preprocessing Summary[/bold]")
    console.print("=" * 60)
    console.print(f"Conversations processed: [green]{stats.get('conversations_processed', 0)}[/green]")
    console.print(f"Conversations skipped: [yellow]{stats.get('conversations_skipped', 0)}[/yellow]")
    console.print(f"Chunks created: [green]{stats.get('chunks_created', 0)}[/green]")
    console.print(f"Chunks existing: [dim]{stats.get('chunks_existing', 0)}[/dim]")
    if stats.get("chunks_incremental", 0) > 0:
        console.print(f"Chunks incremental: [cyan]{stats.get('chunks_incremental', 0)}[/cyan]")
    if stats.get("profiles_built", 0) > 0 or stats.get("profiles_skipped", 0) > 0:
        console.print(f"Profiles built: [green]{stats.get('profiles_built', 0)}[/green]")
        console.print(f"Profiles skipped: [dim]{stats.get('profiles_skipped', 0)}[/dim]")
    total_errors = stats.get("errors", 0) + stats.get("profile_errors", 0)
    if total_errors > 0:
        console.print(f"Errors: [red]{total_errors}[/red]")
    else:
        console.print("Errors: [green]0[/green]")
    console.print("=" * 60)

    if stats.get("errors", 0) > 0 or stats.get("profile_errors", 0) > 0 or "error" in stats:
        sys.exit(1)


if __name__ == "__main__":
    main()
