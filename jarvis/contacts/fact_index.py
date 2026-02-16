"""Semantic fact retrieval using sqlite-vec embeddings.

Embeds contact facts into a vec_facts virtual table and retrieves
semantically relevant facts at generation time, replacing the naive
get_facts_for_contact() which returns all facts sorted by confidence.

Usage:
    from jarvis.contacts.fact_index import search_relevant_facts

    # At generation time: find facts relevant to incoming message
    facts = search_relevant_facts("want to grab food?", chat_id, limit=5)
    # Returns food-related facts, not work facts
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime

import numpy as np

from jarvis.contacts.contact_profile import Fact

logger = logging.getLogger(__name__)

# Int8 quantization scale (matches vec_search.py)
_INT8_SCALE = 127.0

# Minimum similarity to include a fact (avoid returning irrelevant results)
_MIN_SIMILARITY = 0.3


def _quantize(embedding: np.ndarray) -> bytes:
    """Quantize float32 embedding to int8 bytes for sqlite-vec."""
    return (embedding * _INT8_SCALE).astype(np.int8).tobytes()


def _distance_to_similarity(distance: float) -> float:
    """Convert int8-quantized L2 distance to approximate cosine similarity.

    For normalized embeddings quantized to int8 via (emb * 127):
        cos_sim = 1 - (L2_int8 / 127)^2 / 2
    """
    cos_sim = 1.0 - (distance / _INT8_SCALE) ** 2 / 2.0
    return max(0.0, min(1.0, cos_sim))


def _ensure_vec_facts_table(conn: sqlite3.Connection) -> None:
    """Create vec_facts table if it doesn't exist."""
    try:
        conn.execute("SELECT 1 FROM vec_facts LIMIT 0")
    except Exception:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_facts USING vec0(
                embedding int8[384] distance_metric=L2,
                contact_id text,
                +fact_id INTEGER,
                +fact_text TEXT
            )
            """
        )


def index_facts(
    facts: list[Fact],
    contact_id: str,
    embeddings: np.ndarray | None = None,
) -> int:
    """Embed and store facts in vec_facts table.

    Batch-encodes all fact texts (unless embeddings provided), quantizes to int8,
    and inserts into the sqlite-vec virtual table. Skips facts already indexed.

    Args:
        facts: Facts to index.
        contact_id: Contact these facts belong to.
        embeddings: Optional pre-computed embeddings from deduplication phase.
            If provided, skips re-encoding to avoid double computation.

    Returns:
        Number of facts indexed.
    """
    if not facts:
        return 0

    from jarvis.db import get_db

    db = get_db()

    # Build searchable text for each fact
    fact_texts = [f.to_searchable_text() for f in facts]

    # Use provided embeddings or encode fresh
    if embeddings is None:
        from jarvis.embedding_adapter import get_embedder

        embedder = get_embedder()
        # Batch encode all facts at once (GPU lock handled inside encode())
        embeddings = embedder.encode(fact_texts, normalize=True)

    indexed = 0
    with db.connection() as conn:
        _ensure_vec_facts_table(conn)

        # Fetch all fact IDs with their lookup keys in ONE query
        rows = conn.execute(
            "SELECT id, category, subject, predicate FROM contact_facts WHERE contact_id = ?",
            (contact_id,),
        ).fetchall()
        # O(1) lookup: (category, subject, predicate) -> fact_id
        fact_id_lookup: dict[tuple[str, str, str], int] = {
            (row["category"], row["subject"], row["predicate"]): row["id"] for row in rows
        }
        all_fact_ids = set(fact_id_lookup.values())

        # Check which are already in vec_facts
        if all_fact_ids:
            placeholders = ",".join("?" * len(all_fact_ids))
            existing = conn.execute(
                f"SELECT fact_id FROM vec_facts WHERE fact_id IN ({placeholders})",  # noqa: S608  # nosec B608
                list(all_fact_ids),
            ).fetchall()
            already_indexed = {row[0] for row in existing}
        else:
            already_indexed = set()

        # Collect rows to insert in batch
        insert_batch = []
        for fact, emb, text in zip(facts, embeddings, fact_texts):
            key = (fact.category, fact.subject, fact.predicate)
            fact_id = fact_id_lookup.get(key)
            if fact_id is None or fact_id in already_indexed:
                continue

            insert_batch.append((_quantize(emb), contact_id, fact_id, text))

        # Batch INSERT all new facts at once
        if insert_batch:
            conn.executemany(
                """
                INSERT INTO vec_facts(embedding, contact_id, fact_id, fact_text)
                VALUES (vec_int8(?), ?, ?, ?)
                """,
                insert_batch,
            )
        indexed = len(insert_batch)

    if indexed:
        logger.info("Indexed %d facts for %s into vec_facts", indexed, contact_id[:16])
    return indexed


def reindex_all_facts() -> int:
    """Reindex all existing facts in the database.

    Clears vec_facts and re-embeds everything. Used by backfill --reindex.

    Returns:
        Total facts indexed.
    """
    from jarvis.db import get_db
    from jarvis.embedding_adapter import get_embedder

    db = get_db()
    embedder = get_embedder()

    with db.connection() as conn:
        _ensure_vec_facts_table(conn)

        # Clear existing index
        conn.execute("DELETE FROM vec_facts")

        # Load all facts
        rows = conn.execute(
            """
            SELECT id, contact_id, category, subject, predicate, value
            FROM contact_facts
            ORDER BY contact_id
            """
        ).fetchall()

    if not rows:
        return 0

    # Build text for each fact
    fact_data = []
    texts = []
    for row in rows:
        fact = Fact(
            category=row["category"],
            subject=row["subject"],
            predicate=row["predicate"],
            value=row["value"] or "",
        )
        text = fact.to_searchable_text()
        texts.append(text)
        fact_data.append((row["id"], row["contact_id"], text))

    # Batch encode all at once
    embeddings = embedder.encode(texts, normalize=True)

    # Batch insert
    with db.connection() as conn:
        _ensure_vec_facts_table(conn)
        batch = []
        for (fact_id, contact_id, text), emb in zip(fact_data, embeddings):
            batch.append((_quantize(emb), contact_id, fact_id, text))

        conn.executemany(
            """
            INSERT INTO vec_facts(embedding, contact_id, fact_id, fact_text)
            VALUES (vec_int8(?), ?, ?, ?)
            """,
            batch,
        )

    total = len(batch)
    logger.info("Reindexed %d facts into vec_facts", total)
    return total


def search_relevant_facts(
    query: str,
    contact_id: str,
    limit: int = 5,
) -> list[Fact]:
    """Find facts most relevant to the incoming message.

    Embeds the query text and searches vec_facts for the closest matches,
    filtered by contact_id. Falls back to get_facts_for_contact() if
    vec_facts is empty or unavailable.

    Args:
        query: Incoming message text.
        contact_id: Contact to search facts for.
        limit: Maximum facts to return.

    Returns:
        List of Fact objects sorted by relevance.
    """
    from jarvis.db import get_db
    from jarvis.embedding_adapter import get_embedder

    db = get_db()

    # Single connection for all 3 DB operations (was 3 separate connections)
    try:
        with db.connection() as conn:
            # 1. Check if vec_facts has data for this contact
            row = conn.execute(
                "SELECT COUNT(*) FROM vec_facts WHERE contact_id = ?",
                (contact_id,),
            ).fetchone()
            if row is None or row[0] == 0:
                return _fallback_get_facts(contact_id, limit)

            # Encode query (GPU work happens outside DB lock)
            embedder = get_embedder()
            query_emb = embedder.encode([query], normalize=True)[0]
            query_blob = _quantize(query_emb)

            # 2. Vector search with contact_id filter
            results = conn.execute(
                """
                SELECT rowid, distance, fact_id, fact_text, contact_id
                FROM vec_facts
                WHERE embedding MATCH vec_int8(?)
                AND k = ?
                AND contact_id = ?
                """,
                (query_blob, limit * 2, contact_id),
            ).fetchall()

            if not results:
                return _fallback_get_facts(contact_id, limit)

            # Filter by similarity threshold and collect fact IDs
            relevant_ids = []

            # Decay constants
            one_year_s = 365 * 24 * 3600
            two_years_s = 2 * one_year_s

            for row in results:
                sim = _distance_to_similarity(row["distance"])

                # Apply temporal decay to similarity
                # We need the extraction time, but it's not in vec_facts.
                # We'll fetch it in the next step, but for initial filtering,
                # we use the raw similarity.
                if sim >= _MIN_SIMILARITY:
                    relevant_ids.append(row["fact_id"])
                if len(relevant_ids) >= limit:
                    break

            if not relevant_ids:
                return _fallback_get_facts(contact_id, limit)

            # 3. Fetch full fact objects in one query
            fact_rows = conn.execute(
                """
                SELECT category, subject, predicate, value, confidence,
                       source_text, source_message_id, extracted_at,
                       linked_contact_id, valid_from, valid_until
                FROM contact_facts
                WHERE id IN (SELECT value FROM json_each(?))
                """,
                (json.dumps(relevant_ids),),
            ).fetchall()
    except Exception:
        return _fallback_get_facts(contact_id, limit)

    # Filter out expired facts and apply confidence decay
    now = datetime.now()
    now_iso = now.isoformat()
    facts_with_scores: list[tuple[Fact, float]] = []

    # Categories that are considered durable and do not decay over time
    durable_categories = {"relationship", "personal", "health"}
    # Predicates that are considered durable regardless of category
    durable_predicates = {"born_on", "birthday_is", "from", "identity"}

    for r in fact_rows:
        valid_until = r["valid_until"]
        if valid_until and valid_until < now_iso:
            logger.debug(
                "Skipping expired fact: %s %s %s", r["subject"], r["predicate"], r["value"]
            )
            continue

        category = r["category"]
        predicate = r["predicate"]

        # Calculate decay factor
        decay_factor = 1.0

        # Only apply decay if category/predicate is not durable
        is_durable = category in durable_categories or any(
            p in predicate for p in durable_predicates
        )

        if not is_durable:
            try:
                if r["extracted_at"]:
                    # Parse extracted_at (handling various formats)
                    try:
                        ext_at = datetime.fromisoformat(r["extracted_at"].replace("Z", "+00:00"))
                    except ValueError:
                        # Fallback for older sqlite formats
                        ext_at = datetime.strptime(
                            r["extracted_at"].split(".")[0], "%Y-%m-%d %H:%M:%S"
                        )

                    age_seconds = (now - ext_at).total_seconds()

                    if age_seconds > two_years_s:
                        decay_factor = 0.5  # Max decay
                    elif age_seconds > one_year_s:
                        # Linear decay from 1.0 to 0.5 between 1 and 2 years
                        decay_factor = 1.0 - 0.5 * ((age_seconds - one_year_s) / one_year_s)
            except Exception as e:
                logger.debug("Failed to calculate decay for fact: %s", e)

        fact = Fact(
            category=r["category"],
            subject=r["subject"],
            predicate=r["predicate"],
            value=r["value"],
            confidence=r["confidence"] * decay_factor,
            source_text=r["source_text"] or "",
            source_message_id=r["source_message_id"],
            contact_id=contact_id,
            extracted_at=r["extracted_at"] or "",
            linked_contact_id=r["linked_contact_id"],
            valid_from=r["valid_from"],
            valid_until=r["valid_until"],
        )
        # We'll use the final confidence to sort
        facts_with_scores.append((fact, fact.confidence))

    # Sort by confidence (which now includes decay)
    facts_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [f for f, s in facts_with_scores[:limit]]


def find_conflicting_facts(
    fact: Fact,
    contact_id: str,
    threshold: float = 0.7,
) -> list[int]:
    """Find existing facts that semantically conflict with a new fact.

    Used for active degradation: if a new fact is extracted that contradicts
    an old one (e.g., different location for same person), the old one
    should be updated or deprecated.

    Args:
        fact: The new fact to check.
        contact_id: The contact it belongs to.
        threshold: Similarity threshold for identifying related facts.

    Returns:
        List of fact_ids that might be contradicted.
    """
    from jarvis.db import get_db
    from jarvis.embedding_adapter import get_embedder

    db = get_db()
    embedder = get_embedder()

    # Search for related facts (same subject/predicate or semantically similar)
    # We use a broad search first
    search_text = fact.to_searchable_text()
    query_emb = embedder.encode([search_text], normalize=True)[0]
    query_blob = _quantize(query_emb)

    try:
        with db.connection() as conn:
            # Find facts with same subject and predicate but DIFFERENT values
            # This is a direct logical conflict check
            rows = conn.execute(
                """
                SELECT id, value FROM contact_facts
                WHERE contact_id = ? AND subject = ? AND predicate = ? AND attribution = ?
                AND value != ?
                """,
                (contact_id, fact.subject, fact.predicate, fact.attribution, fact.value),
            ).fetchall()

            conflicts = [row["id"] for row in rows]

            # Also do a semantic search for related facts that might conflict
            # (handles "lives in Austin" vs "moved to Dallas")
            results = conn.execute(
                """
                SELECT fact_id, distance FROM vec_facts
                WHERE embedding MATCH vec_int8(?)
                AND k = 10 AND contact_id = ?
                """,
                (query_blob, contact_id),
            ).fetchall()

            for row in results:
                fid = row["fact_id"]
                sim = _distance_to_similarity(row["distance"])
                if sim >= threshold and fid not in conflicts:
                    # Check if it's actually a conflict (different value)
                    # We'll fetch the value to be sure
                    f_row = conn.execute(
                        "SELECT value FROM contact_facts WHERE id = ?", (fid,)
                    ).fetchone()
                    if f_row and f_row["value"] != fact.value:
                        conflicts.append(fid)

            return conflicts
    except Exception as e:
        logger.debug("Conflict detection failed: %s", e)
        return []


def _fallback_get_facts(contact_id: str, limit: int) -> list[Fact]:
    """Fallback to unfiltered fact retrieval when vec_facts is unavailable."""
    from jarvis.contacts.fact_storage import get_facts_for_contact

    return get_facts_for_contact(contact_id)[:limit]
