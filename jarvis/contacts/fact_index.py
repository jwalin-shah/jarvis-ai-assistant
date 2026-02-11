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

import logging

import numpy as np

from jarvis.contacts.contact_profile import Fact

logger = logging.getLogger(__name__)

# Int8 quantization scale (matches vec_search.py)
_INT8_SCALE = 127.0

# Minimum similarity to include a fact (avoid returning irrelevant results)
_MIN_SIMILARITY = 0.3


def _fact_to_text(fact: Fact) -> str:
    """Convert a fact to a searchable text string.

    Examples:
        "likes_food: sushi"
        "works_at: Google (software engineer)"
        "is_family_of: Sarah (sister)"
    """
    text = f"{fact.predicate}: {fact.subject}"
    if fact.value:
        text += f" ({fact.value})"
    return text


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


def _ensure_vec_facts_table(conn) -> None:  # noqa: ANN001
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


def index_facts(facts: list[Fact], contact_id: str) -> int:
    """Embed and store facts in vec_facts table.

    Batch-encodes all fact texts, quantizes to int8, and inserts into
    the sqlite-vec virtual table. Skips facts already indexed.

    Args:
        facts: Facts to index.
        contact_id: Contact these facts belong to.

    Returns:
        Number of facts indexed.
    """
    if not facts:
        return 0

    from jarvis.db import get_db
    from jarvis.embedding_adapter import get_embedder

    db = get_db()
    embedder = get_embedder()

    # Build searchable text for each fact
    fact_texts = [_fact_to_text(f) for f in facts]

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
                f"SELECT fact_id FROM vec_facts WHERE fact_id IN ({placeholders})",  # noqa: S608
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
        text = _fact_to_text(fact)
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

    # Check if vec_facts exists and has data for this contact
    try:
        with db.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM vec_facts WHERE contact_id = ?",
                (contact_id,),
            ).fetchone()
            if row is None or row[0] == 0:
                return _fallback_get_facts(contact_id, limit)
    except Exception:
        return _fallback_get_facts(contact_id, limit)

    # Encode query
    embedder = get_embedder()
    query_emb = embedder.encode([query], normalize=True)[0]
    query_blob = _quantize(query_emb)

    # Vector search with contact_id filter
    with db.connection() as conn:
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
    for row in results:
        sim = _distance_to_similarity(row["distance"])
        if sim >= _MIN_SIMILARITY:
            relevant_ids.append(row["fact_id"])
        if len(relevant_ids) >= limit:
            break

    if not relevant_ids:
        return _fallback_get_facts(contact_id, limit)

    # Fetch full fact objects in one query
    placeholders = ",".join("?" * len(relevant_ids))
    with db.connection() as conn:
        fact_rows = conn.execute(
            f"""
            SELECT category, subject, predicate, value, confidence,
                   source_text, source_message_id, extracted_at,
                   linked_contact_id, valid_from, valid_until
            FROM contact_facts
            WHERE id IN ({placeholders})
            """,  # noqa: S608
            relevant_ids,
        ).fetchall()

    return [
        Fact(
            category=r["category"],
            subject=r["subject"],
            predicate=r["predicate"],
            value=r["value"],
            confidence=r["confidence"],
            source_text=r["source_text"] or "",
            source_message_id=r["source_message_id"],
            contact_id=contact_id,
            extracted_at=r["extracted_at"] or "",
            linked_contact_id=r["linked_contact_id"],
            valid_from=r["valid_from"],
            valid_until=r["valid_until"],
        )
        for r in fact_rows
    ]


def _fallback_get_facts(contact_id: str, limit: int) -> list[Fact]:
    """Fallback to unfiltered fact retrieval when vec_facts is unavailable."""
    from jarvis.contacts.fact_storage import get_facts_for_contact

    return get_facts_for_contact(contact_id)[:limit]
