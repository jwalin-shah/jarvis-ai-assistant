"""SQL Query Builder - Centralized safe SQL generation.

Replaces scattered f-string SQL construction with type-safe, injection-proof
query building. All IN clauses and dynamic SQL is handled here.
"""

from __future__ import annotations

from typing import Any


class QueryBuilder:
    """Build SQL queries safely with automatic parameter handling."""

    # Maximum SQLite parameters per query
    SQLITE_MAX_PARAMS = 900

    @staticmethod
    def in_clause(values: list[Any]) -> tuple[str, list[Any]]:
        """Build safe IN clause with validated placeholders.

        Args:
            values: List of values for IN clause

        Returns:
            Tuple of (placeholder_string, values_list)

        Example:
            ph, vals = QueryBuilder.in_clause([1, 2, 3])
            # ph = "?,?,?", vals = [1, 2, 3]
            sql = f"SELECT * FROM t WHERE id IN ({ph})"
        """
        if not values:
            return "NULL", []  # Empty IN clause

        placeholders = ",".join("?" * len(values))
        # SECURITY: Validate only contains ? and ,
        allowed = set("?, ")
        if not set(placeholders).issubset(allowed):
            raise ValueError(f"Invalid characters in placeholders: {placeholders}")

        return placeholders, values

    @staticmethod
    def chunked_in_clause(values: list[Any], chunk_size: int = 900) -> list[tuple[str, list[Any]]]:
        """Build multiple IN clauses for large value lists.

        SQLite has a limit on parameters per query (999 by default).
        This splits large lists into multiple chunks.

        Args:
            values: List of values
            chunk_size: Max values per chunk (default 900 for safety)

        Returns:
            List of (placeholder_string, values_chunk) tuples
        """
        chunks = []
        for i in range(0, len(values), chunk_size):
            chunk = values[i : i + chunk_size]
            ph, vals = QueryBuilder.in_clause(chunk)
            chunks.append((ph, vals))
        return chunks

    @staticmethod
    def select_in(
        table: str, columns: str, where_column: str, values: list[Any]
    ) -> tuple[str, list[Any]]:
        """Build SELECT ... WHERE col IN (...) query.

        Args:
            table: Table name (not escaped - must be trusted!)
            columns: Column list (e.g., "id, name")
            where_column: Column to filter on
            values: Values for IN clause

        Returns:
            Tuple of (sql_query, parameters)
        """
        placeholders, params = QueryBuilder.in_clause(values)
        sql = f"SELECT {columns} FROM {table} WHERE {where_column} IN ({placeholders})"  # nosec B608
        return sql, params

    @staticmethod
    def delete_in(table: str, where_column: str, values: list[Any]) -> tuple[str, list[Any]]:
        """Build DELETE ... WHERE col IN (...) query."""
        placeholders, params = QueryBuilder.in_clause(values)
        sql = f"DELETE FROM {table} WHERE {where_column} IN ({placeholders})"  # nosec B608
        return sql, params

    @staticmethod
    def select_with_and_in(
        table: str, columns: str, in_conditions: dict[str, list[Any]], additional_where: str = ""
    ) -> tuple[str, list[Any]]:
        """Build SELECT with multiple IN clauses joined by AND.

        Args:
            table: Table name
            columns: Columns to select
            in_conditions: Dict of {column: [values]}
            additional_where: Additional WHERE clause (use ? for params)

        Returns:
            Tuple of (sql_query, all_parameters)
        """
        params = []
        conditions = []

        for col, values in in_conditions.items():
            placeholders, vals = QueryBuilder.in_clause(values)
            conditions.append(f"{col} IN ({placeholders})")
            params.extend(vals)

        if additional_where:
            conditions.append(additional_where)

        where_clause = " AND ".join(conditions)
        sql = f"SELECT {columns} FROM {table} WHERE {where_clause}"  # nosec B608

        return sql, params


class VecSearchQueries:
    """Pre-built queries for vec_search module."""

    @staticmethod
    def insert_vec_chunks(chunk_count: int) -> tuple[str, list[str]]:
        """Get INSERT statement for vec_chunks with RETURNING."""
        columns = [
            "embedding",
            "contact_id",
            "chat_id",
            "response_da_type",
            "source_timestamp",
            "quality_score",
            "topic_label",
            "context_text",
            "reply_text",
            "formatted_text",
            "keywords_json",
            "message_count",
            "source_type",
            "source_id",
        ]
        cols = ", ".join(columns)
        placeholders = ", ".join(["vec_int8(?)", "?" * 13])
        sql = f"INSERT INTO vec_chunks({cols}) VALUES ({placeholders}) RETURNING rowid"  # nosec B608
        return sql, columns

    @staticmethod
    def search_vec_messages(limit: int, filter_chat_id: bool = False) -> tuple[str, list[Any]]:
        """Build semantic search query for vec_messages."""
        base = """
            SELECT rowid, distance, chat_id, text_preview, sender, timestamp, is_from_me
            FROM vec_messages
            WHERE embedding MATCH vec_int8(?)
            AND k = ?
        """
        params: list[Any] = [None, limit]  # embedding placeholder

        if filter_chat_id:
            base += " AND chat_id = ?"
            params.append(None)  # chat_id placeholder

        return base, params

    @staticmethod
    def get_embeddings_by_ids(message_ids: list[int]) -> list[tuple[str, list[int]]]:
        """Build queries to fetch embeddings by IDs (chunked for SQLite limits)."""
        queries = []
        for ph, ids in QueryBuilder.chunked_in_clause(message_ids):
            sql = f"SELECT rowid, embedding FROM vec_messages WHERE rowid IN ({ph})"  # nosec B608
            queries.append((sql, ids))
        return queries

    @staticmethod
    def delete_vec_binary_by_chunkids(chunk_rowids: list[int]) -> tuple[str, list[int]]:
        """Build DELETE for vec_binary by chunk_rowids."""
        ph, params = QueryBuilder.in_clause(chunk_rowids)
        sql = f"DELETE FROM vec_binary WHERE chunk_rowid IN ({ph})"  # nosec B608
        return sql, params


class SegmentStorageQueries:
    """Pre-built queries for segment_storage module."""

    @staticmethod
    def insert_segments(segment_count: int) -> tuple[str, str]:
        """Get INSERT statement for conversation_segments."""
        columns = [
            "segment_id",
            "chat_id",
            "contact_id",
            "start_time",
            "end_time",
            "topic_label",
            "keywords_json",
            "entities_json",
            "confidence",
            "message_count",
            "has_facts",
            "is_active",
        ]
        cols = ", ".join(columns)
        placeholders = ", ".join(["?"] * len(columns))
        sql = f"INSERT INTO conversation_segments ({cols}) VALUES ({placeholders})"  # nosec B608
        return sql, cols

    @staticmethod
    def insert_segment_messages() -> tuple[str, str]:
        """Get INSERT statement for segment_messages."""
        columns = ["segment_id", "message_rowid", "position", "is_from_me"]
        cols = ", ".join(columns)
        placeholders = ", ".join(["?"] * len(columns))
        sql = f"INSERT INTO segment_messages ({cols}) VALUES ({placeholders})"  # nosec B608
        return sql, cols

    @staticmethod
    def delete_segments_by_ids(segment_ids: list[int]) -> tuple[str, list[int]]:
        """Build DELETE for segments by ID."""
        ph, params = QueryBuilder.in_clause(segment_ids)
        sql = f"DELETE FROM conversation_segments WHERE id IN ({ph})"  # nosec B608
        return sql, params

    @staticmethod
    def delete_segment_messages_by_segmentids(segment_ids: list[int]) -> tuple[str, list[int]]:
        """Build DELETE for segment_messages by segment_id."""
        ph, params = QueryBuilder.in_clause(segment_ids)
        sql = f"DELETE FROM segment_messages WHERE segment_id IN ({ph})"  # nosec B608
        return sql, params

    @staticmethod
    def update_vec_chunk_rowids() -> tuple[str, str]:
        """Get UPDATE statement for linking vec_chunk rowids."""
        sql = "UPDATE conversation_segments SET vec_chunk_rowid = ? WHERE id = ?"
        return sql, "vec_chunk_rowid, id"

    @staticmethod
    def mark_facts_extracted(segment_ids: list[int]) -> tuple[str, list[int]]:
        """Build UPDATE to mark segments as having facts extracted."""
        ph, params = QueryBuilder.in_clause(segment_ids)
        sql = f"UPDATE conversation_segments SET has_facts = 1 WHERE id IN ({ph})"  # nosec B608
        return sql, params


class FactStorageQueries:
    """Pre-built queries for fact_storage module."""

    @staticmethod
    def insert_facts() -> tuple[str, str]:
        """Get INSERT statement for contact_facts."""
        columns = [
            "contact_id",
            "category",
            "subject",
            "predicate",
            "value",
            "confidence",
            "source_message_id",
            "source_text",
            "extracted_at",
            "linked_contact_id",
            "valid_from",
            "valid_until",
            "attribution",
            "segment_id",
        ]
        cols = ", ".join(columns)
        placeholders = ", ".join(["?"] * len(columns))
        sql = f"INSERT INTO contact_facts ({cols}) VALUES ({placeholders})"  # nosec B608
        return sql, cols

    @staticmethod
    def select_facts_by_contact(contact_id: str) -> tuple[str, tuple[str]]:
        """Get SELECT for facts by contact."""
        sql = """
            SELECT category, subject, predicate, value, confidence,
                   source_text, source_message_id, extracted_at,
                   valid_from, valid_until, attribution
            FROM contact_facts
            WHERE contact_id = ?
            ORDER BY confidence DESC
        """
        return sql, (contact_id,)


# Convenience exports
qb = QueryBuilder()
