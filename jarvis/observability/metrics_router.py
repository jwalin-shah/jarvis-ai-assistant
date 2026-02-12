"""Routing metrics storage for reply routing decisions.

Provides buffered writes to SQLite to reduce lock contention and overhead
from per-request database connections.

Features:
- Background queue for non-blocking metric recording
- Configurable batching (flush every N records or every M seconds)
- Disable flag for high-throughput scenarios
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.config import MetricsConfig

logger = logging.getLogger(__name__)

DEFAULT_METRICS_DB_PATH = Path.home() / ".jarvis" / "metrics.db"

# Buffering configuration
DEFAULT_BUFFER_SIZE = 100  # Flush after this many metrics
DEFAULT_FLUSH_INTERVAL_SECONDS = 5.0  # Flush at least this often

# SECURITY: Allow-list for migration column names to prevent SQL injection
VALID_METRICS_COLUMNS = {
    "generation_time_ms",
    "tokens_per_second",
    "speculative_enabled",
    "draft_acceptance_rate",
}

# SECURITY: Allow-list for migration column types to prevent SQL injection
VALID_METRICS_COLUMN_TYPES = {
    "REAL NOT NULL DEFAULT 0.0",
    "INTEGER NOT NULL DEFAULT 0",
}


@dataclass
class RoutingMetrics:
    timestamp: float
    query_hash: str
    latency_ms: dict[str, float]
    embedding_computations: int
    vec_candidates: int
    routing_decision: str
    similarity_score: float
    cache_hit: bool
    model_loaded: bool
    generation_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    speculative_enabled: bool = False
    draft_acceptance_rate: float = 0.0


def hash_query(text: str) -> str:
    """Hash a query for metrics logging."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return digest


class RoutingMetricsStore:
    """SQLite-backed storage for routing metrics with buffered writes.

    Buffers metrics in memory and flushes to SQLite in batches to reduce
    lock contention and connection overhead. Flushes occur when:
    - Buffer reaches `buffer_size` items
    - `flush_interval_seconds` has elapsed since last flush
    - `flush()` is called explicitly
    - The store is closed or the process exits

    The `record()` method is non-blocking - metrics are queued and written
    by a background thread. This ensures request handlers are never blocked
    by database writes.

    Args:
        db_path: Path to SQLite database. Defaults to ~/.jarvis/metrics.db.
        buffer_size: Number of metrics to buffer before flushing.
        flush_interval_seconds: Maximum time between flushes.
        enable_background_flush: Whether to start a background flush thread.
        enabled: Whether metrics collection is enabled. If False, record() is a no-op.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        flush_interval_seconds: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
        enable_background_flush: bool = True,
        enabled: bool = True,
    ) -> None:
        self._db_path = db_path or DEFAULT_METRICS_DB_PATH
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval_seconds
        self._enabled = enabled
        self._lock = threading.Lock()
        self._buffer: list[RoutingMetrics] = []
        self._last_flush_time = time.monotonic()
        self._initialized = False
        self._closed = False

        # Background queue for non-blocking record() calls
        self._queue: queue.Queue[RoutingMetrics | None] = queue.Queue(maxsize=10000)

        # Background flush thread
        self._flush_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        if enable_background_flush and enabled:
            self._start_background_flush()

    def _initialize(self) -> None:
        """Initialize database schema (idempotent)."""
        if self._initialized:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path, timeout=30.0) as conn:
            # Enable WAL mode for better concurrent read/write performance
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS routing_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    query_hash TEXT NOT NULL,
                    routing_decision TEXT NOT NULL,
                    similarity_score REAL NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    model_loaded INTEGER NOT NULL,
                    embedding_computations INTEGER NOT NULL,
                    faiss_candidates INTEGER NOT NULL,
                    latency_json TEXT NOT NULL,
                    generation_time_ms REAL NOT NULL DEFAULT 0.0,
                    tokens_per_second REAL NOT NULL DEFAULT 0.0
                )
                """
            )
            # Add columns for existing databases
            for col, typ in [
                ("generation_time_ms", "REAL NOT NULL DEFAULT 0.0"),
                ("tokens_per_second", "REAL NOT NULL DEFAULT 0.0"),
                ("speculative_enabled", "INTEGER NOT NULL DEFAULT 0"),
                ("draft_acceptance_rate", "REAL NOT NULL DEFAULT 0.0"),
            ]:
                # SECURITY: Validate column name and type against allow-lists before SQL execution
                if col not in VALID_METRICS_COLUMNS:
                    raise ValueError(f"Invalid migration column name: {col}")
                if typ not in VALID_METRICS_COLUMN_TYPES:
                    raise ValueError(f"Invalid migration column type: {typ}")
                try:
                    # SECURITY: f-string is safe here because col and typ are validated
                    # against strict allow-lists. SQLite ALTER TABLE doesn't support
                    # parameterized column names/types.
                    conn.execute(f"ALTER TABLE routing_metrics ADD COLUMN {col} {typ}")
                except sqlite3.OperationalError:
                    pass  # Column already exists
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_routing_metrics_timestamp
                ON routing_metrics(timestamp)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_routing_metrics_decision
                ON routing_metrics(routing_decision)
                """
            )
        self._initialized = True

    def _start_background_flush(self) -> None:
        """Start background thread for periodic flushing.

        The background thread processes the queue and flushes to database
        either when buffer_size is reached or flush_interval has elapsed.
        """
        if self._flush_thread is not None:
            return

        def flush_loop() -> None:
            while not self._stop_event.is_set():
                try:
                    # Drain the queue into the buffer (non-blocking batch)
                    self._drain_queue()

                    # Check if we need to flush based on size or time
                    self._flush_if_needed()

                    # Sleep briefly to allow batching, but not too long
                    # to delay flushes beyond interval
                    self._stop_event.wait(timeout=min(0.1, self._flush_interval / 10))
                except Exception as e:
                    logger.warning(f"Background flush failed: {e}")

        self._flush_thread = threading.Thread(
            target=flush_loop,
            name="metrics-flush",
            daemon=True,
        )
        self._flush_thread.start()

    def _drain_queue(self) -> None:
        """Move all queued metrics to the buffer."""
        drained = 0
        while True:
            try:
                metric = self._queue.get_nowait()
                if metric is None:
                    # Shutdown sentinel
                    break
                with self._lock:
                    self._buffer.append(metric)
                drained += 1
            except queue.Empty:
                break
        if drained > 0:
            logger.debug(f"Drained {drained} metrics from queue")

    def _flush_if_needed(self) -> None:
        """Flush buffer if interval has elapsed and buffer is non-empty."""
        with self._lock:
            if not self._buffer:
                return
            elapsed = time.monotonic() - self._last_flush_time
            if elapsed >= self._flush_interval:
                self._flush_buffer_locked()

    def _flush_buffer_locked(self) -> None:
        """Flush buffer to database. Must be called with lock held."""
        if not self._buffer:
            return

        self._initialize()
        metrics_to_write = self._buffer[:]
        self._buffer.clear()
        self._last_flush_time = time.monotonic()

        # Write outside the lock to minimize contention
        # (buffer was cleared above, so new records can be added concurrently)
        try:
            with sqlite3.connect(self._db_path, timeout=30.0) as conn:
                # WAL mode already enabled during init, but set pragmas for this connection too
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.executemany(
                    """
                    INSERT INTO routing_metrics (
                        timestamp,
                        query_hash,
                        routing_decision,
                        similarity_score,
                        cache_hit,
                        model_loaded,
                        embedding_computations,
                        faiss_candidates,
                        latency_json,
                        generation_time_ms,
                        tokens_per_second,
                        speculative_enabled,
                        draft_acceptance_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            m.timestamp,
                            m.query_hash,
                            m.routing_decision,
                            m.similarity_score,
                            1 if m.cache_hit else 0,
                            1 if m.model_loaded else 0,
                            m.embedding_computations,
                            m.vec_candidates,
                            json.dumps(m.latency_ms, separators=(",", ":")),
                            m.generation_time_ms,
                            m.tokens_per_second,
                            1 if m.speculative_enabled else 0,
                            m.draft_acceptance_rate,
                        )
                        for m in metrics_to_write
                    ],
                )
            logger.debug(f"Flushed {len(metrics_to_write)} routing metrics to database")
        except Exception as e:
            # On failure, put metrics back in buffer for retry
            # Note: caller already holds self._lock, so mutate directly
            logger.warning(f"Failed to flush metrics: {e}")
            self._buffer = metrics_to_write + self._buffer

    def record(self, metrics: RoutingMetrics) -> None:
        """Record a routing metric (non-blocking).

        Metrics are queued and processed by the background thread. This method
        returns immediately without blocking on database writes.

        If background flush is disabled, metrics are buffered synchronously
        and flushed when the buffer fills.
        """
        if self._closed or not self._enabled:
            return

        if self._flush_thread is not None:
            # Non-blocking: queue for background processing
            self._queue.put(metrics)
        else:
            # Synchronous: add directly to buffer (for testing)
            with self._lock:
                self._buffer.append(metrics)
                if len(self._buffer) >= self._buffer_size:
                    self._flush_buffer_locked()

    @property
    def enabled(self) -> bool:
        """Return whether metrics collection is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable metrics collection at runtime."""
        self._enabled = value

    def flush(self) -> None:
        """Force flush all buffered metrics to database.

        This drains any queued metrics first, then flushes the buffer.
        """
        # Drain queue first (if using background processing)
        self._drain_queue()
        with self._lock:
            self._flush_buffer_locked()

    def close(self) -> None:
        """Stop background thread and flush remaining metrics."""
        if self._closed:
            return

        self._closed = True
        self._stop_event.set()

        # Signal shutdown to queue processor
        self._queue.put(None)

        if self._flush_thread is not None:
            self._flush_thread.join(timeout=2.0)
            self._flush_thread = None

        # Drain any remaining items from queue
        self._drain_queue()

        # Final flush
        self.flush()

    def query_metrics(
        self,
        since: float | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Query metrics for dashboard display.

        Args:
            since: Unix timestamp to filter from (optional)
            limit: Max recent requests to return

        Returns:
            Dict with recent_requests, summary stats, and latency_trend
        """
        # Flush pending metrics first so dashboard shows current data
        self.flush()
        self._initialize()

        try:
            with sqlite3.connect(self._db_path, timeout=10.0) as conn:
                conn.row_factory = sqlite3.Row

                # Build query with parameterized WHERE clause
                params: list[Any] = []
                query_parts = [
                    "SELECT timestamp, query_hash, routing_decision,",
                    "       similarity_score, cache_hit, model_loaded,",
                    "       embedding_computations, faiss_candidates, latency_json,",
                    "       generation_time_ms, tokens_per_second",
                    "FROM routing_metrics",
                ]
                # Build WHERE clause for count query
                if since is not None:
                    query_parts.append("WHERE timestamp >= ?")
                    params.append(since)
                else:
                    pass
                query_parts.append("ORDER BY timestamp DESC")
                query_parts.append("LIMIT ?")
                params.append(limit)

                rows = conn.execute(" ".join(query_parts), params).fetchall()

                recent = []
                total_latencies: list[float] = []
                cache_hits = 0
                decisions: dict[str, int] = {}

                for row in rows:
                    latency = json.loads(row["latency_json"])
                    total_ms = sum(latency.values()) if latency else 0.0
                    total_latencies.append(total_ms)

                    if row["cache_hit"]:
                        cache_hits += 1

                    decision = row["routing_decision"]
                    decisions[decision] = decisions.get(decision, 0) + 1

                    recent.append(
                        {
                            "timestamp": row["timestamp"],
                            "query_hash": row["query_hash"],
                            "routing_decision": decision,
                            "similarity_score": row["similarity_score"],
                            "cache_hit": bool(row["cache_hit"]),
                            "model_loaded": bool(row["model_loaded"]),
                            "embedding_computations": row["embedding_computations"],
                            "vec_candidates": row["faiss_candidates"],
                            "latency": latency,
                            "total_latency_ms": total_ms,
                            "generation_time_ms": row["generation_time_ms"],
                            "tokens_per_second": row["tokens_per_second"],
                        }
                    )

                # Summary stats
                n = len(total_latencies)
                if n > 0:
                    sorted_lat = sorted(total_latencies)
                    avg_lat = sum(sorted_lat) / n
                    p50_lat = sorted_lat[n // 2]
                    p95_idx = min(int(n * 0.95), n - 1)
                    p95_lat = sorted_lat[p95_idx]
                    cache_rate = cache_hits / n * 100
                else:
                    avg_lat = p50_lat = p95_lat = cache_rate = 0.0

                # Total count (all time)
                # SECURITY: Build WHERE clause safely using parameterized query
                where_clause = "WHERE timestamp >= ?" if since is not None else ""
                where_params = [since] if since is not None else []
                count_row = conn.execute(
                    f"SELECT COUNT(*) as cnt FROM routing_metrics {where_clause}",
                    where_params,
                ).fetchone()
                total_count = count_row["cnt"] if count_row else 0

                return {
                    "recent_requests": recent,
                    "summary": {
                        "total_requests": total_count,
                        "avg_latency_ms": round(avg_lat, 1),
                        "p50_latency_ms": round(p50_lat, 1),
                        "p95_latency_ms": round(p95_lat, 1),
                        "cache_hit_rate": round(cache_rate, 1),
                        "decisions": decisions,
                    },
                }

        except Exception as e:
            logger.warning(f"Failed to query metrics: {e}")
            return {"recent_requests": [], "summary": {}}

    def pending_count(self) -> int:
        """Return number of metrics waiting to be flushed.

        Note: This includes both buffered metrics and queued metrics (if using
        background processing). For accurate counts, ensure the queue has been
        drained first.
        """
        with self._lock:
            return len(self._buffer) + self._queue.qsize()


_store: RoutingMetricsStore | None = None
_store_lock = threading.Lock()
_atexit_registered = False


def _cleanup_store() -> None:
    """Cleanup handler for process exit."""
    global _store
    if _store is not None:
        try:
            _store.close()
        except Exception as e:
            logger.debug(f"Error closing metrics store on exit: {e}")


def get_routing_metrics_store(config: MetricsConfig | None = None) -> RoutingMetricsStore:
    """Get the singleton routing metrics store.

    The store uses buffered writes for better performance. Metrics are
    flushed automatically when the buffer fills or periodically by a
    background thread.

    Args:
        config: Optional metrics config. If not provided, loads from global config.
            The config is only used when creating a new store (first call).
    """
    global _store, _atexit_registered
    if _store is None:
        with _store_lock:
            if _store is None:
                # Load config if not provided
                if config is None:
                    from jarvis.config import get_config

                    config = get_config().metrics

                _store = RoutingMetricsStore(
                    enabled=config.enabled,
                    buffer_size=config.buffer_size,
                    flush_interval_seconds=config.flush_interval_seconds,
                )
                if not _atexit_registered:
                    atexit.register(_cleanup_store)
                    _atexit_registered = True
    return _store


def reset_routing_metrics_store() -> None:
    """Reset the routing metrics store singleton.

    Closes the existing store (flushing any pending metrics) before resetting.
    """
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
        _store = None


def flush_routing_metrics() -> None:
    """Force flush any pending routing metrics to the database.

    Useful for ensuring metrics are persisted before reading them back
    or before process exit in testing scenarios.
    """
    global _store
    if _store is not None:
        _store.flush()


def load_routing_metrics(
    db_path: Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load routing metrics rows for analysis scripts."""
    path = db_path or DEFAULT_METRICS_DB_PATH
    if not path.exists():
        return []

    query = "SELECT * FROM routing_metrics ORDER BY timestamp DESC"
    params: tuple[Any, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)

    with sqlite3.connect(path, timeout=30.0) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_FLUSH_INTERVAL_SECONDS",
    "DEFAULT_METRICS_DB_PATH",
    "RoutingMetrics",
    "RoutingMetricsStore",
    "flush_routing_metrics",
    "get_routing_metrics_store",
    "hash_query",
    "load_routing_metrics",
    "reset_routing_metrics_store",
]
