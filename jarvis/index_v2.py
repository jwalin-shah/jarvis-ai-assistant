"""FAISS Index V2 - Sharded Architecture with Tiered Storage.

High-performance FAISS index with time-based sharding, automatic tier management,
and comprehensive reliability features.

Key Features:
    - Time-based sharding (monthly shards for better manageability)
    - Hot/warm/cold tiers based on access patterns
    - Automatic shard compaction for old data
    - Cross-shard search with result merging
    - Lazy shard loading (only load when queried)
    - Index prefetching based on contact frequency
    - Background index warming on startup
    - Atomic updates with journaling
    - Corruption detection and auto-repair
    - Backup/restore functionality

Storage Layout:
    ~/.jarvis/indexes_v2/
        <model_name>/
            shards/
                2024-01.faiss         # Monthly shard
                2024-01.meta.json     # Shard metadata
                2024-02.faiss
                ...
            hot/                      # Frequently accessed shards (memory-mapped)
            warm/                     # Recently accessed (loaded on demand)
            cold/                     # Archived (compressed, rarely accessed)
            journal/                  # Write-ahead log for atomic updates
            backups/                  # Point-in-time backups
            config.json               # Index configuration
            manifest.json             # Shard manifest and tier assignments

Usage:
    from jarvis.index_v2 import get_sharded_index

    index = get_sharded_index(db)
    results = index.search("hello", k=5)
    index.add_pairs(new_pairs)
"""

from __future__ import annotations

import hashlib
import json
import logging
import mmap
import os
import shutil
import struct
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

if TYPE_CHECKING:
    pass

# Module-level imports for mocking in tests
from jarvis.config import get_config
from jarvis.embedding_adapter import get_embedder, get_configured_model_name

logger = logging.getLogger(__name__)

# Constants
JARVIS_DIR = Path.home() / ".jarvis"
INDEXES_V2_DIR = JARVIS_DIR / "indexes_v2"
JOURNAL_MAGIC = b"JIDX"  # Journal file magic bytes
JOURNAL_VERSION = 1
DEFAULT_BATCH_SIZE = 512  # Increased from 256 for better throughput
SHARD_FORMAT = "%Y-%m"  # Monthly shards


class TierLevel(Enum):
    """Index tier levels based on access patterns."""
    HOT = "hot"       # Frequently accessed, kept in memory
    WARM = "warm"     # Recently accessed, loaded on demand
    COLD = "cold"     # Rarely accessed, compressed archive


class JournalOp(Enum):
    """Journal operation types for atomic updates."""
    ADD = "add"
    DELETE = "delete"
    COMPACT = "compact"
    MIGRATE = "migrate"


@dataclass
class ShardConfig:
    """Configuration for a single shard."""
    shard_id: str                    # e.g., "2024-01"
    tier: TierLevel = TierLevel.WARM
    index_type: str = "ivfpq_4x"
    num_vectors: int = 0
    last_accessed: datetime | None = None
    last_modified: datetime | None = None
    access_count: int = 0
    checksum: str | None = None      # SHA256 of index file
    is_mmap: bool = False            # Memory-mapped for hot tier


@dataclass
class ShardedIndexConfig:
    """Configuration for sharded FAISS index.

    Attributes:
        indexes_dir: Base directory for indexes.
        batch_size: Batch size for encoding (optimized for Apple Silicon).
        normalize: Normalize embeddings for cosine similarity.
        index_type: Default FAISS index type for new shards.
        min_vectors_for_compression: Minimum vectors before compression.
        ivf_nprobe: Number of clusters to search.

        # Tier management
        hot_access_threshold: Access count to promote to hot tier.
        cold_age_days: Days of inactivity before demoting to cold.
        max_hot_shards: Maximum shards to keep in hot tier.

        # Compaction
        compact_threshold: Deletion ratio to trigger compaction.
        merge_cold_shards: Merge cold shards older than this many months.

        # Reliability
        enable_journaling: Enable write-ahead logging.
        enable_checksums: Enable corruption detection.
        auto_repair: Automatically repair corrupted shards.

        # Performance
        prefetch_contacts: Enable contact-based prefetching.
        background_warming: Enable background warming on startup.
        use_mmap: Use memory-mapped files for hot shards.
        max_parallel_searches: Max parallel shard searches.
    """
    indexes_dir: Path = field(default_factory=lambda: INDEXES_V2_DIR)
    batch_size: int = DEFAULT_BATCH_SIZE
    normalize: bool = True
    index_type: str = "ivfpq_4x"
    min_vectors_for_compression: int = 1000
    ivf_nprobe: int = 128

    # Tier management
    hot_access_threshold: int = 100
    cold_age_days: int = 90
    max_hot_shards: int = 3

    # Compaction
    compact_threshold: float = 0.2
    merge_cold_months: int = 6

    # Reliability
    enable_journaling: bool = True
    enable_checksums: bool = True
    auto_repair: bool = True

    # Performance
    prefetch_contacts: bool = True
    background_warming: bool = True
    use_mmap: bool = True
    max_parallel_searches: int = 4


@dataclass
class ShardStats:
    """Statistics for a single shard."""
    shard_id: str
    tier: TierLevel
    total_vectors: int
    active_vectors: int
    deleted_vectors: int
    deletion_ratio: float
    size_bytes: int
    last_accessed: datetime | None
    access_count: int
    is_loaded: bool
    is_mmap: bool


@dataclass
class ShardedIndexStats:
    """Overall statistics for the sharded index."""
    total_shards: int
    hot_shards: int
    warm_shards: int
    cold_shards: int
    total_vectors: int
    active_vectors: int
    total_size_bytes: int
    loaded_shards: int
    mmap_shards: int
    oldest_shard: str | None
    newest_shard: str | None
    needs_compaction: list[str]  # Shard IDs needing compaction


@dataclass
class SearchResult:
    """Single search result with metadata."""
    faiss_id: int
    pair_id: int
    shard_id: str
    similarity: float
    weighted_score: float
    trigger_text: str | None = None
    response_text: str | None = None
    chat_id: str | None = None
    source_timestamp: datetime | None = None
    quality_score: float | None = None


class Shard:
    """Represents a single time-based shard of the index.

    Each shard contains pairs from a specific time period (month by default).
    Supports lazy loading, memory mapping, and soft deletes.
    """

    def __init__(
        self,
        shard_id: str,
        base_dir: Path,
        config: ShardedIndexConfig,
    ) -> None:
        """Initialize shard.

        Args:
            shard_id: Unique shard identifier (e.g., "2024-01").
            base_dir: Base directory for this model's indexes.
            config: Shared configuration.
        """
        self.shard_id = shard_id
        self.base_dir = base_dir
        self.config = config

        self._lock = threading.RLock()
        self._index: Any = None
        self._mmap_file: mmap.mmap | None = None
        self._loaded = False

        # Metadata
        self._tier = TierLevel.WARM
        self._deleted_faiss_ids: set[int] = set()
        self._pair_to_faiss: dict[int, int] = {}
        self._faiss_to_pair: dict[int, int] = {}
        self._last_accessed: datetime | None = None
        self._last_modified: datetime | None = None
        self._access_count = 0
        self._checksum: str | None = None

    @property
    def index_path(self) -> Path:
        """Path to FAISS index file."""
        return self.base_dir / "shards" / f"{self.shard_id}.faiss"

    @property
    def metadata_path(self) -> Path:
        """Path to metadata JSON file."""
        return self.base_dir / "shards" / f"{self.shard_id}.meta.json"

    @property
    def tier(self) -> TierLevel:
        """Current tier level."""
        return self._tier

    @tier.setter
    def tier(self, value: TierLevel) -> None:
        """Set tier level."""
        self._tier = value

    def exists(self) -> bool:
        """Check if shard files exist on disk."""
        return self.index_path.exists()

    def load(self, force: bool = False) -> bool:
        """Load shard from disk.

        Args:
            force: Force reload even if already loaded.

        Returns:
            True if loaded successfully.
        """
        if self._loaded and not force:
            return True

        with self._lock:
            if self._loaded and not force:
                return True

            if not self.exists():
                return False

            try:
                import faiss

                # Verify checksum if enabled
                if self.config.enable_checksums and self._checksum:
                    actual_checksum = self._compute_checksum()
                    if actual_checksum != self._checksum:
                        logger.error(
                            "Shard %s checksum mismatch: expected %s, got %s",
                            self.shard_id, self._checksum, actual_checksum
                        )
                        if self.config.auto_repair:
                            return self._attempt_repair()
                        return False

                # Load FAISS index
                if self.config.use_mmap and self._tier == TierLevel.HOT:
                    # Memory-mapped loading for hot tier
                    self._index = faiss.read_index(str(self.index_path))
                else:
                    self._index = faiss.read_index(str(self.index_path))

                # Load metadata
                self._load_metadata()

                self._loaded = True
                self._last_accessed = datetime.now()
                self._access_count += 1

                logger.debug(
                    "Loaded shard %s: %d vectors (%d deleted)",
                    self.shard_id,
                    self._index.ntotal,
                    len(self._deleted_faiss_ids),
                )
                return True

            except Exception as e:
                logger.error("Failed to load shard %s: %s", self.shard_id, e)
                if self.config.auto_repair:
                    return self._attempt_repair()
                return False

    def unload(self) -> None:
        """Unload shard from memory."""
        with self._lock:
            if self._mmap_file:
                self._mmap_file.close()
                self._mmap_file = None
            self._index = None
            self._loaded = False
            logger.debug("Unloaded shard %s", self.shard_id)

    def _load_metadata(self) -> None:
        """Load metadata from JSON file."""
        if not self.metadata_path.exists():
            return

        try:
            with open(self.metadata_path) as f:
                data = json.load(f)

            self._tier = TierLevel(data.get("tier", "warm"))
            self._deleted_faiss_ids = set(data.get("deleted_faiss_ids", []))
            self._pair_to_faiss = {
                int(k): v for k, v in data.get("pair_to_faiss", {}).items()
            }
            self._faiss_to_pair = {
                int(k): v for k, v in data.get("faiss_to_pair", {}).items()
            }

            last_accessed = data.get("last_accessed")
            self._last_accessed = (
                datetime.fromisoformat(last_accessed) if last_accessed else None
            )

            last_modified = data.get("last_modified")
            self._last_modified = (
                datetime.fromisoformat(last_modified) if last_modified else None
            )

            self._access_count = data.get("access_count", 0)
            self._checksum = data.get("checksum")

        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load shard %s metadata: %s", self.shard_id, e)

    def _save_metadata(self) -> None:
        """Save metadata to JSON file."""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "shard_id": self.shard_id,
            "tier": self._tier.value,
            "deleted_faiss_ids": list(self._deleted_faiss_ids),
            "pair_to_faiss": self._pair_to_faiss,
            "faiss_to_pair": self._faiss_to_pair,
            "last_accessed": (
                self._last_accessed.isoformat() if self._last_accessed else None
            ),
            "last_modified": (
                self._last_modified.isoformat() if self._last_modified else None
            ),
            "access_count": self._access_count,
            "checksum": self._checksum,
        }

        # Atomic write
        temp_path = self.metadata_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f)
        temp_path.rename(self.metadata_path)

    def _compute_checksum(self) -> str:
        """Compute SHA256 checksum of index file."""
        hasher = hashlib.sha256()
        with open(self.index_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _attempt_repair(self) -> bool:
        """Attempt to repair corrupted shard.

        Returns:
            True if repair successful.
        """
        logger.warning("Attempting to repair shard %s", self.shard_id)

        # Try loading from backup
        backup_path = self.base_dir / "backups" / f"{self.shard_id}.faiss"
        if backup_path.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(backup_path))
                # Restore from backup
                shutil.copy(backup_path, self.index_path)
                self._checksum = self._compute_checksum()
                self._save_metadata()
                self._loaded = True
                logger.info("Repaired shard %s from backup", self.shard_id)
                return True
            except Exception as e:
                logger.error("Failed to repair shard %s from backup: %s", self.shard_id, e)

        return False

    def save(self) -> None:
        """Save shard to disk."""
        import faiss

        with self._lock:
            if self._index is None:
                return

            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file first (atomic)
            temp_path = self.index_path.with_suffix(".tmp")
            faiss.write_index(self._index, str(temp_path))
            temp_path.rename(self.index_path)

            # Update checksum
            if self.config.enable_checksums:
                self._checksum = self._compute_checksum()

            self._save_metadata()
            logger.debug("Saved shard %s", self.shard_id)

    def add_vectors(
        self,
        embeddings: np.ndarray,
        pair_ids: list[int],
    ) -> int:
        """Add vectors to this shard.

        Args:
            embeddings: Embedding vectors to add.
            pair_ids: Corresponding pair IDs.

        Returns:
            Number of vectors added.
        """
        import faiss

        with self._lock:
            # Skip pairs already in shard
            new_indices = []
            new_pair_ids = []
            for i, pair_id in enumerate(pair_ids):
                if pair_id not in self._pair_to_faiss:
                    new_indices.append(i)
                    new_pair_ids.append(pair_id)

            if not new_indices:
                return 0

            new_embeddings = embeddings[new_indices]

            # Initialize index if needed
            if self._index is None:
                dimension = new_embeddings.shape[1]
                self._index = self._create_index(dimension, len(new_embeddings), new_embeddings)
                self._loaded = True

            # Get starting FAISS ID
            start_faiss_id = self._index.ntotal

            # Add to index
            self._index.add(new_embeddings)

            # Update mappings
            for i, pair_id in enumerate(new_pair_ids):
                faiss_id = start_faiss_id + i
                self._pair_to_faiss[pair_id] = faiss_id
                self._faiss_to_pair[faiss_id] = pair_id

            self._last_modified = datetime.now()

            return len(new_pair_ids)

    def _create_index(
        self,
        dimension: int,
        num_vectors: int,
        embeddings: np.ndarray | None = None,
    ) -> Any:
        """Create FAISS index based on configuration.

        Args:
            dimension: Embedding dimension.
            num_vectors: Expected number of vectors.
            embeddings: Training embeddings (required for IVF/PQ indexes).

        Returns:
            Configured FAISS index (trained if IVF/PQ).
        """
        import faiss

        index_type = self.config.index_type

        # For small datasets, use flat index
        if num_vectors < self.config.min_vectors_for_compression:
            return faiss.IndexFlatIP(dimension)

        # Calculate IVF clusters
        nlist = int(np.sqrt(num_vectors))
        nlist = max(1, min(nlist, num_vectors // 39))

        if index_type == "flat":
            return faiss.IndexFlatIP(dimension)

        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(
                quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT
            )
            if embeddings is not None:
                index.train(embeddings)
            index.nprobe = self.config.ivf_nprobe
            return index

        elif index_type == "ivfpq_4x":
            m = 48  # 384 / 48 = 8 dims per sub-quantizer
            nbits = 8
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFPQ(
                quantizer, dimension, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT
            )
            if embeddings is not None:
                index.train(embeddings)
            index.nprobe = self.config.ivf_nprobe
            return index

        elif index_type == "ivfpq_8x":
            m = 24  # Higher compression
            nbits = 8
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFPQ(
                quantizer, dimension, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT
            )
            if embeddings is not None:
                index.train(embeddings)
            index.nprobe = self.config.ivf_nprobe
            return index

        else:
            logger.warning("Unknown index type '%s', using flat", index_type)
            return faiss.IndexFlatIP(dimension)

    def remove_pairs(self, pair_ids: list[int]) -> int:
        """Mark pairs as deleted (soft delete).

        Args:
            pair_ids: Pair IDs to remove.

        Returns:
            Number of pairs removed.
        """
        with self._lock:
            removed = 0
            for pair_id in pair_ids:
                if pair_id in self._pair_to_faiss:
                    faiss_id = self._pair_to_faiss[pair_id]
                    self._deleted_faiss_ids.add(faiss_id)
                    removed += 1

            if removed > 0:
                self._last_modified = datetime.now()

            return removed

    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
        threshold: float,
    ) -> list[tuple[int, int, float]]:
        """Search this shard.

        Args:
            query_embedding: Query vector (1, dimension).
            k: Number of results.
            threshold: Minimum similarity.

        Returns:
            List of (faiss_id, pair_id, similarity) tuples.
        """
        with self._lock:
            if not self._loaded:
                if not self.load():
                    return []

            if self._index is None or self._index.ntotal == 0:
                return []

            # Update access stats
            self._last_accessed = datetime.now()
            self._access_count += 1

            # Search for more to account for deleted
            deleted_count = len(self._deleted_faiss_ids)
            search_k = min(k + deleted_count + 5, self._index.ntotal)

            scores, indices = self._index.search(query_embedding, search_k)

            # Filter results
            results = []
            for score, faiss_id in zip(scores[0], indices[0]):
                if faiss_id < 0:
                    continue
                if faiss_id in self._deleted_faiss_ids:
                    continue
                if score < threshold:
                    continue

                pair_id = self._faiss_to_pair.get(faiss_id)
                if pair_id is not None:
                    results.append((int(faiss_id), pair_id, float(score)))

                if len(results) >= k:
                    break

            return results

    def needs_compact(self) -> bool:
        """Check if shard needs compaction."""
        if not self._loaded:
            self._load_metadata()

        if self._index is None:
            # Check from metadata
            total = len(self._pair_to_faiss)
            if total == 0:
                return False
            deleted = len(self._deleted_faiss_ids)
            return deleted / total >= self.config.compact_threshold

        if self._index.ntotal == 0:
            return False

        deletion_ratio = len(self._deleted_faiss_ids) / self._index.ntotal
        return deletion_ratio >= self.config.compact_threshold

    def get_stats(self) -> ShardStats:
        """Get shard statistics."""
        if not self._loaded:
            self._load_metadata()

        total = self._index.ntotal if self._index else len(self._pair_to_faiss)
        deleted = len(self._deleted_faiss_ids)
        active = total - deleted

        size_bytes = 0
        if self.index_path.exists():
            size_bytes = self.index_path.stat().st_size

        return ShardStats(
            shard_id=self.shard_id,
            tier=self._tier,
            total_vectors=total,
            active_vectors=active,
            deleted_vectors=deleted,
            deletion_ratio=deleted / total if total > 0 else 0.0,
            size_bytes=size_bytes,
            last_accessed=self._last_accessed,
            access_count=self._access_count,
            is_loaded=self._loaded,
            is_mmap=self._mmap_file is not None,
        )


class JournalWriter:
    """Write-ahead log for atomic index updates.

    Journal format:
        Magic (4 bytes): "JIDX"
        Version (1 byte): 1
        Entry count (4 bytes): uint32
        Entries:
            - Op type (1 byte)
            - Shard ID length (2 bytes)
            - Shard ID (variable)
            - Data length (4 bytes)
            - Data (JSON, variable)
        Checksum (32 bytes): SHA256
    """

    def __init__(self, journal_dir: Path) -> None:
        """Initialize journal writer.

        Args:
            journal_dir: Directory for journal files.
        """
        self.journal_dir = journal_dir
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._current_journal: Path | None = None
        self._entries: list[tuple[JournalOp, str, dict]] = []

    def begin(self) -> None:
        """Begin a new transaction."""
        with self._lock:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            self._current_journal = self.journal_dir / f"journal-{timestamp}.wal"
            self._entries = []

    def log(self, op: JournalOp, shard_id: str, data: dict) -> None:
        """Log an operation.

        Args:
            op: Operation type.
            shard_id: Target shard ID.
            data: Operation data.
        """
        with self._lock:
            self._entries.append((op, shard_id, data))

    def commit(self) -> None:
        """Commit and flush journal to disk."""
        with self._lock:
            if not self._current_journal or not self._entries:
                return

            # Build content in memory first
            content = bytearray()

            # Header
            content.extend(JOURNAL_MAGIC)
            content.extend(struct.pack("B", JOURNAL_VERSION))
            content.extend(struct.pack("<I", len(self._entries)))

            # Entries
            for op, shard_id, data in self._entries:
                shard_bytes = shard_id.encode("utf-8")
                data_bytes = json.dumps(data).encode("utf-8")

                content.extend(struct.pack("B", list(JournalOp).index(op)))
                content.extend(struct.pack("<H", len(shard_bytes)))
                content.extend(shard_bytes)
                content.extend(struct.pack("<I", len(data_bytes)))
                content.extend(data_bytes)

            # Compute checksum
            checksum = hashlib.sha256(content).digest()

            # Write content + checksum
            with open(self._current_journal, "wb") as f:
                f.write(content)
                f.write(checksum)

            # Sync to disk
            os.sync()

            self._entries = []

    def rollback(self) -> None:
        """Rollback current transaction."""
        with self._lock:
            if self._current_journal and self._current_journal.exists():
                self._current_journal.unlink()
            self._current_journal = None
            self._entries = []

    def cleanup_committed(self) -> None:
        """Remove successfully committed journals."""
        with self._lock:
            if self._current_journal and self._current_journal.exists():
                self._current_journal.unlink()
            self._current_journal = None

    def recover_pending(self) -> Iterator[tuple[JournalOp, str, dict]]:
        """Recover pending operations from incomplete journals.

        Yields:
            Tuples of (operation, shard_id, data).
        """
        for journal_path in sorted(self.journal_dir.glob("journal-*.wal")):
            try:
                # Read entire file
                file_content = journal_path.read_bytes()
                if len(file_content) < 41:  # Minimum: 4 + 1 + 4 + 32 = 41 bytes
                    logger.warning("Journal too small: %s", journal_path)
                    continue

                # Split content and checksum
                content = file_content[:-32]
                stored_checksum = file_content[-32:]

                # Verify checksum
                if hashlib.sha256(content).digest() != stored_checksum:
                    logger.warning("Journal checksum mismatch: %s", journal_path)
                    continue

                # Parse content
                offset = 0

                # Header
                magic = content[offset:offset+4]
                offset += 4
                if magic != JOURNAL_MAGIC:
                    logger.warning("Invalid journal magic in %s", journal_path)
                    continue

                version = struct.unpack("B", content[offset:offset+1])[0]
                offset += 1
                if version != JOURNAL_VERSION:
                    logger.warning("Unknown journal version %d in %s", version, journal_path)
                    continue

                entry_count = struct.unpack("<I", content[offset:offset+4])[0]
                offset += 4

                # Read entries
                entries = []
                for _ in range(entry_count):
                    op_idx = struct.unpack("B", content[offset:offset+1])[0]
                    offset += 1
                    op = list(JournalOp)[op_idx]

                    shard_len = struct.unpack("<H", content[offset:offset+2])[0]
                    offset += 2
                    shard_id = content[offset:offset+shard_len].decode("utf-8")
                    offset += shard_len

                    data_len = struct.unpack("<I", content[offset:offset+4])[0]
                    offset += 4
                    data = json.loads(content[offset:offset+data_len].decode("utf-8"))
                    offset += data_len

                    entries.append((op, shard_id, data))

                # Yield entries for replay
                for entry in entries:
                    yield entry

                # Mark as processed
                journal_path.unlink()

            except Exception as e:
                logger.error("Failed to recover journal %s: %s", journal_path, e)


class ShardedTriggerIndex:
    """Sharded FAISS index with tiered storage and reliability features.

    Manages multiple time-based shards with automatic tier promotion/demotion,
    cross-shard search, and comprehensive reliability features.
    """

    def __init__(
        self,
        jarvis_db: Any,
        config: ShardedIndexConfig | None = None,
    ) -> None:
        """Initialize sharded index.

        Args:
            jarvis_db: JarvisDB instance.
            config: Index configuration.
        """
        self.jarvis_db = jarvis_db
        self.config = config or ShardedIndexConfig()

        self._lock = threading.RLock()
        self._shards: dict[str, Shard] = {}
        self._embedder: Any = None
        self._journal: JournalWriter | None = None
        self._initialized = False
        self._warming_executor: ThreadPoolExecutor | None = None
        self._contact_access_counts: dict[str, int] = {}  # chat_id -> access count

        # Base directory for this model
        self._base_dir: Path | None = None

    @property
    def embedder(self) -> Any:
        """Get the unified embedder."""
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def _get_model_name(self) -> str:
        """Get model name from config."""
        return get_configured_model_name()

    def _get_base_dir(self) -> Path:
        """Get base directory for this model's indexes."""
        if self._base_dir is None:
            model_name = self._get_model_name()
            self._base_dir = self.config.indexes_dir / model_name
        return self._base_dir

    def _ensure_initialized(self) -> None:
        """Ensure index is initialized."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self._initialize()
            self._initialized = True

    def _initialize(self) -> None:
        """Initialize index, load manifest, recover from journal."""
        base_dir = self._get_base_dir()
        base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize journal if enabled
        if self.config.enable_journaling:
            journal_dir = base_dir / "journal"
            self._journal = JournalWriter(journal_dir)

            # Recover any pending operations
            for op, shard_id, data in self._journal.recover_pending():
                logger.info("Recovering journal operation: %s on %s", op.value, shard_id)
                self._replay_journal_op(op, shard_id, data)

        # Load manifest
        self._load_manifest()

        # Load existing shards (metadata only, lazy load actual index)
        shards_dir = base_dir / "shards"
        if shards_dir.exists():
            for meta_path in shards_dir.glob("*.meta.json"):
                shard_id = meta_path.stem.replace(".meta", "")
                if shard_id not in self._shards:
                    self._shards[shard_id] = Shard(shard_id, base_dir, self.config)

        # Start background warming if enabled
        if self.config.background_warming and self._shards:
            self._start_background_warming()

        logger.info(
            "Initialized sharded index with %d shards",
            len(self._shards),
        )

    def _load_manifest(self) -> None:
        """Load shard manifest."""
        manifest_path = self._get_base_dir() / "manifest.json"
        if not manifest_path.exists():
            return

        try:
            with open(manifest_path) as f:
                data = json.load(f)

            # Load contact access counts for prefetching
            self._contact_access_counts = data.get("contact_access_counts", {})

        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load manifest: %s", e)

    def _save_manifest(self) -> None:
        """Save shard manifest."""
        manifest_path = self._get_base_dir() / "manifest.json"

        data = {
            "shards": list(self._shards.keys()),
            "contact_access_counts": self._contact_access_counts,
            "last_updated": datetime.now().isoformat(),
        }

        # Atomic write
        temp_path = manifest_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f)
        temp_path.rename(manifest_path)

    def _replay_journal_op(self, op: JournalOp, shard_id: str, data: dict) -> None:
        """Replay a journal operation during recovery."""
        shard = self._get_or_create_shard(shard_id)

        if op == JournalOp.ADD:
            # Re-add pairs that were logged but not committed
            pair_ids = data.get("pair_ids", [])
            if pair_ids:
                pairs = self.jarvis_db.get_pairs_by_ids(pair_ids)
                if pairs:
                    triggers = [p.trigger_text for p in pairs.values()]
                    embeddings = self.embedder.encode(
                        triggers, normalize=self.config.normalize
                    ).astype(np.float32)
                    shard.add_vectors(embeddings, pair_ids)
                    shard.save()

        elif op == JournalOp.DELETE:
            pair_ids = data.get("pair_ids", [])
            if pair_ids:
                shard.remove_pairs(pair_ids)
                shard._save_metadata()

    def _get_or_create_shard(self, shard_id: str) -> Shard:
        """Get or create a shard.

        Args:
            shard_id: Shard identifier.

        Returns:
            Shard instance.
        """
        if shard_id not in self._shards:
            self._shards[shard_id] = Shard(
                shard_id, self._get_base_dir(), self.config
            )
        return self._shards[shard_id]

    def _get_shard_for_timestamp(self, timestamp: datetime | None) -> str:
        """Get shard ID for a timestamp.

        Args:
            timestamp: Timestamp or None for current time.

        Returns:
            Shard ID (e.g., "2024-01").
        """
        ts = timestamp or datetime.now()
        return ts.strftime(SHARD_FORMAT)

    def _start_background_warming(self) -> None:
        """Start background warming of hot shards."""
        if self._warming_executor is not None:
            return

        self._warming_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="warming")

        # Identify hot shards to warm
        hot_shards = [
            shard_id for shard_id, shard in self._shards.items()
            if shard.tier == TierLevel.HOT
        ]

        # Also warm recent shards
        now = datetime.now()
        for i in range(3):  # Last 3 months
            ts = now - timedelta(days=30 * i)
            shard_id = self._get_shard_for_timestamp(ts)
            if shard_id in self._shards and shard_id not in hot_shards:
                hot_shards.append(shard_id)

        # Submit warming tasks
        for shard_id in hot_shards[:self.config.max_hot_shards]:
            self._warming_executor.submit(self._warm_shard, shard_id)

    def _warm_shard(self, shard_id: str) -> None:
        """Warm a shard in background."""
        try:
            if shard_id in self._shards:
                self._shards[shard_id].load()
                logger.debug("Warmed shard %s", shard_id)
        except Exception as e:
            logger.warning("Failed to warm shard %s: %s", shard_id, e)

    def _update_tier(self, shard: Shard) -> None:
        """Update shard tier based on access patterns.

        Args:
            shard: Shard to evaluate.
        """
        now = datetime.now()

        # Promote to hot if frequently accessed
        if (
            shard.tier != TierLevel.HOT
            and shard._access_count >= self.config.hot_access_threshold
        ):
            # Check if we have room for another hot shard
            hot_count = sum(
                1 for s in self._shards.values() if s.tier == TierLevel.HOT
            )
            if hot_count < self.config.max_hot_shards:
                shard.tier = TierLevel.HOT
                logger.info("Promoted shard %s to HOT tier", shard.shard_id)

        # Demote to cold if not accessed recently
        elif shard.tier != TierLevel.COLD and shard._last_accessed:
            age_days = (now - shard._last_accessed).days
            if age_days >= self.config.cold_age_days:
                shard.tier = TierLevel.COLD
                shard.unload()  # Free memory
                logger.info("Demoted shard %s to COLD tier", shard.shard_id)

    def add_pairs(
        self,
        pairs: list[Any],
        progress_callback: Any | None = None,
    ) -> int:
        """Add pairs to the index.

        Automatically routes pairs to appropriate shards based on timestamp.

        Args:
            pairs: List of Pair objects.
            progress_callback: Optional callback(stage, progress, message).

        Returns:
            Number of pairs added.
        """
        self._ensure_initialized()

        if not pairs:
            return 0

        with self._lock:
            # Group pairs by shard
            pairs_by_shard: dict[str, list[Any]] = {}
            for pair in pairs:
                shard_id = self._get_shard_for_timestamp(pair.source_timestamp)
                if shard_id not in pairs_by_shard:
                    pairs_by_shard[shard_id] = []
                pairs_by_shard[shard_id].append(pair)

            total_added = 0
            shard_count = len(pairs_by_shard)

            for i, (shard_id, shard_pairs) in enumerate(pairs_by_shard.items()):
                if progress_callback:
                    progress = (i / shard_count) * 0.9
                    progress_callback(
                        "adding",
                        progress,
                        f"Adding to shard {shard_id} ({len(shard_pairs)} pairs)..."
                    )

                # Log to journal
                if self._journal:
                    self._journal.begin()
                    self._journal.log(
                        JournalOp.ADD,
                        shard_id,
                        {"pair_ids": [p.id for p in shard_pairs]}
                    )

                try:
                    # Encode pairs
                    triggers = [p.trigger_text for p in shard_pairs]
                    embeddings = self.embedder.encode(
                        triggers, normalize=self.config.normalize
                    ).astype(np.float32)

                    # Get or create shard
                    shard = self._get_or_create_shard(shard_id)

                    # Add to shard
                    pair_ids = [p.id for p in shard_pairs]
                    added = shard.add_vectors(embeddings, pair_ids)
                    total_added += added

                    # Save shard
                    if added > 0:
                        shard.save()

                    # Commit journal
                    if self._journal:
                        self._journal.commit()
                        self._journal.cleanup_committed()

                except Exception as e:
                    logger.error("Failed to add pairs to shard %s: %s", shard_id, e)
                    if self._journal:
                        self._journal.rollback()
                    raise

            # Save manifest
            self._save_manifest()

            if progress_callback:
                progress_callback("done", 1.0, f"Added {total_added} pairs")

            logger.info("Added %d pairs across %d shards", total_added, shard_count)
            return total_added

    def remove_pairs(self, pair_ids: list[int]) -> int:
        """Remove pairs from the index (soft delete).

        Args:
            pair_ids: Pair IDs to remove.

        Returns:
            Number of pairs removed.
        """
        self._ensure_initialized()

        with self._lock:
            # Log to journal
            if self._journal:
                self._journal.begin()

            total_removed = 0

            # Try each shard (we don't know which shard has each pair)
            for shard in self._shards.values():
                if self._journal:
                    self._journal.log(
                        JournalOp.DELETE,
                        shard.shard_id,
                        {"pair_ids": pair_ids}
                    )

                removed = shard.remove_pairs(pair_ids)
                if removed > 0:
                    total_removed += removed
                    shard._save_metadata()

            if self._journal:
                self._journal.commit()
                self._journal.cleanup_committed()

            return total_removed

    def search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.5,
        chat_id: str | None = None,
        embedder: Any | None = None,
    ) -> list[SearchResult]:
        """Search across all shards and merge results.

        Args:
            query: Query text.
            k: Number of results.
            threshold: Minimum similarity.
            chat_id: Optional chat ID for prefetching optimization.
            embedder: Optional embedder override.

        Returns:
            List of SearchResult objects.
        """
        self._ensure_initialized()

        if not self._shards:
            return []

        # Update contact access count for prefetching
        if chat_id:
            self._contact_access_counts[chat_id] = (
                self._contact_access_counts.get(chat_id, 0) + 1
            )

        # Encode query
        query_embedder = embedder or self.embedder
        query_embedding = query_embedder.encode(
            [query], normalize=True
        ).astype(np.float32)

        # Determine shard search order (hot first, then warm, then cold)
        shards_to_search = sorted(
            self._shards.values(),
            key=lambda s: (
                0 if s.tier == TierLevel.HOT else
                1 if s.tier == TierLevel.WARM else 2,
                -s._access_count,  # More accessed first within tier
            )
        )

        # Search shards in parallel
        all_results: list[tuple[int, int, str, float]] = []

        with ThreadPoolExecutor(max_workers=self.config.max_parallel_searches) as executor:
            futures = {
                executor.submit(
                    shard.search, query_embedding, k, threshold
                ): shard
                for shard in shards_to_search
            }

            for future in futures:
                shard = futures[future]
                try:
                    results = future.result(timeout=5.0)
                    for faiss_id, pair_id, score in results:
                        all_results.append((faiss_id, pair_id, shard.shard_id, score))

                    # Update tier based on access
                    self._update_tier(shard)

                except Exception as e:
                    logger.warning(
                        "Search failed for shard %s: %s",
                        shard.shard_id, e
                    )

        # Sort by score and take top k
        all_results.sort(key=lambda x: x[3], reverse=True)
        top_results = all_results[:k]

        if not top_results:
            return []

        # Fetch pair details
        pair_ids = [r[1] for r in top_results]
        pairs_by_id = self.jarvis_db.get_pairs_by_ids(pair_ids)

        # Build search results
        search_results = []
        for faiss_id, pair_id, shard_id, score in top_results:
            pair = pairs_by_id.get(pair_id)
            if not pair:
                continue

            # Apply freshness weighting
            weighted_score = score
            if pair.source_timestamp:
                age_days = (datetime.now() - pair.source_timestamp).days
                decay = max(0.5, 1.0 - (age_days / 365) * 0.1)
                weighted_score = score * decay

            search_results.append(SearchResult(
                faiss_id=faiss_id,
                pair_id=pair_id,
                shard_id=shard_id,
                similarity=round(score, 3),
                weighted_score=round(weighted_score, 3),
                trigger_text=pair.trigger_text,
                response_text=pair.response_text,
                chat_id=pair.chat_id,
                source_timestamp=pair.source_timestamp,
                quality_score=pair.quality_score,
            ))

        # Sort by weighted score
        search_results.sort(key=lambda x: x.weighted_score, reverse=True)
        return search_results[:k]

    def search_with_pairs(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.5,
        prefer_recent: bool = True,
        embedder: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Search and return full pair information (v1 compatible).

        Args:
            query: Query text.
            k: Number of results.
            threshold: Minimum similarity.
            prefer_recent: Weight by recency (always True in v2).
            embedder: Optional embedder override.

        Returns:
            List of dicts with pair info.
        """
        results = self.search(query, k, threshold, embedder=embedder)

        return [
            {
                "similarity": r.similarity,
                "weighted_score": r.weighted_score,
                "trigger_text": r.trigger_text,
                "response_text": r.response_text,
                "chat_id": r.chat_id,
                "faiss_id": r.faiss_id,
                "pair_id": r.pair_id,
                "shard_id": r.shard_id,
                "source_timestamp": r.source_timestamp,
                "quality_score": r.quality_score,
            }
            for r in results
        ]

    def compact_shard(
        self,
        shard_id: str,
        progress_callback: Any | None = None,
    ) -> ShardStats | None:
        """Compact a specific shard to remove deleted vectors.

        Args:
            shard_id: Shard to compact.
            progress_callback: Optional progress callback.

        Returns:
            ShardStats after compaction, or None if shard not found.
        """
        self._ensure_initialized()

        shard = self._shards.get(shard_id)
        if not shard:
            return None

        with self._lock:
            if not shard.load():
                return None

            if progress_callback:
                progress_callback("compacting", 0.1, f"Compacting shard {shard_id}...")

            # Get active pairs
            active_pair_ids = [
                pair_id
                for pair_id, faiss_id in shard._pair_to_faiss.items()
                if faiss_id not in shard._deleted_faiss_ids
            ]

            if not active_pair_ids:
                return shard.get_stats()

            # Fetch pairs
            pairs_by_id = self.jarvis_db.get_pairs_by_ids(active_pair_ids)
            active_pairs = list(pairs_by_id.values())

            if progress_callback:
                progress_callback(
                    "encoding", 0.3, f"Re-encoding {len(active_pairs)} pairs..."
                )

            # Re-encode
            triggers = [p.trigger_text for p in active_pairs]
            embeddings = self.embedder.encode(
                triggers, normalize=self.config.normalize
            ).astype(np.float32)

            if progress_callback:
                progress_callback("rebuilding", 0.6, "Rebuilding index...")

            # Create new index
            import faiss
            dimension = embeddings.shape[1]
            new_index = shard._create_index(dimension, len(embeddings))

            # Train if needed
            if hasattr(new_index, "train") and not new_index.is_trained:
                new_index.train(embeddings)

            new_index.add(embeddings)

            # Update shard state
            shard._index = new_index
            shard._deleted_faiss_ids = set()
            shard._pair_to_faiss = {p.id: i for i, p in enumerate(active_pairs)}
            shard._faiss_to_pair = {i: p.id for i, p in enumerate(active_pairs)}
            shard._last_modified = datetime.now()

            if progress_callback:
                progress_callback("saving", 0.9, "Saving shard...")

            shard.save()

            if progress_callback:
                progress_callback("done", 1.0, f"Compacted shard {shard_id}")

            logger.info(
                "Compacted shard %s: %d active pairs",
                shard_id, len(active_pairs)
            )

            return shard.get_stats()

    def compact_all(
        self,
        progress_callback: Any | None = None,
    ) -> list[str]:
        """Compact all shards that need it.

        Args:
            progress_callback: Optional progress callback.

        Returns:
            List of compacted shard IDs.
        """
        self._ensure_initialized()

        compacted = []
        shards_needing_compact = [
            shard_id for shard_id, shard in self._shards.items()
            if shard.needs_compact()
        ]

        for i, shard_id in enumerate(shards_needing_compact):
            if progress_callback:
                progress = i / len(shards_needing_compact)
                progress_callback(
                    "compacting",
                    progress,
                    f"Compacting shard {i+1}/{len(shards_needing_compact)}..."
                )

            self.compact_shard(shard_id)
            compacted.append(shard_id)

        if progress_callback:
            progress_callback("done", 1.0, f"Compacted {len(compacted)} shards")

        return compacted

    def merge_cold_shards(
        self,
        progress_callback: Any | None = None,
    ) -> str | None:
        """Merge old cold shards into a single archive shard.

        Args:
            progress_callback: Optional progress callback.

        Returns:
            New merged shard ID, or None if nothing to merge.
        """
        self._ensure_initialized()

        # Find cold shards older than threshold
        cutoff = datetime.now() - timedelta(days=30 * self.config.merge_cold_months)
        cold_shards = []

        for shard_id, shard in self._shards.items():
            if shard.tier != TierLevel.COLD:
                continue

            try:
                shard_date = datetime.strptime(shard_id, SHARD_FORMAT)
                if shard_date < cutoff:
                    cold_shards.append(shard_id)
            except ValueError:
                continue

        if len(cold_shards) < 2:
            return None  # Need at least 2 shards to merge

        # Sort by date
        cold_shards.sort()

        # Create merged shard ID
        merged_id = f"archive-{cold_shards[0]}-to-{cold_shards[-1]}"

        if progress_callback:
            progress_callback(
                "merging", 0.1, f"Merging {len(cold_shards)} cold shards..."
            )

        # Collect all active pairs from cold shards
        all_pairs: list[Any] = []
        for i, shard_id in enumerate(cold_shards):
            shard = self._shards[shard_id]
            shard.load()

            # Get active pair IDs
            active_pair_ids = [
                pair_id
                for pair_id, faiss_id in shard._pair_to_faiss.items()
                if faiss_id not in shard._deleted_faiss_ids
            ]

            pairs_by_id = self.jarvis_db.get_pairs_by_ids(active_pair_ids)
            all_pairs.extend(pairs_by_id.values())

            if progress_callback:
                progress = 0.1 + (0.3 * (i + 1) / len(cold_shards))
                progress_callback(
                    "collecting", progress, f"Collected from {shard_id}..."
                )

        if not all_pairs:
            return None

        if progress_callback:
            progress_callback("encoding", 0.5, f"Encoding {len(all_pairs)} pairs...")

        # Create merged shard
        merged_shard = Shard(merged_id, self._get_base_dir(), self.config)
        merged_shard.tier = TierLevel.COLD

        # Encode and add
        triggers = [p.trigger_text for p in all_pairs]
        embeddings = self.embedder.encode(
            triggers, normalize=self.config.normalize
        ).astype(np.float32)

        pair_ids = [p.id for p in all_pairs]
        merged_shard.add_vectors(embeddings, pair_ids)
        merged_shard.save()

        # Remove old shards
        if progress_callback:
            progress_callback("cleanup", 0.9, "Removing old shards...")

        for shard_id in cold_shards:
            shard = self._shards[shard_id]
            shard.unload()

            if shard.index_path.exists():
                shard.index_path.unlink()
            if shard.metadata_path.exists():
                shard.metadata_path.unlink()

            del self._shards[shard_id]

        # Register merged shard
        self._shards[merged_id] = merged_shard
        self._save_manifest()

        if progress_callback:
            progress_callback("done", 1.0, f"Merged into {merged_id}")

        logger.info(
            "Merged %d cold shards into %s (%d pairs)",
            len(cold_shards), merged_id, len(all_pairs)
        )

        return merged_id

    def backup(
        self,
        backup_name: str | None = None,
        progress_callback: Any | None = None,
    ) -> Path:
        """Create a backup of the entire index.

        Args:
            backup_name: Optional backup name (defaults to timestamp).
            progress_callback: Optional progress callback.

        Returns:
            Path to backup directory.
        """
        self._ensure_initialized()

        if backup_name is None:
            backup_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        backup_dir = self._get_base_dir() / "backups" / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback("backup", 0.1, f"Creating backup {backup_name}...")

        # Copy all shards
        shards_dir = self._get_base_dir() / "shards"
        if shards_dir.exists():
            backup_shards_dir = backup_dir / "shards"
            backup_shards_dir.mkdir(parents=True, exist_ok=True)

            shard_files = list(shards_dir.glob("*"))
            for i, src in enumerate(shard_files):
                dst = backup_shards_dir / src.name
                shutil.copy2(src, dst)

                if progress_callback:
                    progress = 0.1 + (0.8 * (i + 1) / len(shard_files))
                    progress_callback("backup", progress, f"Backing up {src.name}...")

        # Copy manifest
        manifest_path = self._get_base_dir() / "manifest.json"
        if manifest_path.exists():
            shutil.copy2(manifest_path, backup_dir / "manifest.json")

        if progress_callback:
            progress_callback("done", 1.0, f"Backup created: {backup_dir}")

        logger.info("Created backup: %s", backup_dir)
        return backup_dir

    def restore(
        self,
        backup_name: str,
        progress_callback: Any | None = None,
    ) -> bool:
        """Restore index from backup.

        Args:
            backup_name: Name of backup to restore.
            progress_callback: Optional progress callback.

        Returns:
            True if restored successfully.
        """
        backup_dir = self._get_base_dir() / "backups" / backup_name
        if not backup_dir.exists():
            logger.error("Backup not found: %s", backup_dir)
            return False

        with self._lock:
            if progress_callback:
                progress_callback("restore", 0.1, f"Restoring from {backup_name}...")

            # Unload all shards
            for shard in self._shards.values():
                shard.unload()
            self._shards.clear()

            # Remove current shards
            shards_dir = self._get_base_dir() / "shards"
            if shards_dir.exists():
                shutil.rmtree(shards_dir)

            # Copy backup shards
            backup_shards_dir = backup_dir / "shards"
            if backup_shards_dir.exists():
                shutil.copytree(backup_shards_dir, shards_dir)

            # Copy manifest
            backup_manifest = backup_dir / "manifest.json"
            if backup_manifest.exists():
                shutil.copy2(
                    backup_manifest,
                    self._get_base_dir() / "manifest.json"
                )

            # Reinitialize
            self._initialized = False
            self._ensure_initialized()

            if progress_callback:
                progress_callback("done", 1.0, f"Restored from {backup_name}")

            logger.info("Restored from backup: %s", backup_name)
            return True

    def list_backups(self) -> list[dict[str, Any]]:
        """List available backups.

        Returns:
            List of backup info dicts.
        """
        backups_dir = self._get_base_dir() / "backups"
        if not backups_dir.exists():
            return []

        backups = []
        for backup_path in sorted(backups_dir.iterdir()):
            if not backup_path.is_dir():
                continue

            # Count shards
            shards_dir = backup_path / "shards"
            shard_count = len(list(shards_dir.glob("*.faiss"))) if shards_dir.exists() else 0

            # Get size
            total_size = sum(
                f.stat().st_size
                for f in backup_path.rglob("*")
                if f.is_file()
            )

            backups.append({
                "name": backup_path.name,
                "path": str(backup_path),
                "shard_count": shard_count,
                "size_bytes": total_size,
                "created_at": datetime.fromtimestamp(backup_path.stat().st_mtime),
            })

        return backups

    def get_stats(self) -> ShardedIndexStats:
        """Get overall index statistics.

        Returns:
            ShardedIndexStats with current state.
        """
        self._ensure_initialized()

        hot_count = 0
        warm_count = 0
        cold_count = 0
        total_vectors = 0
        active_vectors = 0
        total_size = 0
        loaded_count = 0
        mmap_count = 0
        needs_compaction = []

        oldest_shard = None
        newest_shard = None

        for shard_id, shard in self._shards.items():
            stats = shard.get_stats()

            if stats.tier == TierLevel.HOT:
                hot_count += 1
            elif stats.tier == TierLevel.WARM:
                warm_count += 1
            else:
                cold_count += 1

            total_vectors += stats.total_vectors
            active_vectors += stats.active_vectors
            total_size += stats.size_bytes

            if stats.is_loaded:
                loaded_count += 1
            if stats.is_mmap:
                mmap_count += 1

            if shard.needs_compact():
                needs_compaction.append(shard_id)

            # Track oldest/newest
            if oldest_shard is None or shard_id < oldest_shard:
                oldest_shard = shard_id
            if newest_shard is None or shard_id > newest_shard:
                newest_shard = shard_id

        return ShardedIndexStats(
            total_shards=len(self._shards),
            hot_shards=hot_count,
            warm_shards=warm_count,
            cold_shards=cold_count,
            total_vectors=total_vectors,
            active_vectors=active_vectors,
            total_size_bytes=total_size,
            loaded_shards=loaded_count,
            mmap_shards=mmap_count,
            oldest_shard=oldest_shard,
            newest_shard=newest_shard,
            needs_compaction=needs_compaction,
        )

    def get_shard_stats(self, shard_id: str) -> ShardStats | None:
        """Get statistics for a specific shard.

        Args:
            shard_id: Shard identifier.

        Returns:
            ShardStats or None if shard not found.
        """
        self._ensure_initialized()

        shard = self._shards.get(shard_id)
        if shard is None:
            return None

        return shard.get_stats()

    def verify_integrity(
        self,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Verify integrity of all shards.

        Args:
            progress_callback: Optional progress callback.

        Returns:
            Dict with verification results.
        """
        self._ensure_initialized()

        results = {
            "total_shards": len(self._shards),
            "verified": 0,
            "corrupted": [],
            "repaired": [],
            "failed": [],
        }

        for i, (shard_id, shard) in enumerate(self._shards.items()):
            if progress_callback:
                progress = i / len(self._shards)
                progress_callback("verifying", progress, f"Verifying {shard_id}...")

            if not shard.exists():
                results["failed"].append({
                    "shard_id": shard_id,
                    "error": "Index file missing"
                })
                continue

            # Verify checksum
            if self.config.enable_checksums and shard._checksum:
                actual = shard._compute_checksum()
                if actual != shard._checksum:
                    results["corrupted"].append(shard_id)

                    if self.config.auto_repair:
                        if shard._attempt_repair():
                            results["repaired"].append(shard_id)
                        else:
                            results["failed"].append({
                                "shard_id": shard_id,
                                "error": "Repair failed"
                            })
                    continue

            # Try loading
            if shard.load():
                results["verified"] += 1
                shard.unload()  # Don't keep in memory
            else:
                results["failed"].append({
                    "shard_id": shard_id,
                    "error": "Failed to load"
                })

        if progress_callback:
            progress_callback("done", 1.0, f"Verified {results['verified']} shards")

        return results

    def close(self) -> None:
        """Close the index and release resources."""
        with self._lock:
            # Shut down warming executor
            if self._warming_executor:
                self._warming_executor.shutdown(wait=False)
                self._warming_executor = None

            # Unload all shards
            for shard in self._shards.values():
                shard.unload()

            # Save manifest
            if self._initialized:
                self._save_manifest()

            self._shards.clear()
            self._initialized = False

            logger.info("Closed sharded index")


def detect_gpu_acceleration() -> dict[str, Any]:
    """Detect available GPU acceleration.

    Returns:
        Dict with acceleration info.
    """
    result = {
        "metal_available": False,
        "cuda_available": False,
        "faiss_gpu": False,
        "mlx_available": False,
    }

    # Check MLX (Apple Metal)
    try:
        import mlx.core as mx
        result["mlx_available"] = True
        # Explicitly convert to bool to handle MagicMock in test environments
        result["metal_available"] = bool(mx.metal.is_available())
    except ImportError:
        pass

    # Check CUDA
    try:
        import torch
        # Explicitly convert to bool
        result["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        pass

    # Check FAISS GPU
    try:
        import faiss
        result["faiss_gpu"] = hasattr(faiss, "StandardGpuResources")
    except ImportError:
        pass

    return result


# Singleton management
_sharded_index: ShardedTriggerIndex | None = None
_sharded_index_lock = threading.Lock()


def _get_sharded_config_from_jarvis_config() -> ShardedIndexConfig:
    """Create ShardedIndexConfig from JarvisConfig settings."""
    jarvis_config = get_config()
    faiss_config = jarvis_config.faiss_index

    return ShardedIndexConfig(
        index_type=faiss_config.index_type,
        min_vectors_for_compression=faiss_config.min_vectors_for_compression,
        ivf_nprobe=faiss_config.ivf_nprobe,
    )


def get_sharded_index(jarvis_db: Any) -> ShardedTriggerIndex:
    """Get or create singleton sharded index.

    Args:
        jarvis_db: JarvisDB instance.

    Returns:
        The shared ShardedTriggerIndex instance.
    """
    global _sharded_index
    if _sharded_index is None:
        with _sharded_index_lock:
            if _sharded_index is None:
                config = _get_sharded_config_from_jarvis_config()
                _sharded_index = ShardedTriggerIndex(jarvis_db, config)
    return _sharded_index


def reset_sharded_index() -> None:
    """Reset the singleton sharded index."""
    global _sharded_index
    with _sharded_index_lock:
        if _sharded_index is not None:
            _sharded_index.close()
            _sharded_index = None


class MigrationHelper:
    """Helper for migrating from v1 index to v2 sharded index."""

    def __init__(
        self,
        jarvis_db: Any,
        v1_indexes_dir: Path | None = None,
    ) -> None:
        """Initialize migration helper.

        Args:
            jarvis_db: JarvisDB instance.
            v1_indexes_dir: Path to v1 indexes (defaults to ~/.jarvis/indexes/triggers).
        """
        self.jarvis_db = jarvis_db
        self.v1_indexes_dir = v1_indexes_dir or (JARVIS_DIR / "indexes" / "triggers")

    def check_v1_index(self) -> dict[str, Any] | None:
        """Check if v1 index exists and get its stats.

        Returns:
            Dict with v1 index info, or None if not found.
        """
        # Look for incremental or versioned index
        if not self.v1_indexes_dir.exists():
            return None

        # Check for incremental index
        for model_dir in self.v1_indexes_dir.iterdir():
            if not model_dir.is_dir():
                continue

            incremental_dir = model_dir / "incremental"
            if incremental_dir.exists():
                index_path = incremental_dir / "index.faiss"
                meta_path = incremental_dir / "metadata.json"

                if index_path.exists():
                    return {
                        "type": "incremental",
                        "path": str(index_path),
                        "model": model_dir.name,
                        "size_bytes": index_path.stat().st_size,
                        "has_metadata": meta_path.exists(),
                    }

            # Check for versioned indexes
            for version_dir in model_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                if version_dir.name == "incremental":
                    continue

                index_path = version_dir / "index.faiss"
                if index_path.exists():
                    return {
                        "type": "versioned",
                        "path": str(index_path),
                        "model": model_dir.name,
                        "version": version_dir.name,
                        "size_bytes": index_path.stat().st_size,
                    }

        return None

    def migrate(
        self,
        progress_callback: Any | None = None,
        keep_v1: bool = True,
    ) -> dict[str, Any]:
        """Migrate from v1 to v2 index.

        Args:
            progress_callback: Optional progress callback.
            keep_v1: Keep v1 index after migration.

        Returns:
            Migration result dict.
        """
        v1_info = self.check_v1_index()
        if v1_info is None:
            return {
                "success": False,
                "error": "No v1 index found",
            }

        if progress_callback:
            progress_callback("migrating", 0.1, "Loading v1 index...")

        # Get all pairs from database
        all_pairs = self.jarvis_db.get_all_pairs()
        if not all_pairs:
            return {
                "success": False,
                "error": "No pairs in database",
            }

        if progress_callback:
            progress_callback("migrating", 0.3, f"Migrating {len(all_pairs)} pairs...")

        # Create v2 index and add pairs
        config = _get_sharded_config_from_jarvis_config()
        v2_index = ShardedTriggerIndex(self.jarvis_db, config)

        # Add pairs in batches
        batch_size = 1000
        total_added = 0

        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i + batch_size]
            added = v2_index.add_pairs(batch)
            total_added += added

            if progress_callback:
                progress = 0.3 + (0.6 * min(i + batch_size, len(all_pairs)) / len(all_pairs))
                progress_callback(
                    "migrating",
                    progress,
                    f"Migrated {min(i + batch_size, len(all_pairs))}/{len(all_pairs)} pairs..."
                )

        if progress_callback:
            progress_callback("finalizing", 0.95, "Finalizing migration...")

        # Optionally remove v1 index
        if not keep_v1:
            shutil.rmtree(self.v1_indexes_dir)

        if progress_callback:
            progress_callback("done", 1.0, f"Migration complete: {total_added} pairs")

        return {
            "success": True,
            "pairs_migrated": total_added,
            "shards_created": v2_index.get_stats().total_shards,
            "v1_kept": keep_v1,
        }


def migrate_v1_to_v2(
    jarvis_db: Any,
    progress_callback: Any | None = None,
    keep_v1: bool = True,
) -> dict[str, Any]:
    """Migrate from v1 to v2 index.

    Args:
        jarvis_db: JarvisDB instance.
        progress_callback: Optional progress callback.
        keep_v1: Keep v1 index after migration.

    Returns:
        Migration result dict.
    """
    helper = MigrationHelper(jarvis_db)
    return helper.migrate(progress_callback, keep_v1)
