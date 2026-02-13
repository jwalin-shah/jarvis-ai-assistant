"""JARVIS Database Management - SQLite database for contacts, segments, and facts.

Manages ~/.jarvis/jarvis.db which stores:
- Contacts with relationship labels and handle mappings
- Conversation segments (topic-coherent chunks) from iMessage history
- Extracted facts (biographical, preferences, work) from segments
- FAISS vector index metadata and versioning

Usage:
    jarvis db init                     # Create database
    jarvis db add-contact --name "Sarah" --relationship "sister"
    jarvis db list-contacts            # View contacts
    jarvis db extract                  # Extract segments and facts from chat.db
    jarvis db build-index              # Build FAISS index for segments
"""

import threading
from pathlib import Path

from jarvis.db.artifacts import ArtifactMixin
from jarvis.db.clusters import ClusterMixin
from jarvis.db.contacts import ContactMixin
from jarvis.db.core import JarvisDBBase
from jarvis.db.embeddings import EmbeddingMixin
from jarvis.db.index_versions import IndexVersionMixin
from jarvis.cache import TTLCache
from jarvis.db.models import (
    INDEXES_DIR,
    JARVIS_DB_PATH,
    Cluster,
    Contact,
    ContactStyleTargets,
    IndexVersion,
    Pair,
    PairArtifact,
    PairEmbedding,
    _convert_timestamp,
)
from jarvis.db.pairs import PairMixin
from jarvis.db.schema import (
    CURRENT_SCHEMA_VERSION,
    EXPECTED_INDICES,
    SCHEMA_SQL,
    VALID_COLUMN_TYPES,
    VALID_MIGRATION_COLUMNS,
)
from jarvis.db.search import PairSearchMixin
from jarvis.db.stats import StatsMixin


class JarvisDB(
    JarvisDBBase,
    ContactMixin,
    PairMixin,
    PairSearchMixin,
    ClusterMixin,
    EmbeddingMixin,
    IndexVersionMixin,
    ArtifactMixin,
    StatsMixin,
):
    """Manager for the JARVIS SQLite database.

    Thread-safe connection management with context manager support.
    Includes TTL-based caching for frequently accessed data.

    Composed from focused mixin classes for maintainability:
    - JarvisDBBase: Connection management, schema init, index verification
    - ContactMixin: Contact CRUD operations
    - PairMixin: Pair CRUD, bulk ops, DA/cluster updates
    - PairSearchMixin: DA queries, pattern matching, train/test split
    - ClusterMixin: Cluster CRUD
    - EmbeddingMixin: FAISS embedding refs, batch lookups
    - IndexVersionMixin: FAISS index version management
    - ArtifactMixin: Pair artifacts, style targets, validated pairs
    - StatsMixin: Database and gate statistics
    """

    pass


# Singleton instance
_db: JarvisDB | None = None
_db_lock = threading.Lock()


def get_db(db_path: Path | None = None) -> JarvisDB:
    """Get or create the singleton database instance."""
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:  # Double-check after acquiring lock
                _db = JarvisDB(db_path)
    return _db


def reset_db() -> None:
    """Reset the singleton database instance (closes any open connections)."""
    global _db
    with _db_lock:
        if _db is not None:
            _db.close()
        _db = None


__all__ = [
    # Main class and singletons
    "JarvisDB",
    "get_db",
    "reset_db",
    # Data models
    "Contact",
    "Pair",
    "PairArtifact",
    "ContactStyleTargets",
    "Cluster",
    "PairEmbedding",
    "IndexVersion",
    "TTLCache",
    # Constants
    "JARVIS_DB_PATH",
    "INDEXES_DIR",
    "SCHEMA_SQL",
    "EXPECTED_INDICES",
    "CURRENT_SCHEMA_VERSION",
    "VALID_MIGRATION_COLUMNS",
    "VALID_COLUMN_TYPES",
    # Internal (used by tests)
    "_convert_timestamp",
]
