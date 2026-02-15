from __future__ import annotations

from pathlib import Path

from jarvis.infrastructure.cache.memory import MemoryBackend
from jarvis.infrastructure.cache.sqlite import SQLiteBackend
from jarvis.infrastructure.cache.unified import UnifiedCache
from jarvis.utils.singleton import thread_safe_singleton


@thread_safe_singleton
def get_unified_cache() -> UnifiedCache:
    """Get the singleton instance of the unified cache.

    Configures a two-tier cache:
    - L1: Memory (300s TTL, 1000 items)
    - L2: SQLite (~/.jarvis/cache.db, 3600s TTL)
    """
    cache_dir = Path.home() / ".jarvis"
    cache_dir.mkdir(parents=True, exist_ok=True)

    l1 = MemoryBackend(ttl=300.0, maxsize=1000)
    l2 = SQLiteBackend(db_path=cache_dir / "cache.db", default_ttl=3600.0)

    return UnifiedCache(l1=l1, l2=l2)
