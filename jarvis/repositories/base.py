"""Base repository with shared database access pattern."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.db import JarvisDB

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base class for repositories that access JarvisDB.

    Accepts an optional db instance for testability. Falls back to
    the singleton ``get_db()`` when none is provided.
    """

    def __init__(self, db: JarvisDB | None = None) -> None:
        self._db = db

    @property
    def db(self) -> JarvisDB:
        if self._db is None:
            from jarvis.db import get_db

            self._db = get_db()
        return self._db
