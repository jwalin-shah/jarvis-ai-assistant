"""Repository pattern for data access decoupling.

Repositories wrap direct database and storage access behind clean interfaces,
enabling service-layer code to depend on abstractions rather than concrete
storage implementations.
"""

from jarvis.repositories.base import BaseRepository
from jarvis.repositories.contact_repository import ContactRepository
from jarvis.repositories.search_repository import SearchRepository

__all__ = ["BaseRepository", "ContactRepository", "SearchRepository"]
