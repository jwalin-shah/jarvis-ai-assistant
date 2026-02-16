from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from jarvis.handlers.base import BaseHandler, rpc_handler

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ContactHandler(BaseHandler):
    """Handler for contact-related RPC methods."""

    def register(self) -> None:
        """Register contact-related RPC methods."""
        self.server.register("resolve_contacts", self._resolve_contacts)
        self.server.register("get_contacts", self._get_contacts)

    @rpc_handler("Contact resolution failed")
    async def _resolve_contacts(self, identifiers: list[str]) -> dict[str, str | None]:
        """Resolve contact identifiers to names."""
        from jarvis.contacts.resolver import get_contact_resolver

        resolver = get_contact_resolver()
        return resolver.resolve_batch(identifiers)

    @rpc_handler("Failed to get contacts")
    async def _get_contacts(self, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """Get list of contacts."""

        # Note: resolver available via get_contact_resolver() if needed
        # This is a placeholder as the actual implementation might vary
        # based on what's available in the resolver
        return {"contacts": [], "total_count": 0}
