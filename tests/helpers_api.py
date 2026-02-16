"""Shared helpers for API tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import get_imessage_reader


@contextmanager
def api_client_with_reader(
    app: FastAPI,
    reader: object | None = None,
    *,
    raise_server_exceptions: bool = False,
) -> Iterator[TestClient]:
    """Create a TestClient with optional iMessage reader override."""
    if reader is not None:
        app.dependency_overrides[get_imessage_reader] = lambda: reader
    try:
        with TestClient(app, raise_server_exceptions=raise_server_exceptions) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
