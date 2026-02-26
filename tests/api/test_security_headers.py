"""Tests for security headers middleware."""
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

from api.main import create_app


def get_test_app():
    """Create an app instance with disabled lifespan for testing."""
    app = create_app()

    # Override lifespan to avoid starting background services (model warmer, socket server)
    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    app.router.lifespan_context = noop_lifespan
    return app

def test_security_headers_api():
    """Test that API endpoints have strict security headers."""
    app = get_test_app()
    client = TestClient(app)

    # Use a non-existent endpoint to verify headers are applied even on errors (404)
    # This also avoids any side effects from actual endpoints
    response = client.get("/api/v1/non-existent")

    headers = response.headers

    # Verify strict security headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert headers["Permissions-Policy"] == "geolocation=(), camera=(), microphone=()"

    # Verify strict Content-Security-Policy
    csp = headers.get("Content-Security-Policy", "")
    assert "default-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "object-src 'none'" in csp
    assert "base-uri 'none'" in csp

def test_security_headers_docs():
    """Test that documentation endpoints have relaxed security headers."""
    app = get_test_app()
    client = TestClient(app)

    # Request the Swagger UI
    response = client.get("/docs")

    headers = response.headers

    # Basic security headers should still be present
    assert headers["X-Content-Type-Options"] == "nosniff"

    # Content-Security-Policy should be absent to allow inline scripts/styles for Swagger UI
    assert "Content-Security-Policy" not in headers
