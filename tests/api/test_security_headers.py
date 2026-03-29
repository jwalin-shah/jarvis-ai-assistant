"""Tests for security headers middleware.

Verifies that all responses include the required security headers.
"""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

def test_security_headers_present():
    """Verify that security headers are present in responses."""
    # Request to a simple endpoint (e.g., health check or openapi.json)
    # We use openapi.json as it doesn't require database access/permissions
    response = client.get("/openapi.json")

    assert response.status_code == 200

    headers = response.headers

    # X-Content-Type-Options
    assert "x-content-type-options" in headers
    assert headers["x-content-type-options"] == "nosniff"

    # X-Frame-Options
    assert "x-frame-options" in headers
    assert headers["x-frame-options"] == "DENY"

    # Referrer-Policy
    assert "referrer-policy" in headers
    assert headers["referrer-policy"] == "strict-origin-when-cross-origin"

    # Permissions-Policy
    assert "permissions-policy" in headers
    assert headers["permissions-policy"] == "geolocation=(), camera=(), microphone=()"

def test_security_headers_on_error():
    """Verify that security headers are present even on error responses."""
    # Request to a non-existent endpoint
    response = client.get("/non-existent-endpoint")

    assert response.status_code == 404

    headers = response.headers

    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert headers["permissions-policy"] == "geolocation=(), camera=(), microphone=()"
