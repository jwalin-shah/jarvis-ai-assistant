"""Tests for security headers middleware."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_security_headers_present() -> None:
    """Test that security headers are present in responses."""
    # Use the health endpoint as it's simple and public
    response = client.get("/health")
    assert response.status_code == 200

    headers = response.headers

    # Check for critical security headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert headers["Permissions-Policy"] == "geolocation=(), microphone=(), camera=()"
