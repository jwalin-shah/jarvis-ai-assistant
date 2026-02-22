"""Tests for SecurityHeadersMiddleware."""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def app():
    """Create a fresh app instance for each test."""
    return create_app()


@pytest.fixture
def client(app):
    """Create a TestClient for the app."""
    return TestClient(app)


def test_security_headers_present(client):
    """Test that security headers are present in the response."""
    response = client.get("/health")
    assert response.status_code == 200

    # Check for security headers
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "camera=()" in response.headers.get("Permissions-Policy", "")
    assert "microphone=()" in response.headers.get("Permissions-Policy", "")
    assert "geolocation=()" in response.headers.get("Permissions-Policy", "")
