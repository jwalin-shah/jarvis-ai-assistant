"""Tests for Security Headers Middleware.

Ensures that:
1. Standard security headers (HSTS, CSP, etc.) are present on all responses.
2. CSP is strict by default.
3. CSP is relaxed for documentation endpoints to allow Swagger UI.
4. Security headers are applied even on error responses.
"""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_security_headers_present_on_success():
    """Verify security headers are present on a successful response."""
    # Use the health check endpoint as a standard successful response
    response = client.get("/health")
    assert response.status_code == 200

    headers = response.headers

    # Check for presence and correct values of standard security headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["X-XSS-Protection"] == "1; mode=block"
    assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"

    # Check default strict CSP
    assert headers["Content-Security-Policy"] == "default-src 'self'"


def test_security_headers_present_on_error():
    """Verify security headers are present on error responses (e.g. 404)."""
    response = client.get("/non-existent-endpoint")
    assert response.status_code == 404

    headers = response.headers

    # Security headers should still be present
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Content-Security-Policy"] == "default-src 'self'"


def test_csp_relaxed_for_docs():
    """Verify CSP is relaxed for documentation endpoints to allow Swagger UI."""
    # Test /docs
    response = client.get("/docs")
    assert response.status_code == 200
    csp = response.headers["Content-Security-Policy"]
    assert "script-src 'self' 'unsafe-inline'" in csp
    assert "style-src 'self' 'unsafe-inline'" in csp
    assert "img-src 'self' data: https:" in csp

    # Test /redoc
    response = client.get("/redoc")
    assert response.status_code == 200
    csp = response.headers["Content-Security-Policy"]
    assert "script-src 'self' 'unsafe-inline'" in csp

    # Test /openapi.json
    response = client.get("/openapi.json")
    assert response.status_code == 200
    csp = response.headers["Content-Security-Policy"]
    assert "script-src 'self' 'unsafe-inline'" in csp


def test_csp_strict_for_api_endpoints():
    """Verify CSP remains strict for normal API endpoints."""
    # Use a standard API endpoint (e.g. /conversations)
    # We expect a 403 because we're not mocking the DB reader, but headers should be there
    response = client.get("/conversations")

    # Even if it errors (403/500), we just care about headers
    csp = response.headers["Content-Security-Policy"]
    assert csp == "default-src 'self'"
    assert "unsafe-inline" not in csp
