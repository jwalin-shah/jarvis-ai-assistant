"""Tests for security middleware."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware import SecurityHeadersMiddleware


def test_security_headers_middleware():
    """Verify that security headers are added to responses."""
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert (
        response.headers["Permissions-Policy"]
        == "geolocation=(), camera=(), microphone=()"
    )
