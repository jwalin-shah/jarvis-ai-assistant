"""Test security headers middleware."""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_security_headers_on_health():
    """Verify that security headers are present on API endpoints."""
    # Use /health endpoint
    response = client.get("/health")
    assert response.status_code == 200

    headers = response.headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "geolocation=()" in headers["Permissions-Policy"]
    assert "default-src 'none'" in headers["Content-Security-Policy"]
    assert "max-age=31536000" in headers["Strict-Transport-Security"]

def test_security_headers_on_docs_skip_csp():
    """Verify that CSP is skipped on documentation endpoints."""
    for path in ["/docs", "/redoc", "/openapi.json"]:
        response = client.get(path)
        # Even if docs fail to render (which testclient doesn't check), headers should be set
        # But for testclient, it just gets the response
        assert response.status_code == 200, f"Failed for {path}"

        headers = response.headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        # CSP should be missing or permissive
        assert "Content-Security-Policy" not in headers

def test_security_headers_on_404():
    """Verify that security headers are present on 404 responses."""
    response = client.get("/non-existent-route")
    assert response.status_code == 404

    headers = response.headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    # 404s are not docs, so should have CSP
    assert "default-src 'none'" in headers["Content-Security-Policy"]
