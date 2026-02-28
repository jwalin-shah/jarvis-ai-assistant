from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_security_headers_standard_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"
    assert (
        response.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"
    )
    assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert (
        response.headers.get("Content-Security-Policy")
        == "default-src 'none'; frame-ancestors 'none'"
    )


def test_security_headers_docs_endpoint():
    response = client.get("/docs")
    assert response.status_code == 200
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"
    assert (
        response.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"
    )
    assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    # Assert less restrictive CSP for docs
    expected_csp = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https://fastapi.tiangolo.com;"
    assert response.headers.get("Content-Security-Policy") == expected_csp
