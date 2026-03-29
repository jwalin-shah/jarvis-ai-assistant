from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware import SecurityHeadersMiddleware

app = FastAPI()
app.add_middleware(SecurityHeadersMiddleware)


@app.get("/test")
def test_endpoint():
    return {"message": "success"}


@app.get("/docs")
def docs_endpoint():
    return {"message": "docs"}


def test_security_headers_standard_endpoint():
    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200
    assert (
        response.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"
    )
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"
    assert (
        response.headers.get("Content-Security-Policy")
        == "default-src 'self'; frame-ancestors 'none'"
    )


def test_security_headers_docs_endpoint():
    client = TestClient(app)
    response = client.get("/docs")

    assert response.status_code == 200
    assert (
        response.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"
    )
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"

    # Assert relaxed CSP for docs
    csp = response.headers.get("Content-Security-Policy")
    assert "default-src 'self'" in csp
    assert "script-src 'self' 'unsafe-inline' 'unsafe-eval'" in csp
    assert "style-src 'self' 'unsafe-inline'" in csp
    assert "img-src 'self' data:" in csp
