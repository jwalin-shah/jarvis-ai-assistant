import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware import SecurityHeadersMiddleware

app = FastAPI()
app.add_middleware(SecurityHeadersMiddleware)


@app.get("/normal")
def normal_endpoint():
    return {"message": "hello"}


@app.get("/docs")
def docs_endpoint():
    return {"message": "docs"}


@app.get("/openapi.json")
def openapi_endpoint():
    return {"message": "openapi"}


client = TestClient(app)


def test_normal_endpoint_security_headers():
    response = client.get("/normal")
    assert response.status_code == 200

    # Check standard security headers
    assert response.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    # Check strict CSP
    assert (
        response.headers["Content-Security-Policy"] == "default-src 'self'; frame-ancestors 'none'"
    )


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"])
def test_docs_endpoints_relaxed_csp(path):
    # Setup dummy endpoints for the parametrized paths if they don't exist
    if path not in ["/docs", "/openapi.json"]:

        @app.get(path)
        def dummy():
            return {"message": "dummy"}

    response = client.get(path)
    assert response.status_code == 200

    # Check standard headers still apply
    assert response.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"
    assert response.headers["X-Content-Type-Options"] == "nosniff"

    # Check relaxed CSP
    assert "unsafe-inline" in response.headers["Content-Security-Policy"]
    assert "cdn.jsdelivr.net" in response.headers["Content-Security-Policy"]
