import typing

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware that adds standard security headers to all responses.

    This includes:
    - Strict-Transport-Security (HSTS)
    - Content-Security-Policy (CSP)
    - X-Content-Type-Options
    - X-Frame-Options
    - X-XSS-Protection
    """

    async def dispatch(self, request: Request, call_next: typing.Callable) -> Response:
        response = await call_next(request)

        # Determine if this is a documentation endpoint
        path = request.url.path
        is_doc_endpoint = path in ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect")

        # Standard security headers
        headers_to_add = {
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
        }

        # Strict CSP for standard endpoints, relaxed for documentation
        if is_doc_endpoint:
            # Allow inline scripts/styles for Swagger UI / ReDoc
            headers_to_add["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
            )
        else:
            # Strict CSP for API endpoints
            headers_to_add["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"

        # Add headers using setdefault to not overwrite if endpoint set them explicitly
        for key, value in headers_to_add.items():
            response.headers.setdefault(key, value)

        return response
