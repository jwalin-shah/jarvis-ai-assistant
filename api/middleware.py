"""Middleware for API security headers.

Adds standard security headers to all responses to protect against common web vulnerabilities.
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware that adds security headers to all responses.

    Headers added:
    - X-Content-Type-Options: nosniff (Prevents MIME sniffing)
    - X-Frame-Options: DENY (Prevents clickjacking)
    - Referrer-Policy: strict-origin-when-cross-origin (Controls referrer information)
    - Permissions-Policy: geolocation=(), camera=(), microphone=() (Restrict sensitive features)
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"

        return response
