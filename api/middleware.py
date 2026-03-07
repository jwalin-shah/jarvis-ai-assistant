"""Middleware for the FastAPI application."""

from typing import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce security headers on all responses."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process the request and add security headers to the response."""
        response = await call_next(request)

        # Standard security headers
        if "Strict-Transport-Security" not in response.headers:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        if "X-Content-Type-Options" not in response.headers:
            response.headers["X-Content-Type-Options"] = "nosniff"

        if "X-Frame-Options" not in response.headers:
            response.headers["X-Frame-Options"] = "DENY"

        if "X-XSS-Protection" not in response.headers:
            response.headers["X-XSS-Protection"] = "1; mode=block"

        # Content Security Policy (CSP)
        # Exempt documentation paths from strict CSP to allow swagger/redoc assets to load
        path = request.url.path
        if path in ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"):
            if "Content-Security-Policy" not in response.headers:
                response.headers["Content-Security-Policy"] = (
                    "default-src 'self'; "
                    "img-src 'self' data: https://fastapi.tiangolo.com; "
                    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                    "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net;"
                )
        else:
            if "Content-Security-Policy" not in response.headers:
                response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none';"

        return response
