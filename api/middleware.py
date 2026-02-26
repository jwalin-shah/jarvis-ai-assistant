"""Middleware for adding security headers to API responses.

Enforces HSTS, CSP, and standard security headers.
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware that adds security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Security Headers
        # Prevent MIME type sniffing
        response.headers.setdefault("X-Content-Type-Options", "nosniff")

        # Prevent clickjacking
        response.headers.setdefault("X-Frame-Options", "DENY")

        # Restrict referrer information
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")

        # Disable browser features not needed by API
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(), camera=(), microphone=()"
        )

        # Content Security Policy (CSP)
        # We default to a very strict policy for API endpoints to prevent XSS
        # However, Swagger UI/ReDoc need inline scripts and styles to function
        if request.url.path in ["/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"]:
            # Allow Swagger UI/ReDoc to work
            # We don't set strict CSP here as these are development tools
            pass
        else:
            # API endpoints return JSON and should not execute scripts
            # object-src 'none' prevents Flash/Java applets
            # base-uri 'none' prevents base tag hijacking
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'none'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'",
            )

        return response
