from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce security headers (HSTS, CSP, X-Frame-Options, etc.)."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)

        # Enforce HSTS (Strict-Transport-Security)
        if "Strict-Transport-Security" not in response.headers:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Enforce X-Content-Type-Options
        if "X-Content-Type-Options" not in response.headers:
            response.headers["X-Content-Type-Options"] = "nosniff"

        # Enforce X-Frame-Options
        if "X-Frame-Options" not in response.headers:
            response.headers["X-Frame-Options"] = "DENY"

        # Enforce Content-Security-Policy
        if "Content-Security-Policy" not in response.headers:
            path = request.url.path
            # Documentation endpoints need less strict CSP to load assets
            if path in ["/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"]:
                csp = (
                    "default-src 'self'; "
                    "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                    "style-src 'self' 'unsafe-inline';"
                )
            else:
                csp = "default-src 'self'; frame-ancestors 'none';"
            response.headers["Content-Security-Policy"] = csp

        return response
