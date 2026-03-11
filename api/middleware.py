from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware that adds standard security headers to all responses.

    Exempts documentation endpoints from strict Content-Security-Policy
    to allow loading of external assets (Swagger UI, ReDoc).
    """

    # Endpoints that need relaxed CSP to function properly
    DOCS_PATHS = (
        "/docs",
        "/redoc",
        "/openapi.json",
        "/docs/oauth2-redirect",
    )

    # Standard security headers applied to all responses
    SECURITY_HEADERS = {
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
    }

    # Strict CSP for API endpoints
    STRICT_CSP = "default-src 'self'; frame-ancestors 'none'"

    # Relaxed CSP for documentation endpoints
    RELAXED_CSP = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "img-src 'self' data: https://fastapi.tiangolo.com; "
        "font-src 'self' https://cdn.jsdelivr.net; "
        "connect-src 'self'; "
        "frame-ancestors 'none'"
    )

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process the request and add security headers to the response."""
        response = await call_next(request)

        # Add standard security headers if they don't exist
        for header_name, header_value in self.SECURITY_HEADERS.items():
            if header_name not in response.headers:
                response.headers[header_name] = header_value

        # Determine appropriate CSP based on the request path
        if "Content-Security-Policy" not in response.headers:
            is_docs_path = any(
                request.url.path == path or request.url.path.startswith(path + "/")
                for path in self.DOCS_PATHS
            )
            if is_docs_path:
                response.headers["Content-Security-Policy"] = self.RELAXED_CSP
            else:
                response.headers["Content-Security-Policy"] = self.STRICT_CSP

        return response
