from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        # Add standard security headers using setdefault so we don't overwrite
        # any headers already set by specific endpoints.
        if "X-Content-Type-Options" not in response.headers:
            response.headers["X-Content-Type-Options"] = "nosniff"
        if "X-Frame-Options" not in response.headers:
            response.headers["X-Frame-Options"] = "DENY"
        if "X-XSS-Protection" not in response.headers:
            response.headers["X-XSS-Protection"] = "1; mode=block"
        if "Strict-Transport-Security" not in response.headers:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Paths that need to load assets from external sources (like unpkg for swagger)
        docs_paths = ["/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"]

        if request.url.path not in docs_paths:
            if "Content-Security-Policy" not in response.headers:
                response.headers["Content-Security-Policy"] = "default-src 'self';"

        return response
