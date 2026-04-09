"""Middleware for adding security headers to responses.

Enforces HSTS, CSP, and other standard security headers.
"""

from typing import Any

from starlette.types import ASGIApp, Receive, Scope, Send


class SecurityHeadersMiddleware:
    """Middleware that adds security headers to all responses.

    Headers added:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Referrer-Policy: strict-origin-when-cross-origin
    - Content-Security-Policy: default-src 'self'
    - Strict-Transport-Security: max-age=31536000; includeSubDomains

    The CSP is relaxed for documentation endpoints to allow Swagger UI assets.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process the request and add headers to the response."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])

                # Helper to check if header already exists
                existing_headers = {
                    k.decode("latin-1").lower() for k, v in headers
                }

                def add_header(name: str, value: str) -> None:
                    if name.lower() not in existing_headers:
                        headers.append((name.encode("latin-1"), value.encode("latin-1")))

                # Standard security headers
                add_header("X-Content-Type-Options", "nosniff")
                add_header("X-Frame-Options", "DENY")
                add_header("X-XSS-Protection", "1; mode=block")
                add_header("Referrer-Policy", "strict-origin-when-cross-origin")
                add_header(
                    "Strict-Transport-Security",
                    "max-age=31536000; includeSubDomains",
                )

                # Content Security Policy
                # Relax for docs endpoints to allow Swagger UI
                path = scope.get("path", "")
                if path.startswith(("/docs", "/redoc", "/openapi.json")):
                    # Allow inline scripts/styles/images for Swagger UI
                    csp = (
                        "default-src 'self'; "
                        "script-src 'self' 'unsafe-inline'; "
                        "style-src 'self' 'unsafe-inline'; "
                        "img-src 'self' data: https:; "
                    )
                else:
                    # Strict default
                    csp = "default-src 'self'"

                add_header("Content-Security-Policy", csp)

            await send(message)

        await self.app(scope, receive, send_wrapper)
