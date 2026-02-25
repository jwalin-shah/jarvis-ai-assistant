"""Security headers middleware.

Adds standard security headers to all HTTP responses to protect against
common web vulnerabilities like XSS, clickjacking, and MIME sniffing.
"""

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class SecurityHeadersMiddleware:
    """Middleware that adds security headers to all responses."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process the request and add security headers to the response."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Check if the path is a documentation endpoint
        path = scope.get("path", "")
        is_docs = path.startswith(("/docs", "/redoc", "/openapi.json"))

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)

                # Prevent MIME type sniffing
                headers.setdefault("X-Content-Type-Options", "nosniff")

                # Prevent clickjacking (deny framing)
                headers.setdefault("X-Frame-Options", "DENY")

                # Control referrer information (privacy)
                headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")

                # Restrict browser features (privacy/security)
                headers.setdefault(
                    "Permissions-Policy",
                    "geolocation=(), microphone=(), camera=(), payment=(), usb=(), vr=()"
                )

                # Strict-Transport-Security (HSTS)
                # Enforce HTTPS for 1 year (ignored on localhost usually, but good practice)
                headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")

                # Content Security Policy
                # Skip for docs to allow Swagger UI/ReDoc to load assets
                if not is_docs:
                    headers.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")

            await send(message)

        await self.app(scope, receive, send_wrapper)
