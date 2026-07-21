"""Small ASGI safety middleware used before request-body parsing."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

DEFAULT_MAX_REQUEST_BODY_BYTES = 64 * 1024
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_TRUSTED_HOST_PATTERN = re.compile(
    r"^(?:\*|\*\.[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?|"
    r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?)$"
)
_DEFAULT_DEVELOPMENT_HOSTS = ("testserver", "localhost", "*.localhost", "127.0.0.1")

CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'self'",
        "base-uri 'none'",
        "connect-src 'self'",
        "font-src 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "img-src 'self' data:",
        "object-src 'none'",
        "script-src 'self'",
        # The compiled UI uses bounded, application-authored style attributes
        # for progress and SVG geometry. No student value is interpolated into
        # CSS, but those attributes require this CSP allowance.
        "style-src 'self' 'unsafe-inline'",
    )
)


def trusted_hosts_from_environment(
    *,
    pilot_production: bool,
    environ: Mapping[str, str],
) -> tuple[str, ...]:
    """Parse an explicit, bounded host allowlist for Starlette.

    Development has a loopback-only default. Pilot production has no implicit
    domain and rejects wildcard trust, so a deployment cannot accidentally
    serve with host-header validation disabled.
    """

    raw = environ.get("TUTOR_TRUSTED_HOSTS", "")
    if not raw.strip():
        if pilot_production:
            raise RuntimeError(
                "TUTOR_PILOT_PRODUCTION requires explicit TUTOR_TRUSTED_HOSTS"
            )
        return _DEFAULT_DEVELOPMENT_HOSTS
    hosts = tuple(dict.fromkeys(part.strip().lower() for part in raw.split(",")))
    if any(not host for host in hosts):
        raise ValueError("TUTOR_TRUSTED_HOSTS cannot contain empty entries")
    if len(hosts) > 32:
        raise ValueError("TUTOR_TRUSTED_HOSTS cannot contain more than 32 entries")
    invalid = [host for host in hosts if _TRUSTED_HOST_PATTERN.fullmatch(host) is None]
    if invalid:
        raise ValueError(
            "TUTOR_TRUSTED_HOSTS entries must be hostnames without schemes, ports, or paths"
        )
    if pilot_production and any(host == "*" for host in hosts):
        raise RuntimeError("TUTOR_PILOT_PRODUCTION forbids wildcard trusted hosts")
    return hosts


class HttpSecurityHeadersMiddleware:
    """Attach a fixed browser policy without inspecting response bodies."""

    def __init__(self, app: Any, *, secure_transport: bool = False):
        self.app = app
        self.secure_transport = secure_transport

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def secure_send(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", ()))
                names = {name.lower() for name, _ in headers}

                def add(name: bytes, value: bytes) -> None:
                    if name not in names:
                        headers.append((name, value))
                        names.add(name)

                add(b"content-security-policy", CONTENT_SECURITY_POLICY.encode("ascii"))
                add(b"cross-origin-opener-policy", b"same-origin")
                add(b"cross-origin-resource-policy", b"same-origin")
                add(b"permissions-policy", b"camera=(), microphone=(), geolocation=(), payment=(), usb=()")
                add(b"referrer-policy", b"no-referrer")
                add(b"x-content-type-options", b"nosniff")
                add(b"x-frame-options", b"DENY")
                if self.secure_transport:
                    add(
                        b"strict-transport-security",
                        b"max-age=31536000; includeSubDomains",
                    )
                path = str(scope.get("path", ""))
                if path == "/" or path.startswith("/api/") or path.startswith("/sessions"):
                    add(b"cache-control", b"no-store")
                    add(b"pragma", b"no-cache")
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, secure_send)


class RequestBodyLimitMiddleware:
    """Reject oversized bodies before FastAPI/Pydantic parses JSON.

    The body is buffered only up to the configured limit and replayed once to
    the downstream application.  This also covers chunked requests whose size
    cannot be determined from ``Content-Length``.
    """

    def __init__(self, app: Any, max_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES):
        if type(max_body_bytes) is not int or not 1024 <= max_body_bytes <= 1024 * 1024:
            raise ValueError("max_body_bytes must be between 1024 and 1048576")
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") not in _BODY_METHODS:
            await self.app(scope, receive, send)
            return

        content_length = self._content_length(scope)
        if content_length is not None and content_length > self.max_body_bytes:
            await self._reject(scope, send)
            return

        body = bytearray()
        disconnected = False
        while True:
            message = await receive()
            message_type = message.get("type")
            if message_type == "http.disconnect":
                disconnected = True
                break
            if message_type != "http.request":
                continue
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_body_bytes:
                await self._reject(scope, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        delivered = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal delivered
            if delivered or disconnected:
                return {"type": "http.disconnect"}
            delivered = True
            return {
                "type": "http.request",
                "body": bytes(body),
                "more_body": False,
            }

        await self.app(scope, replay_receive, send)

    @staticmethod
    def _content_length(scope: dict) -> int | None:
        values = [
            value
            for name, value in scope.get("headers", ())
            if name.lower() == b"content-length"
        ]
        if len(values) != 1:
            return None
        try:
            value = int(values[0])
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None

    @staticmethod
    async def _reject(scope: dict, send: Any) -> None:
        is_v2 = str(scope.get("path", "")).startswith("/api/v2/")
        payload = (
            {
                "code": "request_too_large",
                "message": "request body exceeds the 64 KiB limit",
            }
            if is_v2
            else {"detail": "request body too large"}
        )
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


__all__ = [
    "CONTENT_SECURITY_POLICY",
    "DEFAULT_MAX_REQUEST_BODY_BYTES",
    "HttpSecurityHeadersMiddleware",
    "RequestBodyLimitMiddleware",
    "trusted_hosts_from_environment",
]
