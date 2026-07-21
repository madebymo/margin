"""Small ASGI safety middleware used before request-body parsing."""

from __future__ import annotations

import json
from typing import Any

DEFAULT_MAX_REQUEST_BODY_BYTES = 64 * 1024
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


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


__all__ = ["DEFAULT_MAX_REQUEST_BODY_BYTES", "RequestBodyLimitMiddleware"]
