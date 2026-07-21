"""Small lifecycle helper for cached fleet-control adapters.

The request path must never wait for Redis.  Concrete safety providers use
this helper to refresh an immutable local snapshot on a daemon thread while
their public ``snapshot()`` methods remain lock-only operations.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable


class BackgroundRefresher:
    """Periodically invoke one bounded refresh callback.

    Refresh failures are handled by the callback because each safety contract
    has a different fail-closed representation.  The helper deliberately logs
    only the exception class; Redis configuration and provider payloads may
    contain deployment secrets.
    """

    def __init__(
        self,
        refresh: Callable[[], None],
        *,
        interval_seconds: float,
        name: str,
        logger: logging.Logger,
    ) -> None:
        if not callable(refresh):
            raise TypeError("refresh must be callable")
        if not isinstance(interval_seconds, (int, float)) or not (
            0.1 <= float(interval_seconds) <= 30.0
        ):
            raise ValueError("interval_seconds must be between 0.1 and 30")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        self._refresh = refresh
        self._interval_seconds = float(interval_seconds)
        self._name = name
        self._logger = logger
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Populate once and start the refresh loop; safe to call repeatedly."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._run_once()
            thread = threading.Thread(
                target=self._run,
                name=self._name,
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def close(self, timeout_seconds: float = 1.0) -> None:
        """Stop future refreshes without waiting indefinitely."""

        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be nonnegative")
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout_seconds)

    def _run_once(self) -> None:
        try:
            self._refresh()
        except Exception as exc:  # pragma: no cover - callbacks normally absorb
            self._logger.warning(
                "fleet control refresh failed adapter=%s error_type=%s",
                self._name,
                type(exc).__name__,
            )

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._run_once()


__all__ = ["BackgroundRefresher"]
