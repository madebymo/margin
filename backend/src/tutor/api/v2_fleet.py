"""Built-in Redis fleet construction for the production pilot."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any
from urllib.parse import urlsplit


@dataclass(frozen=True)
class RedisFleetSettings:
    """Bounded Redis settings; the URL is excluded from representations."""

    url: str = field(repr=False)
    refresh_interval_seconds: float = 5.0
    safety_max_age_seconds: float = 20.0
    socket_connect_timeout_seconds: float = 1.0
    socket_timeout_seconds: float = 1.0

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> RedisFleetSettings:
        values = os.environ if environ is None else environ
        url = values.get("TUTOR_REDIS_URL")
        if url is None or not url.strip():
            raise ValueError("TUTOR_REDIS_URL is required")
        parsed = urlsplit(url.strip())
        if parsed.scheme not in {"redis", "rediss"} or not parsed.hostname:
            raise ValueError("TUTOR_REDIS_URL must be a redis or rediss URL")
        refresh = _float_setting(
            values,
            "TUTOR_REDIS_CONTROL_REFRESH_SECONDS",
            default=5.0,
            minimum=0.1,
            maximum=30.0,
        )
        max_age = _float_setting(
            values,
            "TUTOR_REDIS_CONTROL_MAX_AGE_SECONDS",
            default=20.0,
            minimum=1.0,
            maximum=60.0,
        )
        if max_age <= refresh:
            raise ValueError(
                "TUTOR_REDIS_CONTROL_MAX_AGE_SECONDS must exceed the refresh interval"
            )
        return cls(
            url=url.strip(),
            refresh_interval_seconds=refresh,
            safety_max_age_seconds=max_age,
            socket_connect_timeout_seconds=_float_setting(
                values,
                "TUTOR_REDIS_CONNECT_TIMEOUT_SECONDS",
                default=1.0,
                minimum=0.1,
                maximum=5.0,
            ),
            socket_timeout_seconds=_float_setting(
                values,
                "TUTOR_REDIS_SOCKET_TIMEOUT_SECONDS",
                default=1.0,
                minimum=0.1,
                maximum=5.0,
            ),
        )

    @property
    def safety_max_age(self) -> timedelta:
        return timedelta(seconds=self.safety_max_age_seconds)


def create_redis_client(settings: RedisFleetSettings) -> Any:
    """Construct a bounded synchronous Redis client without connecting yet."""

    if not isinstance(settings, RedisFleetSettings):
        raise TypeError("settings must be RedisFleetSettings")
    try:
        import redis

        return redis.Redis.from_url(
            settings.url,
            socket_connect_timeout=settings.socket_connect_timeout_seconds,
            socket_timeout=settings.socket_timeout_seconds,
            health_check_interval=15,
            decode_responses=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Redis fleet client initialization failed ({type(exc).__name__})"
        ) from None


def _float_setting(
    environ: Mapping[str, str],
    name: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = environ.get(name)
    try:
        value = default if raw is None else float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


__all__ = ["RedisFleetSettings", "create_redis_client"]
