"""Runtime control boundary for fail-closed v2 mutation admission.

The API deliberately depends on a small provider protocol rather than a
particular configuration service.  A provider may refresh its state at any
time, while :class:`StaticMutationGate` preserves the existing startup-only
environment behaviour for local development and simple deployments.
"""

from __future__ import annotations

import os
import json
import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Protocol, runtime_checkable

from tutor.api.background_refresh import BackgroundRefresher

DEFAULT_MUTATION_PAUSE_ENV = "TUTOR_PAUSE_V2_MUTATIONS"
MAX_MUTATION_GATE_REVISION_LENGTH = 128
MAX_MUTATION_GATE_SOURCE_LENGTH = 128

_ENV_SOURCE_PREFIX = "environment:"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_FAIL_CLOSED_REVISION = "fail-closed-v1"
_FAIL_CLOSED_SOURCE = "fail_closed"
_MAX_FUTURE_SKEW = timedelta(seconds=5)
_MAX_REDIS_CONTROL_BYTES = 16 * 1024

logger = logging.getLogger("tutor.api.v2.controls")


def _bounded_label(name: str, value: object, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or not value.strip():
        raise ValueError(f"{name} must not be blank")
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    if len(value) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} characters")
    if not value.isprintable():
        raise ValueError(f"{name} must contain only printable characters")
    return value


def _as_utc(name: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class MutationGateSnapshot:
    """One validated observation of the mutation gate.

    ``revision`` identifies the control-plane value and ``source`` identifies
    the provider that supplied it.  They are intentionally bounded so the
    snapshot is safe to attach to health data or structured metrics.
    """

    paused: bool
    revision: str
    source: str
    observed_at: datetime

    def __post_init__(self) -> None:
        if type(self.paused) is not bool:
            raise TypeError("paused must be a boolean")
        object.__setattr__(
            self,
            "revision",
            _bounded_label(
                "revision",
                self.revision,
                maximum=MAX_MUTATION_GATE_REVISION_LENGTH,
            ),
        )
        object.__setattr__(
            self,
            "source",
            _bounded_label(
                "source",
                self.source,
                maximum=MAX_MUTATION_GATE_SOURCE_LENGTH,
            ),
        )
        object.__setattr__(
            self,
            "observed_at",
            _as_utc("observed_at", self.observed_at),
        )


@runtime_checkable
class MutationGate(Protocol):
    """Provider of the current mutation-admission decision."""

    def snapshot(self) -> MutationGateSnapshot:
        """Return the provider's latest validated observation."""
        ...


class StaticMutationGate:
    """A mutation gate whose value is captured once at construction."""

    def __init__(
        self,
        paused: bool = False,
        *,
        revision: str | None = None,
        source: str = "static",
        observed_at: datetime | None = None,
    ) -> None:
        state_name = "paused" if paused is True else "open"
        self._snapshot = MutationGateSnapshot(
            paused=paused,
            revision=(revision if revision is not None else f"static-v1:{state_name}"),
            source=source,
            observed_at=(observed_at if observed_at is not None else datetime.now(timezone.utc)),
        )

    @classmethod
    def from_environment(
        cls,
        env_name: str = DEFAULT_MUTATION_PAUSE_ENV,
        *,
        environ: Mapping[str, str] | None = None,
        default: bool = False,
        observed_at: datetime | None = None,
    ) -> StaticMutationGate:
        """Capture a strictly parsed environment switch once.

        A custom mapping makes construction deterministic in tests and avoids
        mutating process-global environment state.  Invalid configured values
        raise without copying the value into the exception message.
        """

        if type(default) is not bool:
            raise TypeError("default must be a boolean")
        maximum_name_length = MAX_MUTATION_GATE_SOURCE_LENGTH - len(_ENV_SOURCE_PREFIX)
        env_name = _bounded_label(
            "env_name",
            env_name,
            maximum=maximum_name_length,
        )
        source_environ = os.environ if environ is None else environ
        raw = source_environ.get(env_name)
        if raw is None:
            paused = default
        elif not isinstance(raw, str):
            raise TypeError(f"{env_name} must be a string")
        else:
            normalized = raw.strip().lower()
            if normalized in _TRUE_VALUES:
                paused = True
            elif normalized in _FALSE_VALUES:
                paused = False
            else:
                raise ValueError(
                    f"{env_name} must be one of 1, 0, true, false, yes, no, on, or off"
                )
        state_name = "paused" if paused else "open"
        return cls(
            paused,
            revision=f"static-env-v1:{state_name}",
            source=f"{_ENV_SOURCE_PREFIX}{env_name}",
            observed_at=observed_at,
        )

    def snapshot(self) -> MutationGateSnapshot:
        """Return the immutable startup snapshot."""

        return self._snapshot


class RedisMutationGate:
    """Redis-backed mutation control with an O(1), nonblocking request path.

    Redis stores a strict JSON document at ``key``::

        {"schema_version": 1, "paused": true, "revision": "incident-42"}

    The background refresher is the only code that performs network I/O.
    Missing, malformed, or unavailable state immediately replaces the cache
    with a paused fail-closed observation.
    """

    def __init__(
        self,
        redis_client: Any,
        *,
        key: str = "tutor:v2:controls:mutations",
        refresh_interval_seconds: float = 5.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if redis_client is None or not callable(getattr(redis_client, "get", None)):
            raise TypeError("redis_client must provide get")
        if not isinstance(key, str) or not key or len(key) > 256 or not key.isprintable():
            raise ValueError("key must be a non-empty printable string of at most 256 characters")
        self._redis = redis_client
        self._key = key
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._snapshot = MutationGateSnapshot(
            paused=True,
            revision="redis-uninitialized-v1",
            source="fail_closed",
            observed_at=self._now(),
        )
        self._refresher = BackgroundRefresher(
            self.refresh_once,
            interval_seconds=refresh_interval_seconds,
            name="tutor-redis-mutation-gate",
            logger=logger,
        )

    def _now(self) -> datetime:
        return _as_utc("clock result", self._clock())

    def start(self) -> None:
        self._refresher.start()

    def close(self) -> None:
        self._refresher.close()

    def refresh_once(self) -> None:
        """Fetch and validate one control value outside the request path."""

        observed_at = self._now()
        try:
            raw = self._redis.get(self._key)
            if raw is None:
                raise ValueError("mutation control is missing")
            if isinstance(raw, bytes):
                if len(raw) > _MAX_REDIS_CONTROL_BYTES:
                    raise ValueError("mutation control is oversized")
                text = raw.decode("utf-8", errors="strict")
            elif isinstance(raw, str):
                if len(raw.encode("utf-8")) > _MAX_REDIS_CONTROL_BYTES:
                    raise ValueError("mutation control is oversized")
                text = raw
            else:
                raise TypeError("mutation control must be text")
            payload = json.loads(text)
            if not isinstance(payload, dict) or set(payload) != {
                "schema_version",
                "paused",
                "revision",
            }:
                raise ValueError("mutation control has an invalid schema")
            if payload["schema_version"] != 1:
                raise ValueError("mutation control schema version is unsupported")
            candidate = MutationGateSnapshot(
                paused=payload["paused"],
                revision=payload["revision"],
                source="redis_control_plane",
                observed_at=observed_at,
            )
        except Exception as exc:  # noqa: BLE001 - payload/config stay private
            candidate = MutationGateSnapshot(
                paused=True,
                revision="redis-refresh-failed-v1",
                source="fail_closed",
                observed_at=observed_at,
            )
            logger.warning(
                "mutation control refresh failed error_type=%s",
                type(exc).__name__,
            )
        with self._lock:
            self._snapshot = candidate

    def snapshot(self) -> MutationGateSnapshot:
        """Return the already-refreshed immutable value without Redis I/O."""

        with self._lock:
            return self._snapshot


def safe_mutation_gate_snapshot(
    gate: MutationGate,
    *,
    now: datetime | None = None,
    max_age: timedelta | None = None,
) -> MutationGateSnapshot:
    """Read ``gate`` and convert provider failures or stale state into pause.

    ``max_age`` is optional because a deliberately static provider has no
    refresh contract.  Dynamic control-plane integrations should set it to a
    duration shorter than their operational safety window.
    """

    observed_now = _as_utc("now", now or datetime.now(timezone.utc))
    if max_age is not None:
        if not isinstance(max_age, timedelta):
            raise TypeError("max_age must be a timedelta")
        if max_age <= timedelta(0):
            raise ValueError("max_age must be positive")

    try:
        candidate = gate.snapshot()
        if not isinstance(candidate, MutationGateSnapshot):
            raise TypeError("mutation gate returned an invalid snapshot")
        if candidate.observed_at - observed_now > _MAX_FUTURE_SKEW:
            raise ValueError("mutation gate snapshot is from the future")
        if max_age is not None and observed_now - candidate.observed_at > max_age:
            raise ValueError("mutation gate snapshot is stale")
        return candidate
    except Exception:
        return MutationGateSnapshot(
            paused=True,
            revision=_FAIL_CLOSED_REVISION,
            source=_FAIL_CLOSED_SOURCE,
            observed_at=observed_now,
        )


__all__ = [
    "DEFAULT_MUTATION_PAUSE_ENV",
    "MAX_MUTATION_GATE_REVISION_LENGTH",
    "MAX_MUTATION_GATE_SOURCE_LENGTH",
    "MutationGate",
    "MutationGateSnapshot",
    "RedisMutationGate",
    "StaticMutationGate",
    "safe_mutation_gate_snapshot",
]
