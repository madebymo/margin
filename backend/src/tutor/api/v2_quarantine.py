"""Fail-closed release quarantine contracts for session API v2.

The request path consumes an immutable, already-cached snapshot.  A Redis or
other fleet control adapter is responsible for refreshing that cache outside
the request path; provider/network I/O must never happen in ``snapshot()``.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

from tutor.api.v2_versions import V2ContentRelease

MAX_QUARANTINE_LABEL_LENGTH = 128
_MAX_FUTURE_SKEW = timedelta(seconds=5)


def _label(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and unpadded")
    if len(value) > MAX_QUARANTINE_LABEL_LENGTH or not value.isprintable():
        raise ValueError(
            f"{name} must contain at most {MAX_QUARANTINE_LABEL_LENGTH} "
            "printable characters"
        )
    return value


def _aware(name: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def release_runtime_digest(
    release: V2ContentRelease,
    policy_versions: dict[str, str],
) -> str:
    """Return a canonical digest for content plus executable policy pins."""

    if not policy_versions or any(
        not isinstance(name, str)
        or not name
        or not isinstance(version, str)
        or not version
        for name, version in policy_versions.items()
    ):
        raise ValueError("policy versions must be a non-empty string mapping")
    payload = {
        "schema_version": 1,
        "graph": release.graph.model_dump(mode="json"),
        "item_bank": release.item_bank.model_dump(mode="json"),
        "pedagogy_catalog": release.pedagogy_catalog.model_dump(mode="json"),
        "policy_versions": dict(sorted(policy_versions.items())),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReleaseQuarantineSnapshot:
    """One cached fleet observation keyed by canonical runtime digest."""

    quarantined_digests: frozenset[str]
    revision: str
    source: str
    observed_at: datetime
    available: bool = True

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise TypeError("available must be a boolean")
        object.__setattr__(self, "revision", _label("revision", self.revision))
        object.__setattr__(self, "source", _label("source", self.source))
        object.__setattr__(self, "observed_at", _aware("observed_at", self.observed_at))
        normalized = frozenset(self.quarantined_digests)
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in normalized
        ):
            raise ValueError("quarantined release digests must be lowercase SHA-256")
        object.__setattr__(self, "quarantined_digests", normalized)

    def is_quarantined(self, release_digest: str) -> bool:
        return release_digest in self.quarantined_digests


@runtime_checkable
class ReleaseQuarantineProvider(Protocol):
    """O(1), nonblocking provider of a pre-refreshed quarantine snapshot."""

    def snapshot(self) -> ReleaseQuarantineSnapshot:
        ...


class StaticReleaseQuarantineProvider:
    """Local/test provider captured once at construction."""

    def __init__(
        self,
        quarantined_digests: frozenset[str] = frozenset(),
        *,
        revision: str = "static-quarantine-v1",
        source: str = "builtin_static",
        observed_at: datetime | None = None,
        available: bool = True,
    ) -> None:
        self._snapshot = ReleaseQuarantineSnapshot(
            quarantined_digests=quarantined_digests,
            revision=revision,
            source=source,
            observed_at=observed_at or datetime.now(timezone.utc),
            available=available,
        )

    def snapshot(self) -> ReleaseQuarantineSnapshot:
        return self._snapshot


class CachedReleaseQuarantineProvider:
    """Thread-safe cache for an out-of-band fleet control refresher.

    The cache starts unavailable so a deployment cannot accidentally serve
    content before its Redis/control-plane consumer has populated state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = ReleaseQuarantineSnapshot(
            quarantined_digests=frozenset(),
            revision="cache-uninitialized-v1",
            source="cached_control_plane",
            observed_at=datetime.now(timezone.utc),
            available=False,
        )

    def replace(self, snapshot: ReleaseQuarantineSnapshot) -> None:
        if not isinstance(snapshot, ReleaseQuarantineSnapshot):
            raise TypeError("snapshot must be a ReleaseQuarantineSnapshot")
        with self._lock:
            self._snapshot = snapshot

    def snapshot(self) -> ReleaseQuarantineSnapshot:
        with self._lock:
            return self._snapshot


def safe_release_quarantine_snapshot(
    provider: ReleaseQuarantineProvider,
    *,
    now: datetime | None = None,
    max_age: timedelta | None = None,
) -> ReleaseQuarantineSnapshot:
    """Normalize provider failures, future values, and staleness as unavailable."""

    observed_now = _aware("now", now or datetime.now(timezone.utc))
    if max_age is not None:
        if not isinstance(max_age, timedelta) or max_age <= timedelta(0):
            raise ValueError("max_age must be a positive timedelta")
    try:
        candidate = provider.snapshot()
        if not isinstance(candidate, ReleaseQuarantineSnapshot):
            raise TypeError("provider returned an invalid quarantine snapshot")
        if candidate.observed_at - observed_now > _MAX_FUTURE_SKEW:
            raise ValueError("quarantine snapshot is from the future")
        if max_age is not None and observed_now - candidate.observed_at > max_age:
            raise ValueError("quarantine snapshot is stale")
        return candidate
    except Exception:
        return ReleaseQuarantineSnapshot(
            quarantined_digests=frozenset(),
            revision="fail-closed-quarantine-v1",
            source="fail_closed",
            observed_at=observed_now,
            available=False,
        )


__all__ = [
    "CachedReleaseQuarantineProvider",
    "ReleaseQuarantineProvider",
    "ReleaseQuarantineSnapshot",
    "StaticReleaseQuarantineProvider",
    "release_runtime_digest",
    "safe_release_quarantine_snapshot",
]
