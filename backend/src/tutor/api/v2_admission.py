"""Fleet-shared, privacy-safe request admission for API v2."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import logging
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

AdmissionOperation = Literal["create", "recover", "reset", "action", "read"]

logger = logging.getLogger("tutor.api.v2.admission")

_IDENTITY_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LIFECYCLE_OPERATIONS = frozenset({"create", "recover", "reset"})
_DEFAULT_POLICIES = MappingProxyType(
    {
        "create": (10, 600),
        "recover": (10, 600),
        "reset": (10, 600),
        "action": (60, 60),
        "read": (120, 60),
    }
)

# Atomic continuous-refill token bucket. Redis TIME keeps workers independent
# of application clock skew. No network address or learner/session identifier
# appears in the key; callers supply only a keyed HMAC identity.
_TOKEN_BUCKET_LUA = """
local now_parts = redis.call('TIME')
local now_ms = (tonumber(now_parts[1]) * 1000) + math.floor(tonumber(now_parts[2]) / 1000)
local capacity = tonumber(ARGV[1])
local period_ms = tonumber(ARGV[2])
local refill_per_ms = capacity / period_ms
local state = redis.call('HMGET', KEYS[1], 'tokens', 'updated_ms')
local tokens = tonumber(state[1])
local updated_ms = tonumber(state[2])
if tokens == nil or updated_ms == nil then
  tokens = capacity
  updated_ms = now_ms
else
  local elapsed = math.max(0, now_ms - updated_ms)
  tokens = math.min(capacity, tokens + (elapsed * refill_per_ms))
end
local allowed = 0
local retry_ms = 0
if tokens >= 1 then
  allowed = 1
  tokens = tokens - 1
else
  retry_ms = math.ceil((1 - tokens) / refill_per_ms)
end
redis.call('HMSET', KEYS[1], 'tokens', tostring(tokens), 'updated_ms', tostring(now_ms))
redis.call('PEXPIRE', KEYS[1], math.ceil(period_ms * 2))
return {allowed, retry_ms}
"""


@dataclass(frozen=True)
class AdmissionDecision:
    """One safe admission result; unavailable is distinct from rate-limited."""

    allowed: bool
    available: bool = True
    retry_after_seconds: int | None = None

    def __post_init__(self) -> None:
        if type(self.allowed) is not bool or type(self.available) is not bool:
            raise TypeError("allowed and available must be booleans")
        if self.allowed and not self.available:
            raise ValueError("an unavailable admission service cannot allow a request")
        if self.allowed and self.retry_after_seconds is not None:
            raise ValueError("an allowed request cannot have Retry-After")
        if self.retry_after_seconds is not None and (
            type(self.retry_after_seconds) is not int
            or self.retry_after_seconds < 1
            or self.retry_after_seconds > 3600
        ):
            raise ValueError("retry_after_seconds must be between 1 and 3600")


@runtime_checkable
class RequestAdmissionGate(Protocol):
    """Fleet-shared request admission keyed by a privacy-safe identity."""

    def admit(
        self,
        operation: AdmissionOperation,
        *,
        peer_host: str | None,
        forwarded_for: tuple[str, ...] = (),
    ) -> AdmissionDecision:
        ...


class NetworkIdentityResolver:
    """Resolve a proxy-safe client address and return only its keyed HMAC."""

    def __init__(
        self,
        secret: bytes | str,
        *,
        trusted_proxy_cidrs: tuple[str, ...] = (),
    ) -> None:
        encoded = secret.encode("utf-8") if isinstance(secret, str) else secret
        if not isinstance(encoded, bytes) or len(encoded) < 32:
            raise ValueError("network HMAC secret must contain at least 32 bytes")
        self._secret = bytes(encoded)
        try:
            self._trusted = tuple(
                ipaddress.ip_network(value, strict=False)
                for value in trusted_proxy_cidrs
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("trusted proxy CIDRs are invalid") from exc

    def identity(
        self,
        peer_host: str | None,
        *,
        forwarded_for: tuple[str, ...] = (),
    ) -> str:
        peer = self._address(peer_host)
        resolved = peer
        if peer is not None and self._is_trusted(peer) and len(forwarded_for) == 1:
            chain = self._forwarded_chain(forwarded_for[0])
            if chain is not None:
                # Walk from the immediate proxy toward the originating client.
                # Stop at the first address outside our explicitly trusted set.
                resolved = chain[0]
                for candidate in reversed(chain):
                    resolved = candidate
                    if not self._is_trusted(candidate):
                        break
        canonical = resolved.compressed if resolved is not None else "invalid-peer"
        if canonical.startswith("::ffff:"):
            canonical = canonical.removeprefix("::ffff:")
        return hmac.new(
            self._secret,
            f"network-v1\0{canonical}".encode("ascii"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _address(value: str | None) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
        if not isinstance(value, str) or not value or len(value) > 64:
            return None
        try:
            return ipaddress.ip_address(value)
        except ValueError:
            return None

    def _is_trusted(self, address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        return any(
            address.version == network.version and address in network
            for network in self._trusted
        )

    def _forwarded_chain(
        self, value: str
    ) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...] | None:
        if not isinstance(value, str) or len(value) > 1024:
            return None
        pieces = tuple(piece.strip() for piece in value.split(","))
        if not pieces or len(pieces) > 16 or any(not piece for piece in pieces):
            return None
        parsed = tuple(self._address(piece) for piece in pieces)
        if any(address is None for address in parsed):
            return None
        return parsed  # type: ignore[return-value]


class NoopRequestAdmissionGate:
    """Development default; production construction never selects this gate."""

    def admit(
        self,
        operation: AdmissionOperation,
        *,
        peer_host: str | None,
        forwarded_for: tuple[str, ...] = (),
    ) -> AdmissionDecision:
        del operation, peer_host, forwarded_for
        return AdmissionDecision(allowed=True)


class RedisTokenBucketRequestAdmissionGate:
    """Atomic token buckets shared by every worker through Redis 7."""

    def __init__(
        self,
        redis_client: Any,
        identity_resolver: NetworkIdentityResolver,
        *,
        key_prefix: str = "tutor:v2:admission",
        policies: Mapping[AdmissionOperation, tuple[int, int]] | None = None,
    ) -> None:
        if redis_client is None or not callable(getattr(redis_client, "eval", None)):
            raise TypeError("redis_client must provide eval")
        if not isinstance(identity_resolver, NetworkIdentityResolver):
            raise TypeError("identity_resolver must be a NetworkIdentityResolver")
        if (
            not isinstance(key_prefix, str)
            or not key_prefix
            or len(key_prefix) > 128
            or not re.fullmatch(r"[A-Za-z0-9:_-]+", key_prefix)
        ):
            raise ValueError("key_prefix is invalid")
        supplied = dict(policies or _DEFAULT_POLICIES)
        if set(supplied) != set(_DEFAULT_POLICIES):
            raise ValueError("policies must configure every admission operation")
        for operation, policy in supplied.items():
            if (
                not isinstance(policy, tuple)
                or len(policy) != 2
                or any(type(value) is not int for value in policy)
                or not (1 <= policy[0] <= 10_000)
                or not (1 <= policy[1] <= 3600)
            ):
                raise ValueError(f"policy for {operation} is invalid")
        self._redis = redis_client
        self._identity = identity_resolver
        self._key_prefix = key_prefix
        self._policies = MappingProxyType(supplied)

    @classmethod
    def from_environment(
        cls,
        redis_client: Any,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> RedisTokenBucketRequestAdmissionGate:
        values = os.environ if environ is None else environ
        secret = values.get("TUTOR_NETWORK_HMAC_SECRET")
        if secret is None:
            raise ValueError("TUTOR_NETWORK_HMAC_SECRET is required")
        cidr_value = values.get("TUTOR_TRUSTED_PROXY_CIDRS")
        if cidr_value is None or not cidr_value.strip():
            raise ValueError("TUTOR_TRUSTED_PROXY_CIDRS is required")
        cidrs = tuple(part.strip() for part in cidr_value.split(","))
        if any(not part for part in cidrs):
            raise ValueError("TUTOR_TRUSTED_PROXY_CIDRS is invalid")
        return cls(
            redis_client,
            NetworkIdentityResolver(secret, trusted_proxy_cidrs=cidrs),
        )

    def admit(
        self,
        operation: AdmissionOperation,
        *,
        peer_host: str | None,
        forwarded_for: tuple[str, ...] = (),
    ) -> AdmissionDecision:
        if operation not in self._policies:
            raise ValueError("unknown admission operation")
        identity = self._identity.identity(
            peer_host,
            forwarded_for=forwarded_for,
        )
        if _IDENTITY_PATTERN.fullmatch(identity) is None:  # defensive invariant
            return AdmissionDecision(allowed=False, available=False)
        capacity, period_seconds = self._policies[operation]
        # Creation, recovery, and reset rotate anonymous-session authority and
        # therefore share one lifecycle budget. Separate buckets here would
        # silently triple the documented 10-per-10-minute admission limit.
        bucket_operation = (
            "lifecycle" if operation in _LIFECYCLE_OPERATIONS else operation
        )
        redis_key = f"{self._key_prefix}:{{{identity}}}:{bucket_operation}"
        try:
            result = self._redis.eval(
                _TOKEN_BUCKET_LUA,
                1,
                redis_key,
                capacity,
                period_seconds * 1000,
            )
            if (
                not isinstance(result, (list, tuple))
                or len(result) != 2
                or type(result[0]) is not int
                or type(result[1]) is not int
            ):
                raise TypeError("Redis admission result is invalid")
            allowed = result[0] == 1
            retry_ms = max(0, result[1])
            return AdmissionDecision(
                allowed=allowed,
                retry_after_seconds=(
                    None if allowed else min(3600, max(1, math.ceil(retry_ms / 1000)))
                ),
            )
        except Exception as exc:  # noqa: BLE001 - never expose Redis details
            logger.warning(
                "request admission unavailable operation=%s error_type=%s",
                operation,
                type(exc).__name__,
            )
            return AdmissionDecision(allowed=False, available=False)


__all__ = [
    "AdmissionDecision",
    "AdmissionOperation",
    "NetworkIdentityResolver",
    "NoopRequestAdmissionGate",
    "RedisTokenBucketRequestAdmissionGate",
    "RequestAdmissionGate",
]
