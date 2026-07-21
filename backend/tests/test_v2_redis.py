"""Opt-in Redis 7 integration gates for fleet-shared v2 safety controls.

Set ``TUTOR_TEST_REDIS_URL`` to a disposable Redis instance. The ordinary
unit suite skips this module without importing the optional Redis package.
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from tutor.api.v2_admission import (
    NetworkIdentityResolver,
    RedisTokenBucketRequestAdmissionGate,
)
from tutor.api.v2_controls import RedisMutationGate, safe_mutation_gate_snapshot
from tutor.api.v2_quarantine import (
    RedisReleaseQuarantineProvider,
    safe_release_quarantine_snapshot,
)

_REDIS_ENV = "TUTOR_TEST_REDIS_URL"


@pytest.fixture(scope="module")
def redis_test_context():
    raw_url = os.environ.get(_REDIS_ENV)
    if not raw_url:
        pytest.skip(f"set {_REDIS_ENV} to run Redis integration gates")
    redis = pytest.importorskip("redis")
    client = redis.Redis.from_url(
        raw_url,
        socket_connect_timeout=1,
        socket_timeout=1,
        decode_responses=False,
    )
    try:
        client.ping()
    except Exception as exc:  # noqa: BLE001 - integration setup boundary
        pytest.fail(f"{_REDIS_ENV} is not reachable ({type(exc).__name__})")

    prefix = f"tutor:test:v2:{uuid4().hex}"
    try:
        yield SimpleNamespace(client=client, prefix=prefix)
    finally:
        keys = list(client.scan_iter(match=f"{prefix}:*", count=100))
        if keys:
            client.delete(*keys)
        client.close()


def test_two_workers_share_mutation_and_quarantine_controls(redis_test_context):
    client = redis_test_context.client
    prefix = redis_test_context.prefix
    mutation_key = f"{prefix}:controls:mutations"
    quarantine_key = f"{prefix}:controls:quarantine"
    digest = "a" * 64
    first_mutation = RedisMutationGate(client, key=mutation_key)
    second_mutation = RedisMutationGate(client, key=mutation_key)
    first_quarantine = RedisReleaseQuarantineProvider(client, key=quarantine_key)
    second_quarantine = RedisReleaseQuarantineProvider(client, key=quarantine_key)

    client.set(
        mutation_key,
        json.dumps(
            {"schema_version": 1, "paused": False, "revision": "open-1"}
        ),
    )
    client.set(
        quarantine_key,
        json.dumps(
            {
                "schema_version": 1,
                "revision": "safe-1",
                "quarantined_digests": [],
            }
        ),
    )
    for provider in (
        first_mutation,
        second_mutation,
        first_quarantine,
        second_quarantine,
    ):
        provider.refresh_once()

    assert first_mutation.snapshot().paused is False
    assert second_mutation.snapshot().paused is False
    assert first_quarantine.snapshot().available is True
    assert second_quarantine.snapshot().available is True

    client.set(
        mutation_key,
        json.dumps(
            {"schema_version": 1, "paused": True, "revision": "incident-2"}
        ),
    )
    client.set(
        quarantine_key,
        json.dumps(
            {
                "schema_version": 1,
                "revision": "incident-2",
                "quarantined_digests": [digest],
            }
        ),
    )
    for provider in (
        first_mutation,
        second_mutation,
        first_quarantine,
        second_quarantine,
    ):
        provider.refresh_once()

    assert first_mutation.snapshot().paused is True
    assert second_mutation.snapshot().paused is True
    assert first_quarantine.snapshot().is_quarantined(digest)
    assert second_quarantine.snapshot().is_quarantined(digest)


def test_two_workers_share_one_lifecycle_admission_budget(redis_test_context):
    client = redis_test_context.client
    prefix = redis_test_context.prefix
    resolver = NetworkIdentityResolver(
        b"redis-integration-network-secret-32-bytes",
        trusted_proxy_cidrs=("10.0.0.0/8",),
    )
    first = RedisTokenBucketRequestAdmissionGate(
        client,
        resolver,
        key_prefix=f"{prefix}:admission",
    )
    second = RedisTokenBucketRequestAdmissionGate(
        client,
        resolver,
        key_prefix=f"{prefix}:admission",
    )
    operations = ("create", "recover", "reset", "create", "recover") * 2

    decisions = [
        (first if index % 2 == 0 else second).admit(
            operation,
            peer_host="10.1.1.2",
            forwarded_for=("198.51.100.8",),
        )
        for index, operation in enumerate(operations)
    ]
    rejected = first.admit(
        "reset",
        peer_host="10.1.1.2",
        forwarded_for=("198.51.100.8",),
    )

    assert all(decision.allowed for decision in decisions)
    assert rejected.available is True
    assert rejected.allowed is False
    keys = list(client.scan_iter(match=f"{prefix}:admission:*"))
    assert len(keys) == 1
    assert b"198.51.100.8" not in keys[0]
    assert keys[0].endswith(b":lifecycle")


def test_stale_cached_controls_fail_closed_without_redis_io(redis_test_context):
    client = redis_test_context.client
    prefix = redis_test_context.prefix
    mutation_key = f"{prefix}:stale:mutations"
    quarantine_key = f"{prefix}:stale:quarantine"
    mutation = RedisMutationGate(client, key=mutation_key)
    quarantine = RedisReleaseQuarantineProvider(
        client,
        key=quarantine_key,
    )
    client.set(
        mutation_key,
        json.dumps(
            {"schema_version": 1, "paused": False, "revision": "open-stale"}
        ),
    )
    client.set(
        quarantine_key,
        json.dumps(
            {
                "schema_version": 1,
                "revision": "safe-stale",
                "quarantined_digests": [],
            }
        ),
    )
    mutation.refresh_once()
    quarantine.refresh_once()
    future = max(
        mutation.snapshot().observed_at,
        quarantine.snapshot().observed_at,
    ) + timedelta(seconds=21)

    stale_mutation = safe_mutation_gate_snapshot(
        mutation,
        now=future,
        max_age=timedelta(seconds=20),
    )
    stale_quarantine = safe_release_quarantine_snapshot(
        quarantine,
        now=future,
        max_age=timedelta(seconds=20),
    )

    assert stale_mutation.paused is True
    assert stale_mutation.source == "fail_closed"
    assert stale_quarantine.available is False
    assert stale_quarantine.source == "fail_closed"
