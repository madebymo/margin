"""Contract tests for dynamic and fail-closed v2 mutation controls."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone, tzinfo

import pytest

from tutor.api.v2_controls import (
    DEFAULT_MUTATION_PAUSE_ENV,
    MAX_MUTATION_GATE_REVISION_LENGTH,
    MAX_MUTATION_GATE_SOURCE_LENGTH,
    MutationGate,
    MutationGateSnapshot,
    StaticMutationGate,
    safe_mutation_gate_snapshot,
)

_NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


class MutableMutationGate:
    """Shared in-process test fake that models a live control-plane provider."""

    def __init__(self, *, paused: bool = False) -> None:
        self._revision = 0
        self._paused = paused

    def set_paused(self, paused: bool) -> None:
        self._revision += 1
        self._paused = paused

    def snapshot(self) -> MutationGateSnapshot:
        return MutationGateSnapshot(
            paused=self._paused,
            revision=f"test-{self._revision}",
            source="mutable_test_fake",
            observed_at=_NOW + timedelta(seconds=self._revision),
        )


def _snapshot(**overrides: object) -> MutationGateSnapshot:
    values = {
        "paused": False,
        "revision": "control-17",
        "source": "test-provider",
        "observed_at": _NOW,
    }
    values.update(overrides)
    return MutationGateSnapshot(**values)  # type: ignore[arg-type]


def test_snapshot_is_frozen_and_normalizes_aware_time_to_utc():
    eastern = timezone(timedelta(hours=-4))
    snapshot = _snapshot(
        observed_at=datetime(2026, 7, 20, 8, 0, tzinfo=eastern),
    )

    assert snapshot.observed_at == _NOW
    assert snapshot.observed_at.tzinfo is timezone.utc
    with pytest.raises(FrozenInstanceError):
        snapshot.paused = True  # type: ignore[misc]


@pytest.mark.parametrize("paused", [0, 1, "false", None])
def test_snapshot_requires_a_strict_boolean(paused):
    with pytest.raises(TypeError, match="paused must be a boolean"):
        _snapshot(paused=paused)


@pytest.mark.parametrize("field", ["revision", "source"])
@pytest.mark.parametrize(
    ("value", "exception"),
    [
        (None, TypeError),
        (7, TypeError),
        ("", ValueError),
        ("   ", ValueError),
        (" padded", ValueError),
        ("padded ", ValueError),
        ("line\nbreak", ValueError),
    ],
)
def test_snapshot_rejects_invalid_revision_and_source(field, value, exception):
    with pytest.raises(exception):
        _snapshot(**{field: value})


@pytest.mark.parametrize(
    ("field", "maximum"),
    [
        ("revision", MAX_MUTATION_GATE_REVISION_LENGTH),
        ("source", MAX_MUTATION_GATE_SOURCE_LENGTH),
    ],
)
def test_snapshot_bounds_revision_and_source(field, maximum):
    assert getattr(_snapshot(**{field: "a" * maximum}), field) == "a" * maximum
    with pytest.raises(ValueError, match=f"at most {maximum}"):
        _snapshot(**{field: "a" * (maximum + 1)})


class _NaiveTimezone(tzinfo):
    def utcoffset(self, dt):
        return None

    def dst(self, dt):
        return None


@pytest.mark.parametrize(
    "observed_at",
    [
        "2026-07-20T12:00:00Z",
        datetime(2026, 7, 20, 12, 0),
        datetime(2026, 7, 20, 12, 0, tzinfo=_NaiveTimezone()),
    ],
)
def test_snapshot_requires_timezone_aware_datetime(observed_at):
    with pytest.raises((TypeError, ValueError), match="observed_at"):
        _snapshot(observed_at=observed_at)


def test_static_gate_is_a_runtime_provider_with_stable_default_snapshot():
    gate = StaticMutationGate(observed_at=_NOW)

    assert isinstance(gate, MutationGate)
    assert gate.snapshot() is gate.snapshot()
    assert gate.snapshot() == MutationGateSnapshot(
        paused=False,
        revision="static-v1:open",
        source="static",
        observed_at=_NOW,
    )


def test_static_gate_accepts_an_explicit_paused_snapshot():
    gate = StaticMutationGate(
        True,
        revision="operator-change-42",
        source="deployment-control",
        observed_at=_NOW,
    )

    assert gate.snapshot().paused is True
    assert gate.snapshot().revision == "operator-change-42"


def test_static_gate_does_not_treat_an_explicit_blank_revision_as_missing():
    with pytest.raises(ValueError, match="revision must not be blank"):
        StaticMutationGate(revision="", observed_at=_NOW)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "On"])
def test_environment_gate_parses_true_values(value):
    gate = StaticMutationGate.from_environment(
        environ={DEFAULT_MUTATION_PAUSE_ENV: value},
        observed_at=_NOW,
    )

    assert gate.snapshot().paused is True
    assert gate.snapshot().revision == "static-env-v1:paused"
    assert gate.snapshot().source == f"environment:{DEFAULT_MUTATION_PAUSE_ENV}"


@pytest.mark.parametrize("value", ["0", "false", "FALSE", " no ", "Off"])
def test_environment_gate_parses_false_values(value):
    gate = StaticMutationGate.from_environment(
        environ={DEFAULT_MUTATION_PAUSE_ENV: value},
        default=True,
        observed_at=_NOW,
    )

    assert gate.snapshot().paused is False
    assert gate.snapshot().revision == "static-env-v1:open"


def test_environment_gate_uses_default_only_when_switch_is_absent():
    assert (
        StaticMutationGate.from_environment(
            environ={},
            observed_at=_NOW,
        )
        .snapshot()
        .paused
        is False
    )
    assert (
        StaticMutationGate.from_environment(
            environ={},
            default=True,
            observed_at=_NOW,
        )
        .snapshot()
        .paused
        is True
    )


def test_environment_gate_captures_configuration_once():
    environ = {DEFAULT_MUTATION_PAUSE_ENV: "false"}
    gate = StaticMutationGate.from_environment(environ=environ, observed_at=_NOW)

    environ[DEFAULT_MUTATION_PAUSE_ENV] = "true"

    assert gate.snapshot().paused is False


def test_environment_gate_rejects_invalid_value_without_echoing_it():
    secret_value = "invalid-secret-control-value"

    with pytest.raises(ValueError) as exc_info:
        StaticMutationGate.from_environment(
            environ={DEFAULT_MUTATION_PAUSE_ENV: secret_value},
            observed_at=_NOW,
        )

    assert DEFAULT_MUTATION_PAUSE_ENV in str(exc_info.value)
    assert secret_value not in str(exc_info.value)


@pytest.mark.parametrize("default", [0, 1, "false", None])
def test_environment_gate_requires_a_strict_boolean_default(default):
    with pytest.raises(TypeError, match="default must be a boolean"):
        StaticMutationGate.from_environment(
            environ={},
            default=default,
            observed_at=_NOW,
        )


@pytest.mark.parametrize("env_name", ["", "  ", " PADDED", "PADDED "])
def test_environment_gate_rejects_invalid_environment_name(env_name):
    with pytest.raises(ValueError, match="env_name"):
        StaticMutationGate.from_environment(
            env_name,
            environ={},
            observed_at=_NOW,
        )


def test_mutable_gate_toggles_live_without_consumer_reconstruction():
    gate = MutableMutationGate()

    assert safe_mutation_gate_snapshot(gate, now=_NOW).paused is False
    gate.set_paused(True)
    assert (
        safe_mutation_gate_snapshot(
            gate,
            now=_NOW + timedelta(seconds=1),
        ).paused
        is True
    )
    gate.set_paused(False)
    assert (
        safe_mutation_gate_snapshot(
            gate,
            now=_NOW + timedelta(seconds=2),
        ).paused
        is False
    )


def test_two_consumers_observe_changes_from_one_shared_gate():
    gate = MutableMutationGate()

    def api_consumer() -> bool:
        return safe_mutation_gate_snapshot(gate, now=_NOW).paused

    def health_consumer() -> bool:
        return safe_mutation_gate_snapshot(gate, now=_NOW).paused

    assert (api_consumer(), health_consumer()) == (False, False)
    gate.set_paused(True)
    assert (api_consumer(), health_consumer()) == (True, True)


def test_mutable_test_fake_satisfies_runtime_protocol():
    assert isinstance(MutableMutationGate(), MutationGate)
    assert not isinstance(object(), MutationGate)


def test_safe_snapshot_returns_a_valid_provider_snapshot_unchanged():
    expected = _snapshot()

    class ValidGate:
        def snapshot(self):
            return expected

    assert safe_mutation_gate_snapshot(ValidGate(), now=_NOW) is expected


def test_safe_snapshot_fails_closed_when_provider_raises_without_leaking_error():
    class BrokenGate:
        def snapshot(self):
            raise RuntimeError("provider-secret-detail")

    snapshot = safe_mutation_gate_snapshot(BrokenGate(), now=_NOW)

    assert snapshot == MutationGateSnapshot(
        paused=True,
        revision="fail-closed-v1",
        source="fail_closed",
        observed_at=_NOW,
    )
    assert "provider-secret-detail" not in repr(snapshot)


@pytest.mark.parametrize("value", [None, False, {}, object()])
def test_safe_snapshot_fails_closed_when_provider_returns_wrong_type(value):
    class InvalidGate:
        def snapshot(self):
            return value

    snapshot = safe_mutation_gate_snapshot(InvalidGate(), now=_NOW)

    assert snapshot.paused is True
    assert snapshot.source == "fail_closed"


def test_safe_snapshot_fails_closed_when_dynamic_observation_is_stale():
    stale = _snapshot(observed_at=_NOW - timedelta(seconds=31))

    class StaleGate:
        def snapshot(self):
            return stale

    snapshot = safe_mutation_gate_snapshot(
        StaleGate(),
        now=_NOW,
        max_age=timedelta(seconds=30),
    )

    assert snapshot.paused is True
    assert snapshot.source == "fail_closed"
    assert snapshot.observed_at == _NOW


def test_safe_snapshot_accepts_observation_at_freshness_boundary():
    fresh = _snapshot(observed_at=_NOW - timedelta(seconds=30))

    class FreshGate:
        def snapshot(self):
            return fresh

    assert (
        safe_mutation_gate_snapshot(
            FreshGate(),
            now=_NOW,
            max_age=timedelta(seconds=30),
        )
        is fresh
    )


def test_safe_snapshot_rejects_a_future_dated_observation():
    future = _snapshot(observed_at=_NOW + timedelta(seconds=6))

    class FutureGate:
        def snapshot(self):
            return future

    snapshot = safe_mutation_gate_snapshot(FutureGate(), now=_NOW)

    assert snapshot.paused is True
    assert snapshot.source == "fail_closed"


@pytest.mark.parametrize("max_age", [timedelta(0), timedelta(seconds=-1)])
def test_safe_snapshot_rejects_nonpositive_max_age(max_age):
    with pytest.raises(ValueError, match="max_age must be positive"):
        safe_mutation_gate_snapshot(
            StaticMutationGate(observed_at=_NOW),
            now=_NOW,
            max_age=max_age,
        )


def test_safe_snapshot_rejects_invalid_callers_time():
    with pytest.raises(ValueError, match="now must be timezone-aware"):
        safe_mutation_gate_snapshot(
            StaticMutationGate(observed_at=_NOW),
            now=datetime(2026, 7, 20, 12, 0),
        )


def test_safe_snapshot_does_not_swallow_process_control_exceptions():
    class InterruptingGate:
        def snapshot(self):
            raise SystemExit(2)

    with pytest.raises(SystemExit):
        safe_mutation_gate_snapshot(InterruptingGate(), now=_NOW)
