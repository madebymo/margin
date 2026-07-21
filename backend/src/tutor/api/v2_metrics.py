"""Privacy-safe fleet metrics contract for the trustworthy-session runtime.

The in-process store remains the source for lightweight health snapshots in
tests and single-process development. Deployments may additionally inject a
``MetricsSink`` to export the same counters to a fleet-wide metrics backend.
Only low-cardinality release versions and reviewed stable item ids are ever
provided as dimensions; session ids, learner ids, prompts, answers, and free
text have no representation in this contract.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger("tutor.api.v2.metrics")

_METRIC_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,126}$")
_LABEL_NAME = re.compile(
    r"^(?:graph_version|item_bank_version|pedagogy_catalog_version|"
    r"learner_parameter_version|capability_manifest_version|release_digest|"
    r"item_id|policy_[a-z0-9_]+_version)$"
)


@dataclass(frozen=True)
class V2MetricDimensions:
    """The immutable release coordinates attached to every exported metric."""

    graph_version: str
    item_bank_version: str
    pedagogy_catalog_version: str
    policy_versions: tuple[tuple[str, str], ...]
    learner_parameter_version: str
    capability_manifest_version: str
    release_digest: str | None = None

    def as_labels(self) -> dict[str, str]:
        labels = {
            "graph_version": self.graph_version,
            "item_bank_version": self.item_bank_version,
            "pedagogy_catalog_version": self.pedagogy_catalog_version,
            "learner_parameter_version": self.learner_parameter_version,
            "capability_manifest_version": self.capability_manifest_version,
        }
        labels.update(
            {
                f"policy_{name}_version": version
                for name, version in self.policy_versions
            }
        )
        if self.release_digest is not None:
            labels["release_digest"] = self.release_digest
        return labels

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        fallback: V2MetricDimensions,
    ) -> V2MetricDimensions:
        """Derive pins from one episode checkpoint without learner content.

        Older runtimes may omit fields, so each unavailable coordinate falls
        back independently to the store's active release dimensions.
        """

        policies = checkpoint.get("policy_versions")
        if not isinstance(policies, Mapping) or not policies or any(
            not isinstance(name, str)
            or not isinstance(version, str)
            or not name
            or not version
            for name, version in policies.items()
        ):
            policy_versions = fallback.policy_versions
        else:
            policy_versions = tuple(sorted((str(k), str(v)) for k, v in policies.items()))
        manifest = checkpoint.get("widget_capability_manifest")
        capability_version = (
            str(manifest.get("version"))
            if isinstance(manifest, Mapping) and manifest.get("version") is not None
            else fallback.capability_manifest_version
        )
        learner_params = checkpoint.get("learner_params")
        learner_version = (
            f"bkt-v{learner_params.get('params_version')}"
            if isinstance(learner_params, Mapping)
            and learner_params.get("params_version") is not None
            else fallback.learner_parameter_version
        )
        return cls(
            graph_version=_coordinate(
                checkpoint.get("graph_version"), fallback.graph_version
            ),
            item_bank_version=_coordinate(
                checkpoint.get("item_bank_version"), fallback.item_bank_version
            ),
            pedagogy_catalog_version=_coordinate(
                checkpoint.get("pedagogy_catalog_version"),
                fallback.pedagogy_catalog_version,
            ),
            policy_versions=policy_versions,
            learner_parameter_version=learner_version,
            capability_manifest_version=capability_version,
            release_digest=_release_digest(
                checkpoint.get("release_digest"),
                None,
            ),
        )

    @classmethod
    def from_orchestrator(
        cls,
        orchestrator: Any,
        *,
        fallback: V2MetricDimensions,
    ) -> V2MetricDimensions:
        """Read already-pinned runtime coordinates in constant time."""

        graph = getattr(orchestrator, "_graph", None)
        bank = getattr(orchestrator, "_bank", None)
        catalog_version = getattr(orchestrator, "pedagogy_catalog_version", None)
        policies_provider = getattr(orchestrator, "_policy_versions", None)
        policies = policies_provider() if callable(policies_provider) else None
        manifest = getattr(orchestrator, "_pinned_widget_capabilities", None)
        learner = getattr(orchestrator, "learner", None)
        params = getattr(learner, "params", None)
        return cls.from_checkpoint(
            {
                "graph_version": getattr(graph, "graph_version", None),
                "item_bank_version": getattr(bank, "bank_version", None),
                "pedagogy_catalog_version": catalog_version,
                "policy_versions": policies,
                "widget_capability_manifest": manifest,
                "learner_params": {
                    "params_version": getattr(params, "params_version", None)
                },
                "release_digest": getattr(orchestrator, "release_digest", None),
            },
            fallback=fallback,
        )


@runtime_checkable
class MetricsSink(Protocol):
    """Injectable adapter for a fleet-wide counter backend."""

    def increment(
        self,
        name: str,
        amount: int = 1,
        *,
        dimensions: Mapping[str, str],
    ) -> None:
        """Increment one counter using only the supplied safe dimensions."""


class OpenTelemetryMetricsSink:
    """Bounded, nonblocking metrics exporter.

    ``increment`` only validates and enqueues. Instrument creation and SDK
    calls happen on a daemon worker, and the SDK's periodic reader owns network
    export. A wedged emitter can therefore fill this bounded queue and raise a
    local drop counter, but it cannot block a tutoring request.
    """

    def __init__(
        self,
        emitter: Callable[[str, int, Mapping[str, str]], None],
        *,
        max_queue_size: int = 4096,
        provider: Any | None = None,
    ) -> None:
        if not callable(emitter):
            raise TypeError("emitter must be callable")
        if type(max_queue_size) is not int or not 64 <= max_queue_size <= 65_536:
            raise ValueError("max_queue_size must be between 64 and 65536")
        self._emitter = emitter
        self._provider = provider
        self._queue: queue.Queue[tuple[str, int, dict[str, str]]] = queue.Queue(
            maxsize=max_queue_size
        )
        self._lock = threading.Lock()
        self._closed = False
        self._dropped = 0
        self._export_failures = 0
        self._worker = threading.Thread(
            target=self._run,
            name="tutor-otel-metrics",
            daemon=True,
        )
        self._worker.start()

    @classmethod
    def from_environment(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> OpenTelemetryMetricsSink:
        values = os.environ if environ is None else environ
        endpoint = values.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint is None or not endpoint.strip():
            raise ValueError("OTEL_EXPORTER_OTLP_ENDPOINT is required")
        queue_size = _bounded_environment_int(
            values,
            "TUTOR_OTEL_METRICS_QUEUE_SIZE",
            default=4096,
            minimum=64,
            maximum=65_536,
        )
        export_interval_ms = _bounded_environment_int(
            values,
            "TUTOR_OTEL_EXPORT_INTERVAL_MS",
            default=10_000,
            minimum=1_000,
            maximum=60_000,
        )
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            exporter = OTLPMetricExporter(endpoint=endpoint.strip())
            reader = PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=export_interval_ms,
            )
            provider = MeterProvider(metric_readers=[reader])
            meter = provider.get_meter("tutor.api.v2", "1")
            instruments: dict[str, Any] = {}

            def emit(name: str, amount: int, dimensions: Mapping[str, str]) -> None:
                counter = instruments.get(name)
                if counter is None:
                    counter = meter.create_counter(name)
                    instruments[name] = counter
                counter.add(amount, attributes=dict(dimensions))

            return cls(
                emit,
                max_queue_size=queue_size,
                provider=provider,
            )
        except Exception as exc:
            raise RuntimeError(
                f"OpenTelemetry metrics initialization failed ({type(exc).__name__})"
            ) from None

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def export_failure_count(self) -> int:
        with self._lock:
            return self._export_failures

    def healthy(self) -> bool:
        with self._lock:
            return (
                not self._closed
                and self._worker.is_alive()
                and not self._queue.full()
            )

    def increment(
        self,
        name: str,
        amount: int = 1,
        *,
        dimensions: Mapping[str, str],
    ) -> None:
        if not isinstance(name, str) or _METRIC_NAME.fullmatch(name) is None:
            raise ValueError("metric name is invalid")
        if type(amount) is not int or amount < 1:
            raise ValueError("metric amount must be a positive integer")
        safe_dimensions = _safe_dimensions(dimensions)
        with self._lock:
            if self._closed:
                self._note_drop_locked()
                return
        try:
            self._queue.put_nowait((name, amount, safe_dimensions))
        except queue.Full:
            with self._lock:
                self._note_drop_locked()

    def close(self, timeout_seconds: float = 1.0) -> None:
        """Bounded shutdown; an exporter hang never holds process shutdown."""

        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be nonnegative")
        with self._lock:
            self._closed = True
        self._worker.join(timeout=timeout_seconds)
        if self._worker.is_alive() or self._provider is None:
            return
        timeout_ms = max(1, int(timeout_seconds * 1000))

        def shutdown_provider() -> None:
            try:
                self._provider.shutdown(timeout_millis=timeout_ms)
            except Exception as exc:  # noqa: BLE001 - shutdown remains isolated
                logger.warning(
                    "metrics provider shutdown failed error_type=%s",
                    type(exc).__name__,
                )

        shutdown = threading.Thread(
            target=shutdown_provider,
            name="tutor-otel-shutdown",
            daemon=True,
        )
        shutdown.start()
        shutdown.join(timeout=timeout_seconds)

    def _run(self) -> None:
        while True:
            with self._lock:
                closed = self._closed
            if closed and self._queue.empty():
                return
            try:
                name, amount, dimensions = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self._emitter(name, amount, dimensions)
            except Exception as exc:  # noqa: BLE001 - telemetry is isolated
                with self._lock:
                    self._export_failures += 1
                logger.warning(
                    "queued metric export failed metric=%s error_type=%s",
                    name,
                    type(exc).__name__,
                )
            finally:
                self._queue.task_done()

    def _note_drop_locked(self) -> None:
        self._dropped += 1


def _bounded_environment_int(
    environ: Mapping[str, str],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _coordinate(value: object, fallback: str) -> str:
    if isinstance(value, (str, int)) and str(value):
        return str(value)
    return fallback


def _release_digest(value: object, fallback: str | None) -> str | None:
    if (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        return value
    return fallback


def _safe_dimensions(dimensions: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(dimensions, Mapping):
        raise TypeError("dimensions must be a mapping")
    result: dict[str, str] = {}
    for name, value in dimensions.items():
        if not isinstance(name, str) or _LABEL_NAME.fullmatch(name) is None:
            raise ValueError("metric dimension name is not permitted")
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 128
            or value != value.strip()
            or not value.isprintable()
        ):
            raise ValueError("metric dimension value is invalid")
        result[name] = value
    return result


__all__ = [
    "MetricsSink",
    "OpenTelemetryMetricsSink",
    "V2MetricDimensions",
]
