"""Pinned graph, item-bank, and pedagogy releases for resumable v2 sessions.

The active release is a deployment choice, but a durable checkpoint may name
an older release.  This registry keeps those immutable documents addressable
by their public versions and rejects silent replacement under an existing
version identifier.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tutor.content.item_bank import validate_item_bank
from tutor.learner.evidence_trust import (
    EvidenceTrustPolicy,
    ReviewedEvidenceTrustRegistry,
)
from tutor.schemas.assessment import ItemBankDocument
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy import PedagogyPackCatalog


class V2VersionUnavailable(KeyError):
    """A checkpoint names a release that this deployment cannot restore."""


class V2VersionConflict(ValueError):
    """One version identifier was reused for different immutable content."""


PolicyRestore = Callable[
    [
        GraphDocument,
        dict[str, Any],
        ItemBankDocument,
        PedagogyPackCatalog,
        EvidenceTrustPolicy,
    ],
    Any,
]


@dataclass(frozen=True)
class V2PolicyRuntime:
    """One retained executable implementation for an exact policy-version set."""

    versions: tuple[tuple[str, str], ...]
    restore: PolicyRestore


class V2PolicyRegistry:
    """Dispatch durable checkpoints to retained versioned policy code."""

    def __init__(self) -> None:
        self._runtimes: dict[tuple[tuple[str, str], ...], V2PolicyRuntime] = {}
        self._lock = threading.RLock()

    @classmethod
    def from_environment(cls) -> "V2PolicyRegistry":
        """Load operator-retained policy runtimes before current registration.

        Each configured Python module must expose
        ``register_v2_policy_runtimes(registry)``. This is deliberately a code
        registry, rather than data deserialization: old checkpoints require the
        actual reviewed implementation that authored their pinned policy set.
        """
        registry = cls()
        configured = os.environ.get("TUTOR_V2_POLICY_RUNTIME_MODULES", "")
        for module_name in (
            value.strip() for value in configured.split(",") if value.strip()
        ):
            module = import_module(module_name)
            register = getattr(module, "register_v2_policy_runtimes", None)
            if not callable(register):
                raise RuntimeError(
                    f"retained policy module {module_name!r} has no callable "
                    "register_v2_policy_runtimes"
                )
            register(registry)
        return registry

    @staticmethod
    def _key(versions: dict[str, str]) -> tuple[tuple[str, str], ...]:
        if not versions or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in versions.items()
        ):
            raise V2VersionUnavailable("checkpoint has invalid policy-version pins")
        return tuple(sorted(versions.items()))

    def register(
        self,
        versions: dict[str, str],
        restore: PolicyRestore,
    ) -> V2PolicyRuntime:
        key = self._key(versions)
        runtime = V2PolicyRuntime(versions=key, restore=restore)
        with self._lock:
            existing = self._runtimes.get(key)
            if existing is not None and existing.restore != restore:
                raise V2VersionConflict(
                    "policy-version set already identifies a different runtime"
                )
            self._runtimes[key] = existing or runtime
            return self._runtimes[key]

    def resolve_checkpoint(self, checkpoint: dict[str, Any]) -> V2PolicyRuntime:
        versions = checkpoint.get("policy_versions")
        if not isinstance(versions, dict):
            raise V2VersionUnavailable(
                "checkpoint is missing exact policy-version pins"
            )
        key = self._key(versions)
        with self._lock:
            runtime = self._runtimes.get(key)
        if runtime is None:
            raise V2VersionUnavailable(
                f"checkpoint policy runtime unavailable: {dict(key)}"
            )
        return runtime

    @property
    def version_sets(self) -> tuple[tuple[tuple[str, str], ...], ...]:
        with self._lock:
            return tuple(sorted(self._runtimes))


@dataclass(frozen=True)
class V2ContentRelease:
    """The exact compatible content triple used by one session."""

    graph: GraphDocument
    item_bank: ItemBankDocument
    pedagogy_catalog: PedagogyPackCatalog


@dataclass(frozen=True)
class _SerializedDocument:
    """Canonical immutable storage for an otherwise mutable Pydantic graph."""

    payload: str
    digest: str

    @classmethod
    def capture(cls, document: BaseModel) -> "_SerializedDocument":
        payload = json.dumps(
            document.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return cls(
            payload=payload,
            digest=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        )


class V2VersionRegistry:
    """Thread-safe registry of immutable v2 content releases.

    A component version cannot be replaced with different content, and only
    explicitly registered triples may be resolved. Independently registered
    components are never assembled as an implicit cross-product release.
    """

    def __init__(
        self,
        releases: Iterable[
            tuple[GraphDocument, ItemBankDocument, PedagogyPackCatalog]
        ] = (),
    ) -> None:
        # Store canonical bytes rather than caller-owned model references.
        # GraphDocument is mutable, and frozen content models still contain
        # mutable list values. Every returned release is reparsed as a detached
        # snapshot so post-registration mutation cannot rebind a version.
        self._graphs: dict[int, _SerializedDocument] = {}
        self._item_banks: dict[str, _SerializedDocument] = {}
        self._pedagogy_catalogs: dict[str, _SerializedDocument] = {}
        self._releases: set[tuple[int, str, str]] = set()
        self._lock = threading.RLock()
        for graph, item_bank, pedagogy_catalog in releases:
            self.register(graph, item_bank, pedagogy_catalog)

    @classmethod
    def from_release_directory(
        cls,
        directory: str | Path,
    ) -> V2VersionRegistry:
        """Load retained release triples from strict JSON bundle files.

        Each file is a schema-versioned object containing exactly one graph,
        item bank, and pedagogy catalog. Invalid, incomplete, legacy two-key,
        or conflicting history fails startup:
        silently omitting it would make an otherwise valid durable session
        impossible to resume after deployment.
        """
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(
                f"v2 release registry directory does not exist: {root}"
            )
        registry = cls()
        for path in sorted(root.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid v2 release bundle {path}") from exc
            expected_keys = {
                "schema_version",
                "graph",
                "item_bank",
                "pedagogy_catalog",
            }
            if (
                not isinstance(payload, dict)
                or set(payload) != expected_keys
                or payload.get("schema_version") != 2
            ):
                raise ValueError(
                    f"v2 release bundle {path} must be schema version 2 and "
                    "contain only schema_version, graph, item_bank, and "
                    "pedagogy_catalog"
                )
            try:
                graph = GraphDocument.model_validate(payload["graph"])
                item_bank = ItemBankDocument.model_validate(payload["item_bank"])
                pedagogy_catalog = PedagogyPackCatalog.model_validate(
                    payload["pedagogy_catalog"]
                )
                registry.register(graph, item_bank, pedagogy_catalog)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid v2 release bundle {path}") from exc
        return registry

    @classmethod
    def from_environment(cls) -> V2VersionRegistry:
        """Load deployment-retained releases when the registry path is set."""
        directory = os.environ.get("TUTOR_V2_RELEASE_REGISTRY_DIR")
        return cls.from_release_directory(directory) if directory else cls()

    def register(
        self,
        graph: GraphDocument,
        item_bank: ItemBankDocument,
        pedagogy_catalog: PedagogyPackCatalog,
    ) -> V2ContentRelease:
        """Validate and retain one exact triple without version aliasing."""
        graph_snapshot = _SerializedDocument.capture(graph)
        bank_snapshot = _SerializedDocument.capture(item_bank)
        catalog_snapshot = _SerializedDocument.capture(pedagogy_catalog)
        # Parsing the captured bytes both detaches validation from the caller
        # and proves that the exact stored representation remains schema-valid.
        registered_graph = GraphDocument.model_validate_json(
            graph_snapshot.payload
        )
        registered_bank = ItemBankDocument.model_validate_json(
            bank_snapshot.payload
        )
        registered_catalog = PedagogyPackCatalog.model_validate_json(
            catalog_snapshot.payload
        )
        if registered_bank.graph_version != registered_graph.graph_version:
            raise V2VersionConflict(
                "item-bank graph_version does not match its registered graph"
            )
        if registered_catalog.graph_version != registered_graph.graph_version:
            raise V2VersionConflict(
                "pedagogy-catalog graph_version does not match its registered graph"
            )
        validation_errors = validate_item_bank(
            registered_bank,
            registered_graph,
            registered_catalog,
        )
        if validation_errors:
            raise V2VersionConflict(
                "content release failed validation: "
                + "; ".join(validation_errors[:5])
            )
        key = (
            registered_graph.graph_version,
            registered_bank.bank_version,
            registered_catalog.catalog_version,
        )
        with self._lock:
            existing_graph = self._graphs.get(registered_graph.graph_version)
            if existing_graph is not None and existing_graph != graph_snapshot:
                raise V2VersionConflict(
                    f"graph version {registered_graph.graph_version} already identifies "
                    "different content"
                )
            existing_bank = self._item_banks.get(registered_bank.bank_version)
            if existing_bank is not None and existing_bank != bank_snapshot:
                raise V2VersionConflict(
                    f"item-bank version {registered_bank.bank_version!r} already "
                    "identifies different content"
                )
            existing_catalog = self._pedagogy_catalogs.get(
                registered_catalog.catalog_version
            )
            if (
                existing_catalog is not None
                and existing_catalog != catalog_snapshot
            ):
                raise V2VersionConflict(
                    "pedagogy-catalog version "
                    f"{registered_catalog.catalog_version!r} already identifies "
                    "different content"
                )
            self._graphs[registered_graph.graph_version] = graph_snapshot
            self._item_banks[registered_bank.bank_version] = bank_snapshot
            self._pedagogy_catalogs[
                registered_catalog.catalog_version
            ] = catalog_snapshot
            self._releases.add(key)
            return self._materialize(key)

    def _materialize(
        self,
        key: tuple[int, str, str],
    ) -> V2ContentRelease:
        """Return a detached model snapshot for one lock-protected key."""
        graph_version, bank_version, catalog_version = key
        return V2ContentRelease(
            graph=GraphDocument.model_validate_json(
                self._graphs[graph_version].payload
            ),
            item_bank=ItemBankDocument.model_validate_json(
                self._item_banks[bank_version].payload
            ),
            pedagogy_catalog=PedagogyPackCatalog.model_validate_json(
                self._pedagogy_catalogs[catalog_version].payload
            ),
        )

    def resolve(
        self,
        graph_version: int,
        item_bank_version: str,
        pedagogy_catalog_version: str,
    ) -> V2ContentRelease:
        """Resolve the exact registered triple pinned in a checkpoint."""
        key = (
            graph_version,
            item_bank_version,
            pedagogy_catalog_version,
        )
        with self._lock:
            graph_snapshot = self._graphs.get(graph_version)
            bank_snapshot = self._item_banks.get(item_bank_version)
            catalog_snapshot = self._pedagogy_catalogs.get(
                pedagogy_catalog_version
            )
        missing: list[str] = []
        if graph_snapshot is None:
            missing.append(f"graph {graph_version}")
        if bank_snapshot is None:
            missing.append(f"item bank {item_bank_version!r}")
        if catalog_snapshot is None:
            missing.append(f"pedagogy catalog {pedagogy_catalog_version!r}")
        if missing:
            raise V2VersionUnavailable(
                f"checkpoint content unavailable: {', '.join(missing)}"
            )
        if key not in self._releases:
            raise V2VersionUnavailable(
                "checkpoint content triple was never registered as a release"
            )
        with self._lock:
            return self._materialize(key)

    def resolve_checkpoint(self, checkpoint: dict[str, Any]) -> V2ContentRelease:
        """Resolve and type-check the version pins from orchestrator state."""
        graph_version = checkpoint.get("graph_version")
        item_bank_version = checkpoint.get("item_bank_version")
        pedagogy_catalog_version = checkpoint.get(
            "pedagogy_catalog_version"
        )
        if (
            not isinstance(graph_version, int)
            or isinstance(graph_version, bool)
            or not isinstance(item_bank_version, str)
            or not item_bank_version
            or not isinstance(pedagogy_catalog_version, str)
            or not pedagogy_catalog_version
            or pedagogy_catalog_version == "legacy"
        ):
            raise V2VersionUnavailable(
                "checkpoint is missing valid graph, item-bank, and "
                "pedagogy-catalog version pins"
            )
        return self.resolve(
            graph_version,
            item_bank_version,
            pedagogy_catalog_version,
        )

    @property
    def graph_versions(self) -> tuple[int, ...]:
        """Registered graph versions, exposed for health/debug metadata."""
        with self._lock:
            return tuple(sorted(self._graphs))

    @property
    def item_bank_versions(self) -> tuple[str, ...]:
        """Registered item-bank versions, exposed for health/debug metadata."""
        with self._lock:
            return tuple(sorted(self._item_banks))

    @property
    def pedagogy_catalog_versions(self) -> tuple[str, ...]:
        """Registered catalog versions, exposed for health/debug metadata."""

        with self._lock:
            return tuple(sorted(self._pedagogy_catalogs))

    @property
    def release_versions(self) -> tuple[tuple[int, str, str], ...]:
        """Exact registered triples, never an implicit component product."""

        with self._lock:
            return tuple(sorted(self._releases))

    @property
    def evidence_trust_registry(self) -> ReviewedEvidenceTrustRegistry:
        """Compile an immutable policy from every exact retained release."""

        with self._lock:
            releases = tuple(
                self._materialize(key) for key in sorted(self._releases)
            )
        return ReviewedEvidenceTrustRegistry.from_releases(
            (
                release.graph,
                release.item_bank,
                release.pedagogy_catalog,
            )
            for release in releases
        )

    @property
    def releases(self) -> tuple[V2ContentRelease, ...]:
        """Return detached snapshots of every explicitly registered triple."""

        with self._lock:
            return tuple(
                self._materialize(key)
                for key in sorted(self._releases)
            )
