"""Pinned graph and item-bank releases available to resumable v2 sessions.

The active release is a deployment choice, but a durable checkpoint may name
an older release.  This registry keeps those immutable documents addressable
by their public versions and rejects silent replacement under an existing
version identifier.
"""

from __future__ import annotations

import json
import os
import threading
from importlib import import_module
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tutor.schemas.assessment import ItemBankDocument
from tutor.schemas.kc import GraphDocument


class V2VersionUnavailable(KeyError):
    """A checkpoint names a release that this deployment cannot restore."""


class V2VersionConflict(ValueError):
    """One version identifier was reused for different immutable content."""


PolicyRestore = Callable[
    [GraphDocument, dict[str, Any], ItemBankDocument],
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
    """The exact compatible graph/item-bank pair used by one session."""

    graph: GraphDocument
    item_bank: ItemBankDocument


class V2VersionRegistry:
    """Thread-safe registry of immutable v2 content releases.

    Multiple item-bank versions may be retained for one graph version.  A
    graph or bank document cannot be replaced with different content while
    keeping the same version identifier.
    """

    def __init__(
        self,
        releases: Iterable[tuple[GraphDocument, ItemBankDocument]] = (),
    ) -> None:
        self._graphs: dict[int, GraphDocument] = {}
        self._item_banks: dict[str, ItemBankDocument] = {}
        self._lock = threading.RLock()
        for graph, item_bank in releases:
            self.register(graph, item_bank)

    @classmethod
    def from_release_directory(
        cls,
        directory: str | Path,
    ) -> V2VersionRegistry:
        """Load retained release pairs from strict JSON bundle files.

        Each ``*.json`` file must contain exactly ``graph`` and ``item_bank``
        objects.  Invalid, incomplete, or conflicting history fails startup:
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
            if not isinstance(payload, dict) or set(payload) != {
                "graph",
                "item_bank",
            }:
                raise ValueError(
                    f"v2 release bundle {path} must contain only graph and item_bank"
                )
            try:
                graph = GraphDocument.model_validate(payload["graph"])
                item_bank = ItemBankDocument.model_validate(payload["item_bank"])
                registry.register(graph, item_bank)
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
    ) -> V2ContentRelease:
        """Retain one compatible pair without permitting version aliasing."""
        if item_bank.graph_version != graph.graph_version:
            raise V2VersionConflict(
                "item-bank graph_version does not match its registered graph"
            )
        with self._lock:
            existing_graph = self._graphs.get(graph.graph_version)
            if existing_graph is not None and existing_graph != graph:
                raise V2VersionConflict(
                    f"graph version {graph.graph_version} already identifies "
                    "different content"
                )
            existing_bank = self._item_banks.get(item_bank.bank_version)
            if existing_bank is not None and existing_bank != item_bank:
                raise V2VersionConflict(
                    f"item-bank version {item_bank.bank_version!r} already "
                    "identifies different content"
                )
            self._graphs[graph.graph_version] = graph
            self._item_banks[item_bank.bank_version] = item_bank
        return V2ContentRelease(graph=graph, item_bank=item_bank)

    def resolve(
        self,
        graph_version: int,
        item_bank_version: str,
    ) -> V2ContentRelease:
        """Resolve the exact compatible pair pinned in a checkpoint."""
        with self._lock:
            graph = self._graphs.get(graph_version)
            item_bank = self._item_banks.get(item_bank_version)
        missing: list[str] = []
        if graph is None:
            missing.append(f"graph {graph_version}")
        if item_bank is None:
            missing.append(f"item bank {item_bank_version!r}")
        if missing:
            raise V2VersionUnavailable(
                f"checkpoint content unavailable: {', '.join(missing)}"
            )
        assert graph is not None
        assert item_bank is not None
        if item_bank.graph_version != graph.graph_version:
            raise V2VersionUnavailable(
                "checkpoint graph and item-bank versions are incompatible"
            )
        return V2ContentRelease(graph=graph, item_bank=item_bank)

    def resolve_checkpoint(self, checkpoint: dict[str, Any]) -> V2ContentRelease:
        """Resolve and type-check the version pins from orchestrator state."""
        graph_version = checkpoint.get("graph_version")
        item_bank_version = checkpoint.get("item_bank_version")
        if (
            not isinstance(graph_version, int)
            or isinstance(graph_version, bool)
            or not isinstance(item_bank_version, str)
            or not item_bank_version
        ):
            raise V2VersionUnavailable(
                "checkpoint is missing valid graph and item-bank version pins"
            )
        return self.resolve(graph_version, item_bank_version)

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
