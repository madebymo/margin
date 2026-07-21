"""Pinned graph, item-bank, and pedagogy releases for resumable v2 sessions.

The active release is a deployment choice, but a durable checkpoint may name
an older release.  This registry keeps those immutable documents addressable
by their public versions and rejects silent replacement under an existing
version identifier.
"""

from __future__ import annotations

import hashlib
import hmac
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
from tutor.content.publication import (
    ReleasePublicationError,
    validate_release_reviews,
)
from tutor.content.release_identity import (
    canonical_bundle_sha256,
    fixture_release_id,
)
from tutor.content.review_artifacts import canonical_digest, canonical_json_bytes
from tutor.learner.evidence_trust import (
    EvidenceTrustPolicy,
    ReviewedEvidenceTrustRegistry,
)
from tutor.schemas.assessment import ItemBankDocument
from tutor.schemas.kc import GraphDocument
from tutor.schemas.pedagogy import PedagogyPackCatalog
from tutor.schemas.release_authoring import (
    PublishedReleaseManifest,
    ReleasePublicationMetadata,
    ReleaseReviewManifest,
)


V2_ACTIVE_RELEASE_BUNDLE_ENV = "TUTOR_V2_ACTIVE_RELEASE_BUNDLE"
V2_ACTIVE_RELEASE_SHA256_ENV = "TUTOR_V2_ACTIVE_RELEASE_SHA256"


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
    checkpoint_schema_versions: tuple[int, ...]


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
        Retained registrations must also declare ``checkpoint_schema_versions``
        so deployment readiness can prove the adapter accepts every live token's
        serialized state before traffic is admitted.
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
        *,
        checkpoint_schema_versions: Iterable[int] | None = None,
    ) -> V2PolicyRuntime:
        key = self._key(versions)
        declared_schemas = (
            checkpoint_schema_versions
            if checkpoint_schema_versions is not None
            else getattr(restore, "supported_checkpoint_schema_versions", ())
        )
        try:
            schemas = tuple(sorted(set(declared_schemas)))
        except TypeError as exc:
            raise ValueError(
                "checkpoint schema versions must be an iterable of integers"
            ) from exc
        if any(
            type(schema_version) is not int or schema_version < 1
            for schema_version in schemas
        ):
            raise ValueError(
                "checkpoint schema versions must be positive integers"
            )
        runtime = V2PolicyRuntime(
            versions=key,
            restore=restore,
            checkpoint_schema_versions=schemas,
        )
        with self._lock:
            existing = self._runtimes.get(key)
            if existing is not None and existing != runtime:
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

    def resolve_restoration_checkpoint(
        self,
        checkpoint: dict[str, Any],
    ) -> V2PolicyRuntime:
        """Resolve an executable adapter that declares this checkpoint schema."""

        runtime = self.resolve_checkpoint(checkpoint)
        schema_version = checkpoint.get("schema_version")
        if (
            type(schema_version) is not int
            or schema_version not in runtime.checkpoint_schema_versions
        ):
            raise V2VersionUnavailable(
                "checkpoint schema has no retained policy restoration adapter"
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
    release_id: str
    release_digest: str
    published: bool


@dataclass(frozen=True)
class _ReleaseIdentity:
    release_id: str
    release_digest: str
    published: bool


@dataclass(frozen=True)
class _LoadedBundle:
    graph: GraphDocument
    item_bank: ItemBankDocument
    pedagogy_catalog: PedagogyPackCatalog
    identity: _ReleaseIdentity


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
        self._release_identities: dict[
            tuple[int, str, str], _ReleaseIdentity
        ] = {}
        self._release_ids: dict[str, tuple[int, str, str]] = {}
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
        bundle_paths = tuple(sorted(root.rglob("bundle.json")))
        release_directories = {path.parent for path in bundle_paths}
        marker_names = {
            "release-manifest.json",
            "release-reviews.json",
            "bundle.sha256",
        }
        partial_directories = {
            path.parent
            for path in root.rglob("*")
            if path.is_file() and path.name in marker_names
        } - release_directories
        if partial_directories:
            raise ValueError(
                "v2 release registry contains an incomplete publication directory"
            )
        standalone_bundles = tuple(
            path
            for path in sorted(root.rglob("*.json"))
            if path.name not in marker_names
            and path.name != "bundle.json"
            and not any(
                directory in path.parents
                for directory in release_directories
            )
        )
        if not bundle_paths and not standalone_bundles:
            raise ValueError("v2 release registry contains no release bundles")

        registry = cls()
        for path in bundle_paths:
            registry.register_bundle(path.parent)
        for path in standalone_bundles:
            registry.register_bundle(path)
        return registry

    @classmethod
    def from_environment(cls) -> V2VersionRegistry:
        """Load deployment-retained releases when the registry path is set."""
        directory = os.environ.get("TUTOR_V2_RELEASE_REGISTRY_DIR")
        return cls.from_release_directory(directory) if directory else cls()

    @staticmethod
    def _load_bundle(
        source: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> _LoadedBundle:
        """Parse one exact schema-v2 release file without logging its payload."""

        source_path = Path(source).expanduser()
        path = source_path / "bundle.json" if source_path.is_dir() else source_path
        manifest_path = path.parent / "release-manifest.json"
        sidecar_path = path.parent / "bundle.sha256"
        reviews_path = path.parent / "release-reviews.json"
        try:
            raw_bundle = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"invalid v2 release bundle {path}") from exc
        if expected_sha256 is not None:
            if len(expected_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in expected_sha256
            ):
                raise ValueError("invalid active v2 release SHA-256 pin")
            actual_sha256 = hashlib.sha256(raw_bundle).hexdigest()
            if not hmac.compare_digest(actual_sha256, expected_sha256):
                # Do not expose the path, configured digest, actual digest, or
                # any payload fragment through an exception that startup
                # infrastructure might record.
                raise V2VersionConflict("active v2 release SHA-256 mismatch")
        try:
            payload = json.loads(raw_bundle)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError(f"invalid v2 release bundle {path}") from None
        expected_keys = {
            "schema_version",
            "graph",
            "item_bank",
            "pedagogy_catalog",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected_keys
            or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 2
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
        except (ValueError, TypeError):
            # Pydantic's rich ValidationError includes rejected input values.
            # Suppress the nested exception so startup traceback capture cannot
            # disclose prompts, answers, or other bundle content.
            raise ValueError(f"invalid v2 release bundle {path}") from None

        publication_files = (
            manifest_path.is_file(),
            sidecar_path.is_file(),
            reviews_path.is_file(),
        )
        if source_path.is_dir() or any(publication_files):
            if not all(publication_files):
                raise ValueError(
                    f"incomplete published v2 release beside {path}"
                )
            try:
                raw_manifest = manifest_path.read_bytes()
                manifest = PublishedReleaseManifest.model_validate_json(
                    raw_manifest
                )
                sidecar = sidecar_path.read_bytes()
                raw_reviews = reviews_path.read_bytes()
                reviews = ReleaseReviewManifest.model_validate_json(raw_reviews)
            except (OSError, ValueError, TypeError):
                raise ValueError(
                    f"incomplete or invalid published v2 release beside {path}"
                ) from None
            actual_sha256 = hashlib.sha256(raw_bundle).hexdigest()
            expected_sidecar = f"{actual_sha256}  bundle.json\n".encode("ascii")
            reviews_sha256 = hashlib.sha256(raw_reviews).hexdigest()
            coordinates_match = (
                manifest.graph_version == graph.graph_version
                and manifest.bank_version == item_bank.bank_version
                and manifest.catalog_version == pedagogy_catalog.catalog_version
            )
            component_digests_match = (
                manifest.graph_digest == canonical_digest(graph)
                and manifest.bank_digest == canonical_digest(item_bank)
                and manifest.catalog_digest == canonical_digest(pedagogy_catalog)
            )
            if (
                path.name != manifest.bundle_file
                or not hmac.compare_digest(
                    raw_manifest,
                    canonical_json_bytes(manifest, trailing_newline=True),
                )
                or not hmac.compare_digest(sidecar, expected_sidecar)
                or not hmac.compare_digest(manifest.bundle_sha256, actual_sha256)
                or not hmac.compare_digest(
                    manifest.reviews_sha256,
                    reviews_sha256,
                )
                or not coordinates_match
                or not component_digests_match
                or manifest.released_kcs != tuple(sorted(item_bank.released_kcs))
            ):
                raise V2VersionConflict(
                    "published v2 release manifest does not bind the exact bundle"
                )
            try:
                candidate, expected_manifest = validate_release_reviews(
                    graph,
                    item_bank,
                    pedagogy_catalog,
                    reviews,
                    ReleasePublicationMetadata(
                        published_by=manifest.published_by,
                        published_at=manifest.published_at,
                    ),
                )
            except (ReleasePublicationError, ValueError, TypeError):
                raise V2VersionConflict(
                    "published v2 release lacks valid exact attestations"
                ) from None
            if (
                not hmac.compare_digest(candidate.bundle_bytes, raw_bundle)
                or expected_manifest != manifest
            ):
                raise V2VersionConflict(
                    "published v2 release receipt does not match exact artifacts"
                )
            identity = _ReleaseIdentity(
                release_id=manifest.release_id,
                release_digest=manifest.bundle_sha256,
                published=True,
            )
        else:
            fixture_sha256 = canonical_bundle_sha256(
                graph,
                item_bank,
                pedagogy_catalog,
            )
            identity = _ReleaseIdentity(
                release_id=fixture_release_id(fixture_sha256),
                release_digest=fixture_sha256,
                published=False,
            )
        return _LoadedBundle(
            graph=graph,
            item_bank=item_bank,
            pedagogy_catalog=pedagogy_catalog,
            identity=identity,
        )

    def register_bundle(
        self,
        source: str | Path,
        *,
        require_released_content: bool = False,
        expected_sha256: str | None = None,
    ) -> V2ContentRelease:
        """Parse, validate, and register one explicitly declared release.

        Retained history may include a release with no currently admitted KC,
        but an operator-selected active bundle must contain at least one
        explicit release declaration. This prevents a draft-only bank from
        being treated as a successful deployment merely because its schemas
        parse.
        """

        loaded = self._load_bundle(
            source,
            expected_sha256=expected_sha256,
        )
        graph = loaded.graph
        item_bank = loaded.item_bank
        pedagogy_catalog = loaded.pedagogy_catalog
        if require_released_content and not item_bank.released_kcs:
            raise V2VersionConflict(
                "active v2 release bundle contains no released knowledge components"
            )
        if require_released_content:
            key = (
                graph.graph_version,
                item_bank.bank_version,
                pedagogy_catalog.catalog_version,
            )
            with self._lock:
                exact_release_exists = key in self._releases
                reuses_registered_component = (
                    item_bank.bank_version in self._item_banks
                    or pedagogy_catalog.catalog_version
                    in self._pedagogy_catalogs
                )
            if reuses_registered_component and not exact_release_exists:
                raise V2VersionConflict(
                    "active v2 release bundle is an unregistered component cross-product"
                )
        return self.register(
            graph,
            item_bank,
            pedagogy_catalog,
            _identity=loaded.identity,
        )

    def register(
        self,
        graph: GraphDocument,
        item_bank: ItemBankDocument,
        pedagogy_catalog: PedagogyPackCatalog,
        *,
        _identity: _ReleaseIdentity | None = None,
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
        if _identity is None:
            bundle_sha256 = canonical_bundle_sha256(
                registered_graph,
                registered_bank,
                registered_catalog,
            )
            identity = _ReleaseIdentity(
                release_id=fixture_release_id(bundle_sha256),
                release_digest=bundle_sha256,
                published=False,
            )
        else:
            identity = _identity
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
            existing_identity = self._release_identities.get(key)
            if existing_identity is not None and existing_identity != identity:
                raise V2VersionConflict(
                    "content release coordinates already identify another release"
                )
            existing_release_key = self._release_ids.get(identity.release_id)
            if existing_release_key is not None and existing_release_key != key:
                raise V2VersionConflict(
                    "release_id already identifies different content coordinates"
                )
            self._graphs[registered_graph.graph_version] = graph_snapshot
            self._item_banks[registered_bank.bank_version] = bank_snapshot
            self._pedagogy_catalogs[
                registered_catalog.catalog_version
            ] = catalog_snapshot
            self._releases.add(key)
            self._release_identities[key] = identity
            self._release_ids[identity.release_id] = key
            return self._materialize(key)

    def _materialize(
        self,
        key: tuple[int, str, str],
    ) -> V2ContentRelease:
        """Return a detached model snapshot for one lock-protected key."""
        graph_version, bank_version, catalog_version = key
        identity = self._release_identities[key]
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
            release_id=identity.release_id,
            release_digest=identity.release_digest,
            published=identity.published,
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
        release = self.resolve(
            graph_version,
            item_bank_version,
            pedagogy_catalog_version,
        )
        release_id = checkpoint.get("release_id")
        if release_id is not None and release_id != release.release_id:
            raise V2VersionUnavailable(
                "checkpoint release_id does not match retained release coordinates"
            )
        release_digest = checkpoint.get("release_digest")
        if (
            release_digest is not None
            and release_digest != release.release_digest
        ):
            raise V2VersionUnavailable(
                "checkpoint release_digest does not match retained release coordinates"
            )
        return release

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
