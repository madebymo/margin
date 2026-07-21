"""Fail-closed trust policy for mastery-bearing v2 evidence.

An evidence row is not reviewed merely because it names a non-legacy catalog.
Trust is derived from exact immutable graph, item-bank, and pedagogy-catalog
documents and requires the event to match the reviewed release on every field
that identifies its mathematical content.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from tutor.content.item_bank import validate_item_bank
from tutor.schemas.assessment import (
    AssessmentSurface,
    ItemBankDocument,
)
from tutor.schemas.common import ResponseClass, ReviewStatus
from tutor.schemas.kc import GraphDocument
from tutor.schemas.learner import EvidenceEvent
from tutor.schemas.pedagogy import PedagogyPackCatalog

ContentReleaseDocuments = tuple[
    GraphDocument,
    ItemBankDocument,
    PedagogyPackCatalog,
]
ContentReleaseVersion = tuple[int, str, str]
EVIDENCE_TRUST_POLICY_VERSION = "evidence-trust-v1"


@runtime_checkable
class EvidenceTrustPolicy(Protocol):
    """Decision seam used by the learner model for every evidence event."""

    def trusts(self, event: EvidenceEvent) -> bool:
        """Return whether ``event`` may receive full reviewed-evidence weight."""


@dataclass(frozen=True, slots=True)
class DenyAllEvidenceTrustPolicy:
    """Default learner policy: no ambient string or version claim is trusted."""

    def trusts(self, event: EvidenceEvent) -> bool:
        del event
        return False


DENY_ALL_EVIDENCE = DenyAllEvidenceTrustPolicy()


@dataclass(frozen=True, slots=True)
class _TrustedItem:
    graph_version: int
    item_bank_version: str
    pedagogy_catalog_version: str
    pedagogy_pack_version: int
    item_id: str
    item_revision: int
    family_id: str
    kc_id: str
    surfaces: frozenset[str]
    content_provenance: str
    response_class: ResponseClass
    misconception_ids: frozenset[str]

    @property
    def release_version(self) -> ContentReleaseVersion:
        return (
            self.graph_version,
            self.item_bank_version,
            self.pedagogy_catalog_version,
        )

    def matches(self, event: EvidenceEvent) -> bool:
        expected_item_id = (
            f"lesson-transition.{self.item_id}"
            if event.surface == "instructional_practice"
            else self.item_id
        )
        expected_versions = {
            "graph": str(self.graph_version),
            "item_bank": self.item_bank_version,
            "pedagogy_catalog": self.pedagogy_catalog_version,
            "pedagogy_pack": str(self.pedagogy_pack_version),
        }
        if (
            event.content_versions != expected_versions
            or event.pedagogy_catalog_version
            != self.pedagogy_catalog_version
            or event.item_id != expected_item_id
            or event.item_revision != self.item_revision
            or event.family_id != self.family_id
            or event.kc_ids != [self.kc_id]
            or event.surface not in self.surfaces
            or event.content_provenance != self.content_provenance
        ):
            return False

        if event.surface == "instructional_practice":
            return (
                event.response_class == ResponseClass.WIDGET
                and event.correct
                and not event.assisted
                and event.hints_used == 0
                and event.misconception_id is None
                and event.learning_opportunity
            )
        if event.learning_opportunity or event.response_class != self.response_class:
            return False
        if event.misconception_id is None:
            return True
        return (
            not event.correct
            and event.misconception_id in self.misconception_ids
        )


@dataclass(frozen=True, slots=True)
class ReviewedEvidenceTrustRegistry:
    """Immutable allow-list compiled from exact reviewed content releases."""

    _items: tuple[_TrustedItem, ...] = ()
    _release_versions: tuple[ContentReleaseVersion, ...] = ()

    @classmethod
    def from_release(
        cls,
        graph: GraphDocument,
        item_bank: ItemBankDocument,
        pedagogy_catalog: PedagogyPackCatalog,
    ) -> "ReviewedEvidenceTrustRegistry":
        """Compile one exact content release into a trust registry."""

        return cls.from_releases(((graph, item_bank, pedagogy_catalog),))

    @classmethod
    def from_releases(
        cls,
        releases: Iterable[ContentReleaseDocuments],
    ) -> "ReviewedEvidenceTrustRegistry":
        """Compile multiple retained exact releases without version aliasing."""

        graphs: dict[int, GraphDocument] = {}
        item_banks: dict[str, ItemBankDocument] = {}
        catalogs: dict[str, PedagogyPackCatalog] = {}
        release_documents: dict[
            ContentReleaseVersion,
            ContentReleaseDocuments,
        ] = {}
        trusted_items: set[_TrustedItem] = set()

        for graph, item_bank, pedagogy_catalog in releases:
            cls._bind_component(
                graphs,
                graph.graph_version,
                graph,
                "graph",
            )
            cls._bind_component(
                item_banks,
                item_bank.bank_version,
                item_bank,
                "item-bank",
            )
            cls._bind_component(
                catalogs,
                pedagogy_catalog.catalog_version,
                pedagogy_catalog,
                "pedagogy-catalog",
            )
            release_version = (
                graph.graph_version,
                item_bank.bank_version,
                pedagogy_catalog.catalog_version,
            )
            documents = (graph, item_bank, pedagogy_catalog)
            previous = release_documents.setdefault(release_version, documents)
            if previous != documents:
                raise ValueError(
                    "content release version tuple identifies different documents"
                )

            errors = validate_item_bank(item_bank, graph, pedagogy_catalog)
            if errors:
                preview = "; ".join(errors[:5])
                suffix = f"; and {len(errors) - 5} more" if len(errors) > 5 else ""
                raise ValueError(
                    f"cannot trust invalid content release: {preview}{suffix}"
                )

            packs = pedagogy_catalog.pack_by_kc
            released_kcs = set(item_bank.released_kcs)
            for item in item_bank.items:
                if (
                    item.kc_id not in released_kcs
                    or item.review_status != ReviewStatus.HUMAN_APPROVED
                ):
                    continue
                pack = packs[item.kc_id]
                surfaces = {
                    surface.value
                    for surface in item.eligible_surfaces
                    if surface != AssessmentSurface.WORKED_EXAMPLE
                }
                if AssessmentSurface.GUIDED_WIDGET in item.eligible_surfaces:
                    surfaces.add("instructional_practice")
                if not surfaces:
                    continue
                trusted_items.add(
                    _TrustedItem(
                        graph_version=graph.graph_version,
                        item_bank_version=item_bank.bank_version,
                        pedagogy_catalog_version=(
                            pedagogy_catalog.catalog_version
                        ),
                        pedagogy_pack_version=pack.version,
                        item_id=item.item_id,
                        item_revision=item.revision,
                        family_id=item.family_id,
                        kc_id=item.kc_id,
                        surfaces=frozenset(surfaces),
                        content_provenance=item.provenance.source,
                        response_class=(
                            ResponseClass.MULTIPLE_CHOICE
                            if item.answer.kind == "choice"
                            else ResponseClass.SYMBOLIC_ENTRY
                        ),
                        misconception_ids=frozenset(
                            signature.misconception_id
                            for signature in item.error_signatures
                            if signature.misconception_id is not None
                        ),
                    )
                )

        ordered_items = tuple(
            sorted(
                trusted_items,
                key=lambda item: (
                    item.release_version,
                    item.item_id,
                    item.item_revision,
                ),
            )
        )
        return cls(
            _items=ordered_items,
            _release_versions=tuple(sorted(release_documents)),
        )

    @staticmethod
    def _bind_component(
        registry: dict[object, object],
        version: object,
        document: object,
        label: str,
    ) -> None:
        previous = registry.setdefault(version, document)
        if previous != document:
            raise ValueError(
                f"{label} version {version!r} identifies different documents"
            )

    def trusts(self, event: EvidenceEvent) -> bool:
        """Require one exact reviewed item claim; unknown claims fail closed."""

        return any(item.matches(event) for item in self._items)

    @property
    def release_versions(self) -> tuple[ContentReleaseVersion, ...]:
        """Exact content triples admitted by this immutable policy."""

        return self._release_versions
