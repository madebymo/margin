"""Human attestations and immutable publication metadata for content releases."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tutor.schemas.kc import KC_ID_PATTERN

_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("review and publication timestamps must include a timezone")
    return value


class FamilyApprovalAttestation(_StrictFrozenModel):
    """Independent approval of one exact source and compiled item family."""

    attestation_id: str = Field(max_length=128, pattern=_ID_PATTERN)
    family_id: str = Field(max_length=128, pattern=_ID_PATTERN)
    source_id: str = Field(max_length=128, pattern=_ID_PATTERN)
    source_revision: int = Field(ge=1)
    source_digest: str = Field(pattern=_SHA256_PATTERN)
    compiled_artifact_digest: str = Field(pattern=_SHA256_PATTERN)
    compiler_version: str = Field(max_length=128, pattern=_ID_PATTERN)
    graph_version: int = Field(ge=1)
    author: str = Field(min_length=1, max_length=256)
    reviewed_by: str = Field(min_length=1, max_length=256)
    reviewed_at: datetime
    mathematical_correctness: Literal[True]
    accessibility: Literal[True]
    instructional_clarity: Literal[True]

    @field_validator("author", "reviewed_by", mode="before")
    @classmethod
    def _normalize_people(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("reviewed_at")
    @classmethod
    def _aware_review(cls, value: datetime) -> datetime:
        return _aware(value)

    @model_validator(mode="after")
    def _independent_review(self) -> "FamilyApprovalAttestation":
        if self.author.casefold() == self.reviewed_by.casefold():
            raise ValueError("a family author cannot approve its own compiled artifact")
        return self


class KCApprovalAttestation(_StrictFrozenModel):
    """Human judgment that one KC's approved families form independent evidence."""

    attestation_id: str = Field(max_length=128, pattern=_ID_PATTERN)
    kc_id: str = Field(pattern=KC_ID_PATTERN)
    family_ids: tuple[str, ...] = Field(min_length=1)
    family_attestation_digest: str = Field(pattern=_SHA256_PATTERN)
    # These fields were added after schema-v1 manifests had already been
    # defined.  They remain optional on the individual record so retained v1
    # releases can still be parsed; ReleaseReviewManifest requires them for a
    # schema-v2 review manifest.
    mastery_claim: str | None = Field(default=None, min_length=1, max_length=1024)
    construct_ids: tuple[str, ...] = ()
    prepared_by: str = Field(min_length=1, max_length=256)
    reviewed_by: str = Field(min_length=1, max_length=256)
    reviewed_at: datetime
    construct_coverage: Literal[True]
    family_independence: Literal[True]
    difficulty_progression: Literal[True]
    first_two_paths_reviewed: Literal[True]

    @field_validator("mastery_claim", "prepared_by", "reviewed_by", mode="before")
    @classmethod
    def _normalize_people(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("reviewed_at")
    @classmethod
    def _aware_review(cls, value: datetime) -> datetime:
        return _aware(value)

    @model_validator(mode="after")
    def _coherent_attestation(self) -> "KCApprovalAttestation":
        if len(self.family_ids) != len(set(self.family_ids)):
            raise ValueError("KC attestation family_ids must be unique")
        if len(self.construct_ids) != len(set(self.construct_ids)):
            raise ValueError("KC attestation construct_ids must be unique")
        if any(not construct_id.strip() for construct_id in self.construct_ids):
            raise ValueError("KC attestation construct_ids must be nonblank")
        if self.prepared_by.casefold() == self.reviewed_by.casefold():
            raise ValueError("a KC attestation requires an independent reviewer")
        return self


class ReleaseApprovalAttestation(_StrictFrozenModel):
    """Final human approval bound to the exact candidate release bytes."""

    attestation_id: str = Field(max_length=128, pattern=_ID_PATTERN)
    release_id: str = Field(max_length=128, pattern=_ID_PATTERN)
    graph_version: int = Field(ge=1)
    graph_digest: str = Field(pattern=_SHA256_PATTERN)
    bank_version: str = Field(max_length=128, pattern=_ID_PATTERN)
    bank_digest: str = Field(pattern=_SHA256_PATTERN)
    catalog_version: str = Field(max_length=128, pattern=_ID_PATTERN)
    catalog_digest: str = Field(pattern=_SHA256_PATTERN)
    released_kcs: tuple[str, ...] = Field(min_length=1)
    kc_attestation_digest: str = Field(pattern=_SHA256_PATTERN)
    bundle_sha256: str = Field(pattern=_SHA256_PATTERN)
    prepared_by: str = Field(min_length=1, max_length=256)
    reviewed_by: str = Field(min_length=1, max_length=256)
    reviewed_at: datetime
    cross_component_compatibility: Literal[True]
    complete_hard_closure: Literal[True]
    exact_bytes_reviewed: Literal[True]

    @field_validator("prepared_by", "reviewed_by", mode="before")
    @classmethod
    def _normalize_people(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("release_id")
    @classmethod
    def _nonproduction_namespace_is_reserved(cls, value: str) -> str:
        if value == "nonproduction.legacy-unpinned" or value.startswith(
            "nonproduction.fixture."
        ):
            raise ValueError(
                "release_id uses a reserved non-production namespace"
            )
        return value

    @field_validator("reviewed_at")
    @classmethod
    def _aware_review(cls, value: datetime) -> datetime:
        return _aware(value)

    @model_validator(mode="after")
    def _coherent_attestation(self) -> "ReleaseApprovalAttestation":
        if len(self.released_kcs) != len(set(self.released_kcs)):
            raise ValueError("release attestation released_kcs must be unique")
        if self.prepared_by.casefold() == self.reviewed_by.casefold():
            raise ValueError("a release attestation requires an independent reviewer")
        return self


class ReleaseReviewManifest(_StrictFrozenModel):
    """Exact family, KC, and release approvals consumed by publication."""

    schema_version: Literal[1, 2] = 1
    family_attestations: tuple[FamilyApprovalAttestation, ...] = Field(min_length=1)
    kc_attestations: tuple[KCApprovalAttestation, ...] = Field(min_length=1)
    release_attestation: ReleaseApprovalAttestation

    @model_validator(mode="after")
    def _unique_attestations(self) -> "ReleaseReviewManifest":
        ids = [
            *(item.attestation_id for item in self.family_attestations),
            *(item.attestation_id for item in self.kc_attestations),
            self.release_attestation.attestation_id,
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("attestation ids must be globally unique")
        family_ids = [item.family_id for item in self.family_attestations]
        if len(family_ids) != len(set(family_ids)):
            raise ValueError("a review manifest may attest each family only once")
        kc_ids = [item.kc_id for item in self.kc_attestations]
        if len(kc_ids) != len(set(kc_ids)):
            raise ValueError("a review manifest may attest each KC only once")
        if self.schema_version >= 2:
            incomplete = [
                item.kc_id
                for item in self.kc_attestations
                if item.mastery_claim is None or not item.construct_ids
            ]
            if incomplete:
                raise ValueError(
                    "schema-v2 KC attestations require a mastery claim and "
                    f"constructor coverage: {sorted(incomplete)}"
                )
        return self


class AttestationReviewDecision(StrEnum):
    """Human disposition for a reviewer-fillable attestation scaffold."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class FamilyApprovalAttestationInput(_StrictFrozenModel):
    """Exact family binding awaiting explicit final review judgments.

    ``reviewed_by`` and ``reviewed_at`` are copied from the already-completed
    source review because publication requires the final family attestation to
    bind that same review event.  A scaffold is nevertheless pending until the
    three explicit judgments below are filled.
    """

    attestation_id: str = Field(max_length=128, pattern=_ID_PATTERN)
    family_id: str = Field(max_length=128, pattern=_ID_PATTERN)
    source_id: str = Field(max_length=128, pattern=_ID_PATTERN)
    source_revision: int = Field(ge=1)
    source_digest: str = Field(pattern=_SHA256_PATTERN)
    compiled_artifact_digest: str = Field(pattern=_SHA256_PATTERN)
    compiler_version: str = Field(max_length=128, pattern=_ID_PATTERN)
    graph_version: int = Field(ge=1)
    author: str = Field(min_length=1, max_length=256)
    reviewed_by: str = Field(min_length=1, max_length=256)
    reviewed_at: datetime
    decision: AttestationReviewDecision = AttestationReviewDecision.PENDING
    mathematical_correctness: bool | None = None
    accessibility: bool | None = None
    instructional_clarity: bool | None = None
    notes: str | None = Field(default=None, min_length=1)

    @field_validator("author", "reviewed_by", mode="before")
    @classmethod
    def _normalize_people(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("reviewed_at")
    @classmethod
    def _aware_review(cls, value: datetime) -> datetime:
        return _aware(value)

    @model_validator(mode="after")
    def _decision_is_truthful(self) -> "FamilyApprovalAttestationInput":
        judgments = (
            self.mathematical_correctness,
            self.accessibility,
            self.instructional_clarity,
        )
        if self.author.casefold() == self.reviewed_by.casefold():
            raise ValueError("a family author cannot review its own compiled artifact")
        if self.decision == AttestationReviewDecision.PENDING:
            if any(value is not None for value in judgments):
                raise ValueError("pending family input cannot claim review judgments")
        elif self.decision == AttestationReviewDecision.APPROVED:
            if any(value is not True for value in judgments):
                raise ValueError("approved family input requires every judgment to be true")
        elif not any(value is False for value in judgments):
            raise ValueError("rejected family input requires at least one failed judgment")
        return self


class KCApprovalAttestationInput(_StrictFrozenModel):
    """Exact KC coverage record awaiting independent human attestation."""

    attestation_id: str = Field(max_length=128, pattern=_ID_PATTERN)
    kc_id: str = Field(pattern=KC_ID_PATTERN)
    family_ids: tuple[str, ...] = Field(min_length=1)
    # Computed by the finalizer only after every bound family review is
    # explicitly approved. Reviewers must not guess this dependent digest.
    family_attestation_digest: Literal[None] = None
    mastery_claim: str = Field(min_length=1, max_length=1024)
    construct_ids: tuple[str, ...] = Field(min_length=1)
    decision: AttestationReviewDecision = AttestationReviewDecision.PENDING
    prepared_by: str | None = Field(default=None, min_length=1, max_length=256)
    reviewed_by: str | None = Field(default=None, min_length=1, max_length=256)
    reviewed_at: datetime | None = None
    construct_coverage: bool | None = None
    family_independence: bool | None = None
    difficulty_progression: bool | None = None
    first_two_paths_reviewed: bool | None = None
    notes: str | None = Field(default=None, min_length=1)

    @field_validator("mastery_claim", "prepared_by", "reviewed_by", mode="before")
    @classmethod
    def _normalize_people_and_claim(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("reviewed_at")
    @classmethod
    def _aware_review(cls, value: datetime | None) -> datetime | None:
        return _aware(value) if value is not None else None

    @model_validator(mode="after")
    def _decision_is_truthful(self) -> "KCApprovalAttestationInput":
        if len(self.family_ids) != len(set(self.family_ids)):
            raise ValueError("KC input family_ids must be unique")
        if len(self.construct_ids) != len(set(self.construct_ids)):
            raise ValueError("KC input construct_ids must be unique")
        judgments = (
            self.construct_coverage,
            self.family_independence,
            self.difficulty_progression,
            self.first_two_paths_reviewed,
        )
        people = (self.prepared_by, self.reviewed_by, self.reviewed_at)
        if self.decision == AttestationReviewDecision.PENDING:
            if any(value is not None for value in (*people, *judgments)):
                raise ValueError("pending KC input cannot claim review provenance or judgments")
        else:
            if any(value is None for value in people):
                raise ValueError("completed KC input requires review provenance")
            assert self.prepared_by is not None and self.reviewed_by is not None
            if self.prepared_by.casefold() == self.reviewed_by.casefold():
                raise ValueError("a KC input requires an independent reviewer")
            if self.decision == AttestationReviewDecision.APPROVED:
                if any(value is not True for value in judgments):
                    raise ValueError("approved KC input requires every judgment to be true")
            elif not any(value is False for value in judgments):
                raise ValueError("rejected KC input requires at least one failed judgment")
        return self


class ReleaseApprovalAttestationInput(_StrictFrozenModel):
    """Exact release-byte binding awaiting final independent approval."""

    attestation_id: str = Field(max_length=128, pattern=_ID_PATTERN)
    release_id: str = Field(max_length=128, pattern=_ID_PATTERN)
    graph_version: int = Field(ge=1)
    graph_digest: str = Field(pattern=_SHA256_PATTERN)
    bank_version: str = Field(max_length=128, pattern=_ID_PATTERN)
    bank_digest: str = Field(pattern=_SHA256_PATTERN)
    catalog_version: str = Field(max_length=128, pattern=_ID_PATTERN)
    catalog_digest: str = Field(pattern=_SHA256_PATTERN)
    released_kcs: tuple[str, ...] = Field(min_length=1)
    # Computed by the finalizer only after every KC approval is complete.
    kc_attestation_digest: Literal[None] = None
    bundle_sha256: str = Field(pattern=_SHA256_PATTERN)
    decision: AttestationReviewDecision = AttestationReviewDecision.PENDING
    prepared_by: str | None = Field(default=None, min_length=1, max_length=256)
    reviewed_by: str | None = Field(default=None, min_length=1, max_length=256)
    reviewed_at: datetime | None = None
    cross_component_compatibility: bool | None = None
    complete_hard_closure: bool | None = None
    exact_bytes_reviewed: bool | None = None
    notes: str | None = Field(default=None, min_length=1)

    @field_validator("prepared_by", "reviewed_by", mode="before")
    @classmethod
    def _normalize_people(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("release_id")
    @classmethod
    def _nonproduction_namespace_is_reserved(cls, value: str) -> str:
        if value == "nonproduction.legacy-unpinned" or value.startswith(
            "nonproduction.fixture."
        ):
            raise ValueError("release_id uses a reserved non-production namespace")
        return value

    @field_validator("reviewed_at")
    @classmethod
    def _aware_review(cls, value: datetime | None) -> datetime | None:
        return _aware(value) if value is not None else None

    @model_validator(mode="after")
    def _decision_is_truthful(self) -> "ReleaseApprovalAttestationInput":
        if len(self.released_kcs) != len(set(self.released_kcs)):
            raise ValueError("release input released_kcs must be unique")
        judgments = (
            self.cross_component_compatibility,
            self.complete_hard_closure,
            self.exact_bytes_reviewed,
        )
        people = (self.prepared_by, self.reviewed_by, self.reviewed_at)
        if self.decision == AttestationReviewDecision.PENDING:
            if any(value is not None for value in (*people, *judgments)):
                raise ValueError(
                    "pending release input cannot claim review provenance or judgments"
                )
        else:
            if any(value is None for value in people):
                raise ValueError("completed release input requires review provenance")
            assert self.prepared_by is not None and self.reviewed_by is not None
            if self.prepared_by.casefold() == self.reviewed_by.casefold():
                raise ValueError("a release input requires an independent reviewer")
            if self.decision == AttestationReviewDecision.APPROVED:
                if any(value is not True for value in judgments):
                    raise ValueError(
                        "approved release input requires every judgment to be true"
                    )
            elif not any(value is False for value in judgments):
                raise ValueError("rejected release input requires at least one failed judgment")
        return self


class ReleaseReviewScaffold(_StrictFrozenModel):
    """Non-publishable exact candidate bindings awaiting human decisions."""

    schema_version: Literal[1] = 1
    artifact_kind: Literal["pending_release_review_scaffold"] = (
        "pending_release_review_scaffold"
    )
    warning: Literal[
        "PENDING HUMAN REVIEW: this scaffold is not a release approval manifest."
    ] = "PENDING HUMAN REVIEW: this scaffold is not a release approval manifest."
    assessment_source_digest: str = Field(pattern=_SHA256_PATTERN)
    assessment_review_manifest_digest: str = Field(pattern=_SHA256_PATTERN)
    pedagogy_source_digest: str = Field(pattern=_SHA256_PATTERN)
    pedagogy_review_manifest_digest: str = Field(pattern=_SHA256_PATTERN)
    family_attestations: tuple[FamilyApprovalAttestationInput, ...] = Field(min_length=1)
    kc_attestations: tuple[KCApprovalAttestationInput, ...] = Field(min_length=1)
    release_attestation: ReleaseApprovalAttestationInput

    @model_validator(mode="after")
    def _identities_are_unique(self) -> "ReleaseReviewScaffold":
        attestation_ids = [
            *(item.attestation_id for item in self.family_attestations),
            *(item.attestation_id for item in self.kc_attestations),
            self.release_attestation.attestation_id,
        ]
        if len(attestation_ids) != len(set(attestation_ids)):
            raise ValueError("scaffold attestation ids must be globally unique")
        family_ids = [item.family_id for item in self.family_attestations]
        if len(family_ids) != len(set(family_ids)):
            raise ValueError("a scaffold may bind each family only once")
        kc_ids = [item.kc_id for item in self.kc_attestations]
        if len(kc_ids) != len(set(kc_ids)):
            raise ValueError("a scaffold may bind each KC only once")
        return self


class ReleasePublicationMetadata(_StrictFrozenModel):
    """Explicit publication facts with no wall-clock defaults."""

    published_by: str = Field(min_length=1, max_length=256)
    published_at: datetime

    @field_validator("published_by", mode="before")
    @classmethod
    def _normalize_publisher(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("published_at")
    @classmethod
    def _aware_publication(cls, value: datetime) -> datetime:
        return _aware(value)


class PublishedReleaseManifest(_StrictFrozenModel):
    """Machine-readable receipt emitted beside an immutable bundle."""

    schema_version: Literal[2] = 2
    release_id: str = Field(max_length=128, pattern=_ID_PATTERN)
    bundle_file: Literal["bundle.json"] = "bundle.json"
    bundle_sha256: str = Field(pattern=_SHA256_PATTERN)
    reviews_file: Literal["release-reviews.json"] = "release-reviews.json"
    reviews_sha256: str = Field(pattern=_SHA256_PATTERN)
    graph_version: int = Field(ge=1)
    graph_digest: str = Field(pattern=_SHA256_PATTERN)
    bank_version: str = Field(max_length=128, pattern=_ID_PATTERN)
    bank_digest: str = Field(pattern=_SHA256_PATTERN)
    catalog_version: str = Field(max_length=128, pattern=_ID_PATTERN)
    catalog_digest: str = Field(pattern=_SHA256_PATTERN)
    released_kcs: tuple[str, ...] = Field(min_length=1)
    family_attestation_ids: tuple[str, ...] = Field(min_length=1)
    kc_attestation_ids: tuple[str, ...] = Field(min_length=1)
    release_attestation_id: str = Field(max_length=128, pattern=_ID_PATTERN)
    release_attestation_digest: str = Field(pattern=_SHA256_PATTERN)
    published_by: str = Field(min_length=1, max_length=256)
    published_at: datetime

    @field_validator("published_at")
    @classmethod
    def _aware_publication(cls, value: datetime) -> datetime:
        return _aware(value)

    @field_validator("release_id")
    @classmethod
    def _nonproduction_namespace_is_reserved(cls, value: str) -> str:
        if value == "nonproduction.legacy-unpinned" or value.startswith(
            "nonproduction.fixture."
        ):
            raise ValueError(
                "release_id uses a reserved non-production namespace"
            )
        return value
