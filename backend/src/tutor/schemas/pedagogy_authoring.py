"""Review-bound source contracts for deterministic pedagogy-pack publication.

Authoring sources deliberately contain no ``review_status`` or reviewer fields.
Those facts live in an exact-digest manifest and are applied only by the
review compiler when every source revision has independent approval.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tutor.schemas.kc import KC_ID_PATTERN
from tutor.schemas.pedagogy import Metaphor, Misconception

_CONTENT_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _StrictFrozenModel(BaseModel):
    """Reject contract drift and make review inputs immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PedagogyPackSource(_StrictFrozenModel):
    """One authored, unreviewed pedagogy-pack revision."""

    source_id: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    revision: int = Field(default=1, ge=1)
    kc_id: str = Field(pattern=KC_ID_PATTERN)
    author: str = Field(min_length=1, max_length=256)
    misconceptions: tuple[Misconception, ...] = Field(min_length=3)
    metaphors: tuple[Metaphor, ...] = Field(min_length=1)
    error_patterns: tuple[str, ...] = Field(min_length=1)
    sources: tuple[str, ...] = Field(min_length=1)

    @field_validator("author", mode="before")
    @classmethod
    def _normalize_author(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("error_patterns", "sources", mode="before")
    @classmethod
    def _normalize_string_sequences(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        return tuple(item.strip() if isinstance(item, str) else item for item in value)

    @model_validator(mode="after")
    def _content_is_reviewable(self) -> "PedagogyPackSource":
        misconception_ids = [item.id for item in self.misconceptions]
        if len(misconception_ids) != len(set(misconception_ids)):
            raise ValueError("misconception ids must be unique within a source")
        metaphor_ids = [item.id for item in self.metaphors]
        if len(metaphor_ids) != len(set(metaphor_ids)):
            raise ValueError("metaphor ids must be unique within a source")
        if len(self.error_patterns) != len(set(self.error_patterns)):
            raise ValueError("error_patterns must be unique within a source")
        if any(not pattern for pattern in self.error_patterns):
            raise ValueError("error_patterns must contain nonblank strings")
        if len(self.sources) != len(set(self.sources)):
            raise ValueError("sources must be unique within a source")
        if any(len(source) < 12 for source in self.sources):
            raise ValueError("sources must contain meaningful citations")
        return self


class PedagogySourceDocument(_StrictFrozenModel):
    """A graph-pinned set of pedagogy sources awaiting review."""

    schema_version: Literal[1] = 1
    source_version: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    graph_version: int = Field(ge=1)
    pack_sources: tuple[PedagogyPackSource, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _identities_are_unique(self) -> "PedagogySourceDocument":
        identities = [(source.source_id, source.revision) for source in self.pack_sources]
        if len(identities) != len(set(identities)):
            raise ValueError("source_id/revision pairs must be unique")
        kc_ids = [source.kc_id for source in self.pack_sources]
        if len(kc_ids) != len(set(kc_ids)):
            raise ValueError("a source document may contain only one pack per KC")

        misconception_owners: dict[str, str] = {}
        metaphor_owners: dict[str, str] = {}
        for source in self.pack_sources:
            for misconception in source.misconceptions:
                previous = misconception_owners.setdefault(misconception.id, source.kc_id)
                if previous != source.kc_id:
                    raise ValueError(
                        f"misconception id {misconception.id!r} spans source packs"
                    )
            for metaphor in source.metaphors:
                previous = metaphor_owners.setdefault(metaphor.id, source.kc_id)
                if previous != source.kc_id:
                    raise ValueError(f"metaphor id {metaphor.id!r} spans source packs")
        return self


class PedagogyReviewDecision(StrEnum):
    """Review disposition for one exact source revision."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class PedagogyReviewEntry(_StrictFrozenModel):
    """A review decision bound to one canonical source digest."""

    source_id: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    revision: int = Field(ge=1)
    source_digest: str = Field(pattern=_SHA256_PATTERN)
    decision: PedagogyReviewDecision
    reviewed_by: str | None = Field(default=None, min_length=1, max_length=256)
    reviewed_at: datetime | None = None
    notes: str | None = Field(default=None, min_length=1)

    @field_validator("reviewed_by", mode="before")
    @classmethod
    def _normalize_reviewer(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("reviewed_at")
    @classmethod
    def _completed_timestamp_is_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("reviewed_at must include a timezone")
        return value

    @model_validator(mode="after")
    def _decision_has_truthful_provenance(self) -> "PedagogyReviewEntry":
        completed = self.decision != PedagogyReviewDecision.PENDING
        if completed and (self.reviewed_by is None or self.reviewed_at is None):
            raise ValueError("completed review requires reviewed_by and reviewed_at")
        if not completed and (self.reviewed_by is not None or self.reviewed_at is not None):
            raise ValueError("pending review cannot claim completed review provenance")
        return self


class PedagogyReviewManifest(_StrictFrozenModel):
    """Exact review coverage pinned to graph and compiler versions."""

    schema_version: Literal[1] = 1
    manifest_version: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    graph_version: int = Field(ge=1)
    compiler_version: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    entries: tuple[PedagogyReviewEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _entries_are_unique(self) -> "PedagogyReviewManifest":
        identities = [(entry.source_id, entry.revision) for entry in self.entries]
        if len(identities) != len(set(identities)):
            raise ValueError("review entries must have unique source_id/revision pairs")
        return self


class PedagogyPublicationMetadata(_StrictFrozenModel):
    """Explicit, reproducible release metadata; no wall-clock defaults."""

    catalog_version: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    published_by: str = Field(min_length=1, max_length=256)
    published_at: datetime

    @field_validator("published_by", mode="before")
    @classmethod
    def _normalize_publisher(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("published_at")
    @classmethod
    def _publication_timestamp_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("published_at must include a timezone")
        return value
