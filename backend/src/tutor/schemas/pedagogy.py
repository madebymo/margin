"""Versioned pedagogy packs and immutable reviewed release catalogs.

Individual packs remain an offline authoring surface: imports and LLM jobs may
produce drafts.  A runtime-trusted release is a :class:`PedagogyPackCatalog`,
which contains only explicitly reviewed packs and is pinned to one graph
version.  Draft sources are never promoted merely because they can be loaded.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tutor.schemas.common import ReviewStatus, WidgetType
from tutor.schemas.kc import KC_ID_PATTERN


_CONTENT_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"


class _StrictFrozenModel(BaseModel):
    """Reject undeclared release fields and prevent model-level mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Misconception(_StrictFrozenModel):
    """A known, named student misconception with a detectable error signature."""

    id: str = Field(max_length=128, pattern=r"^m\.[a-z0-9_.]+$")
    description: str = Field(min_length=1)
    error_signature: str = Field(min_length=1)
    remediation_hint: str = Field(min_length=1)


class Metaphor(_StrictFrozenModel):
    """A teaching metaphor and the widget types it renders well with."""

    id: str = Field(max_length=128, pattern=r"^met\.[a-z0-9_.]+$")
    description: str = Field(min_length=1)
    widget_affinity: list[WidgetType] = Field(min_length=1)


class PedagogyPackProvenance(_StrictFrozenModel):
    """Human authorship and review facts for one released pack revision."""

    author: str = Field(min_length=1, max_length=256)
    reviewed_by: str = Field(min_length=1, max_length=256)
    reviewed_at: datetime

    @field_validator("author", "reviewed_by", mode="before")
    @classmethod
    def _normalize_people(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("reviewed_at")
    @classmethod
    def _review_timestamp_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        return value

    @model_validator(mode="after")
    def _independent_reviewer(self) -> "PedagogyPackProvenance":
        if self.author.casefold() == self.reviewed_by.casefold():
            raise ValueError("reviewed_by must identify someone other than the author")
        return self


class PedagogyPack(_StrictFrozenModel):
    """The complete cached pedagogy asset for one KC."""

    kc_id: str = Field(pattern=KC_ID_PATTERN)
    misconceptions: list[Misconception] = Field(default_factory=list)
    metaphors: list[Metaphor] = Field(default_factory=list)
    error_patterns: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.DRAFT
    version: int = Field(default=1, ge=1)
    provenance: PedagogyPackProvenance | None = None

    @field_validator("sources", mode="before")
    @classmethod
    def _normalize_sources(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        normalized: list[object] = []
        for source in value:
            if isinstance(source, str):
                source = source.strip()
                if not source:
                    raise ValueError("sources must contain meaningful nonblank strings")
            normalized.append(source)
        return normalized

    @model_validator(mode="after")
    def _content_invariants(self) -> "PedagogyPack":
        misconception_ids = [item.id for item in self.misconceptions]
        if len(misconception_ids) != len(set(misconception_ids)):
            raise ValueError("misconception ids must be unique within a pack")
        metaphor_ids = [item.id for item in self.metaphors]
        if len(metaphor_ids) != len(set(metaphor_ids)):
            raise ValueError("metaphor ids must be unique within a pack")
        if len(self.error_patterns) != len(set(self.error_patterns)):
            raise ValueError("error_patterns must be unique")
        if len(self.sources) != len(set(self.sources)):
            raise ValueError("sources must be unique")
        if (
            self.review_status == ReviewStatus.HUMAN_APPROVED
            and self.provenance is None
        ):
            raise ValueError("a human_approved pack requires reviewed provenance")
        if self.review_status == ReviewStatus.HUMAN_APPROVED and not self.sources:
            raise ValueError("a human_approved pack requires at least one source")
        return self


class PedagogyPackCatalog(_StrictFrozenModel):
    """One published, graph-pinned set of reviewed pedagogy pack revisions."""

    schema_version: Literal[1] = 1
    catalog_version: str = Field(
        min_length=1,
        max_length=128,
        pattern=_CONTENT_ID_PATTERN,
    )
    graph_version: int = Field(ge=1)
    published_by: str = Field(min_length=1, max_length=256)
    published_at: datetime
    packs: tuple[PedagogyPack, ...] = ()

    @field_validator("published_at")
    @classmethod
    def _publish_timestamp_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("published_at must include a timezone")
        return value

    @model_validator(mode="after")
    def _release_invariants(self) -> "PedagogyPackCatalog":
        pack_kcs = [pack.kc_id for pack in self.packs]
        if len(pack_kcs) != len(set(pack_kcs)):
            raise ValueError("a catalog may contain only one pack per KC")

        misconception_owners: dict[str, str] = {}
        metaphor_owners: dict[str, str] = {}
        for pack in self.packs:
            if pack.review_status != ReviewStatus.HUMAN_APPROVED:
                raise ValueError(
                    f"catalog pack {pack.kc_id!r} is not human_approved"
                )
            # PedagogyPack enforces provenance for approved content. Keep the
            # catalog check explicit as defense against future schema widening.
            if pack.provenance is None:
                raise ValueError(
                    f"catalog pack {pack.kc_id!r} lacks reviewed provenance"
                )
            for misconception in pack.misconceptions:
                owner = misconception_owners.setdefault(
                    misconception.id, pack.kc_id
                )
                if owner != pack.kc_id:
                    raise ValueError(
                        f"misconception id {misconception.id!r} appears in "
                        f"both {owner!r} and {pack.kc_id!r}"
                    )
            for metaphor in pack.metaphors:
                owner = metaphor_owners.setdefault(metaphor.id, pack.kc_id)
                if owner != pack.kc_id:
                    raise ValueError(
                        f"metaphor id {metaphor.id!r} appears in both "
                        f"{owner!r} and {pack.kc_id!r}"
                    )
        return self

    @property
    def pack_by_kc(self) -> dict[str, PedagogyPack]:
        """Return a fresh KC lookup for the catalog's pinned pack models."""

        return {pack.kc_id: pack for pack in self.packs}
