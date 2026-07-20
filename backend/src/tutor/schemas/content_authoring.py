"""Strict source contracts for deterministic assessment-item compilation.

Blueprints describe independently authored mathematical families. They are
kept separate from the compiled :class:`AssessmentItem` contract so review
approval can bind an exact source digest and compiler version instead of
trusting mutable review fields inside an item-bank JSON document.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from tutor.schemas.assessment import (
    AssessmentSurface,
    AssessmentTaskKind,
    StrictFrozenModel,
)
from tutor.schemas.kc import KC_ID_PATTERN

_AUTHORING_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ProductSameBaseCase(StrictFrozenModel):
    """One explicit parameter set for ``x^a * x^b``."""

    case_id: str = Field(max_length=64, pattern=_AUTHORING_ID_PATTERN)
    left_exponent: int = Field(ge=1, le=20)
    right_exponent: int = Field(ge=1, le=20)

    @model_validator(mode="after")
    def _bounded_result_exponent(self) -> "ProductSameBaseCase":
        if self.left_exponent + self.right_exponent > 20:
            raise ValueError("compiled result exponent cannot exceed verifier limit 20")
        return self


class QuotientSameBaseCase(StrictFrozenModel):
    """One explicit parameter set for ``x^a / x^b``."""

    case_id: str = Field(max_length=64, pattern=_AUTHORING_ID_PATTERN)
    numerator_exponent: int = Field(ge=1, le=20)
    denominator_exponent: int = Field(ge=1, le=20)

    @model_validator(mode="after")
    def _positive_result_exponent(self) -> "QuotientSameBaseCase":
        if self.numerator_exponent <= self.denominator_exponent:
            raise ValueError(
                "numerator_exponent must exceed denominator_exponent in the v1 prototype"
            )
        return self


class PowerOfPowerCase(StrictFrozenModel):
    """One explicit parameter set for ``(x^a)^b``."""

    case_id: str = Field(max_length=64, pattern=_AUTHORING_ID_PATTERN)
    inner_exponent: int = Field(ge=1, le=20)
    outer_exponent: int = Field(ge=1, le=20)

    @model_validator(mode="after")
    def _bounded_result_exponent(self) -> "PowerOfPowerCase":
        if self.inner_exponent * self.outer_exponent > 20:
            raise ValueError("compiled result exponent cannot exceed verifier limit 20")
        return self


class _BlueprintBase(StrictFrozenModel):
    """Metadata shared by every reviewed item-family blueprint."""

    blueprint_id: str = Field(max_length=96, pattern=_AUTHORING_ID_PATTERN)
    revision: int = Field(default=1, ge=1)
    family_id: str = Field(max_length=128, pattern=_AUTHORING_ID_PATTERN)
    kc_id: str = Field(pattern=KC_ID_PATTERN)
    surface: AssessmentSurface
    task_kind: AssessmentTaskKind
    allocation_order: int = Field(ge=0)
    difficulty: Literal["foundation", "core", "stretch"] = "core"
    source: str = Field(min_length=1, max_length=128)
    author: str = Field(min_length=1)


class ProductSameBaseBlueprint(_BlueprintBase):
    """Compile explicit variants of the same-base product rule."""

    prototype_id: Literal["exponent.product_same_base"]
    cases: list[ProductSameBaseCase] = Field(min_length=1)


class QuotientSameBaseBlueprint(_BlueprintBase):
    """Compile explicit variants of the same-base quotient rule."""

    prototype_id: Literal["exponent.quotient_same_base"]
    cases: list[QuotientSameBaseCase] = Field(min_length=1)


class PowerOfPowerBlueprint(_BlueprintBase):
    """Compile explicit variants of the power-of-a-power rule."""

    prototype_id: Literal["exponent.power_of_power"]
    cases: list[PowerOfPowerCase] = Field(min_length=1)


ItemFamilyBlueprint = Annotated[
    Union[
        ProductSameBaseBlueprint,
        QuotientSameBaseBlueprint,
        PowerOfPowerBlueprint,
    ],
    Field(discriminator="prototype_id"),
]


class ItemBlueprintDocument(StrictFrozenModel):
    """Versioned authoring input that compiles into an isolated item bank."""

    schema_version: Literal[1] = 1
    blueprint_version: str = Field(pattern=_AUTHORING_ID_PATTERN)
    output_bank_version: str = Field(pattern=_AUTHORING_ID_PATTERN)
    graph_version: int = Field(ge=1)
    released_kcs: list[str] = Field(default_factory=list)
    family_blueprints: list[ItemFamilyBlueprint] = Field(min_length=1)

    @model_validator(mode="after")
    def _source_identities_are_unambiguous(self) -> "ItemBlueprintDocument":
        if len(self.released_kcs) != len(set(self.released_kcs)):
            raise ValueError("released_kcs must be unique")

        blueprint_ids = [
            (blueprint.blueprint_id, blueprint.revision)
            for blueprint in self.family_blueprints
        ]
        if len(blueprint_ids) != len(set(blueprint_ids)):
            raise ValueError("blueprint_id/revision pairs must be unique")

        family_ids = [blueprint.family_id for blueprint in self.family_blueprints]
        if len(family_ids) != len(set(family_ids)):
            raise ValueError("one blueprint document cannot define a family twice")

        order_keys = [
            (blueprint.kc_id, blueprint.surface, blueprint.allocation_order)
            for blueprint in self.family_blueprints
        ]
        if len(order_keys) != len(set(order_keys)):
            raise ValueError("allocation_order must be unique per KC and surface")

        for blueprint in self.family_blueprints:
            case_ids = [case.case_id for case in blueprint.cases]
            if len(case_ids) != len(set(case_ids)):
                raise ValueError(
                    f"case ids must be unique in blueprint {blueprint.blueprint_id!r}"
                )
        return self


class ReviewDecision(StrEnum):
    """Human review disposition for one exact blueprint revision."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ContentReviewEntry(StrictFrozenModel):
    """Review state bound to a canonical blueprint digest."""

    blueprint_id: str = Field(max_length=96, pattern=_AUTHORING_ID_PATTERN)
    revision: int = Field(ge=1)
    source_digest: str = Field(pattern=_SHA256_PATTERN)
    decision: ReviewDecision
    reviewed_by: str | None = Field(default=None, min_length=1)
    reviewed_at: datetime | None = None
    notes: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _review_provenance_matches_decision(self) -> "ContentReviewEntry":
        has_reviewer = self.reviewed_by is not None
        has_timestamp = self.reviewed_at is not None
        if self.decision == ReviewDecision.PENDING:
            if has_reviewer or has_timestamp:
                raise ValueError("pending review entries cannot claim completed review provenance")
        elif not has_reviewer or not has_timestamp:
            raise ValueError("completed review entries require reviewed_by and reviewed_at")
        return self


class ContentReviewManifest(StrictFrozenModel):
    """Review decisions for one graph and deterministic compiler version."""

    schema_version: Literal[1] = 1
    manifest_version: str = Field(pattern=_AUTHORING_ID_PATTERN)
    graph_version: int = Field(ge=1)
    compiler_version: str = Field(pattern=_AUTHORING_ID_PATTERN)
    entries: list[ContentReviewEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _entries_are_unique(self) -> "ContentReviewManifest":
        identities = [(entry.blueprint_id, entry.revision) for entry in self.entries]
        if len(identities) != len(set(identities)):
            raise ValueError("review entries must have unique blueprint_id/revision pairs")
        return self
