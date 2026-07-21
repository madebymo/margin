"""Strict source contracts for the pending U-substitution content wave."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from tutor.schemas.assessment import AssessmentSurface, StrictFrozenModel
from tutor.schemas.kc import KC_ID_PATTERN

_CONTENT_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"


class PolynomialTerm(StrictFrozenModel):
    coefficient: int = Field(ge=-18, le=18)
    exponent: int = Field(ge=0, le=6)

    @model_validator(mode="after")
    def _nonzero(self) -> "PolynomialTerm":
        if self.coefficient == 0:
            raise ValueError("polynomial term coefficient cannot be zero")
        return self


class InnerPolynomialSpec(StrictFrozenModel):
    """A nonconstant polynomial used as u=g(x)."""

    terms: tuple[PolynomialTerm, ...] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def _canonical_nonconstant_polynomial(self) -> "InnerPolynomialSpec":
        powers = [term.exponent for term in self.terms]
        if powers != sorted(powers, reverse=True) or len(powers) != len(set(powers)):
            raise ValueError("inner-polynomial powers must be unique and descending")
        if not any(power > 0 for power in powers):
            raise ValueError("an inner polynomial must be nonconstant")
        return self

    @property
    def degree(self) -> int:
        return self.terms[0].exponent


class PrimitivePolynomialSpec(StrictFrozenModel):
    """F(x); its derivative is the compiler-derived integrand."""

    terms: tuple[PolynomialTerm, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def _canonical_nonconstant_polynomial(self) -> "PrimitivePolynomialSpec":
        powers = [term.exponent for term in self.terms]
        if powers != sorted(powers, reverse=True) or len(powers) != len(set(powers)):
            raise ValueError("primitive-polynomial powers must be unique and descending")
        if any(power == 0 for power in powers):
            raise ValueError("source primitives omit constants; the compiler adds one +C")
        return self


class DifferentialAffineTask(StrictFrozenModel):
    kind: Literal["differential_affine"] = "differential_affine"
    inner: InnerPolynomialSpec

    @model_validator(mode="after")
    def _affine(self) -> "DifferentialAffineTask":
        if self.inner.degree != 1:
            raise ValueError("affine differential tasks require degree one")
        return self


class DifferentialPolynomialTask(StrictFrozenModel):
    kind: Literal["differential_polynomial"] = "differential_polynomial"
    inner: InnerPolynomialSpec


class DifferentialAtPointTask(StrictFrozenModel):
    kind: Literal["differential_at_point"] = "differential_at_point"
    inner: InnerPolynomialSpec
    input_value: int = Field(ge=-8, le=8)


class DifferentialScaleTask(StrictFrozenModel):
    kind: Literal["differential_scale"] = "differential_scale"
    inner: InnerPolynomialSpec
    scale: int = Field(ge=-12, le=12)

    @model_validator(mode="after")
    def _nonzero_scale(self) -> "DifferentialScaleTask":
        if self.scale == 0:
            raise ValueError("differential scale cannot be zero")
        return self


class DifferentialCorrectionTask(StrictFrozenModel):
    kind: Literal["differential_correction"] = "differential_correction"
    inner: InnerPolynomialSpec
    mistake: Literal["kept_power", "missed_coefficient", "omitted_dx"]


class DifferentialVerificationTask(StrictFrozenModel):
    kind: Literal["differential_verification"] = "differential_verification"
    inner: InnerPolynomialSpec


class DifferentialGuidedMappingTask(StrictFrozenModel):
    kind: Literal["differential_guided_mapping"] = "differential_guided_mapping"
    inner: InnerPolynomialSpec

    @model_validator(mode="after")
    def _three_nonconstant_terms(self) -> "DifferentialGuidedMappingTask":
        if len(self.inner.terms) != 3 or any(
            term.exponent == 0 for term in self.inner.terms
        ):
            raise ValueError("guided differential mapping requires three variable terms")
        return self


class IndefiniteDirectTask(StrictFrozenModel):
    kind: Literal["indefinite_direct"] = "indefinite_direct"
    primitive: PrimitivePolynomialSpec


class IndefiniteOneConstantTask(StrictFrozenModel):
    kind: Literal["indefinite_one_constant"] = "indefinite_one_constant"
    primitive: PrimitivePolynomialSpec


class IndefiniteCorrectionTask(StrictFrozenModel):
    kind: Literal["indefinite_correction"] = "indefinite_correction"
    primitive: PrimitivePolynomialSpec
    mistake: Literal["missing_constant", "constant_per_term", "kept_power"]


class IndefiniteDifferentiateTask(StrictFrozenModel):
    kind: Literal["indefinite_differentiate"] = "indefinite_differentiate"
    primitive: PrimitivePolynomialSpec


class IndefiniteContrastTask(StrictFrozenModel):
    kind: Literal["indefinite_contrast"] = "indefinite_contrast"
    primitive: PrimitivePolynomialSpec
    lower: int = Field(ge=-5, le=5)
    upper: int = Field(ge=-5, le=5)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> "IndefiniteContrastTask":
        if self.lower >= self.upper:
            raise ValueError("contrast bounds must be ordered")
        return self


class IndefiniteEquivalentTask(StrictFrozenModel):
    kind: Literal["indefinite_equivalent"] = "indefinite_equivalent"
    primitive: PrimitivePolynomialSpec


class IndefiniteGuidedMappingTask(StrictFrozenModel):
    kind: Literal["indefinite_guided_mapping"] = "indefinite_guided_mapping"
    primitive: PrimitivePolynomialSpec


class IntegralSingleTask(StrictFrozenModel):
    kind: Literal["integral_single"] = "integral_single"
    primitive: PrimitivePolynomialSpec

    @model_validator(mode="after")
    def _one_term(self) -> "IntegralSingleTask":
        if len(self.primitive.terms) != 1:
            raise ValueError("single-term integration requires one primitive term")
        return self


class IntegralSumTask(StrictFrozenModel):
    kind: Literal["integral_sum"] = "integral_sum"
    primitive: PrimitivePolynomialSpec

    @model_validator(mode="after")
    def _multiple_terms(self) -> "IntegralSumTask":
        if len(self.primitive.terms) < 2:
            raise ValueError("sum integration requires multiple terms")
        return self


class IntegralDifferenceTask(StrictFrozenModel):
    kind: Literal["integral_difference"] = "integral_difference"
    primitive: PrimitivePolynomialSpec

    @model_validator(mode="after")
    def _contains_both_signs(self) -> "IntegralDifferenceTask":
        coefficients = [term.coefficient for term in self.primitive.terms]
        if not any(value > 0 for value in coefficients) or not any(
            value < 0 for value in coefficients
        ):
            raise ValueError("difference integration requires positive and negative terms")
        return self


class IntegralConstantMultipleTask(StrictFrozenModel):
    kind: Literal["integral_constant_multiple"] = "integral_constant_multiple"
    primitive: PrimitivePolynomialSpec


class IntegralSparseTask(StrictFrozenModel):
    kind: Literal["integral_sparse"] = "integral_sparse"
    primitive: PrimitivePolynomialSpec

    @model_validator(mode="after")
    def _has_power_gap(self) -> "IntegralSparseTask":
        powers = [term.exponent for term in self.primitive.terms]
        if len(powers) < 2 or not any(
            left - right > 1 for left, right in zip(powers, powers[1:])
        ):
            raise ValueError("sparse integration requires a missing power")
        return self


class IntegralCoefficientAuditTask(StrictFrozenModel):
    kind: Literal["integral_coefficient_audit"] = "integral_coefficient_audit"
    primitive: PrimitivePolynomialSpec


class IntegralCorrectionTask(StrictFrozenModel):
    kind: Literal["integral_correction"] = "integral_correction"
    primitive: PrimitivePolynomialSpec
    mistake: Literal["kept_power", "multiplied_coefficient", "dropped_term"]


class IntegralGuidedMappingTask(StrictFrozenModel):
    kind: Literal["integral_guided_mapping"] = "integral_guided_mapping"
    primitive: PrimitivePolynomialSpec

    @model_validator(mode="after")
    def _three_terms(self) -> "IntegralGuidedMappingTask":
        if len(self.primitive.terms) != 3:
            raise ValueError("guided integration mapping requires three terms")
        return self


class CompositeSpec(StrictFrozenModel):
    """scale*g'(x)*(g(x))^outer_power, derived from typed parameters."""

    inner: InnerPolynomialSpec
    outer_power: int = Field(ge=2, le=6)
    derivative_scale: int = Field(ge=-12, le=12)

    @model_validator(mode="after")
    def _nonzero_scale(self) -> "CompositeSpec":
        if self.derivative_scale == 0:
            raise ValueError("composite derivative scale cannot be zero")
        return self


class RecognizeAffineTask(StrictFrozenModel):
    kind: Literal["recognize_affine"] = "recognize_affine"
    composite: CompositeSpec

    @model_validator(mode="after")
    def _affine(self) -> "RecognizeAffineTask":
        if self.composite.inner.degree != 1:
            raise ValueError("affine recognition requires a degree-one inner")
        return self


class RecognizeQuadraticTask(StrictFrozenModel):
    kind: Literal["recognize_quadratic"] = "recognize_quadratic"
    composite: CompositeSpec

    @model_validator(mode="after")
    def _quadratic(self) -> "RecognizeQuadraticTask":
        if self.composite.inner.degree != 2:
            raise ValueError("quadratic recognition requires a degree-two inner")
        return self


class RecognizeExactFactorTask(StrictFrozenModel):
    kind: Literal["recognize_exact_factor"] = "recognize_exact_factor"
    composite: CompositeSpec


class RecognizeCorrectionTask(StrictFrozenModel):
    kind: Literal["recognize_correction"] = "recognize_correction"
    composite: CompositeSpec
    mistake: Literal["outer_as_inner", "missing_scale", "wrong_derivative"]


class RecognizeOrderedTask(StrictFrozenModel):
    kind: Literal["recognize_ordered"] = "recognize_ordered"
    composite: CompositeSpec


class RecognizeReverseTask(StrictFrozenModel):
    kind: Literal["recognize_reverse"] = "recognize_reverse"
    composite: CompositeSpec


class RecognizeGuidedMappingTask(StrictFrozenModel):
    kind: Literal["recognize_guided_mapping"] = "recognize_guided_mapping"
    composite: CompositeSpec


class USubstitutionSpec(StrictFrozenModel):
    """k*(p+1)*g'(x)*g(x)^p, whose primitive coefficient is k."""

    inner: InnerPolynomialSpec
    outer_power: int = Field(ge=2, le=6)
    result_coefficient: int = Field(ge=-12, le=12)

    @model_validator(mode="after")
    def _nonzero_coefficient(self) -> "USubstitutionSpec":
        if self.result_coefficient == 0:
            raise ValueError("U-substitution result coefficient cannot be zero")
        return self


class USubAffineTask(StrictFrozenModel):
    kind: Literal["u_sub_affine"] = "u_sub_affine"
    substitution: USubstitutionSpec

    @model_validator(mode="after")
    def _affine(self) -> "USubAffineTask":
        if self.substitution.inner.degree != 1:
            raise ValueError("affine substitution requires a degree-one inner")
        return self


class USubQuadraticTask(StrictFrozenModel):
    kind: Literal["u_sub_quadratic"] = "u_sub_quadratic"
    substitution: USubstitutionSpec

    @model_validator(mode="after")
    def _quadratic(self) -> "USubQuadraticTask":
        if self.substitution.inner.degree != 2:
            raise ValueError("quadratic substitution requires a degree-two inner")
        return self


class USubNegativeTask(StrictFrozenModel):
    kind: Literal["u_sub_negative"] = "u_sub_negative"
    substitution: USubstitutionSpec

    @model_validator(mode="after")
    def _negative(self) -> "USubNegativeTask":
        if self.substitution.result_coefficient >= 0:
            raise ValueError("negative substitution task requires a negative result")
        return self


class USubSetupTask(StrictFrozenModel):
    kind: Literal["u_sub_setup"] = "u_sub_setup"
    substitution: USubstitutionSpec


class USubCorrectionTask(StrictFrozenModel):
    kind: Literal["u_sub_correction"] = "u_sub_correction"
    substitution: USubstitutionSpec
    mistake: Literal["forgot_du_scale", "kept_x", "wrong_outer_power"]


class USubVerificationTask(StrictFrozenModel):
    kind: Literal["u_sub_verification"] = "u_sub_verification"
    substitution: USubstitutionSpec


class USubGuidedMappingTask(StrictFrozenModel):
    kind: Literal["u_sub_guided_mapping"] = "u_sub_guided_mapping"
    substitution: USubstitutionSpec


USubConstructId = Literal[
    "differential.affine",
    "differential.polynomial",
    "differential.at_point",
    "differential.scale",
    "differential.correction",
    "differential.verification",
    "differential.guided_mapping",
    "indefinite.direct",
    "indefinite.one_constant",
    "indefinite.correction",
    "indefinite.differentiate",
    "indefinite.contrast",
    "indefinite.equivalent",
    "indefinite.guided_mapping",
    "integral.single",
    "integral.sum",
    "integral.difference",
    "integral.constant_multiple",
    "integral.sparse",
    "integral.coefficient_audit",
    "integral.correction",
    "integral.guided_mapping",
    "recognize.affine",
    "recognize.quadratic",
    "recognize.exact_factor",
    "recognize.correction",
    "recognize.ordered",
    "recognize.reverse",
    "recognize.guided_mapping",
    "u_sub.affine",
    "u_sub.quadratic",
    "u_sub.negative",
    "u_sub.setup",
    "u_sub.correction",
    "u_sub.verification",
    "u_sub.guided_mapping",
]


USubMathTask = Annotated[
    Union[
        DifferentialAffineTask,
        DifferentialPolynomialTask,
        DifferentialAtPointTask,
        DifferentialScaleTask,
        DifferentialCorrectionTask,
        DifferentialVerificationTask,
        DifferentialGuidedMappingTask,
        IndefiniteDirectTask,
        IndefiniteOneConstantTask,
        IndefiniteCorrectionTask,
        IndefiniteDifferentiateTask,
        IndefiniteContrastTask,
        IndefiniteEquivalentTask,
        IndefiniteGuidedMappingTask,
        IntegralSingleTask,
        IntegralSumTask,
        IntegralDifferenceTask,
        IntegralConstantMultipleTask,
        IntegralSparseTask,
        IntegralCoefficientAuditTask,
        IntegralCorrectionTask,
        IntegralGuidedMappingTask,
        RecognizeAffineTask,
        RecognizeQuadraticTask,
        RecognizeExactFactorTask,
        RecognizeCorrectionTask,
        RecognizeOrderedTask,
        RecognizeReverseTask,
        RecognizeGuidedMappingTask,
        USubAffineTask,
        USubQuadraticTask,
        USubNegativeTask,
        USubSetupTask,
        USubCorrectionTask,
        USubVerificationTask,
        USubGuidedMappingTask,
    ],
    Field(discriminator="kind"),
]


class USubFamilyBlueprint(StrictFrozenModel):
    blueprint_id: str = Field(max_length=96, pattern=_CONTENT_ID_PATTERN)
    revision: int = Field(default=1, ge=1)
    item_id: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    family_id: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    kc_id: str = Field(pattern=KC_ID_PATTERN)
    construct_id: USubConstructId
    surface: AssessmentSurface
    allocation_order: int = Field(ge=0)
    difficulty: Literal["foundation", "core", "stretch"] = "core"
    task: USubMathTask


class USubBlueprintDocument(StrictFrozenModel):
    schema_version: Literal[1] = 1
    blueprint_version: str = Field(pattern=_CONTENT_ID_PATTERN)
    output_bank_version: str = Field(pattern=_CONTENT_ID_PATTERN)
    graph_version: int = Field(ge=1)
    authoring_source: str = Field(min_length=1, max_length=128)
    author: str = Field(min_length=1)
    target_kcs: list[str] = Field(min_length=1)
    released_kcs: list[str] = Field(default_factory=list)
    families: list[USubFamilyBlueprint] = Field(min_length=1)

    @model_validator(mode="after")
    def _identities_are_unambiguous(self) -> "USubBlueprintDocument":
        for label, values in (
            ("target_kcs", self.target_kcs),
            ("released_kcs", self.released_kcs),
            ("blueprint identities", [(f.blueprint_id, f.revision) for f in self.families]),
            ("item ids", [f.item_id for f in self.families]),
            ("family ids", [f.family_id for f in self.families]),
            (
                "allocation orders",
                [(f.kc_id, f.surface, f.allocation_order) for f in self.families],
            ),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if not set(self.released_kcs) <= set(self.target_kcs):
            raise ValueError("released_kcs must be a subset of target_kcs")
        if any(family.kc_id not in self.target_kcs for family in self.families):
            raise ValueError("every family KC must occur in target_kcs")
        return self
