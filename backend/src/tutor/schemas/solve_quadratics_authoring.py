"""Strict source contracts for the pending Solve Quadratics content wave.

Every discriminated task stores bounded mathematical parameters only. Prompts,
worked derivations, expected answers, guided truth, and error signatures are
derived by the closed compiler registry.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from tutor.schemas.assessment import AssessmentSurface, StrictFrozenModel
from tutor.schemas.kc import KC_ID_PATTERN

_CONTENT_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"
_SMALL_ARITHMETIC_LIMIT = 120


class CubicPolynomial(StrictFrozenModel):
    """A degree-at-most-three integer polynomial with reviewed display order."""

    cubic: int = Field(ge=-9, le=9)
    quadratic: int = Field(ge=-12, le=12)
    linear: int = Field(ge=-15, le=15)
    constant: int = Field(ge=-20, le=20)
    display_order: Literal["descending", "ascending", "interleaved"] = "descending"

    @model_validator(mode="after")
    def _not_zero(self) -> "CubicPolynomial":
        if not any((self.cubic, self.quadratic, self.linear, self.constant)):
            raise ValueError("a polynomial operand cannot be identically zero")
        return self


class BinomialProduct(StrictFrozenModel):
    left_linear: int = Field(ge=-6, le=6)
    left_constant: int = Field(ge=-9, le=9)
    right_linear: int = Field(ge=-6, le=6)
    right_constant: int = Field(ge=-9, le=9)

    @model_validator(mode="after")
    def _genuine_binomials(self) -> "BinomialProduct":
        if 0 in {
            self.left_linear,
            self.left_constant,
            self.right_linear,
            self.right_constant,
        }:
            raise ValueError("both factors must be genuine linear binomials")
        coefficients = (
            self.left_linear * self.right_linear,
            self.left_linear * self.right_constant
            + self.left_constant * self.right_linear,
            self.left_constant * self.right_constant,
        )
        if max(abs(value) for value in coefficients) > _SMALL_ARITHMETIC_LIMIT:
            raise ValueError("expanded binomial coefficients exceed the arithmetic bound")
        return self


class PolynomialCore(StrictFrozenModel):
    """One full-claim portfolio: addition, subtraction, and expansion."""

    add_left: CubicPolynomial
    add_right: CubicPolynomial
    subtract_left: CubicPolynomial
    subtract_right: CubicPolynomial
    expand: BinomialProduct

    @model_validator(mode="after")
    def _bounded_results(self) -> "PolynomialCore":
        for left, right, sign in (
            (self.add_left, self.add_right, 1),
            (self.subtract_left, self.subtract_right, -1),
        ):
            values = (
                left.cubic + sign * right.cubic,
                left.quadratic + sign * right.quadratic,
                left.linear + sign * right.linear,
                left.constant + sign * right.constant,
            )
            if max(abs(value) for value in values) > _SMALL_ARITHMETIC_LIMIT:
                raise ValueError("polynomial result exceeds the arithmetic bound")
        return self


class PolynomialDirectPortfolioTask(StrictFrozenModel):
    kind: Literal["polynomial_direct_portfolio"] = "polynomial_direct_portfolio"
    core: PolynomialCore


class PolynomialCorrectionPortfolioTask(StrictFrozenModel):
    kind: Literal["polynomial_correction_portfolio"] = "polynomial_correction_portfolio"
    core: PolynomialCore


class PolynomialSparsePortfolioTask(StrictFrozenModel):
    kind: Literal["polynomial_sparse_portfolio"] = "polynomial_sparse_portfolio"
    core: PolynomialCore

    @model_validator(mode="after")
    def _requires_sparse_reordered_inputs(self) -> "PolynomialSparsePortfolioTask":
        operands = (
            self.core.add_left,
            self.core.add_right,
            self.core.subtract_left,
            self.core.subtract_right,
        )
        if not any(0 in (p.cubic, p.quadratic, p.linear, p.constant) for p in operands):
            raise ValueError("sparse portfolio requires a missing power")
        if all(p.display_order == "descending" for p in operands):
            raise ValueError("sparse portfolio requires a noncanonical display order")
        return self


class PolynomialMixedExpressionTask(StrictFrozenModel):
    kind: Literal["polynomial_mixed_expression"] = "polynomial_mixed_expression"
    core: PolynomialCore


class PolynomialCoefficientAuditTask(StrictFrozenModel):
    kind: Literal["polynomial_coefficient_audit"] = "polynomial_coefficient_audit"
    core: PolynomialCore
    add_power: Literal[0, 1, 2, 3]
    subtract_power: Literal[0, 1, 2, 3]
    expand_power: Literal[0, 1, 2]


class PolynomialReversePortfolioTask(StrictFrozenModel):
    kind: Literal["polynomial_reverse_portfolio"] = "polynomial_reverse_portfolio"
    core: PolynomialCore


class PolynomialTablePortfolioTask(StrictFrozenModel):
    kind: Literal["polynomial_table_portfolio"] = "polynomial_table_portfolio"
    core: PolynomialCore


class PolynomialGuidedMatchTask(StrictFrozenModel):
    kind: Literal["polynomial_guided_match"] = "polynomial_guided_match"
    core: PolynomialCore


class GcfFactorSpec(StrictFrozenModel):
    """Normalized signed GCF form g*x^k*(a*x+b), with a positive."""

    common_coefficient: int = Field(ge=-9, le=9)
    common_exponent: int = Field(ge=0, le=2)
    residual_linear: int = Field(ge=1, le=9)
    residual_constant: int = Field(ge=-12, le=12)

    @model_validator(mode="after")
    def _normalized(self) -> "GcfFactorSpec":
        if self.common_coefficient == 0 or self.residual_constant == 0:
            raise ValueError("signed GCF parameters must be nonzero where required")
        if math.gcd(self.residual_linear, abs(self.residual_constant)) != 1:
            raise ValueError("residual coefficients must be relatively prime")
        return self


class MonicFactorSpec(StrictFrozenModel):
    lower_root: int = Field(ge=-8, le=8)
    upper_root: int = Field(ge=-8, le=8)

    @model_validator(mode="after")
    def _ordered(self) -> "MonicFactorSpec":
        if self.lower_root > self.upper_root:
            raise ValueError("roots must be supplied in increasing order")
        return self


class DifferenceSquaresSpec(StrictFrozenModel):
    scale: int = Field(ge=-6, le=6)
    magnitude: int = Field(ge=2, le=8)

    @model_validator(mode="after")
    def _nonzero_scale(self) -> "DifferenceSquaresSpec":
        if self.scale == 0:
            raise ValueError("difference-of-squares scale cannot be zero")
        return self


class FactoringCore(StrictFrozenModel):
    """One full-claim portfolio spanning all three released factoring forms."""

    gcf: GcfFactorSpec
    monic: MonicFactorSpec
    difference: DifferenceSquaresSpec


class FactoringDirectPortfolioTask(StrictFrozenModel):
    kind: Literal["factoring_direct_portfolio"] = "factoring_direct_portfolio"
    core: FactoringCore


class FactoringCorrectionPortfolioTask(StrictFrozenModel):
    kind: Literal["factoring_correction_portfolio"] = "factoring_correction_portfolio"
    core: FactoringCore


class FactoringMissingPortfolioTask(StrictFrozenModel):
    kind: Literal["factoring_missing_portfolio"] = "factoring_missing_portfolio"
    core: FactoringCore


class FactoringTablePortfolioTask(StrictFrozenModel):
    kind: Literal["factoring_table_portfolio"] = "factoring_table_portfolio"
    core: FactoringCore


class FactoringVerificationPortfolioTask(StrictFrozenModel):
    kind: Literal["factoring_verification_portfolio"] = "factoring_verification_portfolio"
    core: FactoringCore
    check_value: int = Field(ge=-3, le=3)


class FactoringTransferPortfolioTask(StrictFrozenModel):
    kind: Literal["factoring_transfer_portfolio"] = "factoring_transfer_portfolio"
    core: FactoringCore


class FactoringGuidedMatchTask(StrictFrozenModel):
    kind: Literal["factoring_guided_match"] = "factoring_guided_match"
    core: FactoringCore


class LinearEquationSpec(StrictFrozenModel):
    left_coefficient: int = Field(ge=-9, le=9)
    left_constant: int = Field(ge=-40, le=40)
    right_coefficient: int = Field(ge=-9, le=9)
    right_constant: int = Field(ge=-40, le=40)

    @model_validator(mode="after")
    def _integer_solution(self) -> "LinearEquationSpec":
        difference = self.left_coefficient - self.right_coefficient
        if difference == 0:
            raise ValueError("linear equation must have a unique solution")
        numerator = self.right_constant - self.left_constant
        if numerator % difference:
            raise ValueError("linear equation must have an integer solution")
        if abs(numerator // difference) > 29:
            raise ValueError("linear solution exceeds the modest arithmetic bound")
        return self


class LinearTwoSidedTask(StrictFrozenModel):
    kind: Literal["linear_two_sided"] = "linear_two_sided"
    equation: LinearEquationSpec


class LinearReversedTask(StrictFrozenModel):
    kind: Literal["linear_reversed"] = "linear_reversed"
    equation: LinearEquationSpec


class LinearCorrectionTask(StrictFrozenModel):
    kind: Literal["linear_correction"] = "linear_correction"
    equation: LinearEquationSpec
    mistake: Literal["one_sided", "sign_transfer", "partial_division"]


class LinearGuidedBalanceTask(StrictFrozenModel):
    kind: Literal["linear_guided_balance"] = "linear_guided_balance"
    equation: LinearEquationSpec


class LinearOneSideTask(StrictFrozenModel):
    kind: Literal["linear_one_side"] = "linear_one_side"
    coefficient: int = Field(ge=-9, le=9)
    constant: int = Field(ge=-30, le=30)
    target: int = Field(ge=-40, le=40)
    variable_side: Literal["left", "right"]

    @model_validator(mode="after")
    def _integer_solution(self) -> "LinearOneSideTask":
        if self.coefficient == 0:
            raise ValueError("one-sided equation requires a variable term")
        if (self.target - self.constant) % self.coefficient:
            raise ValueError("one-sided equation must have an integer solution")
        if abs((self.target - self.constant) // self.coefficient) > 29:
            raise ValueError("one-sided solution exceeds the modest arithmetic bound")
        return self


class LinearDistributedTask(StrictFrozenModel):
    kind: Literal["linear_distributed"] = "linear_distributed"
    multiplier: int = Field(ge=-6, le=6)
    inner_coefficient: int = Field(ge=-6, le=6)
    inner_constant: int = Field(ge=-9, le=9)
    added_constant: int = Field(ge=-15, le=15)
    right_coefficient: int = Field(ge=-9, le=9)
    right_constant: int = Field(ge=-40, le=40)

    @model_validator(mode="after")
    def _valid_equation(self) -> "LinearDistributedTask":
        if self.multiplier == 0 or self.inner_coefficient == 0:
            raise ValueError("distributed equation needs nonzero factors")
        spec = LinearEquationSpec(
            left_coefficient=self.multiplier * self.inner_coefficient,
            left_constant=self.multiplier * self.inner_constant + self.added_constant,
            right_coefficient=self.right_coefficient,
            right_constant=self.right_constant,
        )
        if max(abs(spec.left_coefficient), abs(spec.left_constant)) > _SMALL_ARITHMETIC_LIMIT:
            raise ValueError("distributed equation exceeds the arithmetic bound")
        return self


class LinearGroupedTask(StrictFrozenModel):
    kind: Literal["linear_grouped"] = "linear_grouped"
    left_multiplier: int = Field(ge=-6, le=6)
    left_shift: int = Field(ge=-9, le=9)
    right_multiplier: int = Field(ge=-6, le=6)
    right_shift: int = Field(ge=-9, le=9)

    @model_validator(mode="after")
    def _valid_equation(self) -> "LinearGroupedTask":
        if self.left_multiplier == 0 or self.right_multiplier == 0:
            raise ValueError("grouped equation multipliers must be nonzero")
        LinearEquationSpec(
            left_coefficient=self.left_multiplier,
            left_constant=self.left_multiplier * self.left_shift,
            right_coefficient=self.right_multiplier,
            right_constant=self.right_multiplier * self.right_shift,
        )
        return self


class LinearDoubleDistributedTask(StrictFrozenModel):
    kind: Literal["linear_double_distributed"] = "linear_double_distributed"
    left_multiplier: int = Field(ge=-5, le=5)
    left_coefficient: int = Field(ge=-5, le=5)
    left_constant: int = Field(ge=-8, le=8)
    right_multiplier: int = Field(ge=-5, le=5)
    right_coefficient: int = Field(ge=-5, le=5)
    right_constant: int = Field(ge=-8, le=8)

    @model_validator(mode="after")
    def _valid_equation(self) -> "LinearDoubleDistributedTask":
        if 0 in {
            self.left_multiplier,
            self.left_coefficient,
            self.right_multiplier,
            self.right_coefficient,
        }:
            raise ValueError("double-distributed equation factors must be nonzero")
        LinearEquationSpec(
            left_coefficient=self.left_multiplier * self.left_coefficient,
            left_constant=self.left_multiplier * self.left_constant,
            right_coefficient=self.right_multiplier * self.right_coefficient,
            right_constant=self.right_multiplier * self.right_constant,
        )
        return self


class RootPair(StrictFrozenModel):
    lower: int = Field(ge=-8, le=8)
    upper: int = Field(ge=-8, le=8)

    @model_validator(mode="after")
    def _ordered(self) -> "RootPair":
        if self.lower > self.upper:
            raise ValueError("roots must be supplied in increasing order")
        return self


class QuadraticExpandedTask(StrictFrozenModel):
    kind: Literal["quadratic_expanded"] = "quadratic_expanded"
    roots: RootPair


class QuadraticShiftedTask(StrictFrozenModel):
    kind: Literal["quadratic_shifted"] = "quadratic_shifted"
    roots: RootPair
    right_constant: int = Field(ge=-20, le=20)

    @model_validator(mode="after")
    def _nonzero_shift(self) -> "QuadraticShiftedTask":
        if self.right_constant == 0:
            raise ValueError("shifted equation requires a nonzero right side")
        return self


class QuadraticFactoredTask(StrictFrozenModel):
    kind: Literal["quadratic_factored"] = "quadratic_factored"
    roots: RootPair


class QuadraticReversedTask(StrictFrozenModel):
    kind: Literal["quadratic_reversed"] = "quadratic_reversed"
    roots: RootPair


class QuadraticSparseDifferenceTask(StrictFrozenModel):
    kind: Literal["quadratic_sparse_difference"] = "quadratic_sparse_difference"
    magnitude: int = Field(ge=2, le=8)


class QuadraticRepeatedTask(StrictFrozenModel):
    kind: Literal["quadratic_repeated"] = "quadratic_repeated"
    root: int = Field(ge=-8, le=8)


class QuadraticCorrectionTask(StrictFrozenModel):
    kind: Literal["quadratic_correction"] = "quadratic_correction"
    roots: RootPair
    mistake: Literal["not_zero", "one_root", "factor_sign"]


class QuadraticBothSidesTask(StrictFrozenModel):
    kind: Literal["quadratic_both_sides"] = "quadratic_both_sides"
    roots: RootPair
    right_linear: int = Field(ge=-6, le=6)
    right_constant: int = Field(ge=-15, le=15)


class QuadraticGuidedFactorMapTask(StrictFrozenModel):
    kind: Literal["quadratic_guided_factor_map"] = "quadratic_guided_factor_map"
    roots: RootPair


SolveQuadraticsConstructId = Literal[
    "polynomial.direct_portfolio",
    "polynomial.correction_portfolio",
    "polynomial.sparse_portfolio",
    "polynomial.mixed_expression",
    "polynomial.coefficient_audit",
    "polynomial.reverse_portfolio",
    "polynomial.table_portfolio",
    "polynomial.guided_match",
    "factoring.direct_portfolio",
    "factoring.correction_portfolio",
    "factoring.missing_portfolio",
    "factoring.table_portfolio",
    "factoring.verification_portfolio",
    "factoring.transfer_portfolio",
    "factoring.guided_match",
    "linear.two_sided",
    "linear.reversed",
    "linear.correction",
    "linear.guided_balance",
    "linear.one_side",
    "linear.distributed",
    "linear.grouped",
    "linear.double_distributed",
    "quadratic.expanded",
    "quadratic.shifted",
    "quadratic.factored",
    "quadratic.reversed",
    "quadratic.sparse_difference",
    "quadratic.repeated",
    "quadratic.correction",
    "quadratic.both_sides",
    "quadratic.guided_factor_map",
]


SolveQuadraticsMathTask = Annotated[
    Union[
        PolynomialDirectPortfolioTask,
        PolynomialCorrectionPortfolioTask,
        PolynomialSparsePortfolioTask,
        PolynomialMixedExpressionTask,
        PolynomialCoefficientAuditTask,
        PolynomialReversePortfolioTask,
        PolynomialTablePortfolioTask,
        PolynomialGuidedMatchTask,
        FactoringDirectPortfolioTask,
        FactoringCorrectionPortfolioTask,
        FactoringMissingPortfolioTask,
        FactoringTablePortfolioTask,
        FactoringVerificationPortfolioTask,
        FactoringTransferPortfolioTask,
        FactoringGuidedMatchTask,
        LinearTwoSidedTask,
        LinearReversedTask,
        LinearCorrectionTask,
        LinearGuidedBalanceTask,
        LinearOneSideTask,
        LinearDistributedTask,
        LinearGroupedTask,
        LinearDoubleDistributedTask,
        QuadraticExpandedTask,
        QuadraticShiftedTask,
        QuadraticFactoredTask,
        QuadraticReversedTask,
        QuadraticSparseDifferenceTask,
        QuadraticRepeatedTask,
        QuadraticCorrectionTask,
        QuadraticBothSidesTask,
        QuadraticGuidedFactorMapTask,
    ],
    Field(discriminator="kind"),
]


class SolveQuadraticsFamilyBlueprint(StrictFrozenModel):
    blueprint_id: str = Field(max_length=96, pattern=_CONTENT_ID_PATTERN)
    revision: int = Field(default=1, ge=1)
    item_id: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    family_id: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    kc_id: str = Field(pattern=KC_ID_PATTERN)
    construct_id: SolveQuadraticsConstructId
    surface: AssessmentSurface
    allocation_order: int = Field(ge=0)
    difficulty: Literal["foundation", "core", "stretch"] = "core"
    task: SolveQuadraticsMathTask


class SolveQuadraticsBlueprintDocument(StrictFrozenModel):
    schema_version: Literal[1] = 1
    blueprint_version: str = Field(pattern=_CONTENT_ID_PATTERN)
    output_bank_version: str = Field(pattern=_CONTENT_ID_PATTERN)
    graph_version: int = Field(ge=1)
    authoring_source: str = Field(min_length=1, max_length=128)
    author: str = Field(min_length=1)
    target_kcs: list[str] = Field(min_length=1)
    released_kcs: list[str] = Field(default_factory=list)
    families: list[SolveQuadraticsFamilyBlueprint] = Field(min_length=1)

    @model_validator(mode="after")
    def _identities_are_unambiguous(self) -> "SolveQuadraticsBlueprintDocument":
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
