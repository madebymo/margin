"""Typed authoring contracts for the Product/Quotient Rules pilot closure.

The source document stores mathematical parameters, never an expected-answer
string.  The matching compiler owns the small, reviewed set of transformations
that turn those parameters into prompts, hints, and answer contracts.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from tutor.schemas.assessment import AssessmentSurface, StrictFrozenModel
from tutor.schemas.kc import KC_ID_PATTERN

_CONTENT_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"


class PolynomialTerm(StrictFrozenModel):
    """One non-zero integer-coefficient polynomial term."""

    coefficient: int = Field(ge=-20, le=20)
    exponent: int = Field(ge=0, le=15)

    @model_validator(mode="after")
    def _coefficient_is_nonzero(self) -> "PolynomialTerm":
        if self.coefficient == 0:
            raise ValueError("polynomial coefficients must be non-zero")
        return self


class PolynomialSpec(StrictFrozenModel):
    """A canonical sparse polynomial; terms must be in descending order."""

    terms: list[PolynomialTerm] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def _canonical_terms(self) -> "PolynomialSpec":
        exponents = [term.exponent for term in self.terms]
        if len(exponents) != len(set(exponents)):
            raise ValueError("polynomial exponents must be unique")
        if exponents != sorted(exponents, reverse=True):
            raise ValueError("polynomial terms must use descending exponents")
        return self


class ExponentProductTask(StrictFrozenModel):
    """Simplify ``base^a * base^b`` by adding exponents."""

    kind: Literal["exponent_product"] = "exponent_product"
    base: Literal["z"] = "z"
    left_exponent: int = Field(ge=1, le=19)
    right_exponent: int = Field(ge=1, le=19)

    @model_validator(mode="after")
    def _bounded_result(self) -> "ExponentProductTask":
        if self.left_exponent + self.right_exponent > 20:
            raise ValueError("result exponent exceeds verifier bound")
        return self


class ExponentQuotientTask(StrictFrozenModel):
    """Simplify ``base^a / base^b`` by subtracting exponents."""

    kind: Literal["exponent_quotient"] = "exponent_quotient"
    base: Literal["z"] = "z"
    numerator_exponent: int = Field(ge=1, le=20)
    denominator_exponent: int = Field(ge=1, le=20)

    @model_validator(mode="after")
    def _nonzero_result(self) -> "ExponentQuotientTask":
        if self.numerator_exponent == self.denominator_exponent:
            raise ValueError("use exponent_zero for an identity-result family")
        return self


class ExponentPowerTask(StrictFrozenModel):
    """Simplify ``(base^a)^b`` by multiplying exponents."""

    kind: Literal["exponent_power"] = "exponent_power"
    base: Literal["z"] = "z"
    inner_exponent: int = Field(ge=1, le=20)
    outer_exponent: int = Field(ge=1, le=20)

    @model_validator(mode="after")
    def _bounded_result(self) -> "ExponentPowerTask":
        if self.inner_exponent * self.outer_exponent > 20:
            raise ValueError("result exponent exceeds verifier bound")
        return self


class ExponentNegativeTask(StrictFrozenModel):
    """Rewrite a negative exponent as a reciprocal."""

    kind: Literal["exponent_negative"] = "exponent_negative"
    base: Literal["z"] = "z"
    magnitude: int = Field(ge=1, le=20)


class ExponentZeroTask(StrictFrozenModel):
    """Apply the zero-exponent identity to a declared non-zero base."""

    kind: Literal["exponent_zero"] = "exponent_zero"
    base: Literal["z"] = "z"


class ExponentCompoundTask(StrictFrozenModel):
    """Combine zero, negative, power, product, and quotient exponent rules."""

    kind: Literal["exponent_compound"] = "exponent_compound"
    base: Literal["z"] = "z"
    negative_magnitude: int = Field(ge=1, le=20)
    inner_exponent: int = Field(ge=1, le=20)
    outer_exponent: int = Field(ge=1, le=20)
    product_exponent: int = Field(ge=1, le=20)
    denominator_exponent: int = Field(ge=1, le=20)

    @model_validator(mode="after")
    def _bounded_nonzero_result(self) -> "ExponentCompoundTask":
        result = (
            -self.negative_magnitude
            + self.inner_exponent * self.outer_exponent
            + self.product_exponent
            - self.denominator_exponent
        )
        if result == 0:
            raise ValueError("compound exponent result must be non-zero")
        if abs(result) > 20:
            raise ValueError("compound exponent result exceeds verifier bound")
        return self


class PowerDerivativeTask(StrictFrozenModel):
    """Differentiate one integer power with an integer constant multiple."""

    kind: Literal["power_derivative"] = "power_derivative"
    coefficient: int = Field(ge=-20, le=20)
    exponent: int = Field(ge=-8, le=15)

    @model_validator(mode="after")
    def _meaningful_power(self) -> "PowerDerivativeTask":
        if self.coefficient == 0:
            raise ValueError("coefficient must be non-zero")
        if self.exponent in {0, 1}:
            raise ValueError("pilot power-rule tasks require exponent other than zero or one")
        return self


class RationalPowerDerivativeTask(StrictFrozenModel):
    """Differentiate a constant multiple of a reduced rational power."""

    kind: Literal["rational_power_derivative"] = "rational_power_derivative"
    coefficient: int = Field(ge=-20, le=20)
    numerator: int = Field(ge=-19, le=19)
    denominator: int = Field(ge=2, le=20)

    @model_validator(mode="after")
    def _meaningful_reduced_power(self) -> "RationalPowerDerivativeTask":
        if self.coefficient == 0:
            raise ValueError("coefficient must be non-zero")
        if self.numerator == 0:
            raise ValueError("rational exponent must be non-zero")
        if math.gcd(abs(self.numerator), self.denominator) != 1:
            raise ValueError("rational exponent must be reduced")
        return self


class RadicalPowerDerivativeTask(StrictFrozenModel):
    """Differentiate a square root or its reciprocal without hiding the radical."""

    kind: Literal["radical_power_derivative"] = "radical_power_derivative"
    coefficient: int = Field(ge=-20, le=20)
    form: Literal["sqrt", "reciprocal_sqrt"]

    @model_validator(mode="after")
    def _coefficient_is_nonzero(self) -> "RadicalPowerDerivativeTask":
        if self.coefficient == 0:
            raise ValueError("coefficient must be non-zero")
        return self


class PolynomialDerivativeTask(StrictFrozenModel):
    """Differentiate a polynomial term by term."""

    kind: Literal["polynomial_derivative"] = "polynomial_derivative"
    polynomial: PolynomialSpec

    @model_validator(mode="after")
    def _has_nonconstant_term(self) -> "PolynomialDerivativeTask":
        if not any(term.exponent > 0 for term in self.polynomial.terms):
            raise ValueError("polynomial derivative task cannot be constant-only")
        return self


class FunctionAtPointData(StrictFrozenModel):
    """Opaque function and derivative values at one point."""

    point: int = Field(ge=-12, le=12)
    f_value: int = Field(ge=-30, le=30)
    g_value: int = Field(ge=-30, le=30)
    f_derivative: int = Field(ge=-30, le=30)
    g_derivative: int = Field(ge=-30, le=30)


class ProductAtPointTask(StrictFrozenModel):
    """Find the derivative of an opaque product from local function data."""

    kind: Literal["product_at_point"] = "product_at_point"
    data: FunctionAtPointData


class QuotientAtPointTask(StrictFrozenModel):
    """Find the derivative of an opaque quotient from local function data."""

    kind: Literal["quotient_at_point"] = "quotient_at_point"
    data: FunctionAtPointData

    @model_validator(mode="after")
    def _denominator_is_defined(self) -> "QuotientAtPointTask":
        if self.data.g_value == 0:
            raise ValueError("g(point) must be non-zero for a quotient task")
        return self


ReleaseConstructId = Literal[
    "exponent.product",
    "exponent.quotient",
    "exponent.power",
    "exponent.negative",
    "exponent.zero",
    "exponent.compound",
    "power.positive_integer",
    "power.negative_integer",
    "power.rational",
    "power.sqrt",
    "power.reciprocal_radical",
    "sum.polynomial_termwise",
    "product_quotient.product_at_point",
    "product_quotient.quotient_at_point",
]


ReleaseMathTask = Annotated[
    Union[
        ExponentProductTask,
        ExponentQuotientTask,
        ExponentPowerTask,
        ExponentNegativeTask,
        ExponentZeroTask,
        ExponentCompoundTask,
        PowerDerivativeTask,
        RationalPowerDerivativeTask,
        RadicalPowerDerivativeTask,
        PolynomialDerivativeTask,
        ProductAtPointTask,
        QuotientAtPointTask,
    ],
    Field(discriminator="kind"),
]


class ReleaseFamilyBlueprint(StrictFrozenModel):
    """One independently reviewable family producing one initial pilot item."""

    blueprint_id: str = Field(max_length=96, pattern=_CONTENT_ID_PATTERN)
    revision: int = Field(default=1, ge=1)
    item_id: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    family_id: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    kc_id: str = Field(pattern=KC_ID_PATTERN)
    construct_id: ReleaseConstructId
    surface: AssessmentSurface
    allocation_order: int = Field(ge=0)
    difficulty: Literal["foundation", "core", "stretch"] = "core"
    task: ReleaseMathTask


class ProductQuotientBlueprintDocument(StrictFrozenModel):
    """Pending source inventory for the exact Product/Quotient hard closure."""

    schema_version: Literal[2] = 2
    blueprint_version: str = Field(pattern=_CONTENT_ID_PATTERN)
    output_bank_version: str = Field(pattern=_CONTENT_ID_PATTERN)
    graph_version: int = Field(ge=1)
    authoring_source: str = Field(min_length=1, max_length=128)
    author: str = Field(min_length=1)
    target_kcs: list[str] = Field(min_length=1)
    released_kcs: list[str] = Field(default_factory=list)
    families: list[ReleaseFamilyBlueprint] = Field(min_length=1)

    @model_validator(mode="after")
    def _identities_and_orders_are_unambiguous(self) -> "ProductQuotientBlueprintDocument":
        for label, values in (
            ("target_kcs", self.target_kcs),
            ("released_kcs", self.released_kcs),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if not set(self.released_kcs) <= set(self.target_kcs):
            raise ValueError("released_kcs must be a subset of target_kcs")

        for label, values in (
            ("blueprint identities", [(item.blueprint_id, item.revision) for item in self.families]),
            ("item ids", [item.item_id for item in self.families]),
            ("family ids", [item.family_id for item in self.families]),
            (
                "allocation orders",
                [
                    (item.kc_id, item.surface, item.allocation_order)
                    for item in self.families
                ],
            ),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if any(item.kc_id not in self.target_kcs for item in self.families):
            raise ValueError("every family KC must occur in target_kcs")
        return self
