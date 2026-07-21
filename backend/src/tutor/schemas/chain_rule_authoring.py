"""Typed authoring contracts for the pending Chain Rule content wave.

Blueprints contain only bounded mathematical parameters. Expected answers,
prompts, hints, error signatures, and guided scoring truth are derived by the
deterministic compiler; none can be supplied as free-form authoring fields.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from tutor.schemas.assessment import AssessmentSurface, StrictFrozenModel
from tutor.schemas.kc import KC_ID_PATTERN

_CONTENT_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"
_SMALL_RESULT_LIMIT = 300


def _require_small_result(value: int, label: str) -> None:
    if abs(value) > _SMALL_RESULT_LIMIT:
        raise ValueError(f"{label} must have absolute value at most {_SMALL_RESULT_LIMIT}")


class AffineFunctionValueTask(StrictFrozenModel):
    kind: Literal["function_affine_value"] = "function_affine_value"
    coefficient: int = Field(ge=-12, le=12)
    constant: int = Field(ge=-30, le=30)
    input_value: int = Field(ge=-20, le=20)

    @model_validator(mode="after")
    def _valid_task(self) -> "AffineFunctionValueTask":
        if self.coefficient == 0:
            raise ValueError("an affine evaluation requires a nonzero coefficient")
        _require_small_result(
            self.coefficient * self.input_value + self.constant,
            "affine evaluation",
        )
        return self


class QuadraticFunctionValueTask(StrictFrozenModel):
    kind: Literal["function_quadratic_value"] = "function_quadratic_value"
    quadratic: int = Field(ge=-6, le=6)
    linear: int = Field(ge=-12, le=12)
    constant: int = Field(ge=-30, le=30)
    input_value: int = Field(ge=-8, le=8)

    @model_validator(mode="after")
    def _valid_task(self) -> "QuadraticFunctionValueTask":
        if self.quadratic == 0:
            raise ValueError("a quadratic evaluation requires a nonzero quadratic term")
        _require_small_result(
            self.quadratic * self.input_value**2
            + self.linear * self.input_value
            + self.constant,
            "quadratic evaluation",
        )
        return self


class FunctionExpressionValueTask(StrictFrozenModel):
    kind: Literal["function_expression_value"] = "function_expression_value"
    function_coefficient: int = Field(ge=-10, le=10)
    function_constant: int = Field(ge=-20, le=20)
    input_coefficient: int = Field(ge=-10, le=10)
    input_constant: int = Field(ge=-20, le=20)

    @model_validator(mode="after")
    def _valid_task(self) -> "FunctionExpressionValueTask":
        if self.function_coefficient == 0 or self.input_coefficient == 0:
            raise ValueError("function and input expressions must both be nonconstant")
        return self


class FunctionTableCombinationTask(StrictFrozenModel):
    """Combine two outputs selected from an accessible value table."""

    kind: Literal["function_table_combination"] = "function_table_combination"
    left_input: int = Field(ge=-20, le=20)
    left_output: int = Field(ge=-100, le=100)
    right_input: int = Field(ge=-20, le=20)
    right_output: int = Field(ge=-100, le=100)
    operation: Literal["sum", "left_minus_right"]

    @model_validator(mode="after")
    def _valid_task(self) -> "FunctionTableCombinationTask":
        if self.left_input == self.right_input:
            raise ValueError("table inputs must be distinct")
        result = (
            self.left_output + self.right_output
            if self.operation == "sum"
            else self.left_output - self.right_output
        )
        _require_small_result(result, "table combination")
        return self


class FunctionPlotChangeTask(StrictFrozenModel):
    """Read two exact values from an accessible static plot and find a change."""

    kind: Literal["function_plot_change"] = "function_plot_change"
    start_input: int = Field(ge=-20, le=20)
    start_output: int = Field(ge=-100, le=100)
    end_input: int = Field(ge=-20, le=20)
    end_output: int = Field(ge=-100, le=100)

    @model_validator(mode="after")
    def _valid_task(self) -> "FunctionPlotChangeTask":
        if self.start_input >= self.end_input:
            raise ValueError("plot inputs must increase from start to end")
        _require_small_result(
            self.end_output - self.start_output,
            "plotted output change",
        )
        return self


class FunctionOrderedValuesTask(StrictFrozenModel):
    """Produce two function values in a declared order."""

    kind: Literal["function_ordered_values"] = "function_ordered_values"
    coefficient: int = Field(ge=-12, le=12)
    constant: int = Field(ge=-30, le=30)
    first_input: int = Field(ge=-20, le=20)
    second_input: int = Field(ge=-20, le=20)

    @model_validator(mode="after")
    def _valid_task(self) -> "FunctionOrderedValuesTask":
        if self.coefficient == 0 or self.first_input == self.second_input:
            raise ValueError("ordered values require a nonconstant rule and distinct inputs")
        for value in (self.first_input, self.second_input):
            _require_small_result(
                self.coefficient * value + self.constant,
                "ordered function value",
            )
        return self


class AffineCompositionTask(StrictFrozenModel):
    kind: Literal["composition_affine"] = "composition_affine"
    f_coefficient: int = Field(ge=-10, le=10)
    f_constant: int = Field(ge=-20, le=20)
    g_coefficient: int = Field(ge=-10, le=10)
    g_constant: int = Field(ge=-20, le=20)
    order: Literal["f_after_g", "g_after_f"]

    @model_validator(mode="after")
    def _valid_task(self) -> "AffineCompositionTask":
        if self.f_coefficient == 0 or self.g_coefficient == 0:
            raise ValueError("both composed affine functions must be nonconstant")
        return self


class QuadraticOuterCompositionTask(StrictFrozenModel):
    kind: Literal["composition_quadratic_outer"] = "composition_quadratic_outer"
    outer_scale: int = Field(ge=-6, le=6)
    outer_constant: int = Field(ge=-20, le=20)
    inner_coefficient: int = Field(ge=-8, le=8)
    inner_constant: int = Field(ge=-20, le=20)

    @model_validator(mode="after")
    def _valid_task(self) -> "QuadraticOuterCompositionTask":
        if self.outer_scale == 0 or self.inner_coefficient == 0:
            raise ValueError("quadratic outer and affine inner scales must be nonzero")
        return self


class CompositionAtPointTask(StrictFrozenModel):
    """Evaluate one declared composition order from two affine rules."""

    kind: Literal["composition_at_point"] = "composition_at_point"
    f_coefficient: int = Field(ge=-10, le=10)
    f_constant: int = Field(ge=-20, le=20)
    g_coefficient: int = Field(ge=-10, le=10)
    g_constant: int = Field(ge=-20, le=20)
    point: int = Field(ge=-10, le=10)
    order: Literal["f_after_g", "g_after_f"]

    @model_validator(mode="after")
    def _valid_task(self) -> "CompositionAtPointTask":
        if self.f_coefficient == 0 or self.g_coefficient == 0:
            raise ValueError("both composed affine functions must be nonconstant")
        f_at_point = self.f_coefficient * self.point + self.f_constant
        g_at_point = self.g_coefficient * self.point + self.g_constant
        result = (
            self.f_coefficient * g_at_point + self.f_constant
            if self.order == "f_after_g"
            else self.g_coefficient * f_at_point + self.g_constant
        )
        _require_small_result(result, "point composition")
        return self


class CompositionTablePathTask(StrictFrozenModel):
    """Follow and combine outputs along one table-defined composition path."""

    kind: Literal["composition_table_path"] = "composition_table_path"
    point: int = Field(ge=-20, le=20)
    g_at_point: int = Field(ge=-100, le=100)
    f_at_g: int = Field(ge=-100, le=100)
    distractor_input: int = Field(ge=-20, le=20)
    f_at_distractor: int = Field(ge=-100, le=100)

    @model_validator(mode="after")
    def _valid_task(self) -> "CompositionTablePathTask":
        if len({self.point, self.g_at_point, self.distractor_input}) < 3:
            raise ValueError("table path keys must be distinct")
        _require_small_result(
            self.f_at_g - self.g_at_point,
            "table composition difference",
        )
        return self


class CompositionPairedOrdersTask(StrictFrozenModel):
    """Evaluate both composition orders and preserve their declared order."""

    kind: Literal["composition_paired_orders"] = "composition_paired_orders"
    f_coefficient: int = Field(ge=-8, le=8)
    f_constant: int = Field(ge=-20, le=20)
    g_coefficient: int = Field(ge=-8, le=8)
    g_constant: int = Field(ge=-20, le=20)
    point: int = Field(ge=-10, le=10)

    @model_validator(mode="after")
    def _valid_task(self) -> "CompositionPairedOrdersTask":
        if self.f_coefficient == 0 or self.g_coefficient == 0:
            raise ValueError("both composed affine functions must be nonconstant")
        f_at_point = self.f_coefficient * self.point + self.f_constant
        g_at_point = self.g_coefficient * self.point + self.g_constant
        for value in (
            self.f_coefficient * g_at_point + self.f_constant,
            self.g_coefficient * f_at_point + self.g_constant,
        ):
            _require_small_result(value, "paired composition value")
        return self


class CompositionPlotSumTask(StrictFrozenModel):
    """Follow two composition paths through exact static-plot data."""

    kind: Literal["composition_plot_sum"] = "composition_plot_sum"
    point: int = Field(ge=-20, le=20)
    g_at_point: int = Field(ge=-20, le=20)
    f_at_g: int = Field(ge=-100, le=100)
    f_at_point: int = Field(ge=-20, le=20)
    g_at_f: int = Field(ge=-100, le=100)

    @model_validator(mode="after")
    def _valid_task(self) -> "CompositionPlotSumTask":
        _require_small_result(self.f_at_g + self.g_at_f, "plotted composition sum")
        return self


class AffinePowerChainTask(StrictFrozenModel):
    kind: Literal["chain_affine_power"] = "chain_affine_power"
    inner_coefficient: int = Field(ge=-10, le=10)
    inner_constant: int = Field(ge=-20, le=20)
    outer_power: int = Field(ge=2, le=8)

    @model_validator(mode="after")
    def _valid_task(self) -> "AffinePowerChainTask":
        if self.inner_coefficient == 0:
            raise ValueError("the affine inner function must be nonconstant")
        return self


class QuadraticPowerChainTask(StrictFrozenModel):
    kind: Literal["chain_quadratic_power"] = "chain_quadratic_power"
    quadratic_coefficient: int = Field(ge=-8, le=8)
    inner_constant: int = Field(ge=-20, le=20)
    outer_power: int = Field(ge=2, le=7)

    @model_validator(mode="after")
    def _valid_task(self) -> "QuadraticPowerChainTask":
        if self.quadratic_coefficient == 0:
            raise ValueError("the inner quadratic coefficient must be nonzero")
        return self


class ChainAtPointTask(StrictFrozenModel):
    kind: Literal["chain_at_point"] = "chain_at_point"
    point: int = Field(ge=-20, le=20)
    inner_value: int = Field(ge=-6, le=6)
    inner_derivative: int = Field(ge=-12, le=12)
    outer_power: int = Field(ge=2, le=5)

    @model_validator(mode="after")
    def _valid_task(self) -> "ChainAtPointTask":
        if self.inner_value == 0 or self.inner_derivative == 0:
            raise ValueError("point-data chain tasks require nonzero inner data")
        _require_small_result(
            self.outer_power
            * self.inner_value ** (self.outer_power - 1)
            * self.inner_derivative,
            "point derivative",
        )
        return self


class ChainTableValuesTask(StrictFrozenModel):
    """Compute two derivative values from an accessible local-data table."""

    kind: Literal["chain_table_values"] = "chain_table_values"
    outer_power: int = Field(ge=2, le=4)
    first_point: int = Field(ge=-20, le=20)
    first_inner_value: int = Field(ge=-6, le=6)
    first_inner_derivative: int = Field(ge=-12, le=12)
    second_point: int = Field(ge=-20, le=20)
    second_inner_value: int = Field(ge=-6, le=6)
    second_inner_derivative: int = Field(ge=-12, le=12)

    @model_validator(mode="after")
    def _valid_task(self) -> "ChainTableValuesTask":
        if self.first_point == self.second_point:
            raise ValueError("table points must be distinct")
        if 0 in {
            self.first_inner_value,
            self.first_inner_derivative,
            self.second_inner_value,
            self.second_inner_derivative,
        }:
            raise ValueError("chain table values and derivatives must be nonzero")
        for value, derivative in (
            (self.first_inner_value, self.first_inner_derivative),
            (self.second_inner_value, self.second_inner_derivative),
        ):
            _require_small_result(
                self.outer_power * value ** (self.outer_power - 1) * derivative,
                "tabulated derivative",
            )
        return self


class ChainFactorTupleTask(StrictFrozenModel):
    """Produce the outer factor and inner derivative as an ordered tuple."""

    kind: Literal["chain_factor_tuple"] = "chain_factor_tuple"
    inner_coefficient: int = Field(ge=-10, le=10)
    inner_constant: int = Field(ge=-20, le=20)
    outer_power: int = Field(ge=2, le=8)

    @model_validator(mode="after")
    def _valid_task(self) -> "ChainFactorTupleTask":
        if self.inner_coefficient == 0:
            raise ValueError("the affine inner function must be nonconstant")
        return self


class ChainCorrectionTask(StrictFrozenModel):
    """Correct a task-derived derivative that omits the inner factor."""

    kind: Literal["chain_correction"] = "chain_correction"
    inner_kind: Literal["affine", "quadratic"]
    inner_coefficient: int = Field(ge=-8, le=8)
    inner_constant: int = Field(ge=-20, le=20)
    outer_power: int = Field(ge=2, le=7)

    @model_validator(mode="after")
    def _valid_task(self) -> "ChainCorrectionTask":
        if self.inner_coefficient == 0:
            raise ValueError("the inner function must be nonconstant")
        return self


ChainRuleConstructId = Literal[
    "function.affine_value",
    "function.quadratic_value",
    "function.expression_value",
    "function.table_combination",
    "function.plot_change",
    "function.ordered_values",
    "composition.affine",
    "composition.quadratic_outer",
    "composition.at_point",
    "composition.table_path",
    "composition.paired_orders",
    "composition.plot_sum",
    "chain.affine_power",
    "chain.quadratic_power",
    "chain.at_point",
    "chain.table_values",
    "chain.factor_tuple",
    "chain.correction",
]


ChainRuleMathTask = Annotated[
    Union[
        AffineFunctionValueTask,
        QuadraticFunctionValueTask,
        FunctionExpressionValueTask,
        FunctionTableCombinationTask,
        FunctionPlotChangeTask,
        FunctionOrderedValuesTask,
        AffineCompositionTask,
        QuadraticOuterCompositionTask,
        CompositionAtPointTask,
        CompositionTablePathTask,
        CompositionPairedOrdersTask,
        CompositionPlotSumTask,
        AffinePowerChainTask,
        QuadraticPowerChainTask,
        ChainAtPointTask,
        ChainTableValuesTask,
        ChainFactorTupleTask,
        ChainCorrectionTask,
    ],
    Field(discriminator="kind"),
]


class ChainRuleFamilyBlueprint(StrictFrozenModel):
    """One independently reviewable family in the Chain Rule wave."""

    blueprint_id: str = Field(max_length=96, pattern=_CONTENT_ID_PATTERN)
    revision: int = Field(default=1, ge=1)
    item_id: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    family_id: str = Field(max_length=128, pattern=_CONTENT_ID_PATTERN)
    kc_id: str = Field(pattern=KC_ID_PATTERN)
    construct_id: ChainRuleConstructId
    surface: AssessmentSurface
    allocation_order: int = Field(ge=0)
    difficulty: Literal["foundation", "core", "stretch"] = "core"
    task: ChainRuleMathTask


class ChainRuleBlueprintDocument(StrictFrozenModel):
    """Graph-pinned, unreviewed source for exactly three new Wave 2 KCs."""

    schema_version: Literal[1] = 1
    blueprint_version: str = Field(pattern=_CONTENT_ID_PATTERN)
    output_bank_version: str = Field(pattern=_CONTENT_ID_PATTERN)
    graph_version: int = Field(ge=1)
    authoring_source: str = Field(min_length=1, max_length=128)
    author: str = Field(min_length=1)
    target_kcs: list[str] = Field(min_length=1)
    released_kcs: list[str] = Field(default_factory=list)
    families: list[ChainRuleFamilyBlueprint] = Field(min_length=1)

    @model_validator(mode="after")
    def _identities_are_unambiguous(self) -> "ChainRuleBlueprintDocument":
        for label, values in (
            ("target_kcs", self.target_kcs),
            ("released_kcs", self.released_kcs),
            (
                "blueprint identities",
                [(family.blueprint_id, family.revision) for family in self.families],
            ),
            ("item ids", [family.item_id for family in self.families]),
            ("family ids", [family.family_id for family in self.families]),
            (
                "allocation orders",
                [
                    (family.kc_id, family.surface, family.allocation_order)
                    for family in self.families
                ],
            ),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if not set(self.released_kcs) <= set(self.target_kcs):
            raise ValueError("released_kcs must be a subset of target_kcs")
        if any(family.kc_id not in self.target_kcs for family in self.families):
            raise ValueError("every family KC must occur in target_kcs")
        return self
