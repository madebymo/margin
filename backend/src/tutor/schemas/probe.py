"""Diagnostic probe: a scaffolded worked problem with one blank tagged to a KC."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from tutor.schemas.kc import KC_ID_PATTERN


class DiagnosticProbe(BaseModel):
    """One probe item used during adaptive diagnosis.

    The student sees all scaffold steps with the blank step redacted; filling
    the blank is direct evidence for the tagged KC.
    """

    probe_id: str = Field(min_length=1)
    kc_id: str = Field(pattern=KC_ID_PATTERN)
    scaffold_steps: list[str] = Field(min_length=2)
    blank_index: int = Field(ge=0)
    expected: str = Field(min_length=1)
    checker: Literal["sympy_equiv", "numeric"]
    hint_ladder: list[str] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def _blank_in_range(self) -> "DiagnosticProbe":
        if self.blank_index >= len(self.scaffold_steps):
            raise ValueError(
                f"blank_index {self.blank_index} out of range for "
                f"{len(self.scaffold_steps)} scaffold steps"
            )
        return self
