"""Closed registry for deterministic, typed assessment-task compilers.

Registrations bind a discriminated source task to one construct, one KC, and
one deterministic compiler. The registry never accepts source-authored truth
strings and refuses ambiguous or mismatched registrations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel


class TaskCompilerRegistryError(ValueError):
    """A compiler registration or lookup violates the closed registry."""


@dataclass(frozen=True)
class TaskCompilerRegistration:
    """One typed task compiler and its release taxonomy."""

    kind: str
    task_type: type[BaseModel]
    construct_id: str
    kc_id: str
    compile: Callable[[BaseModel], Any]


class TaskCompilerRegistry:
    """Immutable lookup for reviewed deterministic task constructors."""

    def __init__(self, registrations: Iterable[TaskCompilerRegistration]) -> None:
        by_kind: dict[str, TaskCompilerRegistration] = {}
        by_type: dict[type[BaseModel], TaskCompilerRegistration] = {}
        for registration in registrations:
            if not registration.kind.strip():
                raise TaskCompilerRegistryError("task compiler kind must be nonblank")
            if not registration.construct_id.strip():
                raise TaskCompilerRegistryError(
                    "task compiler construct_id must be nonblank"
                )
            if not registration.kc_id.strip():
                raise TaskCompilerRegistryError("task compiler kc_id must be nonblank")
            if registration.kind in by_kind:
                raise TaskCompilerRegistryError(
                    f"duplicate task compiler kind {registration.kind!r}"
                )
            if registration.task_type in by_type:
                raise TaskCompilerRegistryError(
                    "duplicate task compiler type "
                    f"{registration.task_type.__name__!r}"
                )
            by_kind[registration.kind] = registration
            by_type[registration.task_type] = registration
        if not by_kind:
            raise TaskCompilerRegistryError("a task compiler registry cannot be empty")
        self._by_kind = MappingProxyType(by_kind)
        self._by_type = MappingProxyType(by_type)

    def resolve(self, task: BaseModel) -> TaskCompilerRegistration:
        """Return the exact registration for a validated task instance."""
        kind = getattr(task, "kind", None)
        if not isinstance(kind, str):
            raise TaskCompilerRegistryError("compiled tasks require a string kind")
        registration = self._by_kind.get(kind)
        if registration is None:
            raise TaskCompilerRegistryError(f"unsupported task kind {kind!r}")
        if type(task) is not registration.task_type:
            raise TaskCompilerRegistryError(
                f"task kind {kind!r} requires {registration.task_type.__name__}, "
                f"not {type(task).__name__}"
            )
        return registration

    def compile(self, task: BaseModel) -> Any:
        """Compile a task only through its exact registered constructor."""
        return self.resolve(task).compile(task)

    def validate_taxonomy(
        self,
        task: BaseModel,
        *,
        construct_id: str,
        kc_id: str,
    ) -> None:
        """Reject author metadata that disagrees with the typed constructor."""
        registration = self.resolve(task)
        if registration.construct_id != construct_id:
            raise TaskCompilerRegistryError(
                f"task kind {registration.kind!r} belongs to construct "
                f"{registration.construct_id!r}, not {construct_id!r}"
            )
        if registration.kc_id != kc_id:
            raise TaskCompilerRegistryError(
                f"task kind {registration.kind!r} belongs to KC "
                f"{registration.kc_id!r}, not {kc_id!r}"
            )

    @property
    def registrations(self) -> tuple[TaskCompilerRegistration, ...]:
        """Return registrations in deterministic kind order."""
        return tuple(self._by_kind[kind] for kind in sorted(self._by_kind))
