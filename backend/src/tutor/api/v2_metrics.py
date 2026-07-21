"""Privacy-safe fleet metrics contract for the trustworthy-session runtime.

The in-process store remains the source for lightweight health snapshots in
tests and single-process development. Deployments may additionally inject a
``MetricsSink`` to export the same counters to a fleet-wide metrics backend.
Only low-cardinality release versions and reviewed stable item ids are ever
provided as dimensions; session ids, learner ids, prompts, answers, and free
text have no representation in this contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class V2MetricDimensions:
    """The immutable release coordinates attached to every exported metric."""

    graph_version: str
    item_bank_version: str
    pedagogy_catalog_version: str
    policy_versions: tuple[tuple[str, str], ...]
    learner_parameter_version: str
    capability_manifest_version: str

    def as_labels(self) -> dict[str, str]:
        labels = {
            "graph_version": self.graph_version,
            "item_bank_version": self.item_bank_version,
            "pedagogy_catalog_version": self.pedagogy_catalog_version,
            "learner_parameter_version": self.learner_parameter_version,
            "capability_manifest_version": self.capability_manifest_version,
        }
        labels.update(
            {
                f"policy_{name}_version": version
                for name, version in self.policy_versions
            }
        )
        return labels


@runtime_checkable
class MetricsSink(Protocol):
    """Injectable adapter for a fleet-wide counter backend."""

    def increment(
        self,
        name: str,
        amount: int = 1,
        *,
        dimensions: Mapping[str, str],
    ) -> None:
        """Increment one counter using only the supplied safe dimensions."""
