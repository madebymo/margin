"""Ports for the LLM call sites, plus deterministic Phase 1 implementations.

The template implementations generate probes and check-ins from KC canonical
examples with no LLM involved, so the whole control plane runs and is testable
offline. Real diagnostician / lesson-writer adapters replace these in later
phases behind the same Protocols.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from tutor.schemas.kc import KCNode
from tutor.schemas.probe import DiagnosticProbe


class PracticeItem(BaseModel):
    """A displayable practice item with its hidden answer and hint ladder."""

    prompt: str
    expected: str
    checker: str = "sympy_equiv"
    hints: list[str]


def item_from_example(example: str) -> tuple[str, str]:
    """Turn a canonical example into (prompt, expected answer).

    Splits on the rightmost separator so "3x + 5 = 17 gives x = 4" becomes
    the prompt "3x + 5 = 17 gives x = ?" with expected answer "4".
    """
    for separator in (" = ", " is "):
        if separator in example:
            prompt, expected = example.rsplit(separator, 1)
            return f"{prompt}{separator}?", expected.strip()
    return f"In your own words, restate: {example}", example


def hints_for(node: KCNode) -> list[str]:
    """Generic three-step hint ladder: nudge, targeted hint, worked reference."""
    return [
        f"Think about {node.name.lower()}.",
        node.description,
        f"It works like this example: {node.canonical_examples[-1]}",
    ]


@runtime_checkable
class DiagnosticianPort(Protocol):
    """Generates diagnostic probes and analyzes wrong answers."""

    def generate_probe(self, node: KCNode) -> DiagnosticProbe:
        """Build a probe whose blank is direct evidence for this KC."""
        ...

    def analyze_error(self, kc_id: str) -> str | None:
        """Return the kc id of an implicated prerequisite, or None."""
        ...


@runtime_checkable
class LessonWriterPort(Protocol):
    """Writes lesson narrative and check-in items for one KC."""

    def lesson_text(self, node: KCNode) -> str:
        """Return the mini-lesson narrative."""
        ...

    def checkin_item(self, node: KCNode, attempt: int) -> PracticeItem:
        """Return the check-in variation for the given attempt index."""
        ...


class TemplateDiagnostician:
    """Deterministic, LLM-free diagnostician built from canonical examples."""

    def generate_probe(self, node: KCNode) -> DiagnosticProbe:
        """Probe = the KC's first canonical example with its answer blanked."""
        prompt, expected = item_from_example(node.canonical_examples[0])
        return DiagnosticProbe(
            probe_id=f"probe.{node.id}",
            kc_id=node.id,
            scaffold_steps=[prompt, "____"],
            blank_index=1,
            expected=expected,
            checker="sympy_equiv",
            hint_ladder=hints_for(node),
        )

    def analyze_error(self, kc_id: str) -> str | None:
        """No misconception library yet: defer to binary search."""
        return None


class TemplateLessonWriter:
    """Deterministic, LLM-free lesson writer built from node metadata."""

    def lesson_text(self, node: KCNode) -> str:
        """Narrative = description plus worked examples."""
        examples = "\n".join(f"  - {example}" for example in node.canonical_examples)
        return f"{node.name}\n\n{node.description}\n\nWorked examples:\n{examples}"

    def checkin_item(self, node: KCNode, attempt: int) -> PracticeItem:
        """Cycle through canonical examples as near-transfer variations."""
        example = node.canonical_examples[attempt % len(node.canonical_examples)]
        prompt, expected = item_from_example(example)
        return PracticeItem(
            prompt=f"Check-in: {prompt}",
            expected=expected,
            checker="sympy_equiv",
            hints=hints_for(node),
        )
