"""Lesson planning: narrative + interactive element + evaluation gates.

Implements the plan's content pipeline at the orchestrator level:
lesson writer -> interaction generator (2-3 candidates) -> deterministic hard
gates -> evaluator verdict -> repair loop (bounded) -> static worked-example
fallback. The fallback is mandatory: a bad generator or evaluator can never
block a session.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from tutor.orchestrator.ports import (
    LessonWriterPort,
    TemplateLessonWriter,
    item_from_example,
)
from tutor.schemas.kc import KCNode
from tutor.schemas.widgets import LiveInputWidget, MappingWidget, WidgetConfig
from tutor.verify.checker import MathVerificationError, parse_restricted


class EvaluationVerdict(BaseModel):
    """Evaluator outcome for one widget candidate."""

    accepted: bool
    feedback: str = ""


@runtime_checkable
class InteractionGeneratorPort(Protocol):
    """Produces widget-config candidates for one KC's mini-lesson."""

    def candidates(
        self, node: KCNode, attempt: int, feedback: list[str]
    ) -> list[WidgetConfig]:
        """Return up to 3 candidates; ``feedback`` carries prior rejections."""
        ...


@runtime_checkable
class EvaluatorPort(Protocol):
    """Judges one widget candidate against its lesson."""

    def evaluate(
        self, node: KCNode, narrative: str, widget: WidgetConfig
    ) -> EvaluationVerdict:
        """Return accept/reject with feedback for the repair loop."""
        ...


def deterministic_gates(widget: WidgetConfig) -> list[str]:
    """Hard gates that need no LLM: math parseability and answer leakage."""
    problems: list[str] = []
    if isinstance(widget, LiveInputWidget):
        try:
            parse_restricted(widget.checker.expected)
        except MathVerificationError as exc:
            problems.append(f"expected answer is not safely parseable: {exc}")
        expected = widget.checker.expected.strip()
        if len(expected) >= 3 and expected in widget.prompt:
            problems.append("expected answer leaks into the widget prompt")
    return problems


class TemplateInteractionGenerator:
    """Deterministic candidates built from canonical examples (no LLM)."""

    def candidates(
        self, node: KCNode, attempt: int, feedback: list[str]
    ) -> list[WidgetConfig]:
        """A live-input drill, plus a mapping widget when examples allow."""
        results: list[WidgetConfig] = []
        example = node.canonical_examples[attempt % len(node.canonical_examples)]
        prompt, expected = item_from_example(example)
        results.append(
            LiveInputWidget(
                learning_objective=f"Practice {node.name.lower()}",
                prompt=f"Try it: {prompt}",
                input_kind="expression",
                checker={"equivalence": "sympy_equiv", "expected": expected},
            )
        )
        if len(node.canonical_examples) >= 2:
            pairs = [item_from_example(e) for e in node.canonical_examples[:3]]
            left = [p for p, _ in pairs]
            right = [e for _, e in pairs]
            if len(set(left)) == len(left) and len(set(right)) == len(right):
                results.append(
                    MappingWidget(
                        learning_objective=f"Match outcomes for {node.name.lower()}",
                        prompt="Match each expression to its result.",
                        left=left,
                        right=right,
                        correct_pairs=[(p, e) for p, e in pairs],
                    )
                )
        return results


class TemplateEvaluator:
    """Gates-only evaluator used when no LLM judge is wired."""

    def evaluate(
        self, node: KCNode, narrative: str, widget: WidgetConfig
    ) -> EvaluationVerdict:
        """Deterministic gates already ran in the planner; accept."""
        return EvaluationVerdict(accepted=True)


@dataclass
class PlannedLesson:
    """The planner's output for one KC: narrative plus optional interaction."""

    kc_id: str
    narrative: str
    widget: WidgetConfig | None
    fallback_used: bool
    evaluator_feedback: list[str] = field(default_factory=list)


class LessonPlanner:
    """Composes writer + interaction generator + evaluator with a repair loop."""

    def __init__(
        self,
        writer: LessonWriterPort | None = None,
        generator: InteractionGeneratorPort | None = None,
        evaluator: EvaluatorPort | None = None,
        max_iterations: int = 3,
    ) -> None:
        self._writer = writer or TemplateLessonWriter()
        self._generator = generator or TemplateInteractionGenerator()
        self._evaluator = evaluator or TemplateEvaluator()
        self._max_iterations = max_iterations

    def plan_lesson(self, node: KCNode) -> PlannedLesson:
        """Produce a gated lesson; fall back to a worked example, never block."""
        narrative = self._writer.lesson_text(node)
        feedback: list[str] = []
        for attempt in range(self._max_iterations):
            try:
                candidates = self._generator.candidates(node, attempt, list(feedback))
            except Exception as exc:  # noqa: BLE001 — generator must never block
                feedback.append(f"generator error: {exc}")
                candidates = []
            for candidate in candidates:
                problems = deterministic_gates(candidate)
                if problems:
                    feedback.extend(problems)
                    continue
                verdict = self._evaluator.evaluate(node, narrative, candidate)
                if verdict.accepted:
                    return PlannedLesson(
                        kc_id=node.id,
                        narrative=narrative,
                        widget=candidate,
                        fallback_used=False,
                        evaluator_feedback=feedback,
                    )
                if verdict.feedback:
                    feedback.append(verdict.feedback)
        worked = "\n".join(f"- {example}" for example in node.canonical_examples)
        return PlannedLesson(
            kc_id=node.id,
            narrative=f"{narrative}\n\nStudy this worked example instead:\n{worked}",
            widget=None,
            fallback_used=True,
            evaluator_feedback=feedback,
        )
