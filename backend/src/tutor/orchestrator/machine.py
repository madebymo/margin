"""SessionOrchestrator: the deterministic control plane for one tutoring session.

Phases: INTAKE -> DIAGNOSE -> PLAN -> TEACH -> CAPSTONE -> DONE (or STOPPED).
The machine owns all sequencing; generation and judgment are delegated to
ports (template implementations in Phase 1), scoring to the math verifier,
and belief updates to the learner model service. Routing decisions come from
the pure ``route`` function over the episode envelope.
"""

import logging
import math
import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

from tutor.db.persistence import PersistenceService
from tutor.graph import service as graph_service
from tutor.learner.service import LearnerModelService
from tutor.orchestrator.diagnosis import DiagnosisController, ProbeResult
from tutor.orchestrator.envelope import CheckinOutcome, EpisodeEnvelope, RoutingAction
from tutor.orchestrator.planner import (
    EvaluatorPort,
    InteractionGeneratorPort,
    LessonPlanner,
    PlannedLesson,
)
from tutor.orchestrator.ports import (
    DiagnosticianPort,
    ErrorAnalysis,
    LessonWriterPort,
    TemplateDiagnostician,
    TemplateLessonWriter,
)
from tutor.orchestrator.routing import route
from tutor.schemas.common import ResponseClass
from tutor.schemas.kc import GraphDocument
from tutor.schemas.learner import EvidenceEvent, LearnerProfile
from tutor.schemas.widgets import (
    ClickRegionWidget,
    LiveInputWidget,
    MappingWidget,
    SliderWidget,
    WidgetConfig,
)
from tutor.verify.checker import check_answer, parse_restricted

_CALC_ASSUMED_FLOOR = {"Algebra 1", "Algebra 2", "Precalculus"}
_FEEDBACK_COMPARISON = re.compile(
    r"\s*([A-Za-z][A-Za-z0-9]*)\s*(<=|>=|<|>)\s*(.+?)\s*"
)
_MATH_TOKEN = re.compile(
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z][A-Za-z0-9]*"
)
_PLOT_FUNCTIONS = {"sin", "cos", "tan", "sec", "exp", "log", "ln", "sqrt"}
_SERVER_ONLY_WIDGET_FIELDS = {
    "success_condition",
    "correct_region_ids",
    "correct_pairs",
    "checker",
    "feedback_rules",
}
_SHAPE_COORDINATES = {
    "point": ("x", "y"),
    "rect": ("x", "y", "w", "h"),
    "circle": ("cx", "cy", "r"),
}

logger = logging.getLogger("tutor.orchestrator")


class SessionPhase(StrEnum):
    """Lifecycle phases of a tutoring session."""

    INTAKE = "intake"
    DIAGNOSE = "diagnose"
    PLAN = "plan"
    TEACH = "teach"
    CAPSTONE = "capstone"
    DONE = "done"
    STOPPED = "stopped"


class Interaction(BaseModel):
    """One unit of tutor output for the UI to render."""

    key: str
    kind: Literal["message", "probe", "lesson", "checkin", "capstone"]
    kc_id: str | None = None
    text: str
    prompt_segments: list[dict] | None = None
    widget: dict | None = None


class _Pending(BaseModel):
    """The item currently awaiting a student answer (answer stays server-side)."""

    key: str
    kind: Literal["probe", "checkin", "capstone"]
    kc_id: str
    prompt: str
    expected: str
    checker: str
    hints: list[str]
    hints_given: int = 0


def _score_widget(widget: WidgetConfig, response: dict) -> bool:
    """Authoritative server-side scoring for one widget attempt."""
    if isinstance(widget, SliderWidget):
        try:
            value = float(response.get("value"))
        except (TypeError, ValueError):
            return False
        condition = widget.success_condition
        return abs(value - condition.target) <= condition.tolerance
    if isinstance(widget, ClickRegionWidget):
        selected = response.get("selected")
        if not isinstance(selected, list):
            return False
        return {str(item) for item in selected} == set(widget.correct_region_ids)
    if isinstance(widget, MappingWidget):
        pairs = response.get("pairs")
        if not isinstance(pairs, list):
            return False
        try:
            chosen = {(str(left), str(right)) for left, right in pairs}
        except (TypeError, ValueError):
            return False
        return chosen == {(left, right) for left, right in widget.correct_pairs}
    if isinstance(widget, LiveInputWidget):
        text = response.get("text")
        if not isinstance(text, str) or not text.strip():
            return False
        return check_answer(
            widget.checker.expected,
            text,
            widget.checker.equivalence,
            widget.checker.tolerance,
        )
    return False


def _client_shape(shape: dict) -> dict:
    """Project free-form geometry onto the documented light-field contract."""
    shape_type = shape.get("type")
    coordinates = _SHAPE_COORDINATES.get(shape_type)
    if coordinates is None:
        return {}
    projected = {"type": shape_type}
    for name in coordinates:
        value = shape.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return {}
        if isinstance(value, (int, float)):
            try:
                finite_value = math.isfinite(float(value))
            except OverflowError:
                return {}
            if not finite_value:
                return {}
        projected[name] = value
    return projected


def _client_render(render: dict) -> dict:
    """Project live-input render data without forwarding invented nested keys."""
    if not render:
        return {}
    plot = render.get("plot")
    variable = render.get("var")
    if not isinstance(plot, str) or not isinstance(variable, str):
        return {}
    return {"plot": plot, "var": variable}


def _widget_for_client(widget: WidgetConfig) -> dict:
    """Serialize one widget through the explicit client-safe projection."""
    client = widget.model_dump(exclude=_SERVER_ONLY_WIDGET_FIELDS)
    if isinstance(widget, ClickRegionWidget):
        for client_region, server_region in zip(
            client["regions"], widget.regions, strict=True
        ):
            client_region["shape"] = _client_shape(server_region.shape)
    elif isinstance(widget, LiveInputWidget):
        client["render"] = _client_render(widget.render)
    return client


def _infer_slider_parameter(plot: str | None) -> str:
    """Return the sole non-``x`` symbol in a validated slider plot."""
    if plot is None:
        raise ValueError("params.plot is missing")
    left, separator, right = plot.partition("=")
    if separator != "=" or "=" in right or left.strip() != "y":
        raise ValueError("params.plot must be one equation of the form 'y = <expression>'")
    expression = parse_restricted(plot)
    if getattr(expression, "free_symbols", None) is None:
        raise ValueError("params.plot must contain one math expression")
    identifiers = [
        match
        for match in _MATH_TOKEN.finditer(right)
        if match.group()[0].isalpha()
    ]
    for match in identifiers:
        tail = right[match.end() :].lstrip()
        if tail.startswith("(") and match.group() not in _PLOT_FUNCTIONS:
            raise ValueError(f"params.plot contains unknown function {match.group()!r}")
    symbols = {
        match.group()
        for match in identifiers
        if match.group() not in _PLOT_FUNCTIONS | {"pi"}
    }
    if "x" not in symbols or len(symbols) != 2:
        raise ValueError("params.plot must contain exactly x and one slider parameter")
    parameter = next(symbol for symbol in symbols if symbol != "x")
    if parameter == "y":
        raise ValueError("params.plot cannot use y as the slider parameter")
    return parameter


def _parse_feedback_condition(condition: str, parameter: str) -> tuple[str, float]:
    """Parse one safe ``parameter <op> exact-number`` comparison."""
    match = _FEEDBACK_COMPARISON.fullmatch(condition)
    if match is None:
        raise ValueError("expected one comparison using <, <=, >, or >=")
    left, comparison, raw_threshold = match.groups()
    if left != parameter:
        raise ValueError(
            f"left identifier {left!r} does not match slider parameter {parameter!r}"
        )
    if "=" in raw_threshold:
        raise ValueError("threshold must be one math expression, not an assignment")

    threshold_expression = parse_restricted(raw_threshold)
    free_symbols = getattr(threshold_expression, "free_symbols", None)
    if free_symbols is None:
        raise ValueError("threshold must be one math expression")
    if free_symbols:
        raise ValueError("threshold must not contain free symbols")
    if getattr(threshold_expression, "is_finite", None) is not True:
        raise ValueError("threshold must be finite")
    if getattr(threshold_expression, "is_real", None) is not True:
        raise ValueError("threshold must be real")
    try:
        threshold = float(threshold_expression.evalf())
    except Exception as exc:  # noqa: BLE001 — malformed model output is ignored
        raise ValueError("threshold must resolve to a finite real number") from exc
    if not math.isfinite(threshold):
        raise ValueError("threshold must resolve to a finite real number")
    return comparison, threshold


def _matching_slider_feedback(widget: SliderWidget, response: dict) -> str | None:
    """Return the first matching server-only feedback message, if any."""
    if not widget.feedback_rules:
        return None
    try:
        value = float(response.get("value"))
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value):
        return None

    try:
        parameter = _infer_slider_parameter(widget.params.plot)
    except ValueError as exc:
        logger.warning("ignoring slider feedback rules: %s", exc)
        return None

    for rule in widget.feedback_rules:
        try:
            comparison, threshold = _parse_feedback_condition(rule.when, parameter)
        except ValueError as exc:
            logger.warning("ignoring slider feedback rule %r: %s", rule.when, exc)
            continue
        matches = (
            (comparison == "<" and value < threshold)
            or (comparison == "<=" and value <= threshold)
            or (comparison == ">" and value > threshold)
            or (comparison == ">=" and value >= threshold)
        )
        if matches:
            return rule.say
    return None


class SessionOrchestrator:
    """Drives one session from intake to capstone."""

    def __init__(
        self,
        graph: GraphDocument,
        target_kc: str,
        profile: LearnerProfile,
        diagnostician: DiagnosticianPort | None = None,
        lesson_writer: LessonWriterPort | None = None,
        interaction_generator: InteractionGeneratorPort | None = None,
        evaluator: EvaluatorPort | None = None,
        persistence: PersistenceService | None = None,
        probe_budget: int = 8,
        exit_consecutive: int = 2,
        interaction_budget: int = 40,
    ) -> None:
        self._graph = graph
        self._nodes = {node.id: node for node in graph.nodes}
        if target_kc not in self._nodes:
            raise KeyError(f"unknown kc: {target_kc}")
        self._target = target_kc
        self._profile = profile
        floor = _CALC_ASSUMED_FLOOR if "calc" in profile.course.lower() else set()
        self.learner = LearnerModelService(graph, assumed_floor_levels=floor)
        self._diagnostician = diagnostician or TemplateDiagnostician()
        self._writer = lesson_writer or TemplateLessonWriter()
        self._planner = LessonPlanner(
            writer=self._writer,
            generator=interaction_generator,
            evaluator=evaluator,
        )
        self._planned_lessons: dict[str, PlannedLesson] = {}
        self._active_widgets: dict[str, tuple[str, WidgetConfig]] = {}
        self._widget_attempts: dict[str, int] = {}
        self._diag = DiagnosisController(graph, target_kc, self.learner, probe_budget)
        self._exit_consecutive = exit_consecutive
        self.envelope = EpisodeEnvelope(
            target_kc=target_kc, interaction_budget=interaction_budget
        )
        self.phase = SessionPhase.INTAKE
        self.path: list[str] = []
        self._path_pos = 0
        self._consecutive = 0
        self._checkin_attempts: dict[str, int] = {}
        self._pending: _Pending | None = None
        self._counter = 0
        self._capstone_attempts = 0
        self._mastered_in_session: list[str] = []
        self._fallback_kcs: list[str] = []
        self._frontier_at_diagnosis: list[str] = []
        self._persistence = persistence
        self._episode_id: int | None = None
        if persistence is not None:
            try:
                persistence.ensure_learner(self.learner.learner_id, profile)
                self._episode_id = persistence.start_episode(
                    self.learner.learner_id, target_kc, self.envelope.model_dump()
                )
            except Exception as exc:  # noqa: BLE001 — persistence never blocks a session
                logger.warning("persistence disabled for this session: %s", exc)
                self._persistence = None

    # -- small helpers ---------------------------------------------------------

    def _next_key(self) -> str:
        self._counter += 1
        return f"i{self._counter:03d}"

    def _msg(self, text: str) -> Interaction:
        return Interaction(key=self._next_key(), kind="message", text=text)

    @property
    def pending_kind(self) -> str | None:
        """Kind of the item awaiting an answer (for UIs/tests)."""
        return self._pending.kind if self._pending else None

    @property
    def pending_kc(self) -> str | None:
        """KC of the item awaiting an answer (for UIs/tests)."""
        return self._pending.kc_id if self._pending else None

    @property
    def pending_expected(self) -> str | None:
        """Hidden expected answer (for tooling, tests, and 'reveal')."""
        return self._pending.expected if self._pending else None

    def hint(self) -> str | None:
        """Serve the next rung of the hint ladder; marks the response assisted."""
        if self._pending is None:
            return None
        if self._pending.hints_given >= len(self._pending.hints):
            return None
        text = self._pending.hints[self._pending.hints_given]
        self._pending.hints_given += 1
        return text

    def answer_widget(self, key: str, response: dict) -> tuple[bool, str]:
        """Score a widget attempt (authoritative, server-side).

        Widget practice is formative: it never advances the state machine or
        consumes routing budget. Every attempt is retained so the trajectory is
        auditable; v2 learner models exclude widget events from mastery.
        """
        if self.phase in (SessionPhase.DONE, SessionPhase.STOPPED):
            raise RuntimeError("session is over")
        if key not in self._active_widgets:
            raise KeyError(f"unknown widget: {key}")
        kc, widget = self._active_widgets[key]
        correct = _score_widget(widget, response)
        attempts = self._widget_attempts.get(key, 0)
        self._widget_attempts[key] = attempts + 1
        self._apply_event(
            EvidenceEvent(
                event_id=uuid4(),
                learner_id=self.learner.learner_id,
                t=datetime.now(timezone.utc),
                item_id=key,
                kc_ids=[kc],
                correct=correct,
                response_class=ResponseClass.WIDGET,
                surface="guided_widget",
                attempt_number=attempts + 1,
                content_versions={
                    "graph": str(self._graph.graph_version),
                    "generator": "template-v1",
                },
            )
        )
        message = (
            "Nice — that's it."
            if correct
            else "Not yet — adjust your answer and try again."
        )
        if not correct and isinstance(widget, SliderWidget):
            feedback = _matching_slider_feedback(widget, response)
            if feedback is not None:
                message = f"{message} {feedback}"
        return correct, message

    def _ancestors_of(self, kc: str) -> set[str]:
        return (
            graph_service.ancestor_subgraph(self._graph, kc, hard_only=True).node_ids()
            - {kc}
        )

    # -- session flow ------------------------------------------------------------

    def begin(self) -> list[Interaction]:
        """Start the session: welcome plus the first diagnostic probe."""
        if self.phase != SessionPhase.INTAKE:
            raise RuntimeError("session already started")
        self.phase = SessionPhase.DIAGNOSE
        target_name = self._nodes[self._target].name
        opener = self._msg(
            f"Let's find the best starting point for {target_name}. "
            "I'll ask a few quick questions. Use the dedicated hint control if needed."
        )
        interactions = [opener, *self._issue_next_probe()]
        self._checkpoint()
        return interactions

    def submit(self, answer: str) -> list[Interaction]:
        """Score the pending item and advance the state machine."""
        if self.phase in (SessionPhase.DONE, SessionPhase.STOPPED):
            raise RuntimeError("session is over")
        if self._pending is None:
            raise RuntimeError("no pending item to answer")
        pending = self._pending
        self._pending = None
        correct = check_answer(pending.expected, answer, pending.checker)
        analysis = ErrorAnalysis()
        if not correct and pending.kind in ("probe", "checkin"):
            analysis = self._diagnostician.analyze_error(
                self._nodes[pending.kc_id], pending.prompt, pending.expected, answer
            )
        self._record_event(pending, correct, analysis.misconception_id)
        if pending.kind == "probe":
            interactions = self._after_probe(pending, correct, analysis)
        elif pending.kind == "checkin":
            interactions = self._after_checkin(pending, correct, analysis)
        else:
            interactions = self._after_capstone(pending, correct)
        self._checkpoint()
        return interactions

    def _record_event(
        self, pending: _Pending, correct: bool, misconception_id: str | None = None
    ) -> None:
        event = EvidenceEvent(
            event_id=uuid4(),
            learner_id=self.learner.learner_id,
            t=datetime.now(timezone.utc),
            item_id=pending.key,
            kc_ids=[pending.kc_id],
            correct=correct,
            response_class=ResponseClass.SYMBOLIC_ENTRY,
            hints_used=pending.hints_given,
            assisted=pending.hints_given > 0,
            misconception_id=misconception_id,
            content_versions={
                "graph": str(self._graph.graph_version),
                "generator": "template-v1",
            },
        )
        self._apply_event(event)

    def _apply_event(self, event: EvidenceEvent) -> None:
        """Update the learner model and durably append the event when enabled."""
        self.learner.apply_event(event)
        if self._persistence is None:
            return
        try:
            self._persistence.record_event(event)
        except Exception as exc:  # noqa: BLE001 — persistence never blocks a session
            logger.warning("event persistence failed; disabling: %s", exc)
            self._persistence = None

    def _checkpoint(self) -> None:
        """Persist episode phase + envelope; derived mastery at terminal phases."""
        if self._persistence is None or self._episode_id is None:
            return
        try:
            self._persistence.update_episode(
                self._episode_id, self.phase.value, self.envelope.model_dump()
            )
            if self.phase in (SessionPhase.DONE, SessionPhase.STOPPED):
                self._persistence.save_derived(self.learner.snapshot())
        except Exception as exc:  # noqa: BLE001 — persistence never blocks a session
            logger.warning("checkpoint persistence failed; disabling: %s", exc)
            self._persistence = None

    # -- diagnosis ---------------------------------------------------------------

    def _issue_next_probe(self) -> list[Interaction]:
        kc = self._diag.next_probe_kc()
        if kc is None:
            return self._finish_diagnosis()
        node = self._nodes[kc]
        probe = self._diagnostician.generate_probe(node)
        rendered = "\n".join(
            "____" if index == probe.blank_index else step
            for index, step in enumerate(probe.scaffold_steps)
        )
        text = f"Fill in the blank:\n{rendered}"
        self._pending = _Pending(
            key=self._next_key(),
            kind="probe",
            kc_id=kc,
            prompt=text,
            expected=probe.expected,
            checker=probe.checker,
            hints=list(probe.hint_ladder),
        )
        return [Interaction(key=self._pending.key, kind="probe", kc_id=kc, text=text)]

    def _after_probe(
        self, pending: _Pending, correct: bool, analysis: ErrorAnalysis
    ) -> list[Interaction]:
        implicated = None
        candidate = analysis.implicated_prereq
        if not correct and candidate in self._nodes and candidate != pending.kc_id:
            implicated = candidate
        self._diag.record_result(
            ProbeResult(kc_id=pending.kc_id, correct=correct, implicated_prereq=implicated)
        )
        note = (
            "Nice — that's right."
            if correct
            else "No problem — that tells me where to look."
        )
        return [self._msg(note), *self._issue_next_probe()]

    def _finish_diagnosis(self) -> list[Interaction]:
        self.phase = SessionPhase.PLAN
        frontier = self._diag.frontier()
        self._frontier_at_diagnosis = list(frontier)
        self.path = self._diag.plan_path()
        target_name = self._nodes[self._target].name
        messages: list[Interaction] = []
        if frontier:
            names = ", ".join(self._nodes[kc].name for kc in frontier)
            messages.append(
                self._msg(
                    f"Diagnosis done. The gap starts at: {names}. "
                    f"We'll build from there up to {target_name}."
                )
            )
        else:
            messages.append(
                self._msg(f"Diagnosis done — no gaps found on the way to {target_name}.")
            )
        if not self.path:
            return [*messages, *self._start_capstone()]
        self.phase = SessionPhase.TEACH
        self._path_pos = 0
        return [*messages, *self._issue_lesson(self.path[0])]

    # -- teach loop ----------------------------------------------------------------

    def _issue_lesson(self, kc: str) -> list[Interaction]:
        self._consecutive = 0
        node = self._nodes[kc]
        planned = self._planned_lessons.get(kc)
        if planned is None:
            planned = self._planner.plan_lesson(node)
            self._planned_lessons[kc] = planned
        client_widget = (
            _widget_for_client(planned.widget) if planned.widget is not None else None
        )
        lesson = Interaction(
            key=self._next_key(),
            kind="lesson",
            kc_id=kc,
            text=planned.narrative,
            widget=client_widget,
        )
        if planned.widget is not None:
            self._active_widgets[lesson.key] = (kc, planned.widget)
        return [lesson, *self._issue_checkin(kc)]

    def _issue_checkin(self, kc: str) -> list[Interaction]:
        attempt = self._checkin_attempts.get(kc, 0)
        self._checkin_attempts[kc] = attempt + 1
        item = self._writer.checkin_item(self._nodes[kc], attempt)
        self._pending = _Pending(
            key=self._next_key(),
            kind="checkin",
            kc_id=kc,
            prompt=item.prompt,
            expected=item.expected,
            checker=item.checker,
            hints=list(item.hints),
        )
        return [
            Interaction(key=self._pending.key, kind="checkin", kc_id=kc, text=item.prompt)
        ]

    def _after_checkin(
        self, pending: _Pending, correct: bool, analysis: ErrorAnalysis
    ) -> list[Interaction]:
        kc = pending.kc_id
        self._consecutive = self._consecutive + 1 if correct else 0
        ancestors = self._ancestors_of(kc)
        implicated = None
        if not correct and analysis.implicated_prereq in ancestors:
            implicated = analysis.implicated_prereq
        outcome = CheckinOutcome(
            kc_id=kc,
            correct=correct,
            interaction_key=pending.key,
            consecutive_correct=self._consecutive,
            implicated_prereq=implicated,
        )
        decision, self.envelope = route(
            self.envelope, outcome, ancestors, self._exit_consecutive
        )
        action = decision.action
        if action == RoutingAction.DUPLICATE:
            return [self._msg("Already handled that one.")]
        if action == RoutingAction.STOP:
            return self._stop(
                "We've used our practice budget for today. Rest up — and consider "
                "walking the tricky spots through with a teacher. Great effort!"
            )
        if action == RoutingAction.CONTINUE:
            return [
                self._msg("Correct! One more to make it stick."),
                *self._issue_checkin(kc),
            ]
        if action == RoutingAction.RETRY:
            node = self._nodes[kc]
            return [
                self._msg(f"Not quite. Remember: {node.description}"),
                *self._issue_checkin(kc),
            ]
        if action == RoutingAction.DESCEND and decision.descend_to is not None:
            prereq = decision.descend_to
            return [
                self._msg(
                    f"That miss points at a building block: {self._nodes[prereq].name}. "
                    "Let's shore that up first, then come back."
                ),
                *self._issue_lesson(prereq),
            ]
        if action == RoutingAction.FALLBACK:
            self._fallback_kcs.append(kc)
            worked = self._nodes[kc].canonical_examples[0]
            return [
                self._msg("Let's study it worked out, then keep moving:"),
                self._msg(worked),
                *self._advance_from(kc),
            ]
        self._mastered_in_session.append(kc)
        return [
            self._msg(f"{self._nodes[kc].name}: locked in."),
            *self._advance_from(kc),
        ]

    def _advance_from(self, kc: str) -> list[Interaction]:
        if self.envelope.resume_stack:
            env = self.envelope.model_copy(deep=True)
            resume = env.resume_stack.pop()
            self.envelope = env
            return [
                self._msg(f"Back to {self._nodes[resume].name}."),
                *self._issue_lesson(resume),
            ]
        if self._path_pos < len(self.path) and self.path[self._path_pos] == kc:
            self._path_pos += 1
        elif kc in self.path:
            self._path_pos = self.path.index(kc) + 1
        while self._path_pos < len(self.path) and self.learner.is_mastered(
            self.path[self._path_pos]
        ):
            self._path_pos += 1
        if self._path_pos < len(self.path):
            return self._issue_lesson(self.path[self._path_pos])
        return self._start_capstone()

    # -- capstone -------------------------------------------------------------------

    def _start_capstone(self) -> list[Interaction]:
        self.phase = SessionPhase.CAPSTONE
        node = self._nodes[self._target]
        probe = self._diagnostician.generate_probe(node)
        rendered = "\n".join(
            "____" if index == probe.blank_index else step
            for index, step in enumerate(probe.scaffold_steps)
        )
        capstone_text = f"Goal problem — no scaffolding:\n{rendered}"
        self._pending = _Pending(
            key=self._next_key(),
            kind="capstone",
            kc_id=self._target,
            prompt=capstone_text,
            expected=probe.expected,
            checker=probe.checker,
            hints=list(probe.hint_ladder),
        )
        return [
            self._msg("You've got all the pieces — time to close the loop."),
            Interaction(
                key=self._pending.key,
                kind="capstone",
                kc_id=self._target,
                text=capstone_text,
            ),
        ]

    def _after_capstone(self, pending: _Pending, correct: bool) -> list[Interaction]:
        if correct:
            self.phase = SessionPhase.DONE
            name = self._nodes[self._target].name
            return [
                self._msg(
                    f"That's it — you just solved {name} on your own. Session complete."
                )
            ]
        self._capstone_attempts += 1
        if self._capstone_attempts < 2:
            self._pending = _Pending(
                key=self._next_key(),
                kind="capstone",
                kc_id=pending.kc_id,
                prompt=pending.prompt,
                expected=pending.expected,
                checker=pending.checker,
                hints=pending.hints,
            )
            return [
                self._msg(f"Close — try once more. {pending.hints[0]}"),
                Interaction(
                    key=self._pending.key,
                    kind="capstone",
                    kc_id=pending.kc_id,
                    text="One more attempt:",
                ),
            ]
        return self._stop(
            "So close. Review today's lessons and try again soon — "
            "or walk this one through with a teacher."
        )

    def _stop(self, text: str) -> list[Interaction]:
        self.phase = SessionPhase.STOPPED
        self._pending = None
        return [self._msg(text)]

    # -- reporting --------------------------------------------------------------------

    def summary(self) -> dict:
        """Machine-readable session summary."""
        return {
            "phase": self.phase.value,
            "target": self._target,
            "probes_used": self._diag.probes_issued,
            "frontier": list(self._frontier_at_diagnosis),
            "remaining_gaps": self._diag.frontier(),
            "path": list(self.path),
            "mastered_in_session": list(self._mastered_in_session),
            "fallback_kcs": list(self._fallback_kcs),
            "interactions_used": self.envelope.interactions_used,
            "events_recorded": len(self.learner.events),
        }
