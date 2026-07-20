"""Trusted item-bank tutoring session used by API v2.

This control plane intentionally stays independent of the legacy template/LLM
ports.  Every scored interaction is allocated from the pinned reviewed bank,
and expected answers remain inside the pending server state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from tutor.content.exposure import AllocationError, ItemAllocator
from tutor.content.item_bank import (
    bundle_leakage_problems,
    load_item_bank,
    validate_item_bank,
)
from tutor.content.visible import extend_visible_texts
from tutor.graph import service as graph_service
from tutor.learner.params import BKTParams, DEFAULT_PARAMS_V2
from tutor.learner.service_v2 import LearnerModelServiceV2
from tutor.orchestrator.diagnosis_v2 import (
    DIAGNOSIS_POLICY_VERSION,
    PINNED_IMPACT_DECAY,
    PINNED_IMPACT_LAMBDA,
    DiagnosticObservation,
    DiagnosisControllerV2,
    DiagnosisState,
    LearningPlanStep,
)
from tutor.orchestrator.machine import Interaction, SessionPhase
from tutor.runtime_capabilities import (
    WIDGET_CAPABILITY_VERSION,
    effective_widget_capability_manifest,
    normalize_widget_capability_manifest,
    widget_capability_manifest,
    widget_supported,
)
from tutor.schemas.assessment import (
    AnswerSpec,
    AssessmentItem,
    AssessmentSurface,
    ContentExposureState,
    ItemBankDocument,
    ItemReservation,
    LessonBundleReservation,
    PromptSegment,
)
from tutor.schemas.common import ResponseClass
from tutor.schemas.kc import GraphDocument
from tutor.schemas.learner import EvidenceEvent, LearnerProfile
from tutor.verify.checker import verify_answer

LESSON_POLICY_VERSION = "lesson-flow-v2.2"
CAPSTONE_POLICY_VERSION = "capstone-v2.0"
ALLOCATOR_POLICY_VERSION = "allocator-v2.1"


class PendingInteractionV2(BaseModel):
    """One server-owned assessment item awaiting a student action."""

    key: str
    kind: Literal["probe", "guided_widget", "checkin", "capstone"]
    kc_id: str
    item_id: str
    item_revision: int
    family_id: str
    reservation: ItemReservation
    prompt: str
    prompt_segments: list[PromptSegment]
    answer_spec: AnswerSpec
    hints: list[str]
    revealing_hints: list[bool]
    hints_given: int = 0
    attempt_number: int = 1
    delivery_mode: Literal["widget", "text"] = "widget"

    @property
    def input_mode(self) -> Literal["math", "choice", "widget"]:
        if self.kind == "guided_widget" and self.delivery_mode == "widget":
            return "widget"
        return "choice" if self.answer_spec.kind == "choice" else "math"

    @property
    def can_hint(self) -> bool:
        return self.kind != "capstone" and self.hints_given < len(self.hints)

    @property
    def assisted(self) -> bool:
        """Only seeing a revealing hint disqualifies the attempt."""
        return any(self.revealing_hints[: self.hints_given])


class WidgetResultV2(BaseModel):
    """Widget verdict plus interactions emitted by a state transition."""

    correct: bool
    message: str
    status: Literal["invalid", "attempted", "solved", "remediated"]
    counted: bool
    attempt_number: int | None = Field(default=None, ge=1)
    interactions: list[Interaction] = Field(default_factory=list)


class HintResultV2(BaseModel):
    """Hint text plus any forced transition after a revealing hint."""

    text: str
    interactions: list[Interaction] = Field(default_factory=list)


def render_assessment_prompt(item: AssessmentItem) -> str:
    """Render structured segments without rewriting their mathematical meaning."""
    chunks: list[str] = []
    for segment in item.prompt:
        if segment.kind == "text":
            chunks.append(segment.text)
        elif segment.kind == "math":
            chunks.append(segment.expression)
        else:
            chunks.append("____")
    return " ".join(chunk.strip() for chunk in chunks if chunk.strip())


def _answer_text(item: AssessmentItem) -> str:
    answer = item.answer
    if answer.kind in {"symbolic", "numeric", "antiderivative"}:
        return answer.expected
    if answer.kind in {"finite_set", "ordered_tuple"}:
        opening, closing = ("{", "}") if answer.kind == "finite_set" else ("(", ")")
        return f"{opening}{', '.join(answer.expected)}{closing}"
    if answer.kind == "interval_set":
        rendered = []
        for interval in answer.expected:
            rendered.append(
                f"{'[' if interval.lower_closed else '('}{interval.lower}, "
                f"{interval.upper}{']' if interval.upper_closed else ')'}"
            )
        return " ∪ ".join(rendered)
    return answer.expected_choice_id


class SessionOrchestratorV2:
    """Confirmation-first session using only reviewed, unexposed item families."""

    def __init__(
        self,
        graph: GraphDocument,
        target_kc: str,
        profile: LearnerProfile,
        item_bank: ItemBankDocument | None = None,
        probe_budget: int = 8,
        *,
        learner_id: UUID | None = None,
        as_of: datetime | None = None,
        learner_params: BKTParams | None = None,
        retention_half_life_days: int = 180,
        confirmation_window_days: int = 90,
        impact_lambda: float = PINNED_IMPACT_LAMBDA,
        impact_decay: float = PINNED_IMPACT_DECAY,
        episode_id: str | None = None,
        widget_capabilities: dict[str, Any] | None = None,
    ) -> None:
        self._graph = graph
        self._nodes = {node.id: node for node in graph.nodes}
        if target_kc not in self._nodes:
            raise KeyError(f"unknown kc: {target_kc}")
        self._target = target_kc
        self._profile = profile
        self._episode_id = episode_id or uuid4().hex
        self._pinned_widget_capabilities = normalize_widget_capability_manifest(
            widget_capabilities or widget_capability_manifest()
        )
        self._runtime_widget_capabilities = normalize_widget_capability_manifest(
            self._pinned_widget_capabilities
        )
        self._bank = item_bank or load_item_bank()
        if self._bank.graph_version != graph.graph_version:
            raise ValueError("item bank and graph versions do not match")
        required_kcs = graph_service.ancestor_subgraph(
            graph, target_kc, hard_only=True
        ).node_ids()
        missing_kcs = required_kcs - set(self._bank.released_kcs)
        if missing_kcs:
            raise ValueError(
                "target hard-ancestor closure is not fully released in the v2 "
                f"item bank; missing {sorted(missing_kcs)}"
            )
        release_errors = validate_item_bank(
            self._bank,
            graph,
            released_kcs=required_kcs,
        )
        if release_errors:
            preview = "; ".join(release_errors[:5])
            suffix = (
                f"; and {len(release_errors) - 5} more"
                if len(release_errors) > 5
                else ""
            )
            raise ValueError(
                f"item bank is not trusted for this target: {preview}{suffix}"
            )

        floor = (
            {"Algebra 1", "Algebra 2", "Precalculus"}
            if "calc" in profile.course.lower()
            else set()
        )
        self._assumed_floor_levels = floor
        self.learner = LearnerModelServiceV2(
            graph,
            params=learner_params or DEFAULT_PARAMS_V2,
            assumed_floor_levels=floor,
            learner_id=learner_id,
            as_of=as_of,
            retention_half_life_days=retention_half_life_days,
            confirmation_window_days=confirmation_window_days,
        )
        self._probe_budget = probe_budget
        self._diag = DiagnosisControllerV2(
            graph,
            target_kc,
            self.learner,
            probe_budget=probe_budget,
            impact_lambda=impact_lambda,
            impact_decay=impact_decay,
        )
        self._allocator = ItemAllocator(self._bank)
        self.exposure_state = ContentExposureState()
        self.phase = SessionPhase.INTAKE
        self._pending: PendingInteractionV2 | None = None
        self._counter = 0
        self._interactions_used = 0
        self._plan: list[LearningPlanStep] = []
        self._plan_index = 0
        self._current_bundle: LessonBundleReservation | None = None
        self._checkin_queue: list[ItemReservation] = []
        self._checkin_attempts: dict[str, int] = {}
        self._checkin_success_families: dict[str, list[str]] = {}
        self._verification_mode_kc: str | None = None
        self._widget_attempts: dict[str, int] = {}
        self._learning_transition_keys: set[str] = set()
        self._capstone_attempts = 0
        self._stop_reason: str | None = None
        # Canonical, de-duplicated learner-visible fragments are part of the
        # authoritative checkpoint.  They include context, tutor content,
        # hints, and submitted answers so an upcoming family can never use a
        # truth that has already appeared on screen.
        self._visible_texts: list[str] = []
        # Widget controls can retain submitted values even though the public
        # transcript deliberately uses a generic student bubble. Keep those
        # values separately so durable widget-attempt rows can reconcile them.
        self._private_visible_inputs: list[str] = []

    def remember_visible_content(self, *values: Any) -> None:
        """Idempotently add content that this learner can already see."""
        extend_visible_texts(self._visible_texts, *values)

    def replace_public_visible_content(self, *values: Any) -> None:
        """Reconcile public history to one authoritative SessionView snapshot."""
        self._visible_texts = []
        self.remember_visible_content(*values)

    def remember_private_visible_input(self, *values: Any) -> None:
        """Retain visible control values redacted from the public transcript."""
        extend_visible_texts(self._private_visible_inputs, *values)

    def _visible_history(self) -> list[str]:
        history = list(self._visible_texts)
        extend_visible_texts(history, self._private_visible_inputs)
        return history

    def set_runtime_widget_capabilities(self, manifest: dict[str, Any]) -> None:
        """Apply a runtime safety ceiling without widening the episode's pins."""
        self._runtime_widget_capabilities = normalize_widget_capability_manifest(
            manifest
        )
        if (
            self._pending is not None
            and self._pending.kind == "guided_widget"
            and self._pending.delivery_mode == "widget"
            and not self._widget_supported("live_input")
        ):
            # A widget already present in the transcript becomes read-only; the
            # same pending practice remains answerable through the text composer.
            self._pending.delivery_mode = "text"

    def _effective_widget_capabilities(self) -> dict[str, Any]:
        return effective_widget_capability_manifest(
            self._pinned_widget_capabilities,
            self._runtime_widget_capabilities,
        )

    def _widget_supported(self, widget_type: str) -> bool:
        return widget_supported(widget_type, self._effective_widget_capabilities())

    def bind_episode_id(self, episode_id: str) -> None:
        """Bind evidence provenance to the authoritative API episode id."""
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        has_current_episode_evidence = any(
            event.episode_id == self._episode_id for event in self.learner.events
        )
        if has_current_episode_evidence and self._episode_id != episode_id:
            raise RuntimeError("cannot rebind an episode after evidence was recorded")
        self._episode_id = episode_id

    @property
    def pending(self) -> PendingInteractionV2 | None:
        return self._pending

    def seed_longitudinal(
        self,
        learner_id: UUID,
        events: list[EvidenceEvent],
        *,
        as_of: datetime,
        exposure_state: ContentExposureState | None = None,
    ) -> None:
        """Seed prior evidence and retired content into a new intake episode."""
        if self.phase != SessionPhase.INTAKE:
            raise RuntimeError("longitudinal evidence must be seeded before begin")
        learner = LearnerModelServiceV2(
            self._graph,
            params=self.learner.params,
            assumed_floor_levels=self._assumed_floor_levels,
            learner_id=learner_id,
            as_of=as_of,
            retention_half_life_days=self.learner.retention_half_life_days,
            confirmation_window_days=self.learner.confirmation_window_days,
        )
        self.learner = learner.replay(events, as_of=as_of)
        if exposure_state is not None:
            self.exposure_state = exposure_state.model_copy(deep=True)
        self._diag = DiagnosisControllerV2(
            self._graph,
            self._target,
            self.learner,
            probe_budget=self._probe_budget,
            impact_lambda=self._diag.state.impact_lambda,
            impact_decay=self._diag.state.impact_decay,
        )

    def fresh_episode(self, *, as_of: datetime) -> "SessionOrchestratorV2":
        """Start over on the same pinned release while retaining prior evidence."""
        fresh = SessionOrchestratorV2(
            self._graph,
            self._target,
            self._profile,
            item_bank=self._bank,
            probe_budget=self._probe_budget,
            learner_id=self.learner.learner_id,
            as_of=as_of,
            learner_params=self.learner.params,
            retention_half_life_days=self.learner.retention_half_life_days,
            confirmation_window_days=self.learner.confirmation_window_days,
            impact_lambda=self._diag.state.impact_lambda,
            impact_decay=self._diag.state.impact_decay,
            widget_capabilities=self._effective_widget_capabilities(),
        )
        fresh.seed_longitudinal(
            self.learner.learner_id,
            list(self.learner.events),
            as_of=as_of,
            exposure_state=self.exposure_state,
        )
        return fresh

    @property
    def pending_key(self) -> str | None:
        return self._pending.key if self._pending else None

    @property
    def pending_kind(self) -> str | None:
        return self._pending.kind if self._pending else None

    @property
    def pending_kc(self) -> str | None:
        return self._pending.kc_id if self._pending else None

    @property
    def pending_expected(self) -> str | None:
        """Test/CLI-only answer projection; never used by public API schemas."""
        if self._pending is None:
            return None
        item = self._item_for(self._pending.reservation)
        return _answer_text(item)

    def _next_key(self) -> str:
        self._counter += 1
        return f"v2i{self._counter:04d}"

    def _interaction(
        self,
        kind: Literal["message", "probe", "lesson", "checkin", "capstone"],
        text: str,
        *,
        key: str | None = None,
        kc_id: str | None = None,
        prompt_segments: list[dict] | None = None,
        widget: dict | None = None,
    ) -> Interaction:
        self._interactions_used += 1
        self.remember_visible_content(text, prompt_segments, widget)
        return Interaction(
            key=key or self._next_key(),
            kind=kind,
            kc_id=kc_id,
            text=text,
            prompt_segments=prompt_segments,
            widget=widget,
        )

    def _message(self, text: str) -> Interaction:
        return self._interaction("message", text)

    def _item_for(self, reservation: ItemReservation) -> AssessmentItem:
        return self._allocator.item_for(reservation)

    def _set_pending(
        self,
        reservation: ItemReservation,
        kind: Literal["probe", "guided_widget", "checkin", "capstone"],
        *,
        attempt_number: int = 1,
        delivery_mode: Literal["widget", "text"] = "widget",
    ) -> PendingInteractionV2:
        item = self._item_for(reservation)
        pending = PendingInteractionV2(
            key=self._next_key(),
            kind=kind,
            kc_id=item.kc_id,
            item_id=item.item_id,
            item_revision=item.revision,
            family_id=item.family_id,
            reservation=reservation,
            prompt=render_assessment_prompt(item),
            prompt_segments=list(item.prompt),
            answer_spec=item.answer,
            hints=[hint.text for hint in item.hints],
            revealing_hints=[hint.revealing for hint in item.hints],
            attempt_number=attempt_number,
            delivery_mode=delivery_mode,
        )
        self._pending = pending
        self.remember_visible_content(
            pending.prompt,
            pending.prompt_segments,
            (
                list(getattr(pending.answer_spec, "option_ids", ()))
                if pending.answer_spec.kind == "choice"
                else None
            ),
        )
        self.exposure_state = self._allocator.record_exposure(
            self.exposure_state, reservation
        )
        return pending

    def begin(self) -> list[Interaction]:
        if self.phase != SessionPhase.INTAKE:
            raise RuntimeError("session already started")
        self.phase = SessionPhase.DIAGNOSE
        target_name = self._nodes[self._target].name
        return [
            self._message(
                f"Let's find a trustworthy starting point for {target_name}. "
                "I will confirm any result that changes your path."
            ),
            *self._issue_next_probe(),
        ]

    def _issue_next_probe(self) -> list[Interaction]:
        selection = self._diag.next_probe()
        if selection is None:
            return self._finish_diagnosis()
        try:
            allocation = self._allocator.reserve_item(
                self.exposure_state,
                kc_id=selection.kc_id,
                surface=AssessmentSurface.DIAGNOSTIC,
                visible_texts=self._visible_history(),
            )
        except AllocationError:
            return self._stop(
                "No unseen reviewed diagnostic family remains for the next skill. "
                "The session stopped without guessing or reusing an answer."
            )
        self.exposure_state = allocation.state
        pending = self._set_pending(allocation.reservation, "probe")
        return [
            self._interaction(
                "probe",
                pending.prompt,
                key=pending.key,
                kc_id=pending.kc_id,
                prompt_segments=[
                    segment.model_dump(mode="json")
                    for segment in pending.prompt_segments
                ],
            )
        ]

    def hint(self) -> HintResultV2 | None:
        if self._pending is None or not self._pending.can_hint:
            return None
        index = self._pending.hints_given
        self._pending.hints_given += 1
        revealing = self._pending.revealing_hints[index]
        self.exposure_state = self._allocator.update_exposure(
            self.exposure_state,
            self._pending.reservation,
            hints_seen=self._pending.hints_given,
            answer_revealed=revealing,
        )
        text = self._pending.hints[index]
        self.remember_visible_content(text)
        if not revealing:
            return HintResultV2(text=text)

        pending = self._pending
        self._pending = None
        if pending.kind == "probe":
            # Record only that this family was assisted so the diagnosis policy
            # allocates another family. No learner evidence is created.
            self._diag.record_result(
                DiagnosticObservation(
                    kc_id=pending.kc_id,
                    family_id=pending.family_id,
                    correct=False,
                    assisted=True,
                    response_class=(
                        ResponseClass.MULTIPLE_CHOICE
                        if pending.answer_spec.kind == "choice"
                        else ResponseClass.SYMBOLIC_ENTRY
                    ),
                )
            )
            transitions = [
                self._message(
                    "That hint revealed the answer, so this item will not be "
                    "scored. Try a fresh independent item."
                ),
                *self._issue_next_probe(),
            ]
        elif pending.kind == "guided_widget":
            transitions = [
                self._message(
                    "That hint completed the walkthrough without counting it as "
                    "practice. Now try a fresh independent check."
                ),
                *self._issue_next_checkin(pending.kc_id),
            ]
        else:
            transitions = [
                self._message(
                    "That hint revealed the answer, so this check will not count."
                ),
                *self._after_checkin(pending, False, None),
            ]
        return HintResultV2(text=text, interactions=transitions)

    def submit(self, answer: str) -> list[Interaction]:
        if self.phase in {SessionPhase.DONE, SessionPhase.STOPPED}:
            raise RuntimeError("session is over")
        if self._pending is None:
            raise RuntimeError("no pending item to answer")
        # Register before verification because a correct/incorrect response can
        # allocate the next item in this same call.  A failed transaction acts
        # on a deep copy, so uncommitted text never reaches the live ledger.
        self.remember_visible_content(answer)
        if (
            self._pending.kind == "guided_widget"
            and self._pending.delivery_mode == "widget"
        ):
            raise RuntimeError("submit the guided widget or choose its text fallback")

        pending = self._pending
        result = verify_answer(pending.answer_spec, answer)
        if result.status in {"invalid", "timeout"}:
            return [
                self._message(
                    "I could not read that as a valid answer. "
                    "Nothing was graded; revise it and try again."
                )
            ]

        self._pending = None
        correct = result.status == "correct"
        item = self._item_for(pending.reservation)
        implicated, misconception = self._match_error_signature(item, answer, correct)
        if pending.kind == "guided_widget":
            attempts = self._widget_attempts.get(pending.key, 0) + 1
            self._widget_attempts[pending.key] = attempts
            pending.attempt_number = attempts
        self._record_event(
            pending,
            correct,
            misconception_id=misconception,
        )
        if pending.kind == "guided_widget":
            return self._after_guided_text_attempt(pending, correct)
        if pending.kind == "probe":
            self._diag.record_result(
                DiagnosticObservation(
                    kc_id=pending.kc_id,
                    family_id=pending.family_id,
                    correct=correct,
                    assisted=pending.assisted,
                    response_class=(
                        ResponseClass.MULTIPLE_CHOICE
                        if pending.answer_spec.kind == "choice"
                        else ResponseClass.SYMBOLIC_ENTRY
                    ),
                    implicated_prereq=implicated,
                )
            )
            status = self._diag.status(pending.kc_id)
            if status == "confirmed_mastered":
                note = "Two independent item families now confirm this strength."
            elif status == "confirmed_gap":
                note = "Two independent item families now confirm this gap."
            else:
                note = (
                    "That response is one piece of evidence. "
                    "The skill remains uncertain until an independent item confirms it."
                )
            return [self._message(note), *self._issue_next_probe()]
        if pending.kind == "checkin":
            return self._after_checkin(pending, correct, implicated)
        return self._after_capstone(pending, correct, implicated)

    def _after_guided_text_attempt(
        self,
        pending: PendingInteractionV2,
        correct: bool,
    ) -> list[Interaction]:
        """Apply the same bounded formative policy to keyboard text practice."""
        attempts = self._widget_attempts[pending.key]
        if correct:
            self._record_learning_transition(pending)
            return [
                self._message(
                    "Guided text practice complete. Now try an unseen check."
                ),
                *self._issue_next_checkin(pending.kc_id),
            ]
        if attempts >= 3:
            item = self._item_for(pending.reservation)
            self.exposure_state = self._allocator.update_exposure(
                self.exposure_state,
                pending.reservation,
                solution_exposed=True,
                answer_revealed=True,
            )
            self._record_learning_transition(pending)
            return [
                self._message(
                    f"Here is the guided answer: {_answer_text(item)}. "
                    "It will not count as mastery; next is a fresh independent item."
                ),
                *self._issue_next_checkin(pending.kc_id),
            ]
        self._pending = pending
        return [
            self._message(
                "Not yet — revise the guided response and try again. "
                f"{3 - attempts} guided attempt(s) remain."
            )
        ]

    def _match_error_signature(
        self, item: AssessmentItem, answer: str, correct: bool
    ) -> tuple[str | None, str | None]:
        if correct:
            return None, None
        for signature in item.error_signatures:
            if answer.strip().casefold() == signature.expected_wrong.strip().casefold():
                return signature.implicated_prereq, signature.misconception_id
        return None, None

    def _record_event(
        self,
        pending: PendingInteractionV2,
        correct: bool,
        *,
        misconception_id: str | None = None,
    ) -> None:
        item = self._item_for(pending.reservation)
        policy_version = {
            "probe": DIAGNOSIS_POLICY_VERSION,
            "guided_widget": LESSON_POLICY_VERSION,
            "checkin": LESSON_POLICY_VERSION,
            "capstone": CAPSTONE_POLICY_VERSION,
        }[pending.kind]
        self.learner.apply_event(
            EvidenceEvent(
                event_id=uuid4(),
                learner_id=self.learner.learner_id,
                t=datetime.now(timezone.utc),
                item_id=pending.item_id,
                kc_ids=[pending.kc_id],
                correct=correct,
                response_class=(
                    ResponseClass.MULTIPLE_CHOICE
                    if pending.answer_spec.kind == "choice"
                    else ResponseClass.SYMBOLIC_ENTRY
                ),
                hints_used=pending.hints_given,
                assisted=pending.assisted,
                misconception_id=misconception_id,
                content_versions={
                    "graph": str(self._graph.graph_version),
                    "item_bank": self._bank.bank_version,
                },
                episode_id=self._episode_id,
                family_id=pending.family_id,
                surface={
                    "probe": "diagnostic",
                    "checkin": "checkin",
                    "capstone": "capstone",
                    "guided_widget": "guided_widget",
                }[pending.kind],
                item_revision=pending.item_revision,
                attempt_number=pending.attempt_number,
                policy_version=policy_version,
                learner_params_version=(
                    f"bkt-v{self.learner.params.params_version}"
                ),
                content_provenance=item.provenance.source[:128],
                learning_opportunity=False,
            )
        )

    def _record_learning_transition(self, pending: PendingInteractionV2) -> None:
        """Apply one lesson transition separately from widget/check evidence."""
        key = f"{pending.item_id}@{pending.item_revision}"
        if key in self._learning_transition_keys:
            return
        item = self._item_for(pending.reservation)
        self._learning_transition_keys.add(key)
        self.learner.apply_event(
            EvidenceEvent(
                event_id=uuid4(),
                learner_id=self.learner.learner_id,
                t=datetime.now(timezone.utc),
                item_id=f"lesson-transition.{pending.item_id}",
                kc_ids=[pending.kc_id],
                correct=True,
                response_class=ResponseClass.WIDGET,
                assisted=False,
                content_versions={
                    "graph": str(self._graph.graph_version),
                    "item_bank": self._bank.bank_version,
                },
                episode_id=self._episode_id,
                family_id=pending.family_id,
                surface="instructional_practice",
                item_revision=pending.item_revision,
                attempt_number=1,
                policy_version=LESSON_POLICY_VERSION,
                learner_params_version=(
                    f"bkt-v{self.learner.params.params_version}"
                ),
                content_provenance=item.provenance.source[:128],
                learning_opportunity=True,
            )
        )

    def _finish_diagnosis(self) -> list[Interaction]:
        self.phase = SessionPhase.PLAN
        self._plan = self._diag.learning_plan()
        self._plan_index = 0
        summary = self._diag.learner_summary()
        gaps = summary["confirmed_gaps"]
        uncertain = summary["uncertain"]
        messages = [
            self._message(
                "Diagnosis complete. "
                f"Confirmed gaps: {len(gaps)}. Still uncertain: {len(uncertain)}."
            )
        ]
        if not self._plan:
            return [*messages, *self._start_capstone()]
        self.phase = SessionPhase.TEACH
        return [*messages, *self._start_current_plan_step()]

    def _start_current_plan_step(self) -> list[Interaction]:
        while self._plan_index < len(self._plan):
            step = self._plan[self._plan_index]
            if (
                step.kind == "practice_target"
                and self.learner.mastery_status(step.kc_id) == "confirmed_mastered"
            ):
                self._plan_index += 1
                continue
            if step.kind == "verify_uncertain":
                return self._issue_verification(step.kc_id)
            return self._issue_lesson(step.kc_id, step)
        return self._start_capstone()

    def _issue_verification(self, kc_id: str) -> list[Interaction]:
        """Assess uncertainty before deciding whether instruction is needed."""
        try:
            allocation = self._allocator.reserve_item(
                self.exposure_state,
                kc_id=kc_id,
                surface=AssessmentSurface.CHECKIN,
                visible_texts=self._visible_history(),
            )
        except AllocationError:
            return self._stop(
                "No unseen reviewed verification item remains. "
                "The skill stays uncertain rather than being treated as a gap."
            )
        self.exposure_state = allocation.state
        self._verification_mode_kc = kc_id
        pending = self._set_pending(allocation.reservation, "checkin")
        return [
            self._message(
                f"{self._nodes[kc_id].name} is still uncertain. "
                "Try one fresh item before I decide whether instruction is needed."
            ),
            self._interaction(
                "checkin",
                pending.prompt,
                key=pending.key,
                kc_id=kc_id,
                prompt_segments=[
                    segment.model_dump(mode="json")
                    for segment in pending.prompt_segments
                ],
            ),
        ]

    def _issue_lesson(
        self, kc_id: str, step: LearningPlanStep
    ) -> list[Interaction]:
        exposure_before_bundle = self.exposure_state
        try:
            allocation = self._allocator.reserve_lesson_bundle(
                self.exposure_state,
                kc_id,
                visible_texts=self._visible_history(),
            )
        except AllocationError:
            return self._stop(
                "No unseen reviewed practice passed the answer-separation gate "
                "for this skill. The session stopped rather than reusing an answer."
            )
        self.exposure_state = allocation.state
        self._current_bundle = allocation.bundle
        self._checkin_queue = list(allocation.bundle.checkins)
        worked = self._item_for(allocation.bundle.worked_example)
        widget_item = self._item_for(allocation.bundle.guided_widget)
        narrative = (
            f"{self._nodes[kc_id].name}\n\n{self._nodes[kc_id].description}\n\n"
            f"Worked example: {render_assessment_prompt(worked)} "
            f"Answer: {_answer_text(worked)}"
        )
        upcoming = sorted(
            (
                item
                for item in self._bank.items
                if item.family_id not in exposure_before_bundle.used_family_ids
                and any(
                    surface in item.eligible_surfaces
                    for surface in (
                        AssessmentSurface.GUIDED_WIDGET,
                        AssessmentSurface.CHECKIN,
                        AssessmentSurface.CAPSTONE,
                    )
                )
            ),
            key=lambda item: (item.kc_id, item.item_id, item.revision),
        )
        leakage = bundle_leakage_problems(
            [
                narrative,
                render_assessment_prompt(widget_item),
                *(hint.text for hint in widget_item.hints[:2]),
            ],
            upcoming,
        )
        if leakage:
            return self._stop(
                "The reserved lesson failed its answer-separation gate. "
                "The session stopped before displaying compromised content."
            )
        self.exposure_state = self._allocator.record_exposure(
            self.exposure_state,
            allocation.bundle.worked_example,
            solution_exposed=True,
            answer_revealed=True,
        )
        live_input_enabled = self._widget_supported("live_input")
        pending = self._set_pending(
            allocation.bundle.guided_widget,
            "guided_widget",
            delivery_mode="widget" if live_input_enabled else "text",
        )
        widget = None
        if live_input_enabled:
            widget = {
                "widget_type": "live_input",
                "learning_objective": (
                    f"Guided practice for {self._nodes[kc_id].name}"
                ),
                "prompt": render_assessment_prompt(widget_item),
                "input_kind": (
                    "number"
                    if widget_item.answer.kind == "numeric"
                    else "expression"
                ),
                "text_fallback": (
                    "Use the same prompt as a text exercise, then continue to an "
                    "independent check."
                ),
            }
        else:
            narrative += (
                "\n\nGuided text practice is ready below. It follows the same "
                "three-attempt formative policy as the visual interaction."
            )
        return [
            self._interaction(
                "lesson",
                narrative,
                key=pending.key,
                kc_id=kc_id,
                widget=widget,
            )
        ]

    def answer_widget(self, key: str, response: dict) -> WidgetResultV2:
        if (
            self._pending is None
            or self._pending.kind != "guided_widget"
            or self._pending.delivery_mode != "widget"
        ):
            raise KeyError("no guided widget is pending")
        if key != self._pending.key:
            raise KeyError("unknown or stale widget")
        pending = self._pending
        # The archived control can retain its populated value even though the
        # transcript uses a generic student bubble.
        self.remember_private_visible_input(response)
        raw = response.get("text", response.get("value", ""))
        result = verify_answer(pending.answer_spec, str(raw))
        if result.status in {"invalid", "timeout"}:
            feedback = "That input was not gradable. Nothing was counted; try again."
            self.remember_visible_content(feedback)
            return WidgetResultV2(
                correct=False,
                message=feedback,
                status="invalid",
                counted=False,
            )
        correct = result.status == "correct"
        attempts = self._widget_attempts.get(key, 0) + 1
        self._widget_attempts[key] = attempts
        pending.attempt_number = attempts
        self._record_event(pending, correct)
        if correct:
            feedback = "Nice — the guided relationship is correct."
            self.remember_visible_content(feedback)
            self._record_learning_transition(pending)
            self._pending = None
            interactions = [
                self._message("Guided practice complete. Now try an unseen check."),
                *self._issue_next_checkin(pending.kc_id),
            ]
            return WidgetResultV2(
                correct=True,
                message=feedback,
                status="solved",
                counted=True,
                attempt_number=attempts,
                interactions=interactions,
            )
        if attempts >= 3:
            item = self._item_for(pending.reservation)
            self.exposure_state = self._allocator.update_exposure(
                self.exposure_state,
                pending.reservation,
                solution_exposed=True,
                answer_revealed=True,
            )
            self._record_learning_transition(pending)
            self._pending = None
            feedback = "Three guided attempts used; showing remediation."
            self.remember_visible_content(feedback)
            interactions = [
                self._message(
                    f"Here is the guided answer: {_answer_text(item)}. "
                    "It will not count as mastery; next is a fresh independent item."
                ),
                *self._issue_next_checkin(pending.kc_id),
            ]
            return WidgetResultV2(
                correct=False,
                message=feedback,
                status="remediated",
                counted=True,
                attempt_number=attempts,
                interactions=interactions,
            )
        feedback = "Not yet — adjust the guided response and try again."
        self.remember_visible_content(feedback)
        return WidgetResultV2(
            correct=False,
            message=feedback,
            status="attempted",
            counted=True,
            attempt_number=attempts,
        )

    def use_text_fallback(self) -> list[Interaction]:
        if self._pending is None or self._pending.kind != "guided_widget":
            raise RuntimeError("text fallback is available only for guided practice")
        if self._pending.delivery_mode == "text":
            raise RuntimeError("text fallback is already active")
        self._pending.delivery_mode = "text"
        return [
            self._message(
                "The same guided prompt is now available as keyboard text practice. "
                "Submit an answer before moving to the independent check."
            ),
        ]

    def _issue_next_checkin(self, kc_id: str) -> list[Interaction]:
        attempt = self._checkin_attempts.get(kc_id, 0) + 1
        self._checkin_attempts[kc_id] = attempt
        if self._checkin_queue:
            reservation = self._checkin_queue.pop(0)
            if not self._allocator.reservation_answer_separated(
                reservation,
                self._visible_history(),
            ):
                # The complete bundle was safe when reserved, but a subsequent
                # widget/check response can collide with a queued truth. Retire
                # that reservation and deterministically replace its bundle slot
                # before anything from the replacement is displayed.
                if self._current_bundle is None:
                    return self._stop(
                        "The next independent check is no longer answer-separated. "
                        "The session stopped before displaying it."
                    )
                bundle_index = (
                    len(self._current_bundle.checkins)
                    - len(self._checkin_queue)
                    - 1
                )
                try:
                    replacement = self._allocator.reserve_item(
                        self.exposure_state,
                        kc_id=kc_id,
                        surface=AssessmentSurface.CHECKIN,
                        visible_texts=self._visible_history(),
                    )
                except AllocationError:
                    return self._stop(
                        "No unseen answer-separated check-in family remains. "
                        "The skill stays unconfirmed rather than reusing an answer."
                    )
                self.exposure_state = replacement.state
                reservation = replacement.reservation
                checkins = list(self._current_bundle.checkins)
                checkins[bundle_index] = reservation
                self._current_bundle = self._current_bundle.model_copy(
                    update={"checkins": checkins}
                )
        else:
            try:
                allocation = self._allocator.reserve_item(
                    self.exposure_state,
                    kc_id=kc_id,
                    surface=AssessmentSurface.CHECKIN,
                    visible_texts=self._visible_history(),
                )
            except AllocationError:
                return self._stop(
                    "No unseen reviewed check-in family remains. "
                    "The skill stays unconfirmed rather than reusing an answer."
                )
            self.exposure_state = allocation.state
            reservation = allocation.reservation
        pending = self._set_pending(
            reservation, "checkin", attempt_number=attempt
        )
        return [
            self._interaction(
                "checkin",
                pending.prompt,
                key=pending.key,
                kc_id=kc_id,
                prompt_segments=[
                    segment.model_dump(mode="json")
                    for segment in pending.prompt_segments
                ],
            )
        ]

    def _after_checkin(
        self,
        pending: PendingInteractionV2,
        correct: bool,
        implicated_prereq: str | None,
    ) -> list[Interaction]:
        kc_id = pending.kc_id
        if self._verification_mode_kc == kc_id:
            self._verification_mode_kc = None
            self._plan_index += 1
            status = self.learner.mastery_status(kc_id)
            if status == "confirmed_mastered":
                note = (
                    f"{self._nodes[kc_id].name} is now independently confirmed; "
                    "instruction is not needed."
                )
            elif status == "confirmed_gap":
                note = (
                    f"{self._nodes[kc_id].name} is now a confirmed gap. "
                    "I will teach it before another independent check."
                )
                if (
                    self._plan_index < len(self._plan)
                    and self._plan[self._plan_index].kc_id == kc_id
                ):
                    self._plan[self._plan_index] = LearningPlanStep(
                        kind="teach_confirmed_gap",
                        kc_id=kc_id,
                    )
            else:
                note = (
                    f"{self._nodes[kc_id].name} remains uncertain. "
                    "I will use instruction and fresh checks without calling it a gap."
                )
            return [self._message(note), *self._start_current_plan_step()]

        successes = self._checkin_success_families.setdefault(kc_id, [])
        if (
            correct
            and not pending.assisted
            and pending.answer_spec.kind != "choice"
            and pending.family_id not in successes
        ):
            successes.append(pending.family_id)
        if len(successes) >= 2 and self.learner.mastery_status(kc_id) == "confirmed_mastered":
            self._plan_index += 1
            return [
                self._message(f"{self._nodes[kc_id].name}: independently confirmed."),
                *self._start_current_plan_step(),
            ]

        attempts = self._checkin_attempts.get(kc_id, 0)
        if attempts < 3:
            note = (
                "Correct. One more independent family will confirm it."
                if correct and not pending.assisted
                else "That attempt was assisted or incorrect, so it does not confirm mastery."
            )
            return [self._message(note), *self._issue_next_checkin(kc_id)]

        worked = (
            self._item_for(self._current_bundle.worked_example)
            if self._current_bundle is not None
            else None
        )
        remediation = (
            f"Review: {render_assessment_prompt(worked)} Answer: {_answer_text(worked)}"
            if worked is not None
            else self._nodes[kc_id].description
        )
        if implicated_prereq and implicated_prereq in self._nodes:
            remediation += (
                f" This response also suggests verifying "
                f"{self._nodes[implicated_prereq].name} directly."
            )
        if kc_id == self._target:
            return self._stop(
                f"Independent mastery is not confirmed yet. {remediation}"
            )
        self._plan_index += 1
        return [
            self._message(
                f"Mastery remains uncertain after three independent checks. {remediation}"
            ),
            *self._start_current_plan_step(),
        ]

    def _start_capstone(self, *, retry_after_remediation: bool = False) -> list[Interaction]:
        self.phase = SessionPhase.CAPSTONE
        if (
            not retry_after_remediation
            and self.learner.mastery_status(self._target) != "confirmed_mastered"
        ):
            # A target confirmed during diagnosis lives in the diagnosis policy,
            # so apply that independent evidence to the learner status as well.
            if self._diag.status(self._target) != "confirmed_mastered":
                return self._stop(
                    "The goal skill is still uncertain, so I will not claim completion."
                )
        try:
            allocation = self._allocator.reserve_item(
                self.exposure_state,
                kc_id=self._target,
                surface=AssessmentSurface.CAPSTONE,
                visible_texts=self._visible_history(),
            )
        except AllocationError:
            return self._stop(
                "No unseen reviewed goal problem remains; the session stopped "
                "instead of reusing an answer."
            )
        self.exposure_state = allocation.state
        pending = self._set_pending(
            allocation.reservation,
            "capstone",
            attempt_number=self._capstone_attempts + 1,
        )
        introduction = (
            "The target is now uncertain after the previous goal attempt. "
            "Use this different unseen problem to re-establish it."
            if retry_after_remediation
            and self.learner.mastery_status(self._target) != "confirmed_mastered"
            else "All required skills are confirmed. Finish with an unseen goal problem."
        )
        return [
            self._message(introduction),
            self._interaction(
                "capstone",
                f"Goal problem — work independently:\n{pending.prompt}",
                key=pending.key,
                kc_id=pending.kc_id,
                prompt_segments=[
                    segment.model_dump(mode="json")
                    for segment in pending.prompt_segments
                ],
            ),
        ]

    def _after_capstone(
        self,
        pending: PendingInteractionV2,
        correct: bool,
        implicated_prereq: str | None,
    ) -> list[Interaction]:
        self._capstone_attempts += 1
        unassisted = not pending.assisted
        if (
            correct
            and unassisted
            and self.learner.mastery_status(self._target) == "confirmed_mastered"
        ):
            self.phase = SessionPhase.DONE
            return [
                self._message(
                    f"{self._nodes[self._target].name} was solved independently. "
                    "Session complete."
                )
            ]
        if self._capstone_attempts < 2:
            reason = (
                "That solution used help"
                if correct
                else "That answer needs another pass"
            )
            remediation = self._capstone_remediation(implicated_prereq)
            if remediation is None:
                return self._stop(
                    "The reviewed remediation failed its answer-separation gate. "
                    "The session stopped before displaying compromised content."
                )
            return [
                self._message(
                    f"{reason}. {remediation} "
                    "Next I will use a different, unseen goal-problem family."
                ),
                *self._start_capstone(retry_after_remediation=True),
            ]
        return self._stop(
            "The goal problem is not yet independently confirmed. Review the "
            "targeted lesson and return for a new session."
        )

    def _capstone_remediation(self, implicated_prereq: str | None) -> str | None:
        """Return answer-separated remediation before the second capstone.

        Reserving a worked family is safe before the gate, but no solution is
        marked exposed and no remediation text is returned until every
        remaining capstone family is proven answer-separated.
        """
        worked_reservation = None
        if (
            self._current_bundle is not None
            and self._current_bundle.worked_example.kc_id == self._target
        ):
            worked_reservation = self._current_bundle.worked_example
        else:
            try:
                allocation = self._allocator.reserve_item(
                    self.exposure_state,
                    kc_id=self._target,
                    surface=AssessmentSurface.WORKED_EXAMPLE,
                    visible_texts=self._visible_history(),
                )
            except AllocationError:
                allocation = None
            if allocation is not None:
                self.exposure_state = allocation.state
                worked_reservation = allocation.reservation

        if worked_reservation is not None:
            worked = self._item_for(worked_reservation)
            remediation = (
                f"Review this worked pattern: {render_assessment_prompt(worked)} "
                f"Answer: {_answer_text(worked)}."
            )
        else:
            remediation = f"Review: {self._nodes[self._target].description}"
        if implicated_prereq and implicated_prereq in self._nodes:
            remediation += (
                f" Also revisit {self._nodes[implicated_prereq].name}; "
                "that connection is only a suspicion until checked directly."
            )

        remaining_capstones = sorted(
            (
                item
                for item in self._bank.items
                if item.kc_id == self._target
                and AssessmentSurface.CAPSTONE in item.eligible_surfaces
                and item.family_id not in self.exposure_state.used_family_ids
            ),
            key=lambda item: (item.item_id, item.revision),
        )
        if not remaining_capstones or bundle_leakage_problems(
            [remediation],
            remaining_capstones,
        ):
            return None

        if worked_reservation is not None:
            self.exposure_state = self._allocator.record_exposure(
                self.exposure_state,
                worked_reservation,
                solution_exposed=True,
                answer_revealed=True,
            )
        return remediation

    def _stop(self, text: str) -> list[Interaction]:
        self.phase = SessionPhase.STOPPED
        self._pending = None
        self._stop_reason = text
        return [self._message(text)]

    def summary(self) -> dict[str, Any]:
        diagnosis = self._diag.learner_summary()
        confirmed_mastery = set(diagnosis["confirmed_mastered"])
        confirmed_gaps = set(diagnosis["confirmed_gaps"])
        uncertain = set(diagnosis["uncertain"])
        relevant_kcs = confirmed_mastery | confirmed_gaps | uncertain
        for kc_id in relevant_kcs:
            status = self.learner.mastery_status(kc_id)
            # Later assessment can invalidate an earlier diagnosis result (a
            # failed capstone is the clearest example). Keep the three public
            # categories mutually exclusive and make current evidence
            # authoritative even when it returns the skill to uncertainty.
            confirmed_mastery.discard(kc_id)
            confirmed_gaps.discard(kc_id)
            uncertain.discard(kc_id)
            if status == "confirmed_mastered":
                confirmed_mastery.add(kc_id)
            elif status == "confirmed_gap":
                confirmed_gaps.add(kc_id)
            else:
                uncertain.add(kc_id)
        plan_step = (
            self._plan[self._plan_index].kind
            if self._plan_index < len(self._plan)
            else None
        )
        return {
            "phase": self.phase.value,
            "target": self._target,
            "confirmed_mastery": sorted(confirmed_mastery),
            "confirmed_gaps": sorted(confirmed_gaps),
            "uncertain": sorted(uncertain),
            "probes_used": self._diag.probes_issued,
            "probe_budget": self._probe_budget,
            "interactions_used": self._interactions_used,
            "plan_step": plan_step,
            "events_recorded": len(self.learner.events),
            "item_bank_version": self._bank.bank_version,
            "policy_versions": self._policy_versions(),
            "stop_reason": self._stop_reason,
        }

    @staticmethod
    def _policy_versions() -> dict[str, str]:
        """Versions that must remain available for exact episode replay."""
        return {
            "diagnosis": DIAGNOSIS_POLICY_VERSION,
            "lesson": LESSON_POLICY_VERSION,
            "capstone": CAPSTONE_POLICY_VERSION,
            "allocator": ALLOCATOR_POLICY_VERSION,
            "widget_capabilities": WIDGET_CAPABILITY_VERSION,
        }

    @staticmethod
    def _reservation_signature(
        reservation: ItemReservation,
    ) -> tuple[str, int, str, str, AssessmentSurface, str | None]:
        """Return every authored identity field shared by reservations/exposures."""
        return (
            reservation.item_id,
            reservation.revision,
            reservation.family_id,
            reservation.kc_id,
            reservation.surface,
            reservation.variant_id,
        )

    def _resolve_checkpoint_reservation(
        self,
        reservation: ItemReservation,
        *,
        role: str,
    ) -> AssessmentItem:
        """Resolve one restored reference and reject capabilities we cannot replay."""
        try:
            item = self._allocator.item_for(reservation)
        except AllocationError as exc:
            raise ValueError(
                f"checkpoint {role} does not resolve in the pinned item bank"
            ) from exc
        if reservation.surface not in item.eligible_surfaces:
            raise ValueError(
                f"checkpoint {role} uses an ineligible assessment surface"
            )
        # The current allocator emits only the authored, unparameterized item.
        # Silently accepting a variant id would claim exact replay without a
        # pinned variant generator or parameters to reproduce it.
        if reservation.variant_id is not None:
            raise ValueError(
                f"checkpoint {role} uses an unsupported item variant"
            )
        return item

    def _validate_checkpoint_content_references(self) -> None:
        """Fail closed when restored private scoring state drifts from its pins.

        Pydantic validates the shape of a checkpoint, but an otherwise valid
        payload can still replace an expected answer or repoint an active
        reservation.  Reconcile every content reference with both the exposure
        ledger and the pinned bank before the restored object can score input.
        """
        reservations_by_key = {
            (reservation.item_id, reservation.revision): reservation
            for reservation in self.exposure_state.reservations
        }
        for reservation in self.exposure_state.reservations:
            self._resolve_checkpoint_reservation(
                reservation,
                role="exposure-ledger reservation",
            )

        exposures_by_key = {
            (exposure.item_id, exposure.revision): exposure
            for exposure in self.exposure_state.exposures
        }
        for key, exposure in exposures_by_key.items():
            reservation = reservations_by_key.get(key)
            if (
                reservation is None
                or self._reservation_signature(exposure)
                != self._reservation_signature(reservation)
            ):
                raise ValueError(
                    "checkpoint exposure identity does not match its reservation"
                )

        def registered_item(
            reservation: ItemReservation,
            *,
            role: str,
        ) -> AssessmentItem:
            stored = reservations_by_key.get(
                (reservation.item_id, reservation.revision)
            )
            if stored is None or stored != reservation:
                raise ValueError(
                    f"checkpoint {role} is not an exact exposure-ledger reservation"
                )
            return self._resolve_checkpoint_reservation(
                reservation,
                role=role,
            )

        bundle_reservations: list[ItemReservation] = []
        if self._current_bundle is not None:
            bundle_reservations = [
                self._current_bundle.worked_example,
                self._current_bundle.guided_widget,
                *self._current_bundle.checkins,
            ]
            for reservation in bundle_reservations:
                registered_item(reservation, role="lesson-bundle reservation")

        queue_signatures = [
            self._reservation_signature(reservation)
            for reservation in self._checkin_queue
        ]
        if len(queue_signatures) != len(set(queue_signatures)):
            raise ValueError("checkpoint check-in queue contains a duplicate item")
        for reservation in self._checkin_queue:
            registered_item(reservation, role="check-in queue reservation")
            if reservation.surface != AssessmentSurface.CHECKIN:
                raise ValueError(
                    "checkpoint check-in queue contains a non-check-in item"
                )

        if self._current_bundle is None:
            if self._checkin_queue:
                raise ValueError(
                    "checkpoint check-in queue has no current lesson bundle"
                )
        else:
            checkins = self._current_bundle.checkins
            if len(self._checkin_queue) > len(checkins):
                raise ValueError("checkpoint check-in queue exceeds its lesson bundle")
            expected_queue = checkins[len(checkins) - len(self._checkin_queue) :]
            if self._checkin_queue != expected_queue:
                raise ValueError(
                    "checkpoint check-in queue is not the remaining bundle suffix"
                )

        pending = self._pending
        if pending is None:
            if self._verification_mode_kc is not None:
                raise ValueError(
                    "checkpoint verification mode has no pending check-in"
                )
            return

        item = registered_item(pending.reservation, role="pending reservation")
        exposure = exposures_by_key.get((item.item_id, item.revision))
        if (
            exposure is None
            or self._reservation_signature(exposure)
            != self._reservation_signature(pending.reservation)
        ):
            raise ValueError(
                "checkpoint pending reservation was not exposed exactly"
            )
        if exposure.hints_seen != pending.hints_given:
            raise ValueError(
                "checkpoint pending hint position does not match its exposure"
            )
        if exposure.solution_exposed or exposure.answer_revealed:
            raise ValueError(
                "checkpoint pending item has already revealed its scoring truth"
            )

        expected_surface = {
            "probe": AssessmentSurface.DIAGNOSTIC,
            "guided_widget": AssessmentSurface.GUIDED_WIDGET,
            "checkin": AssessmentSurface.CHECKIN,
            "capstone": AssessmentSurface.CAPSTONE,
        }[pending.kind]
        if pending.reservation.surface != expected_surface:
            raise ValueError(
                "checkpoint pending kind does not match its assessment surface"
            )

        authored_fields = {
            "kc_id": item.kc_id,
            "item_id": item.item_id,
            "item_revision": item.revision,
            "family_id": item.family_id,
            "prompt": render_assessment_prompt(item),
            "prompt_segments": [
                segment.model_dump(mode="json") for segment in item.prompt
            ],
            "answer_spec": item.answer.model_dump(mode="json"),
            "hints": [hint.text for hint in item.hints],
            "revealing_hints": [hint.revealing for hint in item.hints],
        }
        restored_fields = {
            "kc_id": pending.kc_id,
            "item_id": pending.item_id,
            "item_revision": pending.item_revision,
            "family_id": pending.family_id,
            "prompt": pending.prompt,
            "prompt_segments": [
                segment.model_dump(mode="json")
                for segment in pending.prompt_segments
            ],
            "answer_spec": pending.answer_spec.model_dump(mode="json"),
            "hints": pending.hints,
            "revealing_hints": pending.revealing_hints,
        }
        if restored_fields != authored_fields:
            raise ValueError(
                "checkpoint pending authored content does not match the pinned item bank"
            )
        if pending.hints_given < 0 or pending.hints_given > len(item.hints):
            raise ValueError("checkpoint pending hint position is invalid")
        if pending.kind != "guided_widget" and pending.delivery_mode != "widget":
            raise ValueError("checkpoint pending delivery mode is invalid")

        expected_phase = {
            "probe": SessionPhase.DIAGNOSE,
            "guided_widget": SessionPhase.TEACH,
            "checkin": SessionPhase.TEACH,
            "capstone": SessionPhase.CAPSTONE,
        }[pending.kind]
        if self.phase != expected_phase:
            raise ValueError("checkpoint pending kind does not match its phase")
        if pending.kind == "capstone" and pending.kc_id != self._target:
            raise ValueError("checkpoint capstone does not measure the target skill")

        if self._verification_mode_kc is not None:
            if (
                pending.kind != "checkin"
                or pending.kc_id != self._verification_mode_kc
            ):
                raise ValueError(
                    "checkpoint verification mode does not match its pending check-in"
                )
        elif pending.kind == "guided_widget":
            if (
                self._current_bundle is None
                or pending.reservation != self._current_bundle.guided_widget
                or self._checkin_queue != self._current_bundle.checkins
            ):
                raise ValueError(
                    "checkpoint guided practice does not match the current bundle"
                )
        elif pending.kind == "checkin":
            if self._current_bundle is None:
                raise ValueError(
                    "checkpoint lesson check-in has no current bundle"
                )
            answered_or_pending = (
                len(self._current_bundle.checkins) - len(self._checkin_queue)
            )
            if answered_or_pending < 1:
                raise ValueError(
                    "checkpoint lesson check-in has not advanced its bundle queue"
                )
            expected_pending = self._current_bundle.checkins[
                answered_or_pending - 1
            ]
            if pending.reservation != expected_pending:
                raise ValueError(
                    "checkpoint lesson check-in does not match the current bundle"
                )

    def export_checkpoint(self) -> dict[str, Any]:
        """Serialize every control-plane field needed for exact process recovery."""
        return {
            "schema_version": 2,
            "graph_version": self._graph.graph_version,
            "item_bank_version": self._bank.bank_version,
            "policy_versions": self._policy_versions(),
            "widget_capability_manifest": self._pinned_widget_capabilities,
            "episode_id": self._episode_id,
            "target_kc": self._target,
            "profile": self._profile.model_dump(mode="json"),
            "learner_id": str(self.learner.learner_id),
            "as_of": self.learner.as_of.isoformat(),
            "learner_params": self.learner.params.model_dump(mode="json"),
            "retention_half_life_days": self.learner.retention_half_life_days,
            "confirmation_window_days": self.learner.confirmation_window_days,
            "phase": self.phase.value,
            "counter": self._counter,
            "interactions_used": self._interactions_used,
            "probe_budget": self._probe_budget,
            "diagnosis": self._diag.state.model_dump(mode="json"),
            "exposure_state": self.exposure_state.model_dump(mode="json"),
            "visible_texts": list(self._visible_texts),
            "private_visible_inputs": list(self._private_visible_inputs),
            "events": [event.model_dump(mode="json") for event in self.learner.events],
            "plan": [step.model_dump(mode="json") for step in self._plan],
            "plan_index": self._plan_index,
            "pending": self._pending.model_dump(mode="json") if self._pending else None,
            "current_bundle": (
                self._current_bundle.model_dump(mode="json")
                if self._current_bundle
                else None
            ),
            "checkin_queue": [
                reservation.model_dump(mode="json")
                for reservation in self._checkin_queue
            ],
            "checkin_attempts": dict(self._checkin_attempts),
            "checkin_success_families": dict(self._checkin_success_families),
            "verification_mode_kc": self._verification_mode_kc,
            "widget_attempts": dict(self._widget_attempts),
            "learning_transition_keys": sorted(self._learning_transition_keys),
            "capstone_attempts": self._capstone_attempts,
            "stop_reason": self._stop_reason,
        }

    @classmethod
    def restore(
        cls,
        graph: GraphDocument,
        checkpoint: dict[str, Any],
        item_bank: ItemBankDocument | None = None,
    ) -> "SessionOrchestratorV2":
        """Restore an exact checkpoint without regenerating any content."""
        bank = item_bank or load_item_bank()
        if checkpoint.get("schema_version") != 2:
            raise ValueError("unsupported session checkpoint version")
        if checkpoint.get("graph_version") != graph.graph_version:
            raise ValueError("checkpoint graph version is unavailable")
        if checkpoint.get("item_bank_version") != bank.bank_version:
            raise ValueError("checkpoint item-bank version is unavailable")
        expected_policies = cls._policy_versions()
        if checkpoint.get("policy_versions") != expected_policies:
            raise ValueError("checkpoint policy implementation is unavailable")
        pinned_widget_capabilities = normalize_widget_capability_manifest(
            checkpoint.get("widget_capability_manifest", {})
        )
        orchestrator = cls(
            graph,
            checkpoint["target_kc"],
            LearnerProfile.model_validate(checkpoint["profile"]),
            item_bank=bank,
            probe_budget=int(checkpoint["probe_budget"]),
            learner_id=UUID(checkpoint["learner_id"]),
            as_of=datetime.fromisoformat(checkpoint["as_of"]),
            learner_params=BKTParams.model_validate(checkpoint["learner_params"]),
            retention_half_life_days=int(
                checkpoint["retention_half_life_days"]
            ),
            confirmation_window_days=int(
                checkpoint["confirmation_window_days"]
            ),
            impact_lambda=float(checkpoint["diagnosis"]["impact_lambda"]),
            impact_decay=float(checkpoint["diagnosis"]["impact_decay"]),
            episode_id=str(checkpoint["episode_id"]),
            widget_capabilities=pinned_widget_capabilities,
        )
        events = [
            EvidenceEvent.model_validate(payload)
            for payload in checkpoint.get("events", [])
        ]
        orchestrator.learner = orchestrator.learner.replay(events)
        orchestrator._diag = DiagnosisControllerV2(
            graph,
            orchestrator._target,
            orchestrator.learner,
            probe_budget=orchestrator._probe_budget,
            state=DiagnosisState.model_validate(checkpoint["diagnosis"]),
        )
        orchestrator.phase = SessionPhase(checkpoint["phase"])
        orchestrator._counter = int(checkpoint["counter"])
        orchestrator._interactions_used = int(checkpoint["interactions_used"])
        orchestrator.exposure_state = ContentExposureState.model_validate(
            checkpoint["exposure_state"]
        )
        visible_texts = checkpoint.get("visible_texts")
        if not isinstance(visible_texts, list) or not all(
            isinstance(value, str) for value in visible_texts
        ):
            raise ValueError("checkpoint has no valid learner-visible text ledger")
        orchestrator._visible_texts = []
        orchestrator.remember_visible_content(visible_texts)
        if orchestrator._visible_texts != visible_texts:
            raise ValueError("checkpoint learner-visible text ledger is not canonical")
        private_visible_inputs = checkpoint.get("private_visible_inputs")
        if not isinstance(private_visible_inputs, list) or not all(
            isinstance(value, str) for value in private_visible_inputs
        ):
            raise ValueError("checkpoint has no valid private visible-input ledger")
        orchestrator._private_visible_inputs = []
        orchestrator.remember_private_visible_input(private_visible_inputs)
        if orchestrator._private_visible_inputs != private_visible_inputs:
            raise ValueError("checkpoint private visible-input ledger is not canonical")
        orchestrator._plan = [
            LearningPlanStep.model_validate(step)
            for step in checkpoint.get("plan", [])
        ]
        orchestrator._plan_index = int(checkpoint.get("plan_index", 0))
        pending = checkpoint.get("pending")
        orchestrator._pending = (
            PendingInteractionV2.model_validate(pending) if pending else None
        )
        bundle = checkpoint.get("current_bundle")
        orchestrator._current_bundle = (
            LessonBundleReservation.model_validate(bundle) if bundle else None
        )
        orchestrator._checkin_queue = [
            ItemReservation.model_validate(item)
            for item in checkpoint.get("checkin_queue", [])
        ]
        orchestrator._checkin_attempts = {
            str(key): int(value)
            for key, value in checkpoint.get("checkin_attempts", {}).items()
        }
        orchestrator._checkin_success_families = {
            str(key): list(value)
            for key, value in checkpoint.get(
                "checkin_success_families", {}
            ).items()
        }
        orchestrator._verification_mode_kc = checkpoint.get(
            "verification_mode_kc"
        )
        orchestrator._widget_attempts = {
            str(key): int(value)
            for key, value in checkpoint.get("widget_attempts", {}).items()
        }
        orchestrator._learning_transition_keys = set(
            checkpoint.get("learning_transition_keys", [])
        )
        orchestrator._capstone_attempts = int(
            checkpoint.get("capstone_attempts", 0)
        )
        orchestrator._stop_reason = checkpoint.get("stop_reason")
        orchestrator._validate_checkpoint_content_references()
        return orchestrator
