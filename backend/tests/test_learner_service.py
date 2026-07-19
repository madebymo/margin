"""BKT-lite learner model: update direction, discounts, propagation, replay."""

from datetime import datetime, timezone
from uuid import uuid4

from tutor.learner.service import LearnerModelService
from tutor.schemas.common import ResponseClass
from tutor.schemas.kc import GraphDocument, KCEdge, KCNode
from tutor.schemas.learner import EvidenceEvent


def _node(kc_id: str, level: str = "Calc I / AP Calc AB") -> KCNode:
    return KCNode(
        id=kc_id,
        name=kc_id,
        description="d",
        course_level=level,
        canonical_examples=["e"],
    )


def _graph() -> GraphDocument:
    return GraphDocument(
        graph_version=1,
        nodes=[
            _node("kc.alg.a", level="Algebra 1"),
            _node("kc.alg.b"),
            _node("kc.der.c"),
            _node("kc.alg.x"),
        ],
        edges=[
            KCEdge(from_kc="kc.alg.a", to_kc="kc.alg.b", type="hard", rationale="r"),
            KCEdge(from_kc="kc.alg.b", to_kc="kc.der.c", type="hard", rationale="r"),
            KCEdge(from_kc="kc.alg.x", to_kc="kc.der.c", type="soft", rationale="r"),
        ],
    )


def _event(
    service: LearnerModelService,
    kc_ids: list[str],
    correct: bool,
    response_class: ResponseClass = ResponseClass.SYMBOLIC_ENTRY,
    hints_used: int = 0,
) -> EvidenceEvent:
    return EvidenceEvent(
        event_id=uuid4(),
        learner_id=service.learner_id,
        t=datetime.now(timezone.utc),
        item_id="item",
        kc_ids=kc_ids,
        correct=correct,
        response_class=response_class,
        hints_used=hints_used,
        assisted=hints_used > 0,
    )


def test_correct_raises_and_incorrect_lowers():
    service = LearnerModelService(_graph())
    prior = service.routing_score("kc.der.c")
    service.apply_event(_event(service, ["kc.der.c"], correct=True))
    assert service.routing_score("kc.der.c") > prior

    other = LearnerModelService(_graph())
    other.apply_event(_event(other, ["kc.der.c"], correct=False))
    assert other.routing_score("kc.der.c") < prior


def test_assisted_answers_get_discounted_credit():
    unassisted = LearnerModelService(_graph())
    unassisted.apply_event(_event(unassisted, ["kc.der.c"], correct=True))
    assisted = LearnerModelService(_graph())
    assisted.apply_event(_event(assisted, ["kc.der.c"], correct=True, hints_used=1))
    assert assisted.routing_score("kc.der.c") < unassisted.routing_score("kc.der.c")
    assert assisted.routing_score("kc.der.c") > 0.5  # still some credit


def test_guess_rate_varies_by_response_class():
    symbolic = LearnerModelService(_graph())
    symbolic.apply_event(_event(symbolic, ["kc.der.c"], correct=True))
    multiple_choice = LearnerModelService(_graph())
    multiple_choice.apply_event(
        _event(
            multiple_choice,
            ["kc.der.c"],
            correct=True,
            response_class=ResponseClass.MULTIPLE_CHOICE,
        )
    )
    assert multiple_choice.routing_score("kc.der.c") < symbolic.routing_score("kc.der.c")


def test_propagation_is_hard_only_capped_and_insufficient_for_mastery():
    service = LearnerModelService(_graph())
    for _ in range(10):
        service.apply_event(_event(service, ["kc.der.c"], correct=True))
    snapshot = service.snapshot()
    assert snapshot.mastery["kc.alg.b"].inferred > 0  # hard ancestor, depth 1
    assert snapshot.mastery["kc.alg.a"].inferred > 0  # hard ancestor, depth 2
    assert snapshot.mastery["kc.alg.b"].inferred <= service.params.inferred_cap
    assert snapshot.mastery["kc.alg.x"].inferred == 0  # soft edge: no propagation
    # inferred evidence alone can never confirm mastery
    assert not service.is_mastered("kc.alg.b")


def test_miss_does_not_lower_ancestors():
    service = LearnerModelService(_graph())
    service.apply_event(_event(service, ["kc.der.c"], correct=False))
    snapshot = service.snapshot()
    assert snapshot.mastery["kc.alg.b"].inferred == 0
    assert snapshot.mastery["kc.alg.b"].direct == 0.5  # untouched prior


def test_multi_kc_events_are_routing_only():
    service = LearnerModelService(_graph())
    service.apply_event(_event(service, ["kc.alg.a", "kc.alg.b"], correct=True))
    assert service.observations("kc.alg.a") == 0
    assert service.observations("kc.alg.b") == 0
    assert len(service.events) == 1  # still recorded in the log


def test_floor_levels_get_assumed_prior():
    service = LearnerModelService(_graph(), assumed_floor_levels={"Algebra 1"})
    assert service.routing_score("kc.alg.a") == 0.75
    assert service.is_mastered("kc.alg.a")
    assert service.routing_score("kc.alg.b") == 0.5  # calc-level default


def test_replay_is_deterministic():
    service = LearnerModelService(_graph())
    service.apply_event(_event(service, ["kc.der.c"], correct=True))
    service.apply_event(_event(service, ["kc.der.c"], correct=False, hints_used=1))
    service.apply_event(_event(service, ["kc.alg.b"], correct=True))
    rebuilt = service.replay()
    assert rebuilt.snapshot().model_dump() == service.snapshot().model_dump()
