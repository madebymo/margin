"""Activity separation and independent-family mastery gates."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from tutor.learner.service_v2 import LearnerModelServiceV2
from tutor.schemas.common import ResponseClass
from tutor.schemas.learner import EvidenceEvent
from tutor.seed.load_seed import load_graph

KC = "kc.der.chain_rule"


def _event(
    learner: LearnerModelServiceV2,
    *,
    family: str,
    correct: bool = True,
    surface: str = "diagnostic",
    learning: bool = False,
    t: datetime | None = None,
    response_class: ResponseClass | None = None,
) -> EvidenceEvent:
    return EvidenceEvent(
        event_id=uuid4(),
        learner_id=learner.learner_id,
        t=t or learner.as_of,
        item_id=f"item.{family}",
        family_id=family,
        kc_ids=[KC],
        correct=correct,
        response_class=response_class or (
            ResponseClass.WIDGET
            if surface == "guided_widget"
            else ResponseClass.SYMBOLIC_ENTRY
        ),
        surface=surface,
        policy_version="v2",
        learner_params_version="v2",
        content_provenance="reviewed-bank",
        learning_opportunity=learning,
    )


def test_guided_widget_is_recorded_but_does_not_change_probability():
    learner = LearnerModelServiceV2(load_graph())
    before = learner.routing_score(KC)
    learner.apply_event(_event(learner, family="widget-a", surface="guided_widget"))

    assert len(learner.events) == 1
    assert learner.routing_score(KC) == before
    assert learner.mastery_status(KC) == "uncertain"


def test_learning_transition_only_applies_to_declared_practice():
    observed = LearnerModelServiceV2(load_graph())
    practiced = LearnerModelServiceV2(load_graph())
    observed.apply_event(_event(observed, family="a"))
    practiced.apply_event(_event(practiced, family="a"))
    practiced.apply_event(
        _event(
            practiced,
            family="lesson-a",
            surface="instructional_practice",
            learning=True,
        )
    )

    assert practiced.routing_score(KC) > observed.routing_score(KC)


def test_checkin_observation_cannot_smuggle_a_learning_transition():
    plain = LearnerModelServiceV2(load_graph())
    flagged = LearnerModelServiceV2(load_graph())
    plain.apply_event(_event(plain, family="a", surface="checkin"))
    flagged.apply_event(
        _event(flagged, family="a", surface="checkin", learning=True)
    )

    assert flagged.routing_score(KC) == plain.routing_score(KC)


def test_mastery_requires_two_distinct_recent_families():
    learner = LearnerModelServiceV2(load_graph())
    learner.apply_event(_event(learner, family="a"))
    learner.apply_event(_event(learner, family="a"))
    assert learner.mastery_status(KC) == "uncertain"

    learner.apply_event(_event(learner, family="b"))
    assert learner.mastery_status(KC) == "confirmed_mastered"


def test_recognition_items_cannot_satisfy_production_confirmation():
    learner = LearnerModelServiceV2(load_graph())
    learner.apply_event(
        _event(
            learner,
            family="choice-a",
            response_class=ResponseClass.MULTIPLE_CHOICE,
        )
    )
    learner.apply_event(
        _event(
            learner,
            family="choice-b",
            response_class=ResponseClass.MULTIPLE_CHOICE,
        )
    )

    assert learner.routing_score(KC) >= 0.9
    assert learner.recent_independent_counts(KC) == (0, 0)
    assert learner.mastery_status(KC) == "uncertain"


def test_old_families_do_not_satisfy_confirmation_window():
    learner = LearnerModelServiceV2(load_graph())
    old = learner.as_of - timedelta(days=91)
    learner.apply_event(_event(learner, family="old-a", t=old))
    learner.apply_event(_event(learner, family="old-b", t=old))

    assert learner.recent_independent_counts(KC) == (0, 0)
    assert learner.mastery_status(KC) == "uncertain"


def test_historical_evidence_decays_between_observations_before_replay():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with_old_success = LearnerModelServiceV2(load_graph(), as_of=start)
    recent_only = LearnerModelServiceV2(load_graph(), as_of=start)
    old = start - timedelta(days=360)

    with_old_success.apply_event(
        _event(with_old_success, family="old", correct=True, t=old)
    )
    with_old_success.apply_event(
        _event(with_old_success, family="recent", correct=False, t=start)
    )
    recent_only.apply_event(
        _event(recent_only, family="recent", correct=False, t=start)
    )

    prior = LearnerModelServiceV2(load_graph(), as_of=start).routing_score(KC)
    assert abs(with_old_success.routing_score(KC) - prior) < abs(
        0.9 - prior
    )
    assert with_old_success.routing_score(KC) > recent_only.routing_score(KC)


def test_legacy_evidence_only_weakly_nudges_the_prior():
    learner = LearnerModelServiceV2(load_graph())
    full = LearnerModelServiceV2(load_graph())
    learner.apply_event(_event(learner, family="legacy", surface="legacy"))
    full.apply_event(_event(full, family="reviewed"))

    assert learner.routing_score(KC) < full.routing_score(KC)
    assert learner.recent_independent_counts(KC) == (0, 0)


def test_replay_uses_fixed_as_of():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    learner = LearnerModelServiceV2(load_graph(), as_of=start)
    learner.apply_event(_event(learner, family="a", t=start - timedelta(days=180)))
    replayed = learner.replay()

    assert replayed.as_of == start
    assert replayed.routing_score(KC) == learner.routing_score(KC)
