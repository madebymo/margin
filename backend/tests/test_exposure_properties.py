"""Generated invariants for family allocation and cross-episode exposure."""

from datetime import timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from tutor.content.exposure import AllocationError, ItemAllocator
from tutor.content.item_bank import validate_item_bank
from tutor.orchestrator.session_v2 import SessionOrchestratorV2
from tutor.schemas.assessment import (
    AssessmentSurface,
    ContentExposureState,
)
from tutor.schemas.learner import LearnerProfile

from tests.v2_helpers import (
    POWER_RULE_KC,
    approved_power_rule_bank,
    power_rule_only_graph,
)

_SURFACES = tuple(AssessmentSurface)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.sampled_from(_SURFACES), min_size=1, max_size=30))
def test_generated_allocation_sequences_never_reuse_a_family(surfaces):
    allocator = ItemAllocator(approved_power_rule_bank())
    state = ContentExposureState()
    used: set[str] = set()

    for index, surface in enumerate(surfaces):
        try:
            allocation = allocator.reserve_item(
                state,
                kc_id=POWER_RULE_KC,
                surface=surface,
            )
        except AllocationError:
            continue
        assert allocation.reservation.family_id not in used
        used.add(allocation.reservation.family_id)
        state = allocator.record_exposure(
            allocation.state,
            allocation.reservation,
            hints_seen=index % 4,
            answer_revealed=index % 4 == 3,
        )

    assert len(state.used_family_ids) == len(state.reservations)
    assert state.retired_family_ids <= state.used_family_ids


@settings(max_examples=24, deadline=None)
@given(
    diagnostic_prefix=st.integers(min_value=0, max_value=3),
    capstone_prefix=st.integers(min_value=0, max_value=2),
)
def test_generated_lesson_bundles_are_disjoint_from_every_prior_surface(
    diagnostic_prefix,
    capstone_prefix,
):
    bank = approved_power_rule_bank()
    graph = power_rule_only_graph()
    assert validate_item_bank(bank, graph) == []
    allocator = ItemAllocator(bank)
    state = ContentExposureState()
    for surface, count in (
        (AssessmentSurface.DIAGNOSTIC, diagnostic_prefix),
        (AssessmentSurface.CAPSTONE, capstone_prefix),
    ):
        for _ in range(count):
            state = allocator.reserve_item(
                state,
                kc_id=POWER_RULE_KC,
                surface=surface,
            ).state

    prior_families = state.used_family_ids
    allocated = allocator.reserve_lesson_bundle(state, POWER_RULE_KC)
    reservations = (
        allocated.bundle.worked_example,
        allocated.bundle.guided_widget,
        *allocated.bundle.checkins,
    )
    bundle_families = {reservation.family_id for reservation in reservations}

    assert len(bundle_families) == len(reservations)
    assert not (bundle_families & prior_families)


@settings(max_examples=12, deadline=None)
@given(hints_seen=st.integers(min_value=0, max_value=3))
def test_any_shown_diagnostic_family_stays_unavailable_after_restart(hints_seen):
    session = SessionOrchestratorV2(
        power_rule_only_graph(),
        POWER_RULE_KC,
        LearnerProfile(course="AP Calculus AB", age_band="16-18"),
        item_bank=approved_power_rule_bank(),
    )
    session.begin()
    shown_family = session.pending.family_id
    for _ in range(hints_seen):
        assert session.hint()

    fresh = session.fresh_episode(
        as_of=session.learner.as_of + timedelta(seconds=1)
    )
    fresh.begin()

    assert shown_family in fresh.exposure_state.used_family_ids
    assert fresh.pending.family_id != shown_family
