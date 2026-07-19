"""Routing: unit rules and hypothesis property tests over random sequences."""

from hypothesis import given, settings
from hypothesis import strategies as st

from tutor.orchestrator.envelope import CheckinOutcome, EpisodeEnvelope, RoutingAction
from tutor.orchestrator.routing import route

ANCESTORS = {"kc.alg.a", "kc.alg.b"}


def _outcome(
    key: str,
    correct: bool,
    consecutive: int = 0,
    implicated: str | None = None,
    kc: str = "kc.der.c",
) -> CheckinOutcome:
    return CheckinOutcome(
        kc_id=kc,
        correct=correct,
        interaction_key=key,
        consecutive_correct=consecutive,
        implicated_prereq=implicated,
    )


def test_duplicate_key_is_noop():
    env = EpisodeEnvelope(target_kc="kc.der.c")
    _, env = route(env, _outcome("k1", correct=True, consecutive=1), ANCESTORS)
    used_before = env.interactions_used
    decision, env_after = route(env, _outcome("k1", correct=True, consecutive=1), ANCESTORS)
    assert decision.action == RoutingAction.DUPLICATE
    assert env_after.interactions_used == used_before


def test_correct_continues_then_advances():
    env = EpisodeEnvelope(target_kc="kc.der.c")
    decision, env = route(env, _outcome("k1", correct=True, consecutive=1), ANCESTORS)
    assert decision.action == RoutingAction.CONTINUE
    decision, env = route(env, _outcome("k2", correct=True, consecutive=2), ANCESTORS)
    assert decision.action == RoutingAction.ADVANCE


def test_descend_once_then_never_same_prereq_again():
    env = EpisodeEnvelope(target_kc="kc.der.c")
    decision, env = route(
        env, _outcome("k1", correct=False, implicated="kc.alg.a"), ANCESTORS
    )
    assert decision.action == RoutingAction.DESCEND
    assert decision.descend_to == "kc.alg.a"
    assert env.inserted == ["kc.alg.a"]
    assert env.resume_stack == ["kc.der.c"]
    # same implication again: already inserted -> retry instead
    decision, env = route(
        env, _outcome("k2", correct=False, implicated="kc.alg.a"), ANCESTORS
    )
    assert decision.action == RoutingAction.RETRY


def test_descend_requires_strict_ancestor():
    env = EpisodeEnvelope(target_kc="kc.der.c")
    decision, _ = route(
        env, _outcome("k1", correct=False, implicated="kc.der.zzz"), ANCESTORS
    )
    assert decision.action == RoutingAction.RETRY


def test_retry_cap_then_fallback():
    env = EpisodeEnvelope(target_kc="kc.der.c")
    decision, env = route(env, _outcome("k1", correct=False), ANCESTORS)
    assert decision.action == RoutingAction.RETRY
    decision, env = route(env, _outcome("k2", correct=False), ANCESTORS)
    assert decision.action == RoutingAction.RETRY
    decision, env = route(env, _outcome("k3", correct=False), ANCESTORS)
    assert decision.action == RoutingAction.FALLBACK


def test_stop_on_budget_breach():
    env = EpisodeEnvelope(target_kc="kc.der.c", interaction_budget=2)
    _, env = route(env, _outcome("k1", correct=True, consecutive=1), ANCESTORS)
    _, env = route(env, _outcome("k2", correct=True, consecutive=1), ANCESTORS)
    decision, env = route(env, _outcome("k3", correct=True, consecutive=1), ANCESTORS)
    assert decision.action == RoutingAction.STOP


def test_input_envelope_is_not_mutated():
    env = EpisodeEnvelope(target_kc="kc.der.c")
    route(env, _outcome("k1", correct=False, implicated="kc.alg.a"), ANCESTORS)
    assert env.interactions_used == 0
    assert env.inserted == []
    assert env.seen_interaction_keys == []


@settings(deadline=None, max_examples=100)
@given(steps=st.lists(st.tuples(st.booleans(), st.integers(0, 3)), min_size=1, max_size=80))
def test_random_sequences_terminate_and_respect_caps(steps):
    ancestors = {f"kc.alg.p{i}" for i in range(4)}
    env = EpisodeEnvelope(target_kc="kc.der.c", interaction_budget=30)
    consecutive = 0
    stopped = False
    for index, (correct, implicated_index) in enumerate(steps):
        if stopped:
            break
        consecutive = consecutive + 1 if correct else 0
        implicated = f"kc.alg.p{implicated_index}" if not correct else None
        decision, env = route(
            env,
            _outcome(f"k{index}", correct, consecutive, implicated),
            ancestors,
        )
        assert decision.action in RoutingAction
        assert env.interactions_used <= env.interaction_budget + 1
        assert len(env.inserted) == len(set(env.inserted))
        assert len(env.inserted) <= env.max_inserts
        assert set(env.inserted) <= ancestors
        assert all(count <= env.max_retries_per_kc for count in env.retries.values())
        assert len(env.resume_stack) <= env.max_detour_depth
        if decision.action == RoutingAction.STOP:
            stopped = True
    assert env.interactions_used <= env.interaction_budget + 1
