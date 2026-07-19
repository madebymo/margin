"""Diagnostic-policy simulation: generators, episode metrics, determinism."""

from random import Random

import pytest

from tutor.schemas.common import EdgeType
from tutor.seed.load_seed import load_graph
from tutor.sim.harness import run_episode, run_simulation
from tutor.sim.synthetic import (
    SyntheticLearner,
    generate_population,
    make_monotone_mastery,
)

TARGET = "kc.int.u_substitution"


@pytest.fixture(scope="module")
def graph():
    return load_graph()


def test_monotone_mastery_respects_hard_closure(graph):
    mastered = make_monotone_mastery(graph, Random(3))
    for edge in graph.edges:
        if edge.type == EdgeType.HARD and edge.to_kc in mastered:
            assert edge.from_kc in mastered, (edge.from_kc, edge.to_kc)


def test_population_is_deterministic(graph):
    first = generate_population(graph, 6, seed=11)
    second = generate_population(graph, 6, seed=11)
    assert [learner.mastered for learner in first] == [
        learner.mastered for learner in second
    ]


def test_perfect_learner_short_circuits(graph):
    learner = SyntheticLearner(mastered=set(graph.node_ids()), rng=Random(0))
    result = run_episode(graph, learner, TARGET, budget=8)
    assert result.probes_used == 1
    assert result.path == []
    assert result.next_kc_ok
    assert result.overteach == 0 and result.missed == 0


def test_chain_rule_gap_learner_is_localized(graph):
    # Unmastered set is upward-closed (chain rule and everything above it),
    # so the ground truth stays monotone.
    unmastered = {
        "kc.der.chain_rule",
        "kc.der.implicit_differentiation",
        "kc.int.recognizing_composite",
        "kc.int.u_substitution",
        "kc.int.u_sub_definite",
    }
    learner = SyntheticLearner(
        mastered=set(graph.node_ids()) - unmastered, rng=Random(0)
    )
    result = run_episode(graph, learner, TARGET, budget=8)
    assert result.probes_used <= 8
    assert result.predicted_frontier == ["kc.der.chain_rule"]
    assert result.frontier_sound
    assert "kc.der.chain_rule" in result.path
    assert result.missed == 0
    assert result.next_kc_ok


def test_frontier_is_sound_when_slip_is_zero(graph):
    # With slip=0, a mastered learner never answers wrong, so an observed-bad
    # node is always a true gap — the predicted frontier must be sound.
    summaries = run_simulation(
        graph, budgets=(8,), n=20, seed=13, slip=0.0, guess=0.3
    )
    assert summaries[0].frontier_soundness == 1.0


def test_simulation_is_deterministic(graph):
    kwargs = dict(budgets=(5, 8), n=12, seed=21, slip=0.1, guess=0.15)
    assert run_simulation(graph, **kwargs) == run_simulation(graph, **kwargs)
