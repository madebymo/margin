"""Smoke coverage for the v2 simulation and rollout gate report."""

from tutor.seed.load_seed import load_graph
from tutor.sim.harness_v2 import run_simulation_v2, sweep_policy_v2


def test_v2_simulation_reports_safety_metrics():
    summary = run_simulation_v2(
        load_graph(),
        ["kc.der.chain_rule"],
        n=20,
        seeds=(7,),
    )

    assert summary.episodes == 20
    assert 0 <= summary.frontier_precision <= 1
    assert 0 <= summary.next_kc_accuracy <= 1
    assert 0 <= summary.calibration_ece <= 1
    assert 0 <= summary.v1_brier_score <= 1
    assert summary.brier_improvement <= 1
    assert summary.median_perfect_probes > 0


def test_v2_simulation_covers_noise_profiles_and_ranks_policy_grid():
    graph = load_graph()
    profiles = ((0.0, 0.0), (0.2, 0.2))
    summary = run_simulation_v2(
        graph,
        ["kc.der.chain_rule"],
        n=3,
        seeds=(7,),
        noise_profiles=profiles,
    )
    assert summary.episodes == 6

    candidates = sweep_policy_v2(
        graph,
        ["kc.der.chain_rule"],
        n=2,
        budget=8,
        seeds=(7,),
        noise_profiles=((0.1, 0.15),),
        grid=((0.0, 0.25), (0.35, 0.5)),
    )
    assert {(item.impact_lambda, item.impact_decay) for item in candidates} == {
        (0.0, 0.25),
        (0.35, 0.5),
    }
