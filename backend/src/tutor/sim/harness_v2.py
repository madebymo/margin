"""Simulation and activation-gate report for diagnosis policy v2."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from random import Random
from statistics import mean
from uuid import uuid4

from tutor.graph import service as graph_service
from tutor.learner.service import LearnerModelService
from tutor.learner.service_v2 import LearnerModelServiceV2
from tutor.orchestrator.diagnosis import DiagnosisController, ProbeResult
from tutor.orchestrator.diagnosis_v2 import (
    PINNED_IMPACT_DECAY,
    PINNED_IMPACT_LAMBDA,
    DiagnosticObservation,
    DiagnosisControllerV2,
)
from tutor.schemas.common import ResponseClass
from tutor.schemas.kc import GraphDocument
from tutor.schemas.learner import EvidenceEvent
from tutor.seed.load_seed import load_graph
from tutor.sim.synthetic import SyntheticLearner, generate_population

ASSUMED_FLOOR = {"Algebra 1", "Algebra 2", "Precalculus"}
POLICY_GRID = tuple(
    (impact_lambda, impact_decay)
    for impact_lambda in (0.0, 0.15, 0.35, 0.5, 1.0)
    for impact_decay in (0.25, 0.5, 0.75)
)
DEFAULT_NOISE_PROFILES = (
    (0.0, 0.0),
    (0.05, 0.05),
    (0.1, 0.15),
    (0.2, 0.2),
)


@dataclass(frozen=True)
class EpisodeMetrics:
    probes: int
    false_mastery_skip: bool
    next_kc_correct: bool
    frontier_true: int
    frontier_total: int
    overteach: int
    brier_terms: tuple[float, ...]
    v1_brier_terms: tuple[float, ...]
    calibration_points: tuple[tuple[float, bool], ...]


@dataclass(frozen=True)
class SimulationSummaryV2:
    episodes: int
    false_mastery_skip_rate: float
    frontier_precision: float
    next_kc_accuracy: float
    mean_overteach: float
    brier_score: float
    v1_brier_score: float
    brier_improvement: float
    calibration_ece: float
    median_perfect_probes: float
    p95_perfect_probes: float

    def activation_failures(self) -> list[str]:
        failures: list[str] = []
        if self.false_mastery_skip_rate > 0.05:
            failures.append("false_mastery_skip_rate > 0.05")
        if self.frontier_precision < 0.95:
            failures.append("frontier_precision < 0.95")
        if self.next_kc_accuracy < 0.75:
            failures.append("next_kc_accuracy < 0.75")
        if self.mean_overteach > 1:
            failures.append("mean_overteach > 1")
        if self.calibration_ece > 0.05:
            failures.append("calibration_ece > 0.05")
        if self.v1_brier_score and self.brier_score > 0.90 * self.v1_brier_score:
            failures.append("brier_score improvement over v1 < 10%")
        if self.median_perfect_probes > 2:
            failures.append("median_perfect_probes > 2")
        if self.p95_perfect_probes > 3:
            failures.append("p95_perfect_probes > 3")
        return failures


@dataclass(frozen=True)
class PolicySweepResult:
    impact_lambda: float
    impact_decay: float
    summary: SimulationSummaryV2

    @property
    def passed(self) -> bool:
        return not self.summary.activation_failures()


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * percentile)))
    return float(ordered[index])


def _ece(points: list[tuple[float, bool]], bins: int = 10) -> float:
    if not points:
        return 0.0
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [
            (probability, outcome)
            for probability, outcome in points
            if lower <= probability < upper or (index == bins - 1 and probability == 1)
        ]
        if bucket:
            confidence = mean(probability for probability, _ in bucket)
            accuracy = mean(outcome for _, outcome in bucket)
            error += len(bucket) / len(points) * abs(confidence - accuracy)
    return error


def run_episode_v2(
    graph: GraphDocument,
    synthetic: SyntheticLearner,
    target: str,
    budget: int,
    *,
    impact_lambda: float = PINNED_IMPACT_LAMBDA,
    impact_decay: float = PINNED_IMPACT_DECAY,
) -> EpisodeMetrics:
    baseline_synthetic = deepcopy(synthetic)
    service = LearnerModelServiceV2(
        graph, assumed_floor_levels=ASSUMED_FLOOR
    )
    controller = DiagnosisControllerV2(
        graph,
        target,
        service,
        probe_budget=budget,
        impact_lambda=impact_lambda,
        impact_decay=impact_decay,
    )
    family_number: dict[str, int] = {}
    while (selection := controller.next_probe()) is not None:
        number = family_number.get(selection.kc_id, 0)
        family_number[selection.kc_id] = number + 1
        family = f"sim.{selection.kc_id}.{number}"
        controller.record_result(
            DiagnosticObservation(
                kc_id=selection.kc_id,
                family_id=family,
                correct=synthetic.answer_correct(selection.kc_id),
            )
        )

    hard = graph_service.ancestor_subgraph(graph, target, hard_only=True)
    true_gaps = hard.node_ids() - synthetic.mastered
    plan = controller.learning_plan()
    teaching = [
        step.kc_id
        for step in plan
        if step.kind == "teach_confirmed_gap"
    ]
    verification = [
        step.kc_id for step in plan if step.kind == "verify_uncertain"
    ]
    actionable = [*teaching, *verification]
    predicted_frontier = controller.frontier()
    calibration = tuple(
        (controller.probability(kc), kc in synthetic.mastered)
        for kc in hard.node_ids()
        if controller.observations_for(kc)
    )

    baseline_service = LearnerModelService(
        graph, assumed_floor_levels=ASSUMED_FLOOR
    )
    baseline = DiagnosisController(
        graph, target, baseline_service, probe_budget=budget
    )
    while (kc_id := baseline.next_probe_kc()) is not None:
        correct = baseline_synthetic.answer_correct(kc_id)
        baseline_service.apply_event(
            EvidenceEvent(
                event_id=uuid4(),
                learner_id=baseline_service.learner_id,
                t=datetime(2025, 1, 1, tzinfo=timezone.utc),
                item_id="simulation.v1",
                kc_ids=[kc_id],
                correct=correct,
                response_class=ResponseClass.SYMBOLIC_ENTRY,
            )
        )
        baseline.record_result(ProbeResult(kc_id=kc_id, correct=correct))
    baseline_calibration = tuple(
        (baseline_service.routing_score(kc), kc in baseline_synthetic.mastered)
        for kc in hard.node_ids()
        if baseline_service.observations(kc)
    )

    return EpisodeMetrics(
        probes=controller.probes_issued,
        false_mastery_skip=(
            target in true_gaps
            and controller.status(target) == "confirmed_mastered"
        ),
        next_kc_correct=(
            (bool(actionable) and actionable[0] in true_gaps)
            if true_gaps
            else not teaching
        ),
        frontier_true=len(set(predicted_frontier) & true_gaps),
        frontier_total=len(predicted_frontier),
        overteach=len(set(teaching) - true_gaps),
        brier_terms=tuple(
            (probability - float(mastered)) ** 2
            for probability, mastered in calibration
        ),
        v1_brier_terms=tuple(
            (probability - float(mastered)) ** 2
            for probability, mastered in baseline_calibration
        ),
        calibration_points=calibration,
    )


def run_simulation_v2(
    graph: GraphDocument,
    targets: list[str],
    *,
    n: int = 200,
    budget: int = 8,
    seeds: tuple[int, ...] = (7,),
    impact_lambda: float = PINNED_IMPACT_LAMBDA,
    impact_decay: float = PINNED_IMPACT_DECAY,
    noise_profiles: tuple[tuple[float, float], ...] = ((0.1, 0.15),),
) -> SimulationSummaryV2:
    results: list[EpisodeMetrics] = []
    perfect_probes: list[int] = []
    if not noise_profiles:
        raise ValueError("at least one noise profile is required")
    if any(
        not (0 <= slip <= 1 and 0 <= guess <= 1)
        for slip, guess in noise_profiles
    ):
        raise ValueError("slip and guess probabilities must be between 0 and 1")
    for profile_index, (slip, guess) in enumerate(noise_profiles):
        for seed in seeds:
            population = generate_population(
                graph,
                n,
                seed=seed + profile_index * 1_000_003,
                slip=slip,
                guess=guess,
            )
            for target in targets:
                hard = graph_service.ancestor_subgraph(graph, target, hard_only=True)
                perfect = SyntheticLearner(
                    mastered=hard.node_ids(),
                    slip=0.0,
                    guess=0.0,
                    rng=Random(seed + profile_index * 10_007),
                )
                perfect_probes.append(
                    run_episode_v2(
                        graph,
                        perfect,
                        target,
                        budget,
                        impact_lambda=impact_lambda,
                        impact_decay=impact_decay,
                    ).probes
                )
                for synthetic in population:
                    result = run_episode_v2(
                        graph,
                        synthetic,
                        target,
                        budget,
                        impact_lambda=impact_lambda,
                        impact_decay=impact_decay,
                    )
                    results.append(result)

    frontier_true = sum(result.frontier_true for result in results)
    frontier_total = sum(result.frontier_total for result in results)
    brier = [term for result in results for term in result.brier_terms]
    v1_brier = [term for result in results for term in result.v1_brier_terms]
    calibration = [
        point for result in results for point in result.calibration_points
    ]
    brier_score = mean(brier) if brier else 0.0
    v1_brier_score = mean(v1_brier) if v1_brier else 0.0
    return SimulationSummaryV2(
        episodes=len(results),
        false_mastery_skip_rate=mean(result.false_mastery_skip for result in results),
        frontier_precision=frontier_true / frontier_total if frontier_total else 1.0,
        next_kc_accuracy=mean(result.next_kc_correct for result in results),
        mean_overteach=mean(result.overteach for result in results),
        brier_score=brier_score,
        v1_brier_score=v1_brier_score,
        brier_improvement=(
            (v1_brier_score - brier_score) / v1_brier_score
            if v1_brier_score
            else 0.0
        ),
        calibration_ece=_ece(calibration),
        median_perfect_probes=_percentile(perfect_probes, 0.5),
        p95_perfect_probes=_percentile(perfect_probes, 0.95),
    )


def sweep_policy_v2(
    graph: GraphDocument,
    targets: list[str],
    *,
    n: int,
    budget: int,
    seeds: tuple[int, ...],
    noise_profiles: tuple[tuple[float, float], ...],
    grid: tuple[tuple[float, float], ...] = POLICY_GRID,
) -> list[PolicySweepResult]:
    """Evaluate and rank every requested mutual-information impact pair."""
    results = [
        PolicySweepResult(
            impact_lambda=impact_lambda,
            impact_decay=impact_decay,
            summary=run_simulation_v2(
                graph,
                targets,
                n=n,
                budget=budget,
                seeds=seeds,
                impact_lambda=impact_lambda,
                impact_decay=impact_decay,
                noise_profiles=noise_profiles,
            ),
        )
        for impact_lambda, impact_decay in grid
    ]
    return sorted(
        results,
        key=lambda candidate: (
            not candidate.passed,
            -candidate.summary.next_kc_accuracy,
            candidate.summary.mean_overteach,
            candidate.summary.brier_score,
            candidate.summary.false_mastery_skip_rate,
            candidate.impact_lambda,
            candidate.impact_decay,
        ),
    )


def _parse_noise_profile(value: str) -> tuple[float, float]:
    try:
        slip_text, guess_text = value.split(":", 1)
        profile = (float(slip_text), float(guess_text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("noise profile must be SLIP:GUESS") from exc
    if any(probability < 0 or probability > 1 for probability in profile):
        raise argparse.ArgumentTypeError("noise probabilities must be between 0 and 1")
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learners", type=int, default=2000)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11, 17, 23])
    parser.add_argument(
        "--noise-profile",
        type=_parse_noise_profile,
        action="append",
        dest="noise_profiles",
        help="repeatable SLIP:GUESS pair; defaults cover four noise levels",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="evaluate all 15 lambda/decay candidates before the pinned policy",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=[
            "kc.der.chain_rule",
            "kc.der.product_quotient",
            "kc.int.ftc",
            "kc.int.u_substitution",
            "kc.alg.solve_quadratic",
        ],
    )
    args = parser.parse_args(argv)
    noise_profiles = tuple(args.noise_profiles or DEFAULT_NOISE_PROFILES)
    graph = load_graph()
    if args.sweep:
        candidates = sweep_policy_v2(
            graph,
            args.targets,
            n=args.learners,
            budget=args.budget,
            seeds=tuple(args.seeds),
            noise_profiles=noise_profiles,
        )
        for candidate in candidates:
            print(
                "lambda=",
                candidate.impact_lambda,
                " decay=",
                candidate.impact_decay,
                " passed=",
                candidate.passed,
                " next_kc=",
                round(candidate.summary.next_kc_accuracy, 4),
                " brier=",
                round(candidate.summary.brier_score, 4),
                sep="",
            )
        winner = candidates[0]
        print(
            "Selected candidate: "
            f"lambda={winner.impact_lambda}, decay={winner.impact_decay}"
        )
    summary = run_simulation_v2(
        graph,
        args.targets,
        n=args.learners,
        budget=args.budget,
        seeds=tuple(args.seeds),
        noise_profiles=noise_profiles,
    )
    print(summary)
    failures = summary.activation_failures()
    if failures:
        print("Activation blocked:", ", ".join(failures))
        return 1
    print("Diagnosis policy gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
