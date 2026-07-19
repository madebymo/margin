"""Diagnostic-policy simulation harness (the audit's pre-pilot requirement).

Drives the real DiagnosisController + LearnerModelService against synthetic
learners and reports, per probe budget:
- next_kc_accuracy: the first planned KC is one the learner truly lacks
  (or the plan is empty exactly when there are no true gaps)
- frontier_soundness: every predicted-frontier node is a true gap
- mean probes used, mean overteach (planned but already known), mean missed
  (true gaps absent from the plan; the teach loop's descend can still catch
  these later — this measures diagnosis quality alone)

Run:
    python -m tutor.sim.harness --learners 200 --budgets 5 8 10
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from uuid import uuid4

from tutor.graph import service as graph_service
from tutor.learner.service import LearnerModelService
from tutor.orchestrator.diagnosis import DiagnosisController, ProbeResult
from tutor.schemas.common import ResponseClass
from tutor.schemas.kc import GraphDocument
from tutor.schemas.learner import EvidenceEvent
from tutor.seed.load_seed import load_graph
from tutor.sim.synthetic import SyntheticLearner, generate_population

ASSUMED_FLOOR = {"Algebra 1", "Algebra 2", "Precalculus"}


@dataclass
class EpisodeResult:
    """Diagnosis-stage outcome for one synthetic learner."""

    probes_used: int
    predicted_frontier: list[str]
    path: list[str]
    true_gaps: set[str]
    next_kc_ok: bool
    frontier_sound: bool
    overteach: int
    missed: int


@dataclass
class BudgetSummary:
    """Aggregated metrics for one probe budget."""

    budget: int
    episodes: int
    next_kc_accuracy: float
    frontier_soundness: float
    mean_probes: float
    mean_overteach: float
    mean_missed: float


def run_episode(
    graph: GraphDocument,
    learner: SyntheticLearner,
    target: str,
    budget: int,
) -> EpisodeResult:
    """Diagnose one synthetic learner exactly the way the session machine would."""
    service = LearnerModelService(graph, assumed_floor_levels=ASSUMED_FLOOR)
    controller = DiagnosisController(graph, target, service, probe_budget=budget)
    while (kc := controller.next_probe_kc()) is not None:
        correct = learner.answer_correct(kc)
        service.apply_event(
            EvidenceEvent(
                event_id=uuid4(),
                learner_id=service.learner_id,
                t=datetime.now(timezone.utc),
                item_id="sim-probe",
                kc_ids=[kc],
                correct=correct,
                response_class=ResponseClass.SYMBOLIC_ENTRY,
            )
        )
        controller.record_result(ProbeResult(kc_id=kc, correct=correct))

    hard = graph_service.ancestor_subgraph(graph, target, hard_only=True)
    true_gaps = {kc for kc in hard.node_ids() if kc not in learner.mastered}
    frontier = controller.frontier()
    path = controller.plan_path()
    if true_gaps:
        next_kc_ok = bool(path) and path[0] in true_gaps
    else:
        next_kc_ok = not path
    return EpisodeResult(
        probes_used=controller.probes_issued,
        predicted_frontier=frontier,
        path=path,
        true_gaps=true_gaps,
        next_kc_ok=next_kc_ok,
        frontier_sound=set(frontier) <= true_gaps,
        overteach=len(set(path) - true_gaps),
        missed=len(true_gaps - set(path)),
    )


def run_simulation(
    graph: GraphDocument,
    budgets: tuple[int, ...] = (5, 8, 10),
    target: str = "kc.int.u_substitution",
    n: int = 200,
    seed: int = 7,
    slip: float = 0.1,
    guess: float = 0.15,
    patchy_fraction: float = 0.4,
    holes: int = 2,
) -> list[BudgetSummary]:
    """Run the population against each budget (regenerated per budget: paired runs)."""
    summaries: list[BudgetSummary] = []
    for budget in budgets:
        population = generate_population(
            graph,
            n,
            seed=seed,
            patchy_fraction=patchy_fraction,
            slip=slip,
            guess=guess,
            holes=holes,
        )
        results = [run_episode(graph, learner, target, budget) for learner in population]
        summaries.append(
            BudgetSummary(
                budget=budget,
                episodes=len(results),
                next_kc_accuracy=mean(r.next_kc_ok for r in results),
                frontier_soundness=mean(r.frontier_sound for r in results),
                mean_probes=mean(r.probes_used for r in results),
                mean_overteach=mean(r.overteach for r in results),
                mean_missed=mean(r.missed for r in results),
            )
        )
    return summaries


def format_table(summaries: list[BudgetSummary]) -> str:
    """Plain-text metrics table."""
    header = (
        f"{'budget':>6}  {'episodes':>8}  {'next_kc':>7}  {'frontier_sound':>14}  "
        f"{'probes':>6}  {'overteach':>9}  {'missed':>6}"
    )
    rows = [
        f"{s.budget:>6}  {s.episodes:>8}  {s.next_kc_accuracy:>7.2f}  "
        f"{s.frontier_soundness:>14.2f}  {s.mean_probes:>6.2f}  "
        f"{s.mean_overteach:>9.2f}  {s.mean_missed:>6.2f}"
        for s in summaries
    ]
    return "\n".join([header, *rows])


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="kc.int.u_substitution")
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 8, 10])
    parser.add_argument("--learners", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--slip", type=float, default=0.1)
    parser.add_argument("--guess", type=float, default=0.15)
    parser.add_argument("--patchy-fraction", type=float, default=0.4)
    parser.add_argument("--holes", type=int, default=2)
    args = parser.parse_args(argv)

    graph = load_graph()
    summaries = run_simulation(
        graph,
        budgets=tuple(args.budgets),
        target=args.target,
        n=args.learners,
        seed=args.seed,
        slip=args.slip,
        guess=args.guess,
        patchy_fraction=args.patchy_fraction,
        holes=args.holes,
    )
    print(format_table(summaries))
    print(
        "\nnext_kc: fraction of episodes whose first planned KC is a true gap "
        "(empty plan iff no gaps). frontier_sound: predicted frontier is a subset "
        "of true gaps. missed gaps remain catchable by the teach loop's descend."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
