"""Diagnosis controller: gap localization, budgets, and path planning."""

from collections import Counter
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from tutor.learner.service import LearnerModelService
from tutor.orchestrator.diagnosis import DiagnosisController, ProbeResult
from tutor.schemas.common import ResponseClass
from tutor.schemas.kc import GraphDocument
from tutor.schemas.learner import EvidenceEvent
from tutor.seed.load_seed import load_graph

FLOOR = {"Algebra 1", "Algebra 2", "Precalculus"}
TARGET = "kc.int.u_substitution"


@pytest.fixture(scope="module")
def graph() -> GraphDocument:
    return load_graph()


def _event(service: LearnerModelService, kc: str, correct: bool) -> EvidenceEvent:
    return EvidenceEvent(
        event_id=uuid4(),
        learner_id=service.learner_id,
        t=datetime.now(timezone.utc),
        item_id="probe",
        kc_ids=[kc],
        correct=correct,
        response_class=ResponseClass.SYMBOLIC_ENTRY,
    )


def _drive(graph: GraphDocument, weak: set[str], budget: int = 8):
    learner = LearnerModelService(graph, assumed_floor_levels=FLOOR)
    controller = DiagnosisController(graph, TARGET, learner, probe_budget=budget)
    probed: list[str] = []
    while (kc := controller.next_probe_kc()) is not None:
        probed.append(kc)
        correct = kc not in weak
        learner.apply_event(_event(learner, kc, correct))
        controller.record_result(ProbeResult(kc_id=kc, correct=correct))
    return controller, learner, probed


def test_localizes_chain_rule_gap(graph):
    controller, _, probed = _drive(graph, weak={"kc.der.chain_rule", TARGET})
    assert controller.probes_issued <= 8
    # confirmations may re-probe a node, but never more than once
    assert max(Counter(probed).values()) <= 2
    assert controller.frontier() == ["kc.der.chain_rule"]

    path = controller.plan_path()
    assert path[-1] == TARGET
    assert "kc.der.chain_rule" in path
    assert "kc.fun.composition" not in path  # assumed-floor node stays out
    assert path.index("kc.der.chain_rule") < path.index(TARGET)


def test_knows_everything_short_circuits(graph):
    controller, _, probed = _drive(graph, weak=set())
    assert probed == [TARGET]
    assert controller.frontier() == []
    assert controller.plan_path() == []


def test_budget_respected_when_everything_is_weak(graph):
    weak = set(graph.node_ids())
    controller, _, probed = _drive(graph, weak=weak, budget=8)
    assert controller.probes_issued <= 8
    assert controller.finished
    assert TARGET in controller.frontier() or controller.frontier()


def test_single_slip_recovers_via_confirmation(graph):
    """A lone wrong answer (slip) must not leave a phantom gap behind."""
    learner = LearnerModelService(graph, assumed_floor_levels=FLOOR)
    controller = DiagnosisController(graph, TARGET, learner, probe_budget=8)
    first = True
    while (kc := controller.next_probe_kc()) is not None:
        correct = not first  # slip only on the very first probe (the target)
        first = False
        learner.apply_event(_event(learner, kc, correct))
        controller.record_result(ProbeResult(kc_id=kc, correct=correct))
    # confirmation re-probe recovered the slip: no gaps, nothing to teach
    assert controller.frontier() == []
    assert controller.plan_path() == []
    assert controller.probes_issued <= 4  # slip localization stays cheap
