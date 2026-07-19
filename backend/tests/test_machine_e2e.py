"""End-to-end sessions: diagnose -> plan -> teach -> capstone -> done."""

import pytest

from tutor.orchestrator.machine import SessionOrchestrator, SessionPhase
from tutor.schemas.learner import LearnerProfile
from tutor.seed.load_seed import load_graph

TARGET = "kc.int.u_substitution"
PROFILE = LearnerProfile(course="AP Calculus AB", age_band="16-18")


@pytest.fixture(scope="module")
def graph():
    return load_graph()


def _drive(orchestrator: SessionOrchestrator, weak_during_diagnosis: set[str]) -> None:
    orchestrator.begin()
    guard = 0
    while orchestrator.phase not in (SessionPhase.DONE, SessionPhase.STOPPED):
        guard += 1
        assert guard < 200, "session did not terminate"
        assert orchestrator.pending_expected is not None
        if (
            orchestrator.phase == SessionPhase.DIAGNOSE
            and orchestrator.pending_kc in weak_during_diagnosis
        ):
            answer = "totally wrong"
        else:
            answer = orchestrator.pending_expected
        orchestrator.submit(answer)


def test_full_session_reaches_done(graph):
    orchestrator = SessionOrchestrator(graph, TARGET, PROFILE)
    _drive(orchestrator, weak_during_diagnosis={"kc.der.chain_rule", TARGET})

    assert orchestrator.phase == SessionPhase.DONE
    summary = orchestrator.summary()
    assert summary["frontier"] == ["kc.der.chain_rule"]
    assert "kc.der.chain_rule" in summary["mastered_in_session"]
    assert summary["path"][-1] == TARGET
    assert summary["interactions_used"] <= 40
    assert orchestrator.learner.is_mastered(TARGET)


def test_replaying_the_evidence_log_reproduces_state(graph):
    orchestrator = SessionOrchestrator(graph, TARGET, PROFILE)
    _drive(orchestrator, weak_during_diagnosis={"kc.der.chain_rule", TARGET})
    rebuilt = orchestrator.learner.replay()
    assert rebuilt.snapshot().model_dump() == orchestrator.learner.snapshot().model_dump()


def test_knows_everything_goes_straight_to_capstone(graph):
    orchestrator = SessionOrchestrator(graph, TARGET, PROFILE)
    _drive(orchestrator, weak_during_diagnosis=set())

    assert orchestrator.phase == SessionPhase.DONE
    summary = orchestrator.summary()
    assert summary["path"] == []
    assert summary["probes_used"] == 1


def test_hints_mark_the_response_assisted(graph):
    orchestrator = SessionOrchestrator(graph, TARGET, PROFILE)
    orchestrator.begin()
    first_hint = orchestrator.hint()
    second_hint = orchestrator.hint()
    assert first_hint and second_hint and first_hint != second_hint
    orchestrator.submit(orchestrator.pending_expected)
    last_event = orchestrator.learner.events[-1]
    assert last_event.hints_used == 2
    assert last_event.assisted is True
