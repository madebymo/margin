"""LLM-backed ports: gating, fallbacks, id validation, machine integration."""

import pytest

from tutor.llm.client import LLMError, extract_json
from tutor.llm.diagnostician import LLMDiagnostician
from tutor.llm.lesson_writer import LLMLessonWriter
from tutor.orchestrator.machine import SessionOrchestrator, SessionPhase
from tutor.orchestrator.ports import TemplateDiagnostician, TemplateLessonWriter
from tutor.schemas.learner import LearnerProfile
from tutor.schemas.pedagogy import Metaphor, Misconception, PedagogyPack
from tutor.seed.load_seed import load_graph

PROFILE = LearnerProfile(course="AP Calculus AB", age_band="16-18")

VALID_PROBE = {
    "scaffold_steps": ["Differentiate: d/dx x^3 = ?", "____"],
    "blank_index": 1,
    "expected": "3*x^2",
    "checker": "sympy_equiv",
    "hint_ladder": ["What rule applies?", "Bring the power down.", "d/dx x^n = n*x^(n-1)"],
}
VALID_CHECKIN = {
    "prompt": "Differentiate x^4 with respect to x.",
    "expected": "4*x^3",
    "checker": "sympy_equiv",
    "hints": ["Power rule.", "Bring the 4 down.", "4*x^(4-1)"],
}


class FakeLLM:
    """Tag-prefix-routed fake client. Values: dict, list of dicts, or Exception."""

    def __init__(self, handlers: dict[str, object]) -> None:
        self._handlers = handlers
        self.calls: list[str] = []

    def complete_json(self, *, system: str, user: str, tag: str) -> dict:
        self.calls.append(tag)
        for prefix, response in self._handlers.items():
            if tag.startswith(prefix):
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, list):
                    if not response:
                        raise LLMError("handler exhausted")
                    return response.pop(0)
                return dict(response)
        raise LLMError(f"no handler for {tag}")


@pytest.fixture(scope="module")
def graph():
    return load_graph()


@pytest.fixture(scope="module")
def chain_rule_pack() -> dict[str, PedagogyPack]:
    pack = PedagogyPack(
        kc_id="kc.der.chain_rule",
        misconceptions=[
            Misconception(
                id="m.chain.outer_only",
                description="Differentiates the outer function but forgets the inner derivative",
                error_signature="answer missing the inner derivative factor",
                remediation_hint="Multiply by the derivative of the inside function",
            )
        ],
        metaphors=[
            Metaphor(
                id="met.gears",
                description="Nested gears: the outer gear turns at a rate scaled by the inner gear",
                widget_affinity=["slider"],
            )
        ],
    )
    return {pack.kc_id: pack}


def _node(graph, kc_id):
    return next(node for node in graph.nodes if node.id == kc_id)


def test_extract_json_tolerates_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(LLMError):
        extract_json("no json here")


def test_llm_probe_valid(graph):
    diagnostician = LLMDiagnostician(FakeLLM({"probe:": VALID_PROBE}), graph=graph)
    node = _node(graph, "kc.der.power_rule")
    probe = diagnostician.generate_probe(node)
    assert probe.kc_id == "kc.der.power_rule"
    assert probe.expected == "3*x^2"
    assert probe.scaffold_steps[probe.blank_index] == "____"


def test_llm_probe_bad_math_falls_back(graph):
    bad = {**VALID_PROBE, "expected": "x***"}
    diagnostician = LLMDiagnostician(FakeLLM({"probe:": bad}), graph=graph)
    node = _node(graph, "kc.der.power_rule")
    probe = diagnostician.generate_probe(node)
    template = TemplateDiagnostician().generate_probe(node)
    assert probe.expected == template.expected  # fell back


def test_llm_probe_client_error_falls_back(graph):
    diagnostician = LLMDiagnostician(FakeLLM({"probe:": LLMError("boom")}), graph=graph)
    node = _node(graph, "kc.der.power_rule")
    probe = diagnostician.generate_probe(node)
    assert probe.probe_id == f"probe.{node.id}"  # template id, not probe.llm.*


def test_llm_probe_answer_leak_falls_back(graph):
    leaky = {**VALID_PROBE, "hint_ladder": ["look", "the answer is 3*x^2", "..."]}
    diagnostician = LLMDiagnostician(FakeLLM({"probe:": leaky}), graph=graph)
    node = _node(graph, "kc.der.power_rule")
    probe = diagnostician.generate_probe(node)
    assert probe.probe_id == f"probe.{node.id}"


def test_error_analysis_validates_ids(graph, chain_rule_pack):
    node = _node(graph, "kc.der.chain_rule")
    good = FakeLLM(
        {
            "error:": {
                "misconception_id": "m.chain.outer_only",
                "implicated_prereq": "kc.der.power_rule",
                "rationale": "missing inner derivative",
            }
        }
    )
    analysis = LLMDiagnostician(good, graph=graph, packs=chain_rule_pack).analyze_error(
        node, "d/dx sin(x^2) = ?", "2x cos(x^2)", "cos(x^2)"
    )
    assert analysis.misconception_id == "m.chain.outer_only"
    assert analysis.implicated_prereq == "kc.der.power_rule"  # a hard predecessor

    bogus = FakeLLM(
        {
            "error:": {
                "misconception_id": "m.invented.nonsense",
                "implicated_prereq": "kc.int.u_substitution",  # descendant, not prereq
            }
        }
    )
    analysis = LLMDiagnostician(bogus, graph=graph, packs=chain_rule_pack).analyze_error(
        node, "prompt", "2x cos(x^2)", "cos(x^2)"
    )
    assert analysis.misconception_id is None
    assert analysis.implicated_prereq is None


def test_lesson_writer_narrative_and_checkin(graph, chain_rule_pack):
    fake = FakeLLM(
        {
            "lesson:": {"narrative": "Think of nested gears turning together..."},
            "checkin:": VALID_CHECKIN,
        }
    )
    writer = LLMLessonWriter(fake, packs=chain_rule_pack)
    node = _node(graph, "kc.der.chain_rule")
    assert writer.lesson_text(node).startswith("Think of nested gears")
    item = writer.checkin_item(node, 0)
    assert item.expected == "4*x^3"
    assert len(item.hints) == 3


def test_lesson_writer_bad_checkin_falls_back(graph):
    bad = {**VALID_CHECKIN, "expected": "not )( parseable ["}
    writer = LLMLessonWriter(FakeLLM({"checkin:": bad}))
    node = _node(graph, "kc.der.chain_rule")
    item = writer.checkin_item(node, 0)
    template = TemplateLessonWriter().checkin_item(node, 0)
    assert item.expected == template.expected  # fell back


def test_machine_with_llm_ports_descend_misconception_and_done(graph, chain_rule_pack):
    fake = FakeLLM(
        {
            "probe:": VALID_PROBE,
            "error:": {
                "misconception_id": "m.chain.outer_only",
                "implicated_prereq": "kc.der.power_rule",
            },
            "lesson:": {"narrative": "Nested gears: outer times inner."},
            "checkin:": VALID_CHECKIN,
        }
    )
    orchestrator = SessionOrchestrator(
        graph,
        "kc.der.chain_rule",
        PROFILE,
        diagnostician=LLMDiagnostician(fake, graph=graph, packs=chain_rule_pack),
        lesson_writer=LLMLessonWriter(fake, packs=chain_rule_pack),
    )
    # answer wrong N times per (kind, kc); else answer correctly. The probe must
    # be missed twice: policy v1.1 re-probes a lone miss (confirmation), and a
    # correct confirmation would recover the node as a slip.
    wrong_budget = {
        ("probe", "kc.der.chain_rule"): 2,
        ("checkin", "kc.der.chain_rule"): 1,
    }
    orchestrator.begin()
    guard = 0
    while orchestrator.phase not in (SessionPhase.DONE, SessionPhase.STOPPED):
        guard += 1
        assert guard < 100, "session did not terminate"
        key = (orchestrator.pending_kind, orchestrator.pending_kc)
        if wrong_budget.get(key, 0) > 0:
            wrong_budget[key] -= 1
            orchestrator.submit("wrong answer")
        else:
            orchestrator.submit(orchestrator.pending_expected)

    assert orchestrator.phase == SessionPhase.DONE
    summary = orchestrator.summary()
    # the wrong check-in descended into power rule and came back
    assert orchestrator.envelope.inserted == ["kc.der.power_rule"]
    assert "kc.der.power_rule" in summary["mastered_in_session"]
    assert "kc.der.chain_rule" in summary["mastered_in_session"]
    # the misconception from error analysis landed in the learner model
    assert "m.chain.outer_only" in orchestrator.learner.snapshot().misconception_flags
