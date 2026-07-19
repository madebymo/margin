"""Versioned prompts for the LLM call sites.

PROMPT_VERSION is pinned into generated-content provenance; bump it whenever
any prompt in this module changes.
"""

from tutor.schemas.kc import KCNode
from tutor.schemas.learner import LearnerProfile
from tutor.schemas.pedagogy import Metaphor, Misconception, PedagogyPack

PROMPT_VERSION = "p3"

MATH_FORMAT_RULES = (
    "All math must be plain ASCII parseable by SymPy: use ^ or ** for powers, "
    "* for products where ambiguity is possible, and only the functions "
    "sin, cos, tan, sec, exp, log, ln, sqrt. No LaTeX, no unicode symbols, "
    "and no equals sign inside 'expected'."
)

PROBE_SYSTEM = f"""You are the diagnostician inside an adaptive math tutoring system.
You write ONE short scaffolded probe that tests exactly one knowledge component (KC).
Output ONLY a JSON object — no markdown fences, no commentary.
Keys:
- "scaffold_steps": 2-4 strings; exactly one must be the literal blank "____"
- "blank_index": 0-based index of the blank step
- "expected": the answer that fills the blank
- "checker": "sympy_equiv" or "numeric"
- "hint_ladder": exactly 3 strings — a gentle nudge, a targeted hint, a worked sub-step
{MATH_FORMAT_RULES}
The probe must test the target KC directly, be solvable in under a minute,
match the student's course level, and must not reveal the expected answer in
the visible steps or the first two hints."""

ERROR_SYSTEM = """You are the error analyst inside an adaptive math tutoring system.
Given one math item, its expected answer, and a student's wrong answer, identify:
1. the most likely misconception, chosen ONLY from the provided misconception ids
2. the single prerequisite KC the error points at, chosen ONLY from the provided kc ids
Output ONLY JSON: {"misconception_id": <id or null>, "implicated_prereq": <kc id or null>, "rationale": <one short sentence>}
Use null whenever you are unsure. Never invent ids. If the error looks like a
slip (typo, sign error, dropped constant) rather than a knowledge gap, use null
for both fields."""

LESSON_SYSTEM = """You are the lesson writer inside an adaptive math tutoring system.
Write the narrative for ONE mini-lesson on one knowledge component.
Output ONLY JSON: {"narrative": <string>}
The narrative must be 120-220 words, age-appropriate, plain text with ASCII
math (like x^2), build on ONE metaphor consistently when one is provided,
address the student directly, and end by bridging into a practice question.
No headings, no LaTeX, no bullet lists."""

CHECKIN_SYSTEM = f"""You are the check-in writer inside an adaptive math tutoring system.
Write ONE practice question as a near-transfer variation for the given KC.
Output ONLY JSON: {{"prompt": <string>, "expected": <string>, "checker": "sympy_equiv" or "numeric", "hints": [<3 strings>]}}
{MATH_FORMAT_RULES}
Keep numbers small, keep it solvable in under a minute, do not include the
expected answer inside the prompt, and make different attempt indexes produce
different surface features (numbers, functions) of the same skill."""


def _profile_lines(profile: LearnerProfile | None) -> list[str]:
    if profile is None:
        return ["Student profile: unknown"]
    return [f"Student course: {profile.course}; age band: {profile.age_band}"]


def probe_user(
    node: KCNode,
    profile: LearnerProfile | None = None,
    pack: PedagogyPack | None = None,
) -> str:
    """Build the user message for probe generation."""
    lines = [
        f"Target KC: {node.id} — {node.name}",
        f"Description: {node.description}",
        f"Course level: {node.course_level}",
        "Canonical examples:",
        *[f"- {example}" for example in node.canonical_examples],
        *_profile_lines(profile),
    ]
    if pack and pack.misconceptions:
        lines.append("Known misconceptions (make the probe sensitive to them):")
        lines.extend(f"- {m.id}: {m.description}" for m in pack.misconceptions)
    lines.append("Write the probe JSON now.")
    return "\n".join(lines)


def error_user(
    node: KCNode,
    prompt: str,
    expected: str,
    answer: str,
    misconceptions: list[Misconception],
    prereq_candidates: list[str],
) -> str:
    """Build the user message for error analysis."""
    lines = [
        f"KC being tested: {node.id} — {node.name}",
        f"Item shown to the student:\n{prompt}",
        f"Expected answer: {expected}",
        f"Student answer: {answer}",
        "Allowed misconception ids:",
        *([f"- {m.id}: {m.error_signature}" for m in misconceptions] or ["- (none known)"]),
        "Allowed prerequisite kc ids:",
        *([f"- {kc}" for kc in prereq_candidates] or ["- (none)"]),
        "Analyze the error and output the JSON now.",
    ]
    return "\n".join(lines)


def lesson_user(
    node: KCNode,
    profile: LearnerProfile | None = None,
    metaphor: Metaphor | None = None,
    misconceptions: list[Misconception] | None = None,
) -> str:
    """Build the user message for lesson narrative generation."""
    lines = [
        f"KC: {node.id} — {node.name}",
        f"Description: {node.description}",
        "Worked examples to weave in:",
        *[f"- {example}" for example in node.canonical_examples],
        *_profile_lines(profile),
    ]
    if metaphor is not None:
        lines.append(f"Primary metaphor (use it consistently): {metaphor.description}")
    if misconceptions:
        lines.append("Pre-empt these misconceptions:")
        lines.extend(f"- {m.description}" for m in misconceptions)
    lines.append("Write the lesson JSON now.")
    return "\n".join(lines)


def checkin_user(
    node: KCNode,
    attempt: int,
    profile: LearnerProfile | None = None,
    pack: PedagogyPack | None = None,
) -> str:
    """Build the user message for check-in generation."""
    lines = [
        f"KC: {node.id} — {node.name}",
        f"Description: {node.description}",
        "Reference examples:",
        *[f"- {example}" for example in node.canonical_examples],
        *_profile_lines(profile),
        f"Attempt index (vary surface features by this): {attempt}",
    ]
    if pack and pack.misconceptions:
        lines.append("If natural, make plausible wrong paths correspond to:")
        lines.extend(f"- {m.id}: {m.description}" for m in pack.misconceptions)
    lines.append("Write the check-in JSON now.")
    return "\n".join(lines)


INTERACTION_SYSTEM = f"""You are the interaction generator inside an adaptive math tutoring system.
Design 2-3 candidate interactive widgets for ONE mini-lesson.
Output ONLY JSON: {{"candidates": [<widget>, ...]}}
Each widget must be exactly one of these shapes (discriminated by "widget_type"):
- {{"widget_type": "slider", "learning_objective": str, "prompt": str, "params": {{"min": num, "max": num, "step": num > 0, "plot": str or null}}, "success_condition": {{"target": num, "tolerance": num >= 0}}}}
- {{"widget_type": "click_region", "learning_objective": str, "prompt": str, "regions": [{{"id": str, "label": str}}, ...at least 2], "correct_region_ids": [str, ...]}}
- {{"widget_type": "mapping", "learning_objective": str, "prompt": str, "left": [str, ...at least 2], "right": [str, ...at least 2], "correct_pairs": [[left_item, right_item], ...]}}
- {{"widget_type": "live_input", "learning_objective": str, "prompt": str, "input_kind": "number" or "expression", "checker": {{"equivalence": "sympy_equiv" or "numeric", "expected": str}}}}
{MATH_FORMAT_RULES}
The interaction must let the student DO the skill (production), not merely
recognize it. Never include an expected answer inside any prompt or label.
When repair feedback is provided, fix exactly those problems."""

EVALUATOR_SYSTEM = """You are the content evaluator inside an adaptive math tutoring system.
Judge ONE widget candidate against its mini-lesson. Be adversarial: reject on
any doubt about mathematical correctness.
Output ONLY JSON:
{"hard": {"correctness": bool, "alignment": bool, "consistency": bool, "safety": bool},
 "soft": {"clarity": 1-5, "scaffolding": 1-5, "cognitive_load": 1-5, "engagement": 1-5, "age_fit": 1-5},
 "abstain": bool, "feedback": <one short sentence>}
Hard gates: correctness = the math and the expected/success values are right,
including domains; alignment = the widget exercises exactly the lesson's KC;
consistency = prompt, answer checking, and objective agree; safety = content
appropriate for a school-age student. Set abstain=true when unsure — an
abstention counts as a rejection."""


def interaction_user(
    node: KCNode,
    attempt: int,
    feedback: list[str],
    preferred_types: list[str] | None = None,
    pack: PedagogyPack | None = None,
    profile: LearnerProfile | None = None,
) -> str:
    """Build the user message for interaction generation."""
    lines = [
        f"KC: {node.id} — {node.name}",
        f"Description: {node.description}",
        "Reference examples:",
        *[f"- {example}" for example in node.canonical_examples],
        *_profile_lines(profile),
        f"Attempt index: {attempt}",
    ]
    if preferred_types:
        lines.append(f"Preferred widget types for this KC: {', '.join(preferred_types)}")
    if pack and pack.metaphors:
        lines.append(f"Primary metaphor: {pack.metaphors[0].description}")
    if pack and pack.misconceptions:
        lines.append("Target these misconceptions where natural:")
        lines.extend(f"- {m.id}: {m.description}" for m in pack.misconceptions)
    if feedback:
        lines.append("Repair feedback from previous rejected candidates:")
        lines.extend(f"- {item}" for item in feedback[-5:])
    lines.append("Write the candidates JSON now.")
    return "\n".join(lines)


def evaluator_user(node: KCNode, narrative: str, widget_json: str) -> str:
    """Build the user message for widget evaluation."""
    excerpt = narrative if len(narrative) <= 600 else narrative[:600] + "…"
    return "\n".join(
        [
            f"KC: {node.id} — {node.name}",
            f"Description: {node.description}",
            f"Lesson narrative (excerpt):\n{excerpt}",
            f"Widget candidate JSON:\n{widget_json}",
            "Evaluate and output the verdict JSON now.",
        ]
    )


PACK_SYSTEM = """You are the pedagogy-pack compiler inside an adaptive math tutoring system.
From the provided source excerpts (and established math-education knowledge),
compile ONE knowledge component's pedagogy pack.
Output ONLY JSON:
{"misconceptions": [{"slug": <snake_case str>, "description": <str>, "error_signature": <what the wrong answer looks like>, "remediation_hint": <str>}, ...2-4 items],
 "metaphors": [{"slug": <snake_case str>, "description": <str>, "widget_affinity": [subset of "slider", "click_region", "mapping", "live_input"]}, ...1-2 items],
 "error_patterns": [<str>, ...up to 4 items]}
Misconceptions must be genuine conceptual student misconceptions for THIS KC
(not typos or careless slips), each with a distinct, detectable error
signature. Ground the content in the excerpts when they are relevant;
otherwise rely on well-established pedagogy. Keep every string to one concise
sentence. Do not repeat the same misconception with different wording."""


def pack_user(node: KCNode, excerpts: list[tuple[str, str]]) -> str:
    """Build the user message for pedagogy-pack compilation."""
    lines = [
        f"KC: {node.id} — {node.name}",
        f"Description: {node.description}",
        f"Course level: {node.course_level}",
        "Canonical examples:",
        *[f"- {example}" for example in node.canonical_examples],
    ]
    if excerpts:
        lines.append("Source excerpts:")
        for source, text in excerpts:
            lines.append(f"[source: {source}]\n{text}")
    else:
        lines.append("No source excerpts were retrieved; rely on established pedagogy.")
    lines.append("Compile the pedagogy pack JSON now.")
    return "\n".join(lines)
