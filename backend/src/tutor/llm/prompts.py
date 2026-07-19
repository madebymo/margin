"""Versioned prompts for the LLM call sites.

PROMPT_VERSION is pinned into generated-content provenance; bump it whenever
any prompt in this module changes.
"""

from tutor.schemas.kc import KCNode
from tutor.schemas.learner import LearnerProfile
from tutor.schemas.pedagogy import Metaphor, Misconception, PedagogyPack

PROMPT_VERSION = "p1"

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
