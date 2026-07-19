"""Interactive terminal demo of a tutoring session (Phase 1, text-only).

Usage:
    python -m tutor.cli
    python -m tutor.cli --target kc.der.chain_rule --course "AP Calculus AB"

Commands during the session: answers as free text, 'hint' for the next hint,
'reveal' to show the expected answer, 'quit' to leave.
"""

import argparse

from tutor.orchestrator.machine import Interaction, SessionOrchestrator, SessionPhase
from tutor.schemas.learner import LearnerProfile
from tutor.seed.load_seed import load_graph

_PREFIX = {
    "message": "tutor",
    "probe": "tutor asks",
    "lesson": "lesson",
    "checkin": "check-in",
    "capstone": "capstone",
}


def _render(interactions: list[Interaction]) -> None:
    for interaction in interactions:
        print(f"\n[{_PREFIX[interaction.kind]}] {interaction.text}")


def _build_llm_ports(graph, profile, provider: str):
    """Construct LLM-backed ports, or (None, None) with a warning if unavailable."""
    try:
        from tutor.llm.factory import build_llm_ports

        diagnostician, lesson_writer = build_llm_ports(graph, profile, provider)
    except Exception as exc:  # noqa: BLE001 — degrade to templates with a notice
        print(f"[warn] LLM ports unavailable ({exc}); using template ports.")
        return None, None
    print("[info] LLM-backed diagnostician and lesson writer enabled.")
    return diagnostician, lesson_writer


def main(argv: list[str] | None = None) -> int:
    """Run one interactive session against the seed graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="kc.int.u_substitution", help="target kc id")
    parser.add_argument("--course", default="AP Calculus AB", help="student course")
    parser.add_argument("--age-band", default="16-18", help="student age band")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="use LLM-backed diagnostician/lesson writer (needs OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic"),
        default="openai",
        help="LLM provider for --llm (default: openai; model via TUTOR_LLM_MODEL)",
    )
    args = parser.parse_args(argv)

    graph = load_graph()
    profile = LearnerProfile(course=args.course, age_band=args.age_band)
    diagnostician = lesson_writer = None
    if args.llm:
        diagnostician, lesson_writer = _build_llm_ports(graph, profile, args.provider)
    orchestrator = SessionOrchestrator(
        graph,
        args.target,
        profile,
        diagnostician=diagnostician,
        lesson_writer=lesson_writer,
    )
    _render(orchestrator.begin())

    while orchestrator.phase not in (SessionPhase.DONE, SessionPhase.STOPPED):
        try:
            raw = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        if raw.lower() in ("quit", "exit"):
            break
        if raw.lower() == "hint":
            hint = orchestrator.hint()
            print(f"\n[hint] {hint or 'No more hints — give it your best try.'}")
            continue
        if raw.lower() == "reveal":
            print(f"\n[reveal] {orchestrator.pending_expected or 'nothing pending'}")
            continue
        _render(orchestrator.submit(raw))

    summary = orchestrator.summary()
    print("\n--- session summary ---")
    for field in ("phase", "probes_used", "frontier", "path", "mastered_in_session",
                  "interactions_used"):
        print(f"{field}: {summary[field]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
