"""Ad-hoc timing/quality probe for one KC's lesson-planning LLM calls.

Isolates narrative + interaction + evaluator calls (the TEACH-phase burst)
outside the full session/diagnosis flow, for fast prompt iteration.
Usage: python scripts/try_llm_call.py [kc_id]
"""

import sys
import time

from dotenv import load_dotenv

load_dotenv()

from tutor.llm.client import OpenAILLMClient  # noqa: E402
from tutor.llm.evaluator import LLMEvaluator  # noqa: E402
from tutor.llm.interaction import LLMInteractionGenerator  # noqa: E402
from tutor.llm.lesson_writer import LLMLessonWriter  # noqa: E402
from tutor.packs.loader import load_packs  # noqa: E402
from tutor.schemas.learner import LearnerProfile  # noqa: E402
from tutor.seed.load_seed import load_coverage, load_graph  # noqa: E402


def main() -> None:
    kc_id = sys.argv[1] if len(sys.argv) > 1 else "kc.der.chain_rule"
    graph = load_graph()
    node = next(n for n in graph.nodes if n.id == kc_id)
    packs = load_packs()
    coverage = {kc: list(e.get("widget_types", [])) for kc, e in load_coverage().items()}
    profile = LearnerProfile(course="AP Calculus AB", age_band="16-18")

    client = OpenAILLMClient()
    writer = LLMLessonWriter(client, packs=packs, profile=profile)
    generator = LLMInteractionGenerator(client, packs=packs, profile=profile, coverage=coverage)
    evaluator = LLMEvaluator(client)

    print(f"KC: {kc_id}  model={client._model}\n")

    t0 = time.perf_counter()
    narrative = writer.lesson_text(node)
    t1 = time.perf_counter()
    candidates = generator.candidates(node, 0, [])
    t2 = time.perf_counter()

    print(f"--- narrative ({t1 - t0:.1f}s) ---\n{narrative}\n")
    print(f"--- {len(candidates)} candidate(s) generated ({t2 - t1:.1f}s) ---")

    for i, candidate in enumerate(candidates):
        t_start = time.perf_counter()
        verdict = evaluator.evaluate(node, narrative, candidate)
        t_end = time.perf_counter()
        print(f"\ncandidate {i}: {candidate.widget_type} ({t_end - t_start:.1f}s)")
        print(f"  accepted={verdict.accepted} feedback={verdict.feedback!r}")

    total = sum(c.latency_ms for c in client.calls) / 1000
    print(f"\n--- raw LLMCall log (total {total:.1f}s across {len(client.calls)} calls) ---")
    for call in client.calls:
        print(
            f"{call.tag:30s} model={call.model:14s} "
            f"latency_ms={call.latency_ms:7.0f} "
            f"in_chars={call.input_chars:5d} out_chars={call.output_chars:5d}"
        )


if __name__ == "__main__":
    main()
