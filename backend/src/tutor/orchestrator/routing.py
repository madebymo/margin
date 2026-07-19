"""route(): a pure function from (envelope, outcome) to (decision, new envelope).

Invariants:
- Duplicate interaction keys are no-ops (idempotency).
- Every non-duplicate call consumes global budget, so episodes terminate.
- Descents only go to a strict ancestor not already inserted, bounded by
  max_inserts and max_detour_depth; a node already on the resume stack can
  never be descended from again (acyclic stack).
- Retries are counted per KC and capped; exhaustion falls back, never loops.
- The input envelope is never mutated; a copy is returned.
"""

from tutor.orchestrator.envelope import (
    CheckinOutcome,
    EpisodeEnvelope,
    RoutingAction,
    RoutingDecision,
)


def route(
    envelope: EpisodeEnvelope,
    outcome: CheckinOutcome,
    ancestors_of_current: set[str],
    exit_consecutive: int = 2,
) -> tuple[RoutingDecision, EpisodeEnvelope]:
    """Route one check-in outcome. Returns the decision and the next envelope."""
    if outcome.interaction_key in envelope.seen_interaction_keys:
        return (
            RoutingDecision(
                action=RoutingAction.DUPLICATE, reason="interaction already processed"
            ),
            envelope,
        )

    env = envelope.model_copy(deep=True)
    env.seen_interaction_keys.append(outcome.interaction_key)
    env.interactions_used += 1

    if env.interactions_used > env.interaction_budget:
        return (
            RoutingDecision(action=RoutingAction.STOP, reason="interaction budget exhausted"),
            env,
        )

    current = outcome.kc_id

    if outcome.correct:
        if outcome.consecutive_correct >= exit_consecutive:
            return (
                RoutingDecision(action=RoutingAction.ADVANCE, reason="exit criterion met"),
                env,
            )
        return (
            RoutingDecision(
                action=RoutingAction.CONTINUE, reason="correct; more evidence needed"
            ),
            env,
        )

    prereq = outcome.implicated_prereq
    can_descend = (
        prereq is not None
        and prereq != current
        and prereq in ancestors_of_current
        and prereq not in env.inserted
        and current not in env.resume_stack
        and len(env.inserted) < env.max_inserts
        and len(env.resume_stack) < env.max_detour_depth
    )
    if can_descend:
        env.inserted.append(prereq)  # type: ignore[arg-type]
        env.resume_stack.append(current)
        return (
            RoutingDecision(
                action=RoutingAction.DESCEND,
                descend_to=prereq,
                reason=f"miss implicates prerequisite {prereq}",
            ),
            env,
        )

    used_retries = env.retries.get(current, 0)
    if used_retries < env.max_retries_per_kc:
        env.retries[current] = used_retries + 1
        return (
            RoutingDecision(action=RoutingAction.RETRY, reason="retry with targeted variation"),
            env,
        )

    return (
        RoutingDecision(
            action=RoutingAction.FALLBACK,
            reason="retries exhausted; show worked example and move on",
        ),
        env,
    )
