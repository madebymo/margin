"""Deterministic session orchestration: diagnosis -> plan -> teach -> capstone.

The orchestrator is a coded state machine, not an LLM. LLM call sites attach
via the ports in ``tutor.orchestrator.ports``.
"""

from tutor.orchestrator.machine import Interaction, SessionOrchestrator, SessionPhase

__all__ = ["Interaction", "SessionOrchestrator", "SessionPhase"]
