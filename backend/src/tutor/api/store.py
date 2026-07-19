"""In-memory session store.

v1 keeps sessions in-process, matching the plan's no-external-queue stance.
Durable resume tokens and DB-backed session persistence arrive with the
accounts work; the evidence log already lives on each session's learner model.
"""

import threading
from collections import OrderedDict
from uuid import uuid4

from tutor.orchestrator.machine import SessionOrchestrator


class SessionStore:
    """Thread-safe, FIFO-bounded store of live sessions."""

    def __init__(self, max_sessions: int = 500) -> None:
        self._sessions: OrderedDict[str, SessionOrchestrator] = OrderedDict()
        self._lock = threading.Lock()
        self._max_sessions = max_sessions

    def create(self, orchestrator: SessionOrchestrator) -> str:
        """Store a session and return its id; evicts oldest beyond capacity."""
        session_id = uuid4().hex
        with self._lock:
            while len(self._sessions) >= self._max_sessions:
                self._sessions.popitem(last=False)
            self._sessions[session_id] = orchestrator
        return session_id

    def get(self, session_id: str) -> SessionOrchestrator:
        """Return the session or raise KeyError."""
        with self._lock:
            return self._sessions[session_id]

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
