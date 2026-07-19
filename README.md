# Adaptive Math Tutor

An LLM chat tutor that adaptively diagnoses gaps in foundational math knowledge, teaches a just-in-time path of interactive mini-lessons from the deepest gap up to the student's original question, and finishes with the student solving that question unaided.

Architecture summary: deterministic control plane (state machine, graph service, learner model, math verifier) + four stateless LLM call sites (diagnostician, lesson writer, interaction generator, evaluator) + persistent data assets (versioned KC graph, pedagogy packs, append-only evidence log, widget library).

## Layout
- `backend/` — Python backend
  - `src/tutor/schemas/` — Pydantic v2 models (source of truth for JSON Schemas)
  - `src/tutor/db/` — SQLAlchemy 2.0 models (Postgres; SQLite variant for tests)
  - `src/tutor/graph/` — KC graph service (acyclicity, ancestor subgraph, topo sort)
  - `src/tutor/seed/` — Calc-1 KC graph seed (~40 nodes) + KC-to-affordance coverage matrix
  - `src/tutor/packs/` — pedagogy pack CSV import surface
  - `src/tutor/learner/` — BKT-lite learner model over the evidence log
  - `src/tutor/orchestrator/` — deterministic session state machine, diagnosis, routing
  - `src/tutor/llm/` — LLM-backed diagnostician/lesson writer (OpenAI default)
  - `src/tutor/api/` — FastAPI session service + minimal web chat UI
- `scripts/` — JSON Schema export, utilities
- `docker-compose.yml` — Postgres 16 for local development

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e "backend[dev,api]"
pytest backend/tests
python -m tutor.cli                     # terminal chat demo
uvicorn tutor.api.app:app --reload      # web chat at http://*********:8000
python -m tutor.sim.harness             # diagnostic-policy metrics (budgets 5/8/10)
docker compose up -d db                 # optional: real Postgres
python -m tutor.seed.load_seed --validate
```
LLM mode: `pip install -e "backend[llm]"`, set `OPENAI_API_KEY` (model override via `TUTOR_LLM_MODEL`), then `python -m tutor.cli --llm` or toggle LLM in the web UI.
Persistence: set `DATABASE_URL=postgresql+psycopg://tutor:tutor@localhost/tutor` (after `docker compose up -d db`) to durably store learners, the append-only evidence log, episode checkpoints, and derived mastery; learner state is rebuildable by replaying the evidence log.

## Status
Done: Phase 0 (data layer), Phase 1 (orchestrator, diagnosis, learner model, CLI), LLM agent integration (OpenAI default, Anthropic optional), session API + web chat, diagnostic-policy simulation harness.
Next: diagnosis-policy tuning against harness metrics, widget runtime (Phase 2), pedagogy packs for the remaining KCs.
