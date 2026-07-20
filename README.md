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
  - `src/tutor/api/` — FastAPI session service + committed production web assets
- `frontend/` — Vite + Svelte widget runtime and Elm SVG scene renderer
- `scripts/` — JSON Schema export, utilities
- `docker-compose.yml` — Postgres 16 for local development

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e "backend[dev,api]"
pytest backend/tests
npm --prefix frontend ci
npm --prefix frontend test
npm --prefix frontend run build
python -m tutor.cli                     # terminal chat demo
uvicorn tutor.api.app:app --reload      # web chat at http://*********:8000
python -m tutor.sim.harness             # diagnostic-policy metrics (budgets 5/8/10)
docker compose up -d db                 # optional: real Postgres
python -m tutor.seed.load_seed --validate
```
LLM mode: `pip install -e "backend[llm]"`, set `OPENAI_API_KEY` (model override via `TUTOR_LLM_MODEL`), then `python -m tutor.cli --llm` or toggle LLM in the web UI.
Persistence: set `DATABASE_URL=postgresql+psycopg://tutor:tutor@localhost/tutor` (after `docker compose up -d db`) to durably store learners, the append-only evidence log, episode checkpoints, and derived mastery; learner state is rebuildable by replaying the evidence log.

For frontend development, run FastAPI on port 8000 and `npm --prefix frontend run dev` in a second terminal; Vite proxies `/sessions` and `/healthz`. FastAPI production/local serving uses the committed hashed bundle under `backend/src/tutor/api/static/dist/`. Run `scripts/check_frontend_dist.sh` before committing frontend changes to detect stale or new build artifacts.

## Status
Done: Phase 0 (data layer), the deterministic tutor loop, LLM adapters, session API, diagnostic-policy simulation harness, answer-safe widget serialization, and the Phase 1 Svelte/Elm widget runtime with a rich slider scene.
Next: diagnosis-policy tuning against harness metrics, pedagogy packs for the remaining KCs, richer scenes for the other widget types, and a separately designed secure protocol for live-during-drag coaching.
