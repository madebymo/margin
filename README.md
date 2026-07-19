# Adaptive Math Tutor

An LLM chat tutor that adaptively diagnoses gaps in foundational math knowledge, teaches a just-in-time path of interactive mini-lessons from the deepest gap up to the student's original question, and finishes with the student solving that question unaided.

Architecture summary: deterministic control plane (state machine, graph service, learner model, math verifier) + four stateless LLM call sites (diagnostician, lesson writer, interaction generator, evaluator) + persistent data assets (versioned KC graph, pedagogy packs, append-only evidence log, widget library).

## Layout
- `backend/` — Python data layer and services (FastAPI arrives in Phase 1)
  - `src/tutor/schemas/` — Pydantic v2 models (source of truth for JSON Schemas)
  - `src/tutor/db/` — SQLAlchemy 2.0 models (Postgres; SQLite variant for tests)
  - `src/tutor/graph/` — KC graph service (acyclicity, ancestor subgraph, topo sort)
  - `src/tutor/seed/` — Calc-1 KC graph seed (~40 nodes) + KC-to-affordance coverage matrix
  - `src/tutor/packs/` — pedagogy pack CSV import surface
- `scripts/` — JSON Schema export, utilities
- `docker-compose.yml` — Postgres 16 for local development

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e "backend[dev]"
pytest backend/tests
docker compose up -d db   # optional: real Postgres
python backend/src/tutor/seed/load_seed.py --validate
```

## Status
Phase 0 (foundations): schemas, DB models, graph seed, coverage matrix, pack import surface.
