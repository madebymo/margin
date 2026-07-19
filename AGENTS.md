# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project

Adaptive Math Tutor — an LLM chat tutor that diagnoses gaps in foundational math knowledge, teaches a just-in-time path of mini-lessons from the deepest gap up to the student's original question, and finishes with the student solving that question unaided.

Architecture: **deterministic control plane** (state machine, graph service, learner model, math verifier) + **four stateless LLM call sites** (diagnostician, lesson writer, interaction generator, evaluator) + **versioned data assets** (KC graph, pedagogy packs, append-only evidence log, widget library). The control plane is fully correct and testable with zero LLM; LLM adapters plug in behind the same Protocols.

Single Python package `tutor` under `backend/src/` (src layout), Python ≥3.11. Pydantic v2 schemas are the source of truth; SQLAlchemy 2.0 models mirror them for persistence.

## Commands

Install (run from repo root):
```bash
pip install -e "backend[dev,api]"          # core + test/api dev deps
pip install -e "backend[llm]"              # OpenAI client (default provider)
pip install -e "backend[llm-anthropic]"    # Anthropic client (optional)
```

Tests & lint:
```bash
pytest backend/tests                                       # full suite (in-memory SQLite, no Postgres needed)
pytest backend/tests/test_machine_e2e.py::test_full_session_reaches_done   # single test
ruff check backend/src backend/tests                       # line-length 100 (backend/pyproject.toml)
```

Run the app:
```bash
python -m tutor.cli                        # terminal chat demo (commands: hint, reveal, quit)
python -m tutor.cli --llm --provider openai --target kc.der.chain_rule
uvicorn tutor.api.app:app --reload         # web chat + REST API at http://localhost:8000
```

Diagnostics & data tooling:
```bash
python -m tutor.sim.harness --learners 200 --budgets 5 8 10   # diagnostic-policy simulation metrics
python -m tutor.seed.load_seed --validate                     # validate seed graph + coverage matrix
python -m tutor.seed.load_seed --validate --db postgresql+psycopg://tutor:tutor@localhost/tutor   # publish to Postgres
python -m tutor.packs.import_csv --validate backend/src/tutor/packs/template.csv --out /tmp/packs # validate pack CSV
python scripts/export_json_schemas.py                         # regenerate schemas/json/ (gitignored)
docker compose up -d db                                       # local Postgres 16
```

Environment: `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` (LLM providers), `TUTOR_LLM_MODEL` (model override), `DATABASE_URL` (DB engine; defaults to in-memory SQLite).

## Architecture

### The session loop (`tutor/orchestrator/`)
`SessionOrchestrator` (`machine.py`) drives one session through phases INTAKE → DIAGNOSE → PLAN → TEACH → CAPSTONE → DONE/STOPPED. It owns all sequencing and the single `_Pending` item currently awaiting an answer; **the expected answer never leaves the server** (API responses expose only `pending_kind`/`pending_kc`; `pending_expected` is for tests/CLI `reveal` only). Generation is delegated to ports, scoring to the verifier, belief updates to the learner model, and routing to the pure `route` function.

- `DiagnosisController` (`diagnosis.py`) — adaptive diagnosis over the **hard-ancestor subgraph** of the target. Policy v1.1: probe the target first; on a miss, probe the implicated prereq (if error analysis names one) else binary-search the unresolved hard-ancestor chain by depth; on a hit, re-probe single-observation bad nodes once (slip recovery) then drill toward the deepest bad node; spend leftover budget verifying the shallowest unprobed would-be lesson node (gated on an observed gap so a passing student short-circuits after one probe). No node is probed more than twice. Outputs a **frontier** (deepest observed-bad nodes with no observed-bad hard ancestor) and a **teaching path** (topological order of unmastered nodes, target last).
- `route()` (`routing.py`) — **pure function** `(envelope, outcome) → (decision, new_envelope)`. Idempotent on duplicate interaction keys; enforces global interaction budget; descends only to a strict ancestor not already inserted, bounded by `max_inserts`/`max_detour_depth` with an acyclic resume stack; per-KC retries are capped and exhaustion falls back (never loops). Never mutates its input — returns a deep copy.
- `EpisodeEnvelope` (`envelope.py`) — explicit routing state (budgets, retries, `inserted`, `resume_stack`, `seen_interaction_keys`) that persists across detours so oscillation can't reset counters. Only ever updated through `route()` (plus the machine's resume pop).

### Ports: template vs LLM (`tutor/orchestrator/ports.py`, `tutor/llm/`)
`DiagnosticianPort` and `LessonWriterPort` are `runtime_checkable` Protocols. `TemplateDiagnostician`/`TemplateLessonWriter` are deterministic, LLM-free implementations built from KC canonical examples — **these are the defaults**, so the whole control plane runs and is tested offline. `LLMDiagnostician`/`LLMLessonWriter` (`tutor/llm/`) satisfy the same Protocols and are constructed by `build_llm_ports` (`factory.py`), shared by the CLI and API.

LLM outputs are never trusted raw:
- Every probe and check-in passes the **correctness gate** (`verify/checker.parse_restricted` of the expected answer, blank normalization, and an answer-leak check against visible text) before display; failures retry then **fall back to the deterministic template**.
- Error-analysis ids are **membership-validated**: misconception ids must come from the KC's pedagogy pack and implicated prereqs from the KC's hard predecessors — the model cannot invent either.
- `llm/client.py` thin sync JSON-only adapters: `OpenAILLMClient` (JSON mode, default model `gpt-5.5`) and `AnthropicLLMClient` (prompt caching on the system block, default `claude-sonnet-4-5`). Both record per-call `LLMCall` metadata. The model can be overridden via `TUTOR_LLM_MODEL`.

### Graph, learner model, and verifier
- `graph/service.py` — pure functions over `GraphDocument` (`ancestor_subgraph` with `hard_only`, `topological_order` with lexicographic tie-break via a heap, `roots`, `descendants`, `validate_acyclic`) plus the one DB-touching `publish_graph`. This is the seam for a future dedicated graph store; traverse via this service, not raw graph attributes.
- `learner/service.py` `LearnerModelService` — BKT-lite derived mastery over an **append-only evidence log**. Invariants: direct and inferred evidence are tracked separately and never merged; **inferred evidence alone can never cross the mastery threshold** (capped below it); a dependent miss never lowers prerequisite beliefs (only direct misses do); multi-KC events inform routing only (skip BKT updates); derived state is rebuildable via `replay()`. Unassisted correct answers propagate discounted inferred belief up hard ancestors. Parameters are external and versioned (`learner/params.py`, `DEFAULT_PARAMS_V1`).
- `verify/checker.py` `check_answer` — restricted SymPy parsing under a character whitelist (no underscores/quotes/brackets — blocks dunder/attribute tricks) and a fixed function table; unknown names become inert symbols. `sympy_equiv` or `numeric` checking, with a normalized string-compare fallback when either side fails to parse. Never arbitrary evaluation.

### API (`tutor/api/`)
`app.py` builds a FastAPI app around one graph version and an in-memory `SessionStore` (`store.py`, thread-safe, FIFO-bounded at 500). Endpoints: `POST /sessions`, `POST /sessions/{id}/answer`, `POST /sessions/{id}/hint`, `GET /sessions/{id}`, `GET /healthz`, `GET /` (single-file chat UI in `api/static/index.html`). LLM ports are constructed per-session on request (`llm: true`) and degrade to templates with a warning if unavailable.

### Data assets & persistence
- `schemas/` — Pydantic v2 models, the source of truth for the JSON Schemas exported by `scripts/export_json_schemas.py`. `kc.GraphDocument` validates unique node ids, edge endpoint existence, and acyclicity (three-color DFS `find_cycle`). KC ids must match `^kc\.(alg|fun|lim|der|int)\.[a-z0-9_]+$`; `canonical_examples` holds 1–3 examples. `learner.EvidenceEvent` is `frozen` (immutable). `common.py` defines `EdgeType` (HARD/SOFT), `ResponseClass`, `ReviewStatus`, `JobStatus`, `WidgetType`.
- `seed/` — the Calc-1 KC graph (`kc_graph_calc1.json`, ~40 nodes) and `coverage_matrix.json`. `load_graph()` fully validates. `--validate` checks the coverage matrix covers exactly the node set, uses valid widget types, mandates `text_fallback`, and declares `measures` ∈ {production, recognition, reasoning}.
- `packs/` — pedagogy pack CSV import. The CSV is a **validated import surface only** (the DB is authoritative). Multi-value cells use `|`; rows group by `kc_id` into one `PedagogyPack` per KC, imported as `review_status=draft`. `template.csv` is the bundled pack loaded by `build_llm_ports`.
- `db/` — SQLAlchemy 2.0 models (`models.py`): `graph_versions`, `kc_nodes`, `kc_edges`, `pedagogy_packs`, `learners`, `resume_tokens`, `evidence_events` (append-only — no `updated_at` by design), `derived_mastery`, `episodes`, `generation_jobs`, `mini_lessons`. `JSONVariant` renders JSONB on Postgres and JSON on SQLite so the same models work in tests. `db/session.py` `get_engine` reads `DATABASE_URL` or defaults to in-memory SQLite with a `StaticPool` (shared connection). Row classes carry a `Row` suffix to distinguish them from Pydantic schemas; UUIDs are `String(36)`.
- `sim/` — `harness.py` is the diagnostic-policy simulation harness (the audit's pre-pilot requirement): drives the real `DiagnosisController` + `LearnerModelService` against synthetic learners and reports `next_kc_accuracy`, `frontier_soundness`, and mean probes/overteach/missed per probe budget. `synthetic.py` generates the learner population.

## Conventions to preserve

- **Hard vs soft edges**: hard edges gate path planning, propagation, and diagnosis traversal; soft edges inform only. Use `ancestor_subgraph(..., hard_only=True)` for diagnosis/path planning.
- **Router purity**: never mutate `EpisodeEnvelope` in place outside `route()` (the machine's resume-stack pop is the one exception). Callers get a new envelope back.
- **Evidence log is append-only and authoritative**: mastery is derived and rebuildable. Don't add update paths to `evidence_events`. Keep direct and inferred evidence separate.
- **Expected answers stay server-side**: don't add them to API response models or log them in student-visible output.
- **Gating discipline for LLM output**: any new LLM-generated item must pass the correctness gate and fall back to the template port on failure; any id the model returns must be membership-validated against the graph/pack before use.
- `schemas/json/` and `.ai/` are gitignored (generated/local only). The JSON Schemas are regenerated by the export script, not hand-edited.
