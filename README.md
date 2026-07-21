# Margin

Margin is a deterministic, evidence-aware tutor for curated math goals spanning
calculus and its prerequisites. It probes prerequisite knowledge, keeps
confirmed mastery, confirmed gaps, and uncertainty separate, teaches a short
path of skills, and finishes with an independently authored goal problem.

The trustworthy-session v2 architecture keeps mathematical truth in reviewed,
versioned content and deterministic verification. LLMs may eventually provide
coaching language, but they do not author mastery evidence or choose the
correct answer.

## Built with Codex and GPT-5.6

Margin was developed with Codex running GPT-5.6 as an engineering collaborator
across discovery, architecture, implementation, verification, and
documentation. Codex exercised the no-refresh learner journey, turned observed
failures—including answer reuse, hint grading, duplicate advancement, unsafe
symbolic parsing, widget traps, and recovery gaps—into explicit trust
invariants, and helped carry those invariants through the FastAPI and Svelte
implementation.

Across the repository, Codex helped build and test the revision-checked session
state machine, PostgreSQL checkpoints, Redis-backed safety controls, restricted
symbolic verifier, typed content compilers and reviewer packets, accessible
learner interface, deployment runbook, and diagnosis simulations. The work was
split into reviewable commits and checked with focused unit, concurrency,
recovery, browser, accessibility, and simulation suites.

This is a build-process claim, not a scoring claim. GPT-5.6 is not Margin's
mathematical judge: mastery-bearing items come from versioned curated content,
expected answers remain server-side, and deterministic verification produces
evidence. The optional runtime LLM adapters are separate from the Codex
development workflow, and the pilot interface currently exposes curated mode
only. The accompanying [hackathon video script](HACKATHON_VIDEO_SCRIPT.md)
includes the on-screen evidence and disclosure beats for this workflow.

## Repository

**Submission URL:** [github.com/madebymo/margin](https://github.com/madebymo/margin)

## Judge demo

**Live demo:** [retro-ebony-extended-wear.trycloudflare.com](https://retro-ebony-extended-wear.trycloudflare.com)

The judge build is an explicitly labeled, single-goal engineering demonstration
of the complete no-refresh v2 flow. It uses synthetic Power Rule content so the
interaction, diagnosis, hint, independent-check, capstone, and resume behavior
can be assessed without presenting the pending pilot curriculum as reviewed or
released. Its state is process-local: reload works, but a host restart begins a
new demo session. The normal production entry point remains fail-closed.

Run the same guarded build locally with a fresh secret:

```bash
cd backend
TUTOR_SUBMISSION_DEMO=1 \
TUTOR_RESUME_TOKEN_SECRET="$(openssl rand -hex 32)" \
uvicorn tests.browser_v2_app:app --host 127.0.0.1 --port 8000
```

## Release posture: fail closed

The v2 control plane, API, persistence model, diagnosis policy, allocator,
restricted verifier, and unified Svelte interface are implemented. Student
release is intentionally blocked by content review:

- The packaged bank is `draft-v2.0.0`; its items have `review_status: draft`
  and `released_kcs: []`.
- The packaged pedagogy catalog is an immutable, graph-pinned empty release;
  reviewed packs must be published into a new catalog version before their KCs
  can become student-eligible.
- At local development's default 100% rollout setting, `GET /api/v2/goals` returns an
  empty catalog with `rollout.status: "content_unavailable"`. A target appears
  only when its complete hard-prerequisite closure is declared released and
  passes item-bank validation.
- New v1 sessions return `410 Gone` by default. Set
  `TUTOR_ALLOW_V1_SESSION_CREATION=1` only for local compatibility testing.
- The five pilot goals and their 22 hard-ancestor KCs still need complete,
  independently authored, human-reviewed item families before pilot traffic is
  enabled.

This is deliberate: missing or unreviewed content must make a goal unavailable,
not silently fall back to canonical examples or model-authored scored items.

## Architecture

- `backend/src/tutor/schemas/` — strict Pydantic v2 public and content schemas
- `backend/src/tutor/content/` — item-bank validation, bundle allocation,
  exposure tracking, and leakage detection
- `backend/src/tutor/verify/` — restricted grammar/AST and supervised symbolic
  equivalence worker
- `backend/src/tutor/orchestrator/` — v1 compatibility machine plus the v2
  diagnosis and lesson state machine
- `backend/src/tutor/learner/` — replayable evidence model and v2
  distinct-family confirmation policy
- `backend/src/tutor/api/` — FastAPI v1 compatibility routes, authoritative v2
  snapshots, idempotency, resume, persistence, and release registry
- `backend/src/tutor/db/` — SQLAlchemy models and the additive Alembic
  migration chain
- `backend/src/tutor/seed/` — Calc-1 graph, coverage matrix, packaged draft item
  bank, and exact reviewed pedagogy-catalog release
- `backend/src/tutor/sim/` — v1 and v2 synthetic diagnosis harnesses
- `frontend/` — one Svelte session state model with native accessible widget
  controls and Elm-rendered SVG scenes

Every v2 session pins one explicitly registered graph/item-bank/pedagogy-catalog
release triple plus its diagnosis, lesson, allocator, learner-parameter, and
widget-capability versions. Expected answers and scoring rules have no
representation in the public `SessionView`.

## Local setup

Python 3.11 or newer and Node.js are required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e "backend[dev,api]"

npm --prefix frontend ci

python -m tutor.seed.load_seed --validate
ruff check backend/src backend/tests
pytest backend/tests
npm --prefix frontend test
npm --prefix frontend run build
scripts/check_frontend_dist.sh

# Requires a one-time local Chromium install.
npx --prefix frontend playwright install chromium
npm --prefix frontend run test:e2e
```

Run the API and committed web application:

```bash
uvicorn tutor.api.app:app --reload
```

Open `http://127.0.0.1:8000/`. With the packaged draft bank, the intake screen
honestly reports that no reviewed goal is available. For frontend development,
run `npm --prefix frontend run dev` in a second terminal; Vite proxies the API
to port 8000.

The terminal v1 compatibility demo remains available:

```bash
python -m tutor.cli
```

LLM dependencies are optional:

```bash
pip install -e "backend[llm]"
export OPENAI_API_KEY=...
```

`llm_coaching` is accepted by the v2 intake contract but currently reports an
explicit fallback to `curated`; it is not presented as an LLM runtime failure.

## API v2

The public routes are:

| Method | Route | Contract |
| --- | --- | --- |
| `GET` | `/api/v2/goals` | Cohort-specific rollout status plus goals with a fully released and valid hard-prerequisite closure |
| `GET` | `/api/v2/capabilities` | Versioned widget capability manifest |
| `POST` | `/api/v2/sessions` | Create or replay an anonymous episode |
| `POST` | `/api/v2/sessions/recover` | Recover a committed create/reset response using its client-held request proof |
| `GET` | `/api/v2/sessions/current` | Restore the complete authoritative snapshot from the resume cookie |
| `GET` | `/api/v2/sessions/{id}` | Read the owned session without advancing it |
| `POST` | `/api/v2/sessions/{id}/actions` | Apply one revision-checked, idempotent action |
| `POST` | `/api/v2/sessions/current/reset` | Atomically replace the current episode while retaining prior evidence |

Session creation accepts:

```json
{
  "request_id": "8d46af11-f3c6-49ad-887a-0d3f201fa5ce",
  "goal_id": "goal.der.chain_rule",
  "course": "AP Calculus AB",
  "age_band": "16-18",
  "content_mode": "curated",
  "context": "Optional coaching context",
  "provider": "openai"
}
```

Before a create or reset is sent, the web client keeps only the versioned
operation name and cryptographically generated `request_id` in per-tab
`sessionStorage`. It never stores the creation payload, coaching context, or a
student answer there. If the commit succeeds but its replacement `Set-Cookie`
response is lost, bootstrap posts that proof to `/api/v2/sessions/recover`
before reading the current session. Reset recovery also requires the revoked
old `HttpOnly` cookie; `GET /sessions/current` never treats a revoked token as
an alias for its successor. The client clears its proof only after a confirmed
recovery or a definitive 4xx response.

The optional context is returned as context, not represented as a solved,
verified, or “original” question.

An action body is one of the following discriminated shapes:

```json
{
  "type": "answer",
  "request_id": "4ff95a14-1e0e-442e-9220-4193887601b2",
  "expected_revision": 3,
  "pending_key": "diagnostic:kc.der.chain_rule:2",
  "answer": "2*x"
}
```

The other action types are `request_hint`, `widget_attempt` (with a `response`
object), and `use_text_fallback`. Each includes `request_id`,
`expected_revision`, and `pending_key`. Reset uses the same concurrency fields,
with a nullable `pending_key`.

Every successful action returns a complete, student-safe `SessionView`.
Duplicate requests with the same payload replay the committed view. Reusing a
request ID for a different payload, or sending a stale revision/key, returns a
typed `409` with the authoritative view where available. A failed durable
transaction returns a retryable `503` without replacing live state.

Answers are rejected at the API boundary above the verifier's 256-character
limit. Each episode also has a hard 256-action storage ceiling: exact retries
still replay at the ceiling, while a new request returns `429` without adding a
transcript entry or receipt. An anonymous learner may create at most 32
episodes in the rolling 30-day resume window. These bounds prevent invalid
input and reset loops from growing cumulative snapshots without limit.

Anonymous resume uses a 256-bit token in an `HttpOnly`, `SameSite=Lax` cookie,
`Secure` away from localhost. Mutations enforce same-origin checks. Each token
hash is bound to one exact episode checkpoint and has a rolling 30-day expiry;
raw tokens and expected answers are not logged. Reset atomically revokes the
old token, creates a fresh same-goal episode on the same anonymous learner and
pinned release, replays eligible prior evidence at a new fixed `as_of`, and
returns the replacement `SessionView` with a new token.

`GET /api/v2/goals` includes a `rollout` object with `status`, `reason`, and
`percentage`. Its status is one of `available`, `not_selected`, `paused`, or
`content_unavailable`. These states are intentionally distinct: a browser
outside a limited canary receives `not_selected`, without being told that
curriculum review failed. Only a selected browser can receive
`content_unavailable` when its cohort is open but no goal passes the reviewed
content gate.

## Content release and validation

Each released KC must have, at minimum:

- 3 diagnostic families, including 2 production-answer families
- 4 independent production check-in families
- 1 guided-widget family
- 2 independent production capstone families
- 1 worked-example family

Items carry stable item/revision/family/KC identifiers, structured prompt
segments, three ordered hints, a discriminated answer contract, review status,
and provenance. Reviewed error signatures must reference a human-approved
misconception in the exact graph-pinned pedagogy catalog released with the
item bank. Runtime validation never merges the legacy CSV and generated draft
directories into this trust decision.

Run the release validator after any graph, coverage, pack, or item-bank change:

```bash
python -m tutor.seed.load_seed --validate
```

Validation checks graph and coverage integrity, stable content identifiers,
review/provenance rules, answer-spec parseability, family independence,
surface coverage, production-family requirements, pedagogy-catalog coverage,
reviewed misconception membership, and answer leakage. The current draft
passes structural validation while releasing no KC.

For a non-default authoring release, pass the exact reviewed catalog to the
compiler check rather than relying on ambient pack directories:

```bash
python -m tutor.content.compiler --check \
  --pedagogy-catalog /path/to/pedagogy-catalog.json
```

Do not add a KC to `released_kcs` merely to expose it in the UI. Author and
review its complete family set and the complete hard-ancestor closure first.

Four cumulative pilot waves are now authored as pending review inputs. Together
they contain 221 draft families and 17 draft pedagogy packs, while releasing
zero KCs and exposing zero goals to fresh pilot learners:

| Pending wave | New KCs | Draft families | Cumulative families |
|---|---:|---:|---:|
| Product/Quotient Rules | 4 | 52 | 52 |
| Chain Rule | 3 | 39 | 91 |
| Solve Quadratics | 4 | 52 | 143 |
| Fundamental Theorem of Calculus | 6 | 78 | 221 |

Each wave has a construct-aware typed compiler and exact review manifest. The
compilers derive mathematical truth from bounded source parameters and reject
answer reuse or visible leakage both within the wave and against all preceding
families. Run all four exact draft checks with:

```bash
python -m tutor.content.product_quotient_release --check
python -m tutor.content.chain_rule_release --check
python -m tutor.content.solve_quadratics_release --check
python -m tutor.content.ftc_release --check
```

The FTC check covers 3,003 within-wave answer pairs and 6,006 directed
within-wave family paths. Its mandatory cumulative check covers all 24,310
unordered answer pairs and 48,620 directed family paths in the 221-family
catalog. Candidate math is normalized in bounded workers, and every canonical
match is confirmed by the restricted verifier before being reported as a leak.

Generate deterministic private packets used for independent review directly
from each wave's pending assessment and pedagogy source/review manifests:

```bash
python -m tutor.content.product_quotient_reviewer_packet \
  --check \
  --out-dir /tmp/product-quotient-review
python -m tutor.content.chain_rule_reviewer_packet \
  --check \
  --out-dir /tmp/chain-rule-review
python -m tutor.content.solve_quadratics_reviewer_packet \
  --check \
  --out-dir /tmp/solve-quadratics-review
python -m tutor.content.ftc_reviewer_packet \
  --check \
  --out-dir /tmp/ftc-review
```

The output contains exact learner-visible and spoken rendering, expected-answer
contracts, private widget scoring data, ordered hints and revealing-hint effects,
allocation paths, provenance and citations, and structured similarity warnings.
It is an offline truth-bearing artifact: never place it under an application
static directory or release mount. The command requires all decisions to remain
pending and `released_kcs` to remain empty; it creates no approval records,
changes no review manifest, and publishes no release. Independent reviewers
record their own decisions through the separate review workflow after inspecting
the packet's exact digests.

Every packaged wave is deliberately marked AI-assisted and unreviewed. These
automated integrity checks do not establish instructional validity or
psychometric family independence. Human reviewers must approve construct
coverage and ordering (including what two early successes establish), task
coherence, accessibility, and every bound family digest before promotion. The
remaining U-substitution wave requires five new KCs, 65 families, and five
pedagogy packs before the final 22-KC/286-family catalog can enter review.

## Persistence and deployment

The production deployment, readiness, backup/restore, quarantine, rollback,
and canary procedure is maintained in
[`docs/pilot-runbook.md`](docs/pilot-runbook.md). The commands below remain the
development and component-level setup reference.

Development and tests can run memory-only. A production pilot must use
PostgreSQL and a stable resume-token secret:

```bash
docker compose up -d db
export DATABASE_URL='postgresql+psycopg://tutor:tutor@localhost/tutor'

# Required for fresh and existing databases; safe to rerun. This upgrades to
# the explicit production head recorded in alembic_version.
alembic -c backend/alembic.ini upgrade head

# Validate the data assets and publish the graph only after schema migration.
python -m tutor.seed.load_seed --validate --db "$DATABASE_URL"

# Generate once, store in a secret manager, and reuse across restarts.
export TUTOR_RESUME_TOKEN_SECRET='replace-with-at-least-32-random-bytes'
export TUTOR_REDIS_URL='rediss://redis.example.invalid/0'
export TUTOR_NETWORK_HMAC_SECRET='replace-with-an-independent-32-byte-secret'
export TUTOR_TRUSTED_PROXY_CIDRS='10.0.0.0/8'
export OTEL_EXPORTER_OTLP_ENDPOINT='https://otel.example.invalid'
export TUTOR_PILOT_PRODUCTION=1
export TUTOR_ENABLE_API_SESSION_V2=1
export TUTOR_ENABLE_CONTENT_ALLOCATION_V2=1
export TUTOR_ENABLE_DIAGNOSIS_V2=1
export TUTOR_ENABLE_LESSON_FLOW_V2=1
export TUTOR_ENABLE_RICH_WIDGETS_V2=0
export TUTOR_PAUSE_V2_MUTATIONS=0
export TUTOR_V2_STUDENT_ROLLOUT_PERCENT=0
uvicorn tutor.api.app:app
```

`python -m tutor.db.migrate_session_v2` remains an equivalent compatibility
entry point for existing deployment automation. Both commands require an
explicit `DATABASE_URL`; neither application startup nor the migration command
silently targets an in-memory database.

Pilot mode fails startup if `DATABASE_URL` is not PostgreSQL, persistence
cannot initialize, the v2 schema has not been migrated, the resume secret is
missing/too short, or the Redis safety/admission and OpenTelemetry adapters
cannot be constructed. It also requires all six feature flags and the rollout
percentage explicitly; the example starts closed before the reviewed 5/25/100
canary progression. Non-pilot development reports `memory_only` durability
when no database is configured.

Durable sessions restore only the exact registered graph, item-bank, and
pedagogy-catalog triple pinned in their checkpoint and checkpoint row. Keep
each published release directory, including `bundle.json`,
`release-reviews.json`, `release-manifest.json`, and `bundle.sha256`, for the
full resume window, and configure:

```bash
export TUTOR_V2_RELEASE_REGISTRY_DIR=/srv/tutor/releases
```

Once an exact release bundle has passed independent review, select it rather
than relying on packaged draft content. Pilot deployments must pin the bytes by
SHA-256 so a path replacement cannot silently change the active release:

```bash
export TUTOR_V2_ACTIVE_RELEASE_BUNDLE=/srv/tutor/releases/product-quotient-v1
export TUTOR_V2_ACTIVE_RELEASE_SHA256='<bundle.json 64-character SHA-256 digest>'
```

Do not set these variables for the pending Product/Quotient draft: it contains
no released KCs and is intentionally ineligible for active-session admission.

Startup rejects malformed or partial publication directories, invalid exact
attestations, incompatible graph/bank/catalog triples, unregistered component
cross-products, and reuse of any version identifier for different content. A
policy-version bump must retain the executable restore implementation too. Each
configured Python module must expose
`register_v2_policy_runtimes(registry)` and register the exact version set it
can restore:

```bash
export TUTOR_V2_POLICY_RUNTIME_MODULES='tutor_retained.policy_v20,tutor_retained.policy_v21'
```

Startup fails if a retained module cannot be imported or lacks that hook. Keep
the predecessor implementation in the deployment image for the full 30-day
Resume reconciles the checkpoint against the ordered durable transcript,
evidence, exposure-transition, widget-attempt, and mutation-receipt ledgers.
Missing or divergent rows fail closed with `503` and an integrity metric; the
checkpoint copy cannot silently mask missing append-only evidence. The
additive Alembic revisions label pre-catalog evidence and checkpoints `legacy`
rather than fabricating trust: legacy evidence may weakly seed belief, but cannot
confirm mastery, carry misconception flags, propagate mastery, or receive a
practice-learning transition. A pre-catalog v2 checkpoint is not silently
rebound to the deployment's current catalog and therefore fails closed.
Expired
anonymous checkpoint, transcript, receipt, exposure, widget, and token rows
are purged while learner identity and longitudinal evidence are retained.

### PostgreSQL transaction gates

The ordinary backend suite remains self-contained on SQLite. A dedicated CI
job starts an isolated PostgreSQL 16 service and runs the production-like
contention, rollback, and process-kill tests. Local execution is opt-in so it
cannot accidentally target a developer or production database:

```bash
docker compose up -d db
export TUTOR_TEST_POSTGRES_URL='postgresql+psycopg://tutor:tutor@localhost/tutor'
pytest backend/tests/test_api_v2_postgres.py -v
```

Use a disposable database and a role allowed to create and drop schemas. Each
test creates a randomly named private schema, runs the complete v2 schema
there, and drops it afterward. When `TUTOR_TEST_POSTGRES_URL` is unset, the
module skips cleanly.

These gates exercise two independent application stores contending on the same
PostgreSQL checkpoint. They assert that duplicate actions produce one revision,
one evidence event, one transcript mutation, and one receipt; distinct actions
serialize with one stale loser; and a deliberately injected failure during the
receipt insert rolls back the entire turn. The failed request is then retried
with the same request ID, restored in a fresh app instance, and replayed without
additional durable rows. A fourth gate terminates a child process immediately
after the receipt insert but before commit, then proves Postgres rolled the
whole turn back before retry and exact replay.

## Feature flags, cohort rollout, and widget capabilities

These rollout switches are independent. Non-production development defaults
them to enabled; `TUTOR_PILOT_PRODUCTION=1` requires each value explicitly:

- `TUTOR_ENABLE_API_SESSION_V2`
- `TUTOR_ENABLE_CONTENT_ALLOCATION_V2`
- `TUTOR_ENABLE_DIAGNOSIS_V2`
- `TUTOR_ENABLE_LESSON_FLOW_V2`
- `TUTOR_ENABLE_RICH_WIDGETS_V2`
- `TUTOR_PAUSE_V2_MUTATIONS`

Disabling the API flag removes the v2 routes. Disabling content allocation,
diagnosis, or lesson flow returns an empty catalog with a `paused` rollout
status for selected browsers. Rich widgets are separate: disabling them keeps
the student flow available and serves required guided practice through the
keyboard text equivalent.

`TUTOR_PAUSE_V2_MUTATIONS=1` is the separate emergency write stop. It keeps
goal/capability reads, current-session reads, cookie refresh, and receipt-only
recovery available. It also preserves exact replays of already committed
create, action, and reset request IDs. Any genuinely new create, action, or
reset returns `503` with code `v2_mutations_paused` and `retryable: true`
before invoking the orchestrator or writing revision, transcript, evidence,
exposure, widget-attempt, or receipt state. Clients must retry the identical
payload and request ID after the operator clears the pause.

Pilot production uses a built-in Redis refresher for the mutation switch and
release quarantine. Requests read immutable process-local snapshots; only the
background refresher performs Redis I/O. Provider exceptions, malformed
documents, stale observations, and materially future-dated observations fail
closed. The static pause flag remains a one-way ceiling, and receipt lookup
still precedes both the mutation gate and request bucket. Optional
`TUTOR_V2_MUTATION_GATE_FACTORY` and
`TUTOR_V2_RELEASE_QUARANTINE_FACTORY` adapters override the built-ins when a
deployment supplies the same runtime contracts.

Redis also owns fleet-shared token buckets for create, recover, reset, action,
and read requests. Network identities are HMACs produced from the direct peer;
`X-Forwarded-For` is considered only when that peer belongs to
`TUTOR_TRUSTED_PROXY_CIDRS`. The defaults are 10 create/recover/reset requests
per 10 minutes, 60 actions per minute, and 120 reads per minute. A depleted
bucket returns typed `429 rate_limited` with `Retry-After`. Redis failure closes
new mutations and API reads; liveness and static assets remain available. A custom gate can be supplied through
`TUTOR_V2_REQUEST_ADMISSION_FACTORY`.

New-session admission uses one explicit percentage:

```bash
# The only accepted values are 0, 5, 25, and 100.
export TUTOR_V2_STUDENT_ROLLOUT_PERCENT=5
```

The first catalog request receives a signed, random 256-bit anonymous cohort
cookie. Its deterministic bucket is stable across requests and process
restarts when `TUTOR_RESUME_TOKEN_SECRET` is stable. Increasing rollout from 5
to 25 to 100 preserves every previously selected cohort. The cookie is
`HttpOnly`, `SameSite=Lax`, `Secure` away from localhost, scoped to `/api/v2`,
and contains no account or student content. Invalid or modified values are
replaced.

Rollout admission controls new episodes only. Current-session reads, actions,
reset, and exact idempotent creation replays remain available so lowering or
pausing admission does not strand an existing student. The server enforces the
same assignment on `POST /api/v2/sessions`; hiding goals in the interface is
not the security boundary. Percentage selection never bypasses the
released-content and complete hard-ancestor validation gate.

Legacy-session diagnosis shadowing is separately opt-in:

```bash
export TUTOR_ENABLE_DIAGNOSIS_V2_SHADOW=1
```

The observer consumes only already-scored evidence metadata, never raw
answers, prompts, expected answers, learner IDs, or context. It compares v2's
counterfactual next-KC choice with the unchanged v1 route, stops comparison
after divergence, isolates observer failures, and reports aggregate metrics
under `diagnosis_v2_shadow` in `/healthz`.

The full manifest currently enables mapping and slider controls with keyboard
equivalents. `live_input` stays on the accessible text path until reviewed
`render.plot` semantics are implemented end to end; `click_region` remains
disabled because true target geometry and an equivalent keyboard interaction
are not implemented. Widget attempts are formative; they cannot establish
mastery. Every submission is durably ordered, and invalid, incorrect, solved,
remediated, and text-fallback outcomes remain distinct in the safe transcript
and attempt ledger.

The browser starts with a mapping-only manifest and stays there if capability
fetch or validation fails. The server pins the episode manifest and intersects
it with the runtime switch, so an emergency rich-widget rollback converts an
already pending visual practice to text and prevents new rich generation.

`GET /healthz` reports effective flags, persistence availability, catalog and
session counts, active graph/item-bank/policy/learner-parameter/capability
versions, privacy-safe v2 counters keyed by stable item IDs, resume success
rate, middleware-counted action 5xx rate, duplicate-advance and
missing-evidence detections, and commit-integrity failures. The resume-rate
denominator contains only eligible attempts whose token is known active and
recoverable. Active-token restoration and ledger failures count as eligible
failures, as does a rolling-expiry refresh failure after authorization.
Success is recorded only after the read returns a refreshed cookie. Both
`GET /api/v2/sessions/current` and `GET /api/v2/sessions/{id}` are measured
read/resume journeys; authorization performed for create, action, reset, and
recovery paths is deliberately excluded. Raw cookie attempts remain
informational; no-cookie,
malformed/unknown/revoked, observable expired, and session-id-mismatch requests
remain separate and do not dilute the reliability rate. An expiry already
removed by retention is classified as invalid because no durable row remains.

The process-local snapshot remains useful for tests and one-worker
development, but is not a fleet aggregator. Production constructs a bounded,
nonblocking OpenTelemetry sink from the standard OTLP environment. An optional
`TUTOR_V2_METRICS_SINK_FACTORY=package.module:factory` overrides it; embedded
callers may instead pass `create_app(v2_metrics_sink=...)`. Every exported
increment is tagged only with the originating session's pinned graph,
item-bank, pedagogy-catalog, learner, capability, and policy versions and,
where applicable, a reviewed stable item ID. A blocked exporter fills only a
bounded queue; drops increment an independent local failure counter and never
block a tutoring request. Raw
answers, expected answers, session/learner IDs, prompts, and student context are
excluded from the sink contract. Readiness reports
`fleet_metrics_configured` without exposing provider configuration.

## Diagnosis simulation

The v2 harness exits non-zero when an encoded diagnosis-policy gate fails. A
10,000-episode run across the five pilot targets and five seeds is:

```bash
python -m tutor.sim.harness_v2 \
  --learners 100 \
  --budget 8 \
  --seeds 3 7 11 17 23
```

CI runs this exact 10,000-episode activation gate on every change and fails
closed when any encoded threshold regresses.

The command runs 10,000 episodes across the five goals, five seeds, and four
slip/guess profiles. The latest local run of the selected policy passed the
encoded diagnosis gates: 1.19% false-mastery skips, 96.98% frontier precision,
93.06% next-KC accuracy, 0.026 mean overteach, ECE
0.0166, Brier 0.0372 versus v1's 0.1072, and perfect-learner probe counts of
median 2 / p95 2.

Run the full 15-pair `lambda`/`delta` comparison with `--sweep`; candidates
that fail any diagnosis-policy gate rank behind passing candidates. The completed
150,000-episode sweep selected `lambda=0`, `delta=0.25`: all candidates passed,
and the selected pair produced 93.06% next-KC accuracy versus 92.48% for the
former `lambda=0.35`, `delta=0.5` pin. Diagnosis policy `v2.1` pins the winner.

The harness is synthetic diagnosis-policy evidence, not a complete release
proof. It does not claim to measure allocator or capstone family reuse. Those
invariants are covered by allocator/state-machine property tests and must also
run across every reviewed content family before release.

## Browser and accessibility journey

The dedicated `browser-e2e` CI job builds the unified application, starts a
guarded test-only FastAPI runtime, and runs the Chromium Playwright journey
with axe. The runtime is defined under `backend/tests/`, requires
`TUTOR_E2E_TEST_APP=1`, and explicitly injects the narrow approved power-rule
fixture. It does not alter the packaged draft bank, production app factory, or
fail-closed catalog behavior.

The journeys cover keyboard-only intake, a full no-refresh lesson and
capstone, a competing-tab stale revision, a real committed-response transport
retry, required guided text practice, exact reload/cookie resume, create/reset
response-loss recovery, malformed-widget fallback, desktop and 390px mobile
layouts, horizontal-overflow checks, and WCAG A/AA axe scans. Run them locally
after installing Chromium:

```bash
python -m pip install -e "backend[dev,api]"
npm --prefix frontend ci
npx --prefix frontend playwright install chromium
npm --prefix frontend run build
npm --prefix frontend run test:e2e
```

Test discovery does not require launching Chromium:

```bash
npm --prefix frontend run test:e2e:list
```

CI retains Playwright traces, screenshots, videos, and the HTML report when
the journey fails. These automated checks are necessary but do not replace a
headed visual review in the target deployment environment.

## Remaining pilot blockers

- Author and independently review complete item-bank coverage for the five
  goals and their 22 hard prerequisites; the shipped bank remains a draft.
- Collect and review diagnosis shadow metrics on representative legacy
  traffic, then approve the 5/25/100 canary progression. Stable admission and
  failure-isolated shadowing are implemented, but promotion remains an
  operator decision rather than an automatic metrics controller.
- Run the PostgreSQL contention, rollback, and process-kill suite in the target
  deployment environment in addition to the dedicated PostgreSQL CI job.
- Run and visually review the Playwright/axe journey in the target deployment
  environment; repository CI now executes its automated Chromium checks.
- Verify Redis admission, mutation pause, and release quarantine across two
  target-deployment workers, including outage and trusted-proxy drills.
- Verify pinned-version dimensions in the target telemetry backend and alert
  on queue drops and `metrics_export_failures`.
- Keep `click_region` disabled until geometric hit testing and equivalent
  keyboard semantics are complete.

No pilot should be activated until the reviewed-content, operational, replay,
accessibility, and no-leakage gates all pass together.
