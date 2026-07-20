# Phase 1 Implementation Handoff

Date: 2026-07-20

Status: Phase 1 is implemented, hardened, and ready for Fable's implementation
review. Fable previously accepted Tasks 1-2. The product owner then accepted
submit-only coaching for the pre-pilot and authorized the amended Phase 1 plan.

## Product-owner decision — resolved

The accepted Phase 1 contract changes directional coaching from live-during-drag
feedback to submit-only feedback:

- Raw `feedback_rules` remain server-only.
- A learner receives at most the first matching hint after an incorrect **Check**.
- Each hint therefore costs a client/server round trip.
- There is no coaching while the learner is dragging.
- Secure live coaching is deferred to a separately designed later protocol.

The product owner explicitly accepted that temporary pre-pilot tradeoff before
implementation began. Live-during-drag coaching is not part of this implementation
and is not implied by the current protocol; it can be reconsidered in a later version
only after a secure design review.

## Phase 1 implementation result

### Server authority and answer boundary

- `_score_widget` remains unchanged and authoritative.
- The browser receives only explicitly projected safe widget fields. Nested
  `Region.shape` and live-input `render` dictionaries are now normalized through
  allowlists instead of being copied as free-form data.
- Slider bounds, steps, targets, and tolerances reject non-finite values at schema
  validation.
- Deterministic planner gates reject slider copy that exposes the target, including
  equivalent numeric and exact forms such as scientific notation, `pi/2`, and
  `sqrt(2)`. Authorized coordinate markers remain usable without weakening prompt
  checks.
- After an incorrect slider Check, the server evaluates retained `feedback_rules`
  against the inferred slider parameter and appends at most the first matching hint.
  Thresholds use the existing restricted SymPy parser; malformed, mismatched,
  symbolic, or non-finite rules are logged and ignored.

### Frontend and rendering

- `frontend/` is now the sole UI source. The previous single-file UI was removed
  after the Svelte application reached feature parity.
- The exact toolchain is locked in `package-lock.json`: Svelte, Vite, Elm, and GSAP.
  GSAP is installed but intentionally unused in Phase 1.
- Shared Svelte chrome owns submission, verdict display, request locking,
  disable-on-correct behavior, and request cancellation during session replacement
  or component teardown.
- Slider, click-region, mapping, and live-input widgets retain native controls and
  submit only `value`, `selected`, `pairs`, and `text`, respectively. Mapping choices
  are shuffled once per mount.
- Rich sliders use a small non-evaluating expression parser, one compile cache, and
  an Elm `<tutor-scene>` custom element. The scene supports responsive plots, point
  markers, bounded and one-sided shades, segmented discontinuities, and the fixed
  Phase 1 viewport.
- Invalid expressions, decoder failures, unsupported overlays, and scenes with no
  drawable segment show an explicit status while leaving the native control usable.
  Custom-element reconnects do not duplicate event subscriptions.

### Build, serving, and packaging

- Vite emits a clean, hashed bundle to
  `backend/src/tutor/api/static/dist/`.
- FastAPI serves the built index and assets while preserving API routes and existing
  dotenv initialization.
- Python package data includes the built UI, seed graph JSON, and pedagogy-pack CSV.
  A clean wheel was installed into an isolated directory and served its root and both
  hashed assets successfully.
- `scripts/check_frontend_dist.sh` detects modified, deleted, and untracked build
  output. The generated `dist/` files are included in the Phase 1 milestone commit,
  so the guard can verify the committed bundle against a clean rebuild.

## Completed work

### Task 1: widget-answer redaction

`SessionOrchestrator._issue_lesson` creates a separate client widget dump and removes:

- `success_condition`
- `correct_region_ids`
- `correct_pairs`
- `checker`
- `feedback_rules`

The full `planned.widget` remains in `_active_widgets` for authoritative server
scoring. `_score_widget` and the response fields it reads are unchanged.

The runtime tests exercise all four widget variants through diagnosis, lesson issue,
client serialization, and correct server scoring. The HTTP test checks the actual
wire payload.

### Task 1 follow-up: exhaustive field-classification guard

The tests now derive every concrete member of the `WidgetConfig` union and recursively
walk its Pydantic fields, including nested models.

Every serialized field path must be classified independently in test code as either:

- client-safe; or
- answer-bearing/server-only.

The guard also includes computed fields and rejects models configured with
`extra="allow"`. A future widget variant or field fails CI until someone explicitly
classifies it. The serialization regression then requires the browser payload to
contain exactly the safe paths and none of the server-only paths.

The production exclusion set and test classification remain independent. Updating a
field to server-only in the test does not update production redaction automatically,
so CI still catches an incomplete security change.

The schemas still model `Region.shape` and `LiveInputWidget.render` as
`dict[str, Any]`, so the field walker classifies those containers rather than
arbitrary nested paths. Production serialization no longer forwards either
dictionary wholesale: it projects supported point/rectangle/circle geometry and
live-render plot/variable fields through explicit allowlists, dropping all other
nested keys. Runtime tests pin those projections, while deterministic planner gates
cover semantic target leaks in authorized slider text and expressions.

### Task 2: rich interaction prompt

The accepted p4 work added:

- declarative, render-ready widget guidance;
- concrete goal-state and tight-coupling rules;
- rich evaluator criteria;
- misconception remediation hints and metaphor ids;
- documentation for `plot`, `shade`, `when`, `shape`, and `render`.

Widget types, defaults, constraints, the discriminated union, parsing, and scoring
were not changed.

### Task 2 follow-up: p5 corrections

`PROMPT_VERSION` is now `p5`.

The p5 changes are:

1. The click-region example uses neutral ids `r1` through `r4`, with `r2` correct.
   The angle-bearing ids `p_30`, `p_120`, `p_210`, and `p_300` are gone.
2. The generator hard rules say click-region ids reach the client and must not encode
   answers, coordinates, angles, semantic meaning, or correctness.
3. The evaluator safety gate independently rejects answer-bearing ids.
4. The `interaction_user()` directive requests `feedback_rules` for sliders only.
   It no longer asks live-input widgets for a field their schema does not have.
5. Prompt and schema documentation now state that raw rules are server-only and are
   evaluated after an incorrect slider submission.
6. The evaluator calls the conditions server-evaluated rather than frontend- or
   machine-evaluated.

Prompt-contract tests pin the version, neutral example ids, generator/evaluator id
policies, slider-only directive, and server-side feedback wording.

`PROMPT_VERSION` still has no consumer outside `prompts.py`; Fable recommended leaving
that pre-existing provenance gap as-is.

## Final verification

Automated verification from the repository root:

```text
backend/.venv/bin/python -m pytest backend/tests -q
179 passed, 1 warning

backend/.venv/bin/ruff check backend/src backend/tests
All checks passed!

npm --prefix frontend test -- --run
3 test files passed; 23 tests passed

npm --prefix frontend run build
122 modules transformed; production build passed

npm --prefix frontend audit --audit-level=high
0 vulnerabilities
```

`git diff --check`, shell syntax validation, and the source scan forbidding `eval`
and `new Function` all pass. The one Python warning is the pre-existing Starlette
`TestClient`/httpx deprecation warning.

The clean production bundle contains:

```text
index.html
assets/index-Db7VEdQx.js
assets/index-V3UnIOAS.css
```

The Elm program compiles as part of the production build. A clean wheel contains
exactly the current UI bundle plus the seed and pack data. Its isolated smoke test
loaded all 40 KCs and returned HTTP 200 for `/` and both hashed assets.

Production-browser verification used uvicorn-served bundle assets and covered:

- a complete template session from Start through DONE, including a diagnosis hint;
- a rich slider whose Elm curve changed during drag;
- exact `{key, response: {value}}` submission, server-scored correctness, and a
  matched submit-only coaching hint;
- request locking and cancellation when a session replaces an in-flight widget;
- explicit native fallback for invalid rich plots;
- click-region, mapping, and live-input rendering, payloads, and scoring;
- decoder failure, custom-element disconnect/reconnect, pole segmentation, and
  narrow mobile layout;
- no unexpected console, network, or HTTP errors.

## Existing worktree changes preserved

The worktree already contained unrelated edits in:

- `AGENTS.md`
- `README.md`
- `backend/pyproject.toml`
- `backend/src/tutor/api/app.py`
- `backend/src/tutor/cli.py`
- `backend/src/tutor/packs/ingest.py`
- untracked `backend/uv.lock`
- untracked `scripts/try_llm_call.py`

Those pre-existing changes, including the dotenv work in `app.py`, were preserved.
Phase 1 necessarily adds nearby edits to some of the same files; it does not revert
or overwrite the prior work. The untracked `backend/uv.lock` and
`scripts/try_llm_call.py` remain untouched.

## Implemented Task 3, Phase 1 contract

The following is the complete reviewed contract implemented in this handoff and is
retained as Fable's acceptance checklist.

### 1. Build and serving contract

- Create `frontend/` as the only UI source directory.
- Add Vite, Svelte, Elm, and GSAP.
- Install GSAP as part of the approved toolchain but add no GSAP behavior in Phase 1.
- Relocate the existing HTML, CSS, and vanilla chat/session flow without redesigning
  it.
- Remove `backend/src/tutor/api/static/index.html` only after built-asset parity is
  verified; do not keep a stale legacy fallback.
- Configure Vite with a fixed asset base and build hashed files into
  `backend/src/tutor/api/static/dist/`.
- Serve `dist/index.html` through FastAPI `FileResponse`.
- Mount the hashed asset directory with FastAPI `StaticFiles`.
- Preserve all API routes and the current dotenv initialization.

Development workflow:

1. Run FastAPI/uvicorn on port 8000.
2. Run the Vite development server with `/sessions` and `/healthz` proxied to FastAPI.

Production/local Python workflow:

1. Run the deterministic frontend build.
2. Run uvicorn; FastAPI serves the committed `dist/` output.

Commit `package-lock.json` and the built `dist/` artifacts for the solo pre-pilot.

### 2. Svelte ownership and recipe contract

One shared Svelte `WidgetChrome` owns:

- widget kind tag;
- prompt;
- native-control and scene slots;
- Check button and request;
- server verdict;
- disable-on-correct behavior.

The ambiguous registry `chrome` member is replaced by an explicit per-type control:

```text
recipes[widget_type] -> {
  init(config) -> state,
  normalize(config, state) -> Scene | null,
  responseFrom(state) -> server payload,
  control: SvelteComponent
}
```

Phase 1 has one rich slider normalizer and four native controls/fallbacks:

- slider: range input plus current value;
- click region: toggle buttons;
- mapping: selects with right-side options shuffled once per mount;
- live input: text input.

The mapping shuffle is required so option order does not encode the answer.

A rich scene is only an enhancement. Unknown or unparseable rich data leaves the
correct per-type native control visible and displays an explicit fallback status.
The client submits only `value`, `selected`, `pairs`, or `text`.

### 3. Elm `<tutor-scene>` boundary

- Define one normalized `Scene` union and one total dispatch.
- Add the Phase 1 plot scene: plane, segmented curve, point marker, and region shade.
- Render through a single Elm `Scene -> Svg` path.
- Wrap the Elm program as `<tutor-scene>`.
- Use one bundled downward property containing `{scene, status}`.
- Use one upward `interact` `CustomEvent`.
- Decode initial flags and subsequent values totally.
- On decoder failure, keep the native control visible and show an explicit fallback
  rather than a blank SVG.

### 4. Safe expression parser and only cache

Implement a small recursive-descent or shunting-yard parser, not a second CAS and
never `eval`/`Function`.

Supported grammar:

- finite numeric literals and `pi`;
- `+`, `-`, `*`, `/`, `^`, and `**`;
- unary signs and parentheses;
- `sin`, `cos`, `tan`, `sec`, `exp`, `log`, `ln`, and `sqrt`;
- a plot equation `y = <expression>`.

Slider plots must contain exactly `x` plus one inferred slider parameter. A second
free parameter such as `y = m*x + b`, a non-responsive plot, unknown syntax, or an
invalid goal overlay causes visible rich-scene fallback while preserving the slider.

Use exactly one cache:

```text
Map<raw expression string, compiled expression>
```

The same cache serves plot, marker, and region-bound expressions. Parse once per raw
string; evaluate and sample on each state update.

Use the fixed Phase 1 viewport `[-5, 5] x [-5, 5]`. A schema-level viewport/domain
field remains a likely later addition.

### 5. Point and region shades

Support both p5-authorized shade forms:

- point marker: `point(expr, expr)`;
- one-sided interval: `x <op> constant` or `constant <op> x`;
- bounded interval: `constant <op> x <op> constant`;
- operators: `<`, `<=`, `>`, and `>=`;
- exact bounds such as `pi/2` and `sqrt(2)`.

Clip one-sided and bounded intervals to the fixed viewport. Render a region shade as
the area between the curve and x-axis over the interval. Unsupported shade syntax
falls back to the native slider rather than silently dropping the visual.

This is required so natural integral/u-substitution visuals using
`0 <= x <= 2` do not degrade to bare controls.

### 6. Non-finite curve handling

Each sampled point is either finite or a gap:

- `NaN` and infinity terminate the current SVG subpath.
- Invalid spans are omitted.
- Opposite clipped y-boundaries are not connected across a pole.
- Remaining finite segments render independently.
- Shaded polygons use the same segmentation.
- Rich-scene fallback happens only when no drawable finite segment remains.

Required cases include `ln(x)` over the negative half of the viewport, `tan(x)` near
poles, and an expression with no finite segment.

### 7. Server-side submit-only feedback

Keep `_score_widget` unchanged.

After an incorrect slider submission:

1. Read rules only from the retained full server widget.
2. Split one comparison using `<`, `<=`, `>`, or `>=`.
3. Require the left identifier to match the slider parameter inferred from
   `params.plot`.
4. Parse the right threshold through the existing restricted SymPy parser.
5. Require no free symbols and a real finite result.
6. Compare it with the already parsed finite submitted value.
7. Append only the first matching rule's `say` text to the existing verdict message.
8. Ignore and log malformed, mismatched-symbol, or non-finite rules.

This admits exact thresholds such as `pi/2` and `sqrt(2)` without arbitrary
evaluation. The API response remains `{correct, message}`. The matched `say` text may
reach the learner inside `message`; the raw rule and threshold never reach the
browser.

Add tests for boundaries, exact constants, malformed rules, wrong parameters,
non-finite thresholds, first-match behavior, and absence of rules from client
payloads.

### 8. Verification

Automated:

- parser grammar, rejection, and exact-constant tests;
- cache test proving one parse across repeated slider updates;
- marker and one-sided/bounded region normalization;
- non-finite curve and shade segmentation;
- Elm compilation and total-decoder cases;
- server feedback tests;
- production Vite build;
- FastAPI root and every hashed-asset route;
- complete backend test and lint suites.

Real-browser verification must use uvicorn-served production assets, not the Vite
development server:

1. Complete one template session from Start through DONE.
2. Observe a probe and request at least one hint.
3. Assert no console or network errors.
4. Exercise a rich slider and confirm the curve path changes during drag.
5. Confirm the request body contains only `value`.
6. Confirm the server controls correctness and returns a matched submit-only hint.
7. Confirm an invalid rich plot visibly uses the native slider fallback.
8. Confirm click-region, mapping, and live-input controls render and score as before.

### 9. Committed-build staleness guard

CI and local release preparation must run:

1. `npm ci`
2. A clean production build that empties and recreates `dist/`
3. `git diff --exit-code -- backend/src/tutor/api/static/dist`
4. A scoped `git status --porcelain --untracked-files=all` check for new files in
   `dist/`

This catches modified, deleted, and newly generated hashed assets.

## Phase 1 non-goals

- No rich click-region visualization.
- No live-input visualization.
- No mapping redesign.
- No GSAP slider smoothing or curve tweening.
- No GSAP behavior until a later success-pulse phase.
- No pre-rendering.
- No additional caches.
- No localStorage or content-addressed config cache.
- No schema-version fallback ladder.
- No six-primitive component kernel.
- No new widget type.
- No redesign of the working chat/session flow.
- No secure live-during-drag coaching protocol in Phase 1.

## Sign-off checklist

- [x] Accept Task 1 redaction and tests — Fable accepted.
- [x] Accept server-only `feedback_rules` — Fable accepted.
- [x] Accept Task 2 rich prompt/schema documentation — Fable accepted.
- [x] Add exhaustive schema-field classification — implemented and verified.
- [x] Correct prompt id leaks and use neutral ids — implemented as p5.
- [x] Scope feedback generation to sliders — implemented as p5.
- [x] Correct the documented feedback ownership — implemented as p5.
- [x] Use `frontend/` and `backend/src/tutor/api/static/dist/`.
- [x] Include the generated build output and staleness guard in the Phase 1
  milestone commit.
- [x] Clarify shared Svelte chrome and all four per-type fallbacks.
- [x] Support point and region shades.
- [x] Segment non-finite curves and shades.
- [x] Parse feedback thresholds with restricted SymPy and log rejected rules.
- [x] Product owner explicitly accepts submit-only coaching for Phase 1.
- [x] Product owner explicitly authorizes installation and implementation of Phase 1.
