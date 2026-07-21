# Margin — OpenAI Build Week submission video

Target runtime: 2:45–2:55. The spoken script is intentionally below the
three-minute submission limit. Record the product sequence only against the real
API v2 service, its durable PostgreSQL session store, and an exact reviewed
release bundle. Do not substitute the browser-test fixture when recording.

## 0:00–0:15 — Inspiration

**Picture:** A calculus problem appears. Highlight one step, then zoom out to a
small prerequisite map. Bring in the Margin wordmark.

**Voiceover:**

> A student can miss one calculus problem for a reason hidden three lessons
> earlier. Most tutors repeat the topic or guess. Margin is the space beside the
> work where the right question reveals the real gap—and uncertainty can remain
> uncertainty.

## 0:15–0:40 — What it does

**Picture:** Show the goal picker, then a short diagnosis sequence. Pause on the
three learner-summary groups: confirmed strengths, confirmed gaps, and uncertain
skills.

**Voiceover:**

> Margin is an adaptive tutor for curated math goals spanning calculus and its
> prerequisites. It probes the goal, checks connected skills with distinct,
> separately authored question families, then teaches from the deepest confirmed
> gap. Each lesson combines explanation, guided practice, unseen check-ins, and a
> final goal problem. Hints, widgets, and assumptions never become mastery
> evidence.

## 0:40–1:13 — Product demonstration

**Picture:** Start a real Product and Quotient Rules session. Keep the **Saved**
indicator and **GPT-5.6 coach** attribution visible. Answer one probe, show an
adaptive route decision, complete guided practice, request a conceptual hint,
and answer a fresh check-in. Reload and show the same transcript, draft, widget
state, and progress restored from PostgreSQL.

**Voiceover:**

> One response changes this route. Margin explains the current skill and offers
> keyboard-accessible guided practice, but evidence comes from a different,
> unseen item family. A conceptual hint preserves the learner's draft; a
> revealing hint retires that question. Reloading this saved session restores
> its transcript, draft, widget state, and progress—without duplicate advancement
> or missing evidence.

## 1:13–1:50 — How we built it, with Codex and GPT-5.6

**Picture:** Fast cuts between the original browser audit, a simplified
architecture diagram, Codex diffs, small green commits, and passing test output.
Use the following architecture labels: **deterministic control plane**,
**versioned content**, **restricted verifier**, and **authoritative session view**.

**Voiceover:**

> We used Codex with GPT-5.6 as our development collaborator—not as Margin's
> scoring engine. Codex first operated the live app without refreshing and
> exposed answer reuse, hint grading, double advancement, widget traps, unsafe
> parsing, and broken recovery. It turned those findings into trust invariants
> and an implementation map, then worked repo-wide on versioned contracts, the
> deterministic session state machine behind FastAPI, Postgres checkpoints,
> Redis safety controls, the restricted verifier, content compilers, and the
> Svelte interface. Codex also ran focused, concurrency, recovery, browser, and
> simulation tests, committing each green behavior separately.

## 1:50–2:12 — Challenges we ran into

**Picture:** Show a red “answer leaked” comparison becoming green, two competing
tabs resolving to one revision, and a verifier worker timing out safely.

**Voiceover:**

> The hardest challenge was realizing that a correct-looking lesson can create
> false evidence. A worked example can leak an answer; two tabs can advance
> twice; symbolic math can be unsafe or slow. We addressed each at the boundary
> with disjoint item families, revision-checked idempotency, atomic commits, and
> bounded verifier workers.

## 2:12–2:31 — Accomplishments that we are proud of

**Picture:** Show the no-refresh journey, the content separation report, the
accessibility controls, and a “release blocked: review pending” screen.

**Voiceover:**

> We are proud that Margin is honest when that is inconvenient. It separates
> mastery, gaps, and uncertainty; resumes exactly; keeps scoring truth off the
> client; and blocks incomplete curriculum rather than substituting generated
> questions. Today, 221 draft families across 17 skills pass engineering
> separation checks, and zero are falsely labeled reviewed.

## 2:31–2:42 — What we learned

**Picture:** Replace a generic “AI tutor” box with the layered Margin
architecture.

**Voiceover:**

> We learned that trustworthy AI education depends less on generating words than
> preserving boundaries: what the learner showed, what the system inferred, and
> what remains unknown.

## 2:42–2:58 — What’s next for Margin

**Picture:** Animate the path **65 final families → independent review → exact
release → canary pilot**. End on the Margin wordmark and repository URL.

**Voiceover:**

> Next, we will finish the final U-substitution wave, independently review all
> 286 families and 22 teaching packs, qualify the exact release, and canary the
> five-goal pilot. Margin: adaptive teaching, with evidence you can trust.

## Recording checklist

- Keep the finished video below 3:00; aim for 2:50 to leave upload headroom.
- Upload it as a public YouTube video.
- Keep the recorded product name and wordmark consistently **Margin**.
- Say both “Codex” and “GPT-5.6” in the recorded audio; captions alone do not
  satisfy the stated requirement.
- Show the Codex session/model evidence supporting the GPT-5.6 claim. If the
  recorded build history used a different model, replace the sentence with the
  exact, truthful division of work before recording.
- Record at 1080p or higher and zoom the browser to keep student copy legible.
- Use captions, avoid rapid flashing, and leave each important state visible for
  at least two seconds.
- Do not expose expected answers, cookies, API keys, raw student data, or private
  reviewer artifacts in terminal or browser shots.
- Do not record this script until the displayed release digest has an independent
  family, KC, pedagogy, and exact-bundle attestation. Use final release-count
  claims only after they are factually true.
