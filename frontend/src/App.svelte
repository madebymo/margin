<script>
  import { onMount, tick } from "svelte";

  import { api, ApiError, apiV2 } from "./api.js";
  import { transcriptEntryAnnouncement } from "./accessibility.js";
  import {
    answerDraftScope,
    clearAnswerDraft,
    readAnswerDraft,
    writeAnswerDraft,
  } from "./drafts.js";
  import {
    clearPendingRecovery,
    MutationCoordinator,
    readPendingRecovery,
  } from "./mutations.js";
  import {
    catalogEmptyMessage,
    canPreserveAnswerDraft,
    isWidgetPending,
    normalizeGoalCatalog,
    normalizeSessionView,
    pendingAcceptsText,
    phaseLabel,
  } from "./session.js";
  import PromptSegments from "./components/PromptSegments.svelte";
  import WidgetHost from "./widgets/WidgetHost.svelte";
  import {
    installMinimalWidgetCapabilities,
    installWidgetCapabilities,
  } from "./widgets/capabilities.js";

  let apiMode = "v2";
  let bootState = "loading";
  let goals = [];
  let catalogRollout = {
    status: "paused",
    reason: "Loading the trustworthy goal catalog.",
    percentage: 0,
  };
  let view = null;
  let errorMessage = "";
  let statusMessage = "";
  let busy = false;
  let answerText = "";
  let answerElement;
  let choiceElement;
  let transcriptElement;
  let lastFocusedPendingKey = null;
  let lastTranscriptMarker = "";
  let legacyTranscript = [];
  let legacySessionId = null;

  let goalId = "";
  let courseBand = "calculus_1";
  const contentMode = "curated";

  const coordinator = new MutationCoordinator(async (action) => {
    if (!view?.session_id || !view?.pending?.key) {
      throw new Error("There is no active question to answer.");
    }
    return apiV2.action(view.session_id, {
      ...action,
      expected_revision: view.revision,
      pending_key: view.pending.key,
    });
  });

  const resetCoordinator = new MutationCoordinator(async (mutation) =>
    apiV2.reset(mutation),
    undefined,
    { recoveryOperation: "reset" },
  );

  const createCoordinator = new MutationCoordinator(async (creation) =>
    apiV2.create(creation),
    undefined,
    { recoveryOperation: "create" },
  );

  onMount(() => {
    const controller = new AbortController();
    bootstrap(controller.signal);
    return () => controller.abort();
  });

  async function bootstrap(signal) {
    bootState = "loading";
    errorMessage = "";
    try {
      const goalPayload = await apiV2.goals(signal);
      const catalog = normalizeGoalCatalog(goalPayload);
      goals = catalog.goals;
      catalogRollout = catalog.rollout;
      goalId = goals[0]?.id ?? "";
      apiMode = "v2";
      try {
        installWidgetCapabilities(await apiV2.capabilities(signal));
      } catch (capabilityError) {
        if (capabilityError?.name === "AbortError") throw capabilityError;
        installMinimalWidgetCapabilities();
        statusMessage =
          "Using minimal safe widget capabilities; rich guided visuals are unavailable.";
      }
    } catch (error) {
      if (error?.name === "AbortError") return;
      if (error instanceof ApiError && [404, 405].includes(error.status)) {
        apiMode = "v2";
        goals = [];
        catalogRollout = {
          status: "paused",
          reason: "The trustworthy session API is unavailable on this server.",
          percentage: 0,
        };
        goalId = "";
        bootState = "ready";
        installMinimalWidgetCapabilities();
        statusMessage =
          "New sessions are paused because the trustworthy session API is unavailable.";
        return;
      }
      errorMessage = error.message;
      bootState = "ready";
      return;
    }

    if (!(await recoverPendingSession(signal))) {
      bootState = "ready";
      return;
    }

    try {
      const payload = await apiV2.current(signal);
      applySnapshot(payload, { restoreDraft: true });
      statusMessage = "Your session was restored.";
    } catch (error) {
      if (error?.name === "AbortError") return;
      if (!(error instanceof ApiError) || ![401, 404, 410].includes(error.status)) {
        errorMessage = error.message;
      }
    } finally {
      bootState = "ready";
    }
  }

  async function recoverPendingSession(signal) {
    const pendingRecovery = readPendingRecovery();
    if (!pendingRecovery) return true;
    try {
      await apiV2.recover(pendingRecovery, signal);
      clearPendingRecovery(pendingRecovery);
      statusMessage = "Your committed session change was safely recovered.";
      return true;
    } catch (error) {
      if (error?.name === "AbortError") return false;
      if (error instanceof ApiError && !error.retryable) {
        // A complete 4xx response confirms that no matching active rotation
        // remains. The ordinary authoritative resume check can now proceed.
        clearPendingRecovery(pendingRecovery);
        return true;
      }
      errorMessage = error.message;
      return false;
    }
  }

  function applySnapshot(
    payload,
    {
      clearRetry = true,
      preserveAnswer = false,
      restoreDraft = false,
    } = {},
  ) {
    const previousAnswer = answerText;
    const previousView = view;
    const previousScope = answerDraftScope(previousView);
    const next = normalizeSessionView(payload);
    if (!next.session_id) {
      throw new Error("The server returned a session without an id.");
    }
    view = next;
    const canPreserve =
      preserveAnswer && canPreserveAnswerDraft(previousView, next);
    if (!canPreserve) clearAnswerDraft(previousScope);
    answerText = canPreserve
      ? previousAnswer
      : restoreDraft && pendingAcceptsText(next.pending)
        ? readAnswerDraft(answerDraftScope(next))
        : "";
    if (clearRetry) {
      coordinator.clearRetry();
    }
  }

  function appendLegacyTurn(turn, studentText = null) {
    if (studentText) {
      legacyTranscript = [
        ...legacyTranscript,
        {
          id: `student-${legacyTranscript.length}`,
          key: `student-${legacyTranscript.length}`,
          role: "student",
          kind: "you",
          text: studentText,
        },
      ];
    }
    const newEntries = (turn.interactions ?? []).map((entry, index) => ({
      ...entry,
      id: entry.id ?? `${entry.key ?? "turn"}-${legacyTranscript.length + index}`,
      role: entry.kind === "you" ? "student" : "tutor",
    }));
    legacyTranscript = [...legacyTranscript, ...newEntries];
    legacySessionId = turn.session_id ?? legacySessionId;
    const selectedGoal = goals.find((goal) => goal.id === goalId);
    view = normalizeSessionView({
      ...turn,
      session_id: legacySessionId,
      revision: legacyTranscript.length,
      transcript: legacyTranscript,
      goal: selectedGoal,
      durability: "memory_only",
      content_mode: {
        requested: contentMode,
        effective: turn.llm_enabled ? "llm_coaching" : "curated",
        fallback_reason:
          contentMode === "llm_coaching" && turn.llm_enabled === false
            ? "LLM coaching was unavailable, so reviewed content was used."
            : null,
      },
    });
    answerText = "";
  }

  async function startSession() {
    if (busy || !goalId) return;
    busy = true;
    errorMessage = "";
    statusMessage = "";
    try {
      if (apiMode === "v2") {
        applySnapshot(
          await createCoordinator.execute({
            goal_id: goalId,
            course: courseBand,
            age_band: "adult",
            content_mode: contentMode,
            context: null,
          }),
        );
        createCoordinator.clearRetry();
      } else {
        legacyTranscript = [];
        legacySessionId = null;
        const goal = goals.find((candidate) => candidate.id === goalId) ?? goals[0];
        appendLegacyTurn(
          await api("/sessions", {
            target_kc: goal.target_kc,
            llm: contentMode === "llm_coaching",
          }),
        );
      }
    } catch (error) {
      errorMessage = error.message;
    } finally {
      busy = false;
    }
  }

  async function executeV2Action(action, { preserveDraft = false } = {}) {
    busy = true;
    errorMessage = "";
    statusMessage = "";
    try {
      const payload = await coordinator.execute(action);
      applySnapshot(payload, { preserveAnswer: preserveDraft });
      return payload;
    } catch (error) {
      if (error?.view) {
        applySnapshot(error.view, {
          clearRetry: !error.retryable,
          preserveAnswer: error.retryable || preserveDraft,
        });
        statusMessage = "The session changed elsewhere. The latest state is shown.";
      }
      errorMessage = error.message;
      throw error;
    } finally {
      busy = false;
    }
  }

  async function submitAnswer() {
    const value = answerText.trim();
    if (!value || busy || !view?.pending) return;
    if (apiMode === "v2") {
      try {
        await executeV2Action({ type: "answer", answer: value });
      } catch {
        // Keep the answer in the field so a retry reuses the request id.
      }
      return;
    }
    busy = true;
    errorMessage = "";
    try {
      const turn = await api(`/sessions/${legacySessionId}/answer`, { answer: value });
      appendLegacyTurn(turn, value);
    } catch (error) {
      errorMessage = error.message;
    } finally {
      busy = false;
    }
  }

  async function requestHint() {
    if (busy || !view?.pending) return;
    if (apiMode === "v2") {
      try {
        await executeV2Action(
          { type: "request_hint" },
          { preserveDraft: true },
        );
      } catch {
        // The authoritative stale view, when supplied, has already been applied.
      }
      return;
    }
    busy = true;
    errorMessage = "";
    try {
      const data = await api(`/sessions/${legacySessionId}/hint`, {});
      legacyTranscript = [
        ...legacyTranscript,
        {
          id: `hint-${legacyTranscript.length}`,
          key: `hint-${legacyTranscript.length}`,
          role: "tutor",
          kind: "hint",
          text: data.hint || "No more hints are available for this item.",
        },
      ];
      view = normalizeSessionView({
        ...view.raw,
        session_id: legacySessionId,
        revision: legacyTranscript.length,
        transcript: legacyTranscript,
        pending: view.pending,
      });
    } catch (error) {
      errorMessage = error.message;
    } finally {
      busy = false;
    }
  }

  async function submitWidget({ key, response, signal }) {
    if (apiMode === "v2") {
      const previousKey = view?.pending?.key;
      const payload = await executeV2Action({
        type: "widget_attempt",
        response,
      });
      const snapshot = normalizeSessionView(payload);
      const feedback = [...snapshot.transcript]
        .reverse()
        .find(
          (entry) =>
            entry.kind === "widget_feedback" &&
            (entry.key === key || entry.raw?.interaction_key === key),
        );
      return {
        message: feedback?.text ?? "Practice response recorded.",
        correct: snapshot.pending?.key !== previousKey,
        authoritative: true,
      };
    }
    return api(
      `/sessions/${legacySessionId}/widget`,
      { key, response },
      { signal },
    );
  }

  async function useTextFallback({ key }) {
    if (apiMode === "v2") {
      return executeV2Action({
        type: "use_text_fallback",
      });
    }
    statusMessage =
      "This legacy session cannot switch this widget automatically. Start a curated v2 session for text alternatives.";
    return {};
  }

  async function resetSession() {
    if (busy) return;
    errorMessage = "";
    statusMessage = "";
    if (apiMode === "v1") {
      clearAnswerDraft();
      view = null;
      legacyTranscript = [];
      legacySessionId = null;
      return;
    }
    if (!view) return;
    busy = true;
    try {
      const reset = await resetCoordinator.execute({
        expected_revision: view.revision,
        pending_key: view.pending?.key ?? null,
      });
      if (reset?.session) {
        applySnapshot(reset.session);
      } else {
        clearAnswerDraft(answerDraftScope(view));
        view = null;
        answerText = "";
        lastFocusedPendingKey = null;
      }
      coordinator.clearRetry();
      resetCoordinator.clearRetry();
      statusMessage = "A fresh episode has started with your prior learning retained.";
    } catch (error) {
      if (error?.view) applySnapshot(error.view);
      errorMessage = error.message;
    } finally {
      busy = false;
    }
  }

  function handleAnswerKeydown(event) {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      submitAnswer();
    }
  }

  async function focusPending(key) {
    lastFocusedPendingKey = key;
    await tick();
    if (
      view?.pending?.key === key &&
      pendingAcceptsText(view.pending) &&
      (answerElement || choiceElement)
    ) {
      if (view.pending.input_mode === "choice") {
        choiceElement?.querySelector("input")?.focus();
      } else {
        answerElement?.focus();
      }
    }
  }

  async function scrollTranscript(marker) {
    lastTranscriptMarker = marker;
    await tick();
    transcriptElement?.scrollTo({
      top: transcriptElement.scrollHeight,
      behavior: globalThis.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches
        ? "auto"
        : "smooth",
    });
  }

  function widgetItem(entry) {
    const latestStatus = [...(view?.transcript ?? [])]
      .reverse()
      .find(
        (candidate) =>
          candidate.key === entry.key &&
          (candidate.widget_status || candidate.widget_state),
      );
    const pendingState =
      view?.pending?.key === entry.key ? view.pending.widget_state : null;
    return {
      ...entry.raw,
      key: entry.key,
      widget: entry.widget,
      widget_state:
        pendingState ?? latestStatus?.widget_state ?? entry.widget_state ?? null,
      widget_status: latestStatus?.widget_status ?? entry.widget_status,
      widget_attempt_number:
        latestStatus?.widget_attempt_number ?? entry.widget_attempt_number,
    };
  }

  function widgetIsActive(entry) {
    return Boolean(
      view?.pending?.key === entry.key &&
        (isWidgetPending(view.pending) || apiMode === "v1"),
    );
  }

  function pendingWidgetAppearsInTranscript() {
    if (!view?.pending?.key) return false;
    return view.transcript.some(
      (entry) => entry.key === view.pending.key && entry.widget,
    );
  }

  function pendingPromptAppearsInTranscript() {
    if (!view?.pending?.prompt) return false;
    return view.transcript.some(
      (entry) =>
        (entry.key === view.pending.key &&
          ["probe", "checkin", "capstone"].includes(entry.kind)) ||
        entry.text === view.pending.prompt ||
        (entry.key === view.pending.key &&
          entry.widget?.prompt === view.pending.prompt),
    );
  }

  $: pendingKey = view?.pending?.key ?? null;
  $: draftScope = answerDraftScope(view);
  $: if (draftScope && pendingAcceptsText(view?.pending)) {
    writeAnswerDraft(draftScope, answerText);
  }
  $: if (pendingKey && pendingKey !== lastFocusedPendingKey) {
    focusPending(pendingKey);
  }
  $: transcriptMarker = view?.transcript?.at(-1)?.id ?? "";
  $: latestTranscriptAnnouncement = transcriptEntryAnnouncement(
    view?.transcript?.at(-1),
  );
  $: if (transcriptMarker && transcriptMarker !== lastTranscriptMarker) {
    scrollTranscript(transcriptMarker);
  }
  $: activeGoal = goals.find((goal) => goal.id === goalId) ?? goals[0];
  $: catalogMessage = catalogEmptyMessage(catalogRollout);
</script>

<svelte:head>
  <title>Adaptive Math Tutor</title>
  <meta
    name="description"
    content="A focused math tutor that checks prerequisites and teaches from your starting point."
  >
</svelte:head>

<div class="app-shell">
  <header class="topbar">
    <a class="brand" href="/" aria-label="Adaptive Math Tutor home">
      <span class="brand-mark" aria-hidden="true">∫</span>
      <span>
        <strong>Adaptive Math Tutor</strong>
        <small>Build understanding, one step at a time</small>
      </span>
    </a>
    {#if view}
      <div class="topbar-status">
        <span class:memory-only={view.durability !== "durable"} class="durability">
          {view.durability === "durable" ? "Saved" : "Not durably saved"}
        </span>
        <span class="phase-pill">{phaseLabel(view.phase)}</span>
        <button
          type="button"
          class="quiet-button"
          disabled={busy}
          on:click={resetSession}
        >Restart this goal</button>
      </div>
    {/if}
  </header>

  {#if bootState === "loading"}
    <main class="loading-screen" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span>
      <p>Checking for a saved session…</p>
    </main>
  {:else if !view}
    <main class="welcome-layout">
      <section class="welcome-copy">
        <p class="eyebrow">A calmer way through calculus</p>
        <h1>Start where your understanding actually starts.</h1>
        <p class="lede">
          Choose a goal. The tutor will check a few connected skills, explain what
          it knows and does not know, then build a focused path to the goal problem.
        </p>
        <div class="trust-list" aria-label="How this tutor works">
          <div><span aria-hidden="true">01</span><p><strong>Brief check-in</strong>Different questions confirm important results.</p></div>
          <div><span aria-hidden="true">02</span><p><strong>Guided practice</strong>Examples and interactions stay separate from your checks.</p></div>
          <div><span aria-hidden="true">03</span><p><strong>Independent finish</strong>You complete an unseen goal problem without the answer being shown.</p></div>
        </div>
      </section>

      <section class="start-card" aria-labelledby="start-heading">
        <div>
          <p class="eyebrow">New learning path</p>
          <h2 id="start-heading">What would you like to work toward?</h2>
        </div>

        {#if goals.length === 0}
          <div class="catalog-empty" role="status">
            <span aria-hidden="true">⌁</span>
            <div>
              <h3>{catalogMessage.title}</h3>
              <p>{catalogMessage.body}</p>
            </div>
          </div>
          <button
            type="button"
            class="secondary availability-button"
            disabled={busy}
            on:click={() => bootstrap()}
          >Check availability again</button>
        {:else}
          <label>
            Goal
            <select bind:value={goalId} disabled={busy}>
              {#each goals as goal}
                <option value={goal.id}>{goal.title}</option>
              {/each}
            </select>
          </label>
          {#if activeGoal?.description}
            <p class="field-help">{activeGoal.description}</p>
          {/if}

          <label>
            Current course
            <select bind:value={courseBand} disabled={busy}>
              <option value="calculus_1">Calculus I</option>
              <option value="precalculus">Precalculus</option>
              <option value="algebra_2">Algebra II</option>
              <option value="other">Another course</option>
            </select>
          </label>

          <div class="curated-notice">
            <strong>Reviewed lessons</strong>
            <span>This pilot uses only authored, reviewed teaching and questions.</span>
          </div>

          <button
            type="button"
            class="primary-button start-button"
            disabled={busy || !goalId}
            on:click={startSession}
          >{busy ? "Starting…" : "Start my learning path"}</button>

          {#if apiMode === "v2"}
            <p class="resume-note">Your progress can resume on this browser for 30 days.</p>
          {/if}
        {/if}
      </section>
    </main>
  {:else}
    <main class="session-layout">
      <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
      <aside
        class="session-sidebar"
        aria-label="Learning path status"
        tabindex="0"
      >
        <div class="goal-card">
          <p class="eyebrow">Your goal</p>
          <h1>{view.goal.title || "Math learning path"}</h1>
          {#if view.pending?.skill_label}
            <p class="current-skill">
              <span>Current skill</span>
              <strong>{view.pending.skill_label}</strong>
            </p>
          {/if}
        </div>

        <section class="progress-card" aria-labelledby="progress-heading">
          <div class="section-heading">
            <h2 id="progress-heading">Progress</h2>
            {#if view.progress.total > 0}
              <span>{view.progress.completed}/{view.progress.total}</span>
            {/if}
          </div>
          {#if view.progress.bar_kind}
            <div
              class="progress-track"
              role="progressbar"
              aria-label={view.progress.bar_label}
              aria-valuemin="0"
              aria-valuemax="100"
              aria-valuenow={Math.round(view.progress.percent)}
            >
              <span style={`width: ${view.progress.percent}%`}></span>
            </div>
          {/if}
          <p>{view.progress.label || phaseLabel(view.phase)}</p>
          {#if view.progress.probe_budget != null}
            <p class="budget">
              Diagnosis checks used: {view.progress.probes_used ?? 0} of {view.progress.probe_budget}
            </p>
          {/if}
        </section>

        <section class="understanding-card" aria-labelledby="understanding-heading">
          <h2 id="understanding-heading">What we know so far</h2>
          <div class="understanding-group strengths">
            <h3>Confirmed strengths</h3>
            {#if view.learner_summary.confirmed_strengths.length}
              <ul>
                {#each view.learner_summary.confirmed_strengths as skill}
                  <li>{skill}</li>
                {/each}
              </ul>
            {:else}
              <p>Still gathering evidence.</p>
            {/if}
          </div>
          <div class="understanding-group gaps">
            <h3>Skills to build</h3>
            {#if view.learner_summary.confirmed_gaps.length}
              <ul>
                {#each view.learner_summary.confirmed_gaps as skill}
                  <li>{skill}</li>
                {/each}
              </ul>
            {:else}
              <p>No confirmed gaps yet.</p>
            {/if}
          </div>
          <div class="understanding-group uncertain">
            <h3>Not yet certain</h3>
            {#if view.learner_summary.uncertain.length}
              <ul>
                {#each view.learner_summary.uncertain as skill}
                  <li>{skill}</li>
                {/each}
              </ul>
            {:else}
              <p>Nothing unresolved right now.</p>
            {/if}
          </div>
        </section>

        <div class="content-mode">
          <span>Lesson content</span>
          <strong>
            {view.content_mode.effective === "llm_coaching"
              ? "Adaptive coaching"
              : "Reviewed lessons"}
          </strong>
          {#if view.content_mode.fallback_reason}
            <p>{view.content_mode.fallback_reason}</p>
          {/if}
        </div>
      </aside>

      <section class="conversation-panel" aria-label="Tutor conversation">
        <div class="conversation-heading">
          <div>
            <p class="eyebrow">Learning session</p>
            <h2>{phaseLabel(view.phase)}</h2>
          </div>
          <span class="revision">Update {view.revision}</span>
        </div>

        <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
        <div
          class="transcript"
          bind:this={transcriptElement}
          role="region"
          aria-label="Tutor conversation history"
          tabindex="0"
        >
          {#if view.transcript.length === 0}
            <div class="empty-transcript">
              <span class="spinner" aria-hidden="true"></span>
              <p>The tutor is preparing your first check-in.</p>
            </div>
          {/if}
          {#each view.transcript as entry (entry.id)}
            <article
              class:student-bubble={entry.role === "student"}
              class:tutor-bubble={entry.role === "tutor"}
              class:system-bubble={entry.role === "system"}
              class:hint-bubble={entry.kind === "hint"}
              class:assessment-bubble={["probe", "checkin", "capstone"].includes(entry.kind)}
              class="message"
            >
              {#if entry.role !== "student"}
                <span class="message-label">
                  {entry.kind === "capstone"
                    ? "Goal problem"
                    : entry.kind === "probe"
                      ? "Check-in"
                      : entry.kind === "checkin"
                        ? "Independent check"
                        : entry.kind === "hint"
                          ? "Hint"
                          : "Tutor"}
                </span>
              {/if}
              {#if entry.content_blocks?.length}
                {#each entry.content_blocks as block}
                  <div class={`content-block ${block.kind}`}>
                    {#if block.kind === "worked_example"}
                      <span class="kind-tag">Worked example</span>
                    {:else if block.kind === "remediation"}
                      <span class="kind-tag">Review</span>
                    {/if}
                    {#if block.text}<p>{block.text}</p>{/if}
                    {#if block.segments?.length}
                      <PromptSegments segments={block.segments} />
                    {/if}
                  </div>
                {/each}
              {:else if entry.prompt_segments?.length}
                <PromptSegments segments={entry.prompt_segments} />
              {:else if entry.text}
                <p>{entry.text}</p>
              {/if}
            </article>
            {#if entry.widget}
              <WidgetHost
                item={widgetItem(entry)}
                disabled={!widgetIsActive(entry) || busy}
                onAttempt={submitWidget}
                onTextFallback={useTextFallback}
                onError={(error) => (errorMessage = error.message)}
              />
            {/if}
          {/each}

          {#if view.pending?.prompt && !pendingPromptAppearsInTranscript()}
            <article class="message tutor-bubble assessment-bubble">
              <span class="message-label">
                {view.pending.kind === "capstone" ? "Goal problem" : "Your turn"}
              </span>
              {#if view.pending.prompt_segments?.length}
                <PromptSegments segments={view.pending.prompt_segments} />
              {:else}
                <p>{view.pending.prompt}</p>
              {/if}
            </article>
          {/if}

          {#if view.pending?.widget && !pendingWidgetAppearsInTranscript()}
            <WidgetHost
              item={{
                key: view.pending.key,
                widget: view.pending.widget,
                widget_state: view.pending.widget_state,
              }}
              disabled={busy}
              onAttempt={submitWidget}
              onTextFallback={useTextFallback}
              onError={(error) => (errorMessage = error.message)}
            />
          {/if}

          {#if view.terminal}
            <section class="completion-card" aria-labelledby="completion-heading">
              <span aria-hidden="true">✓</span>
              <div>
                <h3 id="completion-heading">
                  {view.phase === "done" ? "Learning path complete" : "Session ended"}
                </h3>
                <p>
                  The summary at left separates what you demonstrated from what
                  still needs more evidence.
                </p>
              </div>
            </section>
          {/if}
        </div>

        <div class="sr-only" aria-live="polite" aria-atomic="true">
          {latestTranscriptAnnouncement}
        </div>

        <div class="session-feedback" aria-live="assertive">
          {#if errorMessage}
            <div class="notice error-notice" role="alert">
              <span>{errorMessage}</span>
              <button type="button" class="notice-close" on:click={() => (errorMessage = "")}>
                <span class="sr-only">Dismiss error</span>×
              </button>
            </div>
          {:else if statusMessage}
            <div class="notice status-notice" role="status">{statusMessage}</div>
          {/if}
        </div>

        {#if !view.terminal && pendingAcceptsText(view.pending)}
          <form class="answer-composer" on:submit|preventDefault={submitAnswer}>
            {#if view.pending.input_mode === "choice" && view.pending.choice_options?.length}
              <fieldset class="choice-options" bind:this={choiceElement} disabled={busy}>
                <legend>Choose one answer</legend>
                {#each view.pending.choice_options as option}
                  <label>
                    <input type="radio" bind:group={answerText} value={option}>
                    <span>{option}</span>
                  </label>
                {/each}
              </fieldset>
            {:else}
              <label class="sr-only" for="answer">Your answer</label>
              <input
                id="answer"
                bind:this={answerElement}
                bind:value={answerText}
                placeholder={view.pending?.placeholder ?? "Type your answer"}
                maxlength="256"
                autocomplete="off"
                aria-describedby="answer-format-help"
                disabled={busy}
                on:keydown={handleAnswerKeydown}
              >
              <p id="answer-format-help" class="composer-help">
                Use * for multiplication and ^ for powers when needed.
              </p>
            {/if}
            <div class="composer-actions">
              {#if view.pending?.can_hint}
                <button
                  type="button"
                  class="secondary"
                  disabled={busy}
                  on:click={requestHint}
                >{view.pending.hint?.next_reveals_answer
                    ? "Reveal the answer and move to a new problem"
                    : "Get a hint"}</button>
              {/if}
              <button
                type="submit"
                class="primary-button"
                disabled={busy || !answerText.trim()}
              >{busy ? "Checking…" : "Check answer"}</button>
            </div>
          </form>
        {:else if !view.terminal && isWidgetPending(view.pending)}
          <div class="widget-footer" role="status">
            Complete the guided practice above, or choose its text alternative.
          </div>
        {/if}
      </section>
    </main>
  {/if}

  {#if !view && (errorMessage || statusMessage)}
    <div class="global-notice" aria-live="assertive">
      {#if errorMessage}
        <div class="notice error-notice" role="alert">{errorMessage}</div>
      {:else}
        <div class="notice status-notice" role="status">{statusMessage}</div>
      {/if}
    </div>
  {/if}

  <footer class="site-footer">
    <span>Accessible guided practice with keyboard alternatives.</span>
    <span>Expected answers never leave the server.</span>
  </footer>
</div>
