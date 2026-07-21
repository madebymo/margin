<script>
  import { onDestroy } from "svelte";

  import "../scene/TutorSceneElement.js";
  import { widgetCapability } from "./capabilities.js";

  export let item;
  export let recipe;
  export let disabled = false;
  export let onAttempt = async () => {};
  export let onTextFallback = async () => {};
  export let onError = () => {};

  const config = item.widget;
  const capability = widgetCapability(config.widget_type);
  const terminalStatuses = new Set([
    "correct",
    "solved",
    "remediated",
    "text_fallback",
  ]);
  let recipeFailure = "";
  let state = null;
  if (recipe) {
    try {
      state = recipe.init(
        config,
        item.widget_state ?? item.widget_current_state ?? item.state ?? null,
      );
    } catch {
      recipeFailure =
        "This guided visual could not be initialized. Use its text alternative.";
    }
  }
  $: activeRecipe = recipeFailure ? null : recipe;
  let completed = terminalStatuses.has(item.widget_status);
  let checking = false;
  let feedback = item.feedback ?? "";
  let correct = completed;
  let sceneElement;
  let activeController = null;
  let destroyed = false;

  onDestroy(() => {
    destroyed = true;
    activeController?.abort();
    activeController = null;
  });

  const fallbackStatus = (message) =>
    message || "Graph preview unavailable. Use the slider below.";

  function normalizeScene(activeRecipe, widget, widgetState) {
    if (!activeRecipe) {
      return null;
    }
    try {
      const result = activeRecipe.normalize(widget, widgetState);
      if (result == null) {
        return null;
      }
      const scene = result.scene ?? null;
      const status =
        typeof result.status === "string"
          ? result.status
          : scene
            ? "Rich visual ready."
            : fallbackStatus();
      return { scene, status };
    } catch {
      return {
        scene: null,
        status: fallbackStatus(),
      };
    }
  }

  function updateState(nextState) {
    state = nextState;
  }

  async function check() {
    if (!activeRecipe || disabled || completed || checking) {
      return;
    }
    checking = true;
    const controller = new AbortController();
    activeController = controller;
    try {
      const data = await onAttempt({
        key: item.key,
        response: activeRecipe.responseFrom(state),
        signal: controller.signal,
      });
      if (destroyed) return;
      const result = data?.action_result ?? data?.result ?? data?.last_action ?? data ?? {};
      feedback = result.authoritative
        ? ""
        : result.message ??
          result.feedback ??
          (result.correct ? "That works." : "Try adjusting your response.");
      correct = Boolean(result.correct);
      if (correct) {
        completed = true;
      }
    } catch (error) {
      if (!destroyed && error.name !== "AbortError") {
        onError(error);
      }
    } finally {
      if (activeController === controller) {
        activeController = null;
      }
      if (!destroyed) {
        checking = false;
      }
    }
  }

  async function useFallback() {
    if (disabled || checking) return;
    checking = true;
    try {
      await onTextFallback({ key: item.key });
    } catch (error) {
      onError(error);
    } finally {
      checking = false;
    }
  }

  $: sceneState = normalizeScene(activeRecipe, config, state);
  $: if (terminalStatuses.has(item.widget_status)) {
    completed = true;
  }
  $: archiveLabel =
    item.widget_status === "remediated"
      ? "remediated practice"
      : item.widget_status === "text_fallback"
        ? "text alternative selected"
        : item.widget_status === "solved" || completed
          ? "solved practice"
          : disabled
            ? "archived practice"
            : "guided practice";
  $: if (sceneElement && sceneState) {
    sceneElement.sceneState = sceneState;
  }
</script>

<section
  class="widgetbox"
  class:archived={disabled || completed}
  aria-label={archiveLabel}
>
  <span class="kind-tag">{archiveLabel}</span>
  <div class="widget-prompt">{config.prompt || ""}</div>

  {#if activeRecipe}
    {#if sceneState}
      <div class="widget-scene">
        <tutor-scene
          bind:this={sceneElement}
          aria-label="Interactive graph for this slider"
        ></tutor-scene>
      </div>
    {/if}

    <svelte:component
      this={activeRecipe.control}
      {config}
      {state}
      disabled={disabled || completed || checking}
      onChange={updateState}
    />

    {#if !disabled && !completed}
      <div class="widget-actions">
        <button type="button" disabled={checking} on:click={check}>
          {checking ? "Checking…" : "Check practice"}
        </button>
        <button
          type="button"
          class="secondary"
          disabled={checking}
          on:click={useFallback}
        >Use text alternative</button>
      </div>
    {/if}
    <div
      class:ok={feedback && correct}
      class:no={feedback && !correct}
      class="feedback"
      aria-live="polite"
    >{feedback}</div>
  {:else}
    <div class="unsupported-widget" role="status">
      <strong>Accessible alternative needed</strong>
      <span>{recipeFailure || capability.reason || "This visual is unavailable."}</span>
    </div>
    {#if !disabled}
      <button
        type="button"
        class="secondary"
        disabled={checking}
        on:click={useFallback}
      >Continue with a text alternative</button>
    {/if}
  {/if}
</section>
