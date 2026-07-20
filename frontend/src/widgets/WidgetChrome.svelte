<script>
  import { onDestroy } from "svelte";

  import { api } from "../api.js";
  import "../scene/TutorSceneElement.js";

  export let item;
  export let sessionId;
  export let recipe;
  export let onError = () => {};

  const config = item.widget;
  let state = recipe ? recipe.init(config) : null;
  let disabled = false;
  let checking = false;
  let feedback = "";
  let correct = false;
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
    if (!recipe || disabled || checking) {
      return;
    }
    checking = true;
    const controller = new AbortController();
    activeController = controller;
    try {
      const data = await api(
        `/sessions/${sessionId}/widget`,
        {
          key: item.key,
          response: recipe.responseFrom(state),
        },
        { signal: controller.signal },
      );
      if (destroyed) return;
      feedback = data.message;
      correct = Boolean(data.correct);
      if (correct) {
        disabled = true;
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

  $: sceneState = normalizeScene(recipe, config, state);
  $: if (sceneElement && sceneState) {
    sceneElement.sceneState = sceneState;
  }
</script>

<div class="widgetbox">
  <span class="kind-tag">interactive: {config.widget_type}</span>
  <div>{config.prompt || ""}</div>

  {#if recipe}
    {#if sceneState}
      <div class="widget-scene">
        <tutor-scene
          bind:this={sceneElement}
          aria-label="Interactive graph for this slider"
        ></tutor-scene>
      </div>
    {/if}

    <svelte:component
      this={recipe.control}
      {config}
      {state}
      disabled={disabled || checking}
      onChange={updateState}
    />

    <button type="button" disabled={disabled || checking} on:click={check}>
      {checking ? "Checking…" : "Check"}
    </button>
    <div
      class:ok={feedback && correct}
      class:no={feedback && !correct}
      class="feedback"
      aria-live="polite"
    >{feedback}</div>
  {:else}
    <div class="unsupported-widget" role="status">
      This interaction type is not supported yet.
    </div>
  {/if}
</div>
