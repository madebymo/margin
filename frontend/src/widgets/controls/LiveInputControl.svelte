<script>
  export let config;
  export let state;
  export let disabled = false;
  export let onChange = () => {};

  function handleInput(event) {
    onChange({
      ...state,
      text: event.currentTarget.value,
    });
  }
</script>

<div class="widget-controls">
  <input
    type="text"
    value={state.text}
    placeholder="your answer…"
    aria-label={config.prompt || "Your answer"}
    {disabled}
    on:input={handleInput}
  >
</div>
