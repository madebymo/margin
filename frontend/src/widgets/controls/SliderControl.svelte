<script>
  export let config;
  export let state;
  export let disabled = false;
  export let onChange = () => {};

  function handleInput(event) {
    onChange({
      ...state,
      value: Number(event.currentTarget.value),
    });
  }
</script>

<div class="widget-controls">
  <input
    type="range"
    min={config.params.min}
    max={config.params.max}
    step={config.params.step}
    value={state.value}
    {disabled}
    aria-label={config.prompt || "Slider value"}
    on:input={handleInput}
  >
  <span class="slider-value">{state.value}</span>
</div>
