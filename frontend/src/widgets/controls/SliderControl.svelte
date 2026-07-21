<script>
  export let config;
  export let state;
  export let disabled = false;
  export let onChange = () => {};

  $: params = config.params ?? config.presentation ?? config;
  $: minimum = params.min ?? params.minimum;
  $: maximum = params.max ?? params.maximum;
  $: label = params.value_label ?? config.prompt ?? "Slider value";
  $: result = params.result_template
    ? params.result_template.replace("{value}", String(state.value))
    : `${label}: ${state.value}`;

  function handleInput(event) {
    onChange({
      ...state,
      value: Number(event.currentTarget.value),
    });
  }
</script>

<div class="widget-controls slider-controls">
  <label>
    <span>{label}</span>
    <input
      type="range"
      min={minimum}
      max={maximum}
      step={params.step}
      value={state.value}
      {disabled}
      aria-valuetext={result}
      on:input={handleInput}
    >
  </label>
  <output class="slider-value" aria-live="polite">{result}</output>
</div>
