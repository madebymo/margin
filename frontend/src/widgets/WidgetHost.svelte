<script>
  import WidgetChrome from "./WidgetChrome.svelte";
  import { recipeFor } from "./recipes.js";

  export let item;
  export let disabled = false;
  export let onAttempt = async () => {};
  export let onTextFallback = async () => {};
  export let onError = () => {};

  const recipe = recipeFor(item.widget?.widget_type);
</script>

<WidgetChrome
  {item}
  {recipe}
  {disabled}
  {onAttempt}
  {onTextFallback}
  {onError}
/>
