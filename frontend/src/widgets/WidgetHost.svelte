<script>
  import WidgetChrome from "./WidgetChrome.svelte";
  import { recipeFor } from "./recipes.js";

  export let item;
  export let sessionId;
  export let onError = () => {};

  const recipe = recipeFor(item.widget?.widget_type);
</script>

<WidgetChrome {item} {sessionId} {recipe} {onError} />
