<script lang="ts">
  import "../app.css";
  import { onMount, onDestroy } from "svelte";
  import { startPolling, stopPolling } from "$lib/stores/app";

  onMount(() => {
    startPolling(30000);
  });

  onDestroy(() => {
    stopPolling();
  });
</script>

<slot />
