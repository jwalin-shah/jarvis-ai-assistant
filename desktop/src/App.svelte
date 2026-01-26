<script lang="ts">
  import { onMount } from "svelte";
  import { listen } from "@tauri-apps/api/event";
  import Sidebar from "./lib/components/Sidebar.svelte";
  import ConversationList from "./lib/components/ConversationList.svelte";
  import MessageView from "./lib/components/MessageView.svelte";
  import Dashboard from "./lib/components/Dashboard.svelte";
  import HealthStatus from "./lib/components/HealthStatus.svelte";
  import ModelLoadingOverlay from "./lib/components/ModelLoadingOverlay.svelte";
  import { checkApiConnection, modelStatus, preloadModel, fetchModelStatus } from "./lib/stores/health";
  import { clearSelection } from "./lib/stores/conversations";

  let currentView: "messages" | "dashboard" | "health" = "messages";
  let showModelLoading = false;
  let preloadOnStart = false; // Could be loaded from settings

  // Check if we should show loading overlay (only during active loading)
  $: isModelLoading = $modelStatus.state === "loading";

  onMount(async () => {
    // Check API connection on start
    const connected = await checkApiConnection();

    if (connected) {
      // Fetch initial model status
      await fetchModelStatus();

      // Optionally preload model on start (based on settings)
      if (preloadOnStart && $modelStatus.state === "unloaded") {
        showModelLoading = true;
        await preloadModel();
      }
    }

    // Listen for navigation events from tray menu
    const unlisten = await listen<string>("navigate", (event) => {
      if (
        event.payload === "health" ||
        event.payload === "dashboard" ||
        event.payload === "messages"
      ) {
        currentView = event.payload;
        if (event.payload !== "messages") {
          clearSelection();
        }
      }
    });

    // Cleanup on unmount
    return () => {
      unlisten();
    };
  });

  function handleModelLoaded() {
    showModelLoading = false;
  }

  function handleModelError(error: string) {
    showModelLoading = false;
    console.error("Model loading error:", error);
  }
</script>

<main class="app">
  <Sidebar bind:currentView />

  {#if currentView === "dashboard"}
    <Dashboard on:navigate={(e) => currentView = e.detail} />
  {:else if currentView === "health"}
    <HealthStatus />
  {:else}
    <ConversationList />
    <MessageView />
  {/if}
</main>

{#if showModelLoading && isModelLoading}
  <ModelLoadingOverlay
    onLoaded={handleModelLoaded}
    onError={handleModelError}
  />
{/if}

<style>
  :global(*) {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  :global(body) {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      Helvetica, Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    overflow: hidden;
  }

  :global(:root) {
    --bg-primary: #1c1c1e;
    --bg-secondary: #2c2c2e;
    --bg-hover: #3a3a3c;
    --bg-active: #48484a;
    --bg-bubble-me: #0b93f6;
    --bg-bubble-other: #3a3a3c;
    --text-primary: #ffffff;
    --text-secondary: #8e8e93;
    --border-color: #38383a;
    --accent-color: #0b93f6;
    --group-color: #5856d6;
    --error-color: #ff5f57;
  }

  .app {
    display: flex;
    height: 100vh;
    width: 100vw;
  }
</style>
