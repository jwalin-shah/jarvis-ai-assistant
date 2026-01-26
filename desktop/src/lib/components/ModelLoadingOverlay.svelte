<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import LoadingSpinner from "./LoadingSpinner.svelte";

  export let onLoaded: () => void = () => {};
  export let onError: (error: string) => void = () => {};
  export let apiUrl: string = "http://localhost:8742";

  interface ModelStatus {
    state: "unloaded" | "loading" | "loaded" | "error";
    progress: number | null;
    message: string | null;
    memory_usage_mb: number | null;
    load_time_seconds: number | null;
    error: string | null;
  }

  let status: ModelStatus = {
    state: "loading",
    progress: 0,
    message: "Connecting...",
    memory_usage_mb: null,
    load_time_seconds: null,
    error: null,
  };

  let eventSource: EventSource | null = null;

  onMount(() => {
    // Connect to SSE stream for real-time updates
    eventSource = new EventSource(`${apiUrl}/model-status/stream`);

    eventSource.onmessage = (event) => {
      try {
        status = JSON.parse(event.data);

        if (status.state === "loaded") {
          eventSource?.close();
          onLoaded();
        } else if (status.state === "error") {
          eventSource?.close();
          onError(status.error || "Unknown error");
        }
      } catch (e) {
        console.error("Failed to parse model status:", e);
      }
    };

    eventSource.onerror = () => {
      // Connection error - try to get status via regular endpoint
      eventSource?.close();
      fetchStatus();
    };
  });

  onDestroy(() => {
    eventSource?.close();
  });

  async function fetchStatus() {
    try {
      const response = await fetch(`${apiUrl}/model-status`);
      if (response.ok) {
        status = await response.json();
        if (status.state === "loaded") {
          onLoaded();
        } else if (status.state === "error") {
          onError(status.error || "Unknown error");
        }
      }
    } catch (e) {
      status = {
        ...status,
        state: "error",
        message: "Failed to connect to API",
        error: "Connection failed",
      };
      onError("Failed to connect to API");
    }
  }

  $: progressPercent = status.progress ? Math.round(status.progress * 100) : 0;
</script>

<div class="loading-overlay">
  <div class="loading-content">
    <LoadingSpinner size="large" />

    <h3>Loading AI Model</h3>

    <p class="status-message">{status.message || "Please wait..."}</p>

    {#if status.state === "loading" && status.progress !== null}
      <div class="progress-bar">
        <div class="progress" style="width: {progressPercent}%"></div>
      </div>
      <p class="progress-text">{progressPercent}%</p>
    {/if}

    {#if status.state === "error"}
      <p class="error-message">{status.error}</p>
    {/if}

    <p class="hint">This may take 10-15 seconds on first run</p>

    {#if status.load_time_seconds}
      <p class="load-time">
        Loaded in {status.load_time_seconds.toFixed(1)}s
      </p>
    {/if}
  </div>
</div>

<style>
  .loading-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.85);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    backdrop-filter: blur(4px);
  }

  .loading-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
    padding: 32px;
    background: var(--bg-secondary);
    border-radius: 16px;
    max-width: 320px;
    text-align: center;
  }

  h3 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  .status-message {
    font-size: 14px;
    color: var(--text-secondary);
    margin: 0;
  }

  .progress-bar {
    width: 200px;
    height: 4px;
    background: var(--bg-primary);
    border-radius: 2px;
    overflow: hidden;
  }

  .progress {
    height: 100%;
    background: var(--accent-color);
    transition: width 0.3s ease;
  }

  .progress-text {
    font-size: 12px;
    color: var(--text-secondary);
    margin: 0;
    font-variant-numeric: tabular-nums;
  }

  .error-message {
    font-size: 13px;
    color: var(--error-color);
    margin: 0;
    padding: 8px 12px;
    background: rgba(255, 95, 87, 0.1);
    border-radius: 6px;
  }

  .hint {
    font-size: 12px;
    color: var(--text-secondary);
    opacity: 0.7;
    margin: 0;
  }

  .load-time {
    font-size: 11px;
    color: var(--text-secondary);
    opacity: 0.5;
    margin: 0;
  }
</style>
