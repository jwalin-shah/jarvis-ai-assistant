<script lang="ts">
  import { onMount } from "svelte";
  import {
    modelStatus,
    fetchModelStatus,
    preloadModel,
    unloadModel,
  } from "../stores/health";
  import LoadingSpinner from "./LoadingSpinner.svelte";

  export let preloadOnStart: boolean = false;
  export let onPreloadOnStartChange: (value: boolean) => void = () => {};

  let isPreloading = false;
  let isUnloading = false;

  onMount(() => {
    fetchModelStatus();
  });

  async function handlePreload() {
    isPreloading = true;
    await preloadModel();
    // Poll until loaded
    const pollInterval = setInterval(async () => {
      const status = await fetchModelStatus();
      if (status && (status.state === "loaded" || status.state === "error")) {
        clearInterval(pollInterval);
        isPreloading = false;
      }
    }, 500);
  }

  async function handleUnload() {
    isUnloading = true;
    await unloadModel();
    isUnloading = false;
  }

  function togglePreloadOnStart() {
    preloadOnStart = !preloadOnStart;
    onPreloadOnStartChange(preloadOnStart);
  }
</script>

<div class="settings">
  <h2>Settings</h2>

  <section class="setting-section">
    <h3>AI Model</h3>

    <div class="setting-item">
      <div class="setting-info">
        <span class="setting-label">Model Status</span>
        <span class="setting-description">
          {#if $modelStatus.state === "loaded"}
            Model is loaded and ready
          {:else if $modelStatus.state === "loading"}
            Model is loading... {Math.round(($modelStatus.progress || 0) * 100)}%
          {:else if $modelStatus.state === "error"}
            Error: {$modelStatus.error || "Unknown error"}
          {:else}
            Model is not loaded
          {/if}
        </span>
      </div>
      <div class="setting-control">
        {#if $modelStatus.state === "loaded"}
          <button
            class="btn btn-danger"
            on:click={handleUnload}
            disabled={isUnloading}
          >
            {#if isUnloading}
              <LoadingSpinner size="small" />
            {:else}
              Unload
            {/if}
          </button>
        {:else if $modelStatus.state === "loading"}
          <button class="btn" disabled>
            <LoadingSpinner size="small" />
            {Math.round(($modelStatus.progress || 0) * 100)}%
          </button>
        {:else}
          <button
            class="btn btn-primary"
            on:click={handlePreload}
            disabled={isPreloading}
          >
            {#if isPreloading}
              <LoadingSpinner size="small" />
            {:else}
              Preload Model
            {/if}
          </button>
        {/if}
      </div>
    </div>

    <div class="setting-item">
      <div class="setting-info">
        <span class="setting-label">Preload on Start</span>
        <span class="setting-description">
          Automatically load the AI model when the app starts
        </span>
      </div>
      <div class="setting-control">
        <button
          class="toggle"
          class:active={preloadOnStart}
          on:click={togglePreloadOnStart}
          role="switch"
          aria-checked={preloadOnStart}
        >
          <span class="toggle-thumb"></span>
        </button>
      </div>
    </div>

    {#if $modelStatus.state === "loaded"}
      <div class="model-stats">
        {#if $modelStatus.memory_usage_mb}
          <div class="stat">
            <span class="stat-label">Memory:</span>
            <span class="stat-value">{$modelStatus.memory_usage_mb.toFixed(0)} MB</span>
          </div>
        {/if}
        {#if $modelStatus.load_time_seconds}
          <div class="stat">
            <span class="stat-label">Load time:</span>
            <span class="stat-value">{$modelStatus.load_time_seconds.toFixed(2)}s</span>
          </div>
        {/if}
      </div>
    {/if}
  </section>
</div>

<style>
  .settings {
    padding: 20px;
  }

  h2 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 20px 0;
  }

  .setting-section {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
  }

  h3 {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 16px 0;
  }

  .setting-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid var(--border-color);
  }

  .setting-item:last-child {
    border-bottom: none;
  }

  .setting-info {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .setting-label {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .setting-description {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .setting-control {
    display: flex;
    align-items: center;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
    border: 1px solid var(--border-color);
    background: var(--bg-primary);
    color: var(--text-primary);
  }

  .btn:hover:not(:disabled) {
    background: var(--bg-hover);
  }

  .btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .btn-primary {
    background: var(--accent-color);
    border-color: var(--accent-color);
    color: white;
  }

  .btn-primary:hover:not(:disabled) {
    opacity: 0.9;
  }

  .btn-danger {
    border-color: var(--error-color);
    color: var(--error-color);
  }

  .btn-danger:hover:not(:disabled) {
    background: rgba(255, 95, 87, 0.1);
  }

  .toggle {
    position: relative;
    width: 44px;
    height: 24px;
    border-radius: 12px;
    background: var(--bg-active);
    border: none;
    cursor: pointer;
    transition: background 0.2s ease;
    padding: 0;
  }

  .toggle.active {
    background: var(--accent-color);
  }

  .toggle-thumb {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: white;
    transition: transform 0.2s ease;
  }

  .toggle.active .toggle-thumb {
    transform: translateX(20px);
  }

  .model-stats {
    display: flex;
    gap: 16px;
    margin-top: 12px;
    padding: 12px;
    background: var(--bg-primary);
    border-radius: 8px;
  }

  .stat {
    display: flex;
    gap: 6px;
    font-size: 12px;
  }

  .stat-label {
    color: var(--text-secondary);
  }

  .stat-value {
    color: var(--text-primary);
    font-weight: 500;
  }
</style>
