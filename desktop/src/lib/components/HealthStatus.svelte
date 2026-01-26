<script lang="ts">
  import { onMount } from "svelte";
  import {
    healthStatus,
    modelStatus,
    fetchHealthStatus,
    fetchModelStatus,
    preloadModel,
    unloadModel,
  } from "../stores/health";
  import LoadingSpinner from "./LoadingSpinner.svelte";

  let loading = true;
  let isPreloading = false;
  let isUnloading = false;

  onMount(async () => {
    await Promise.all([fetchHealthStatus(), fetchModelStatus()]);
    loading = false;
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

  async function refreshStatus() {
    await Promise.all([fetchHealthStatus(), fetchModelStatus()]);
  }
</script>

<div class="health-status">
  <header>
    <h1>System Health</h1>
    <button class="refresh-btn" on:click={refreshStatus}>
      🔄 Refresh
    </button>
  </header>

  {#if loading}
    <div class="loading">
      <LoadingSpinner size="large" />
      <span>Loading health status...</span>
    </div>
  {:else}
    <div class="sections">
      <!-- Model Status Section -->
      <section class="model-section">
        <h2>AI Model</h2>
        <div class="model-card">
          <div class="model-header">
            <div class="model-status-indicator" class:loaded={$modelStatus.state === "loaded"}>
              {#if $modelStatus.state === "loaded"}
                <span class="status-badge success">Ready</span>
              {:else if $modelStatus.state === "loading"}
                <span class="status-badge loading">Loading</span>
              {:else if $modelStatus.state === "error"}
                <span class="status-badge error">Error</span>
              {:else}
                <span class="status-badge">Unloaded</span>
              {/if}
            </div>
          </div>

          {#if $modelStatus.state === "loading"}
            <div class="loading-progress">
              <div class="progress-bar">
                <div
                  class="progress"
                  style="width: {($modelStatus.progress || 0) * 100}%"
                ></div>
              </div>
              <div class="progress-text">
                {$modelStatus.message || "Loading..."} ({Math.round(($modelStatus.progress || 0) * 100)}%)
              </div>
            </div>
          {/if}

          {#if $modelStatus.state === "error"}
            <div class="error-message">
              {$modelStatus.error || "Unknown error occurred"}
            </div>
          {/if}

          <div class="model-details">
            {#if $modelStatus.memory_usage_mb}
              <div class="detail-row">
                <span class="label">Memory Usage:</span>
                <span class="value">{$modelStatus.memory_usage_mb.toFixed(0)} MB</span>
              </div>
            {/if}
            {#if $modelStatus.load_time_seconds}
              <div class="detail-row">
                <span class="label">Load Time:</span>
                <span class="value">{$modelStatus.load_time_seconds.toFixed(2)}s</span>
              </div>
            {/if}
          </div>

          <div class="model-actions">
            {#if $modelStatus.state === "loaded"}
              <button
                class="action-btn danger"
                on:click={handleUnload}
                disabled={isUnloading}
              >
                {#if isUnloading}
                  <LoadingSpinner size="small" />
                {:else}
                  Unload Model
                {/if}
              </button>
            {:else if $modelStatus.state === "loading"}
              <button class="action-btn" disabled>
                <LoadingSpinner size="small" />
                Loading... {Math.round(($modelStatus.progress || 0) * 100)}%
              </button>
            {:else}
              <button
                class="action-btn primary"
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

          <p class="model-hint">
            {#if $modelStatus.state === "loaded"}
              Model is ready for AI-powered suggestions
            {:else}
              Preload the model for faster response times (10-15s on first load)
            {/if}
          </p>
        </div>
      </section>

      <!-- System Health Section -->
      {#if $healthStatus}
        <section class="system-section">
          <h2>System Status</h2>
          <div class="status-grid">
            <div class="status-card" class:healthy={$healthStatus.status === "healthy"}>
              <div class="status-icon">
                {#if $healthStatus.status === "healthy"}
                  ✓
                {:else if $healthStatus.status === "degraded"}
                  !
                {:else}
                  ✕
                {/if}
              </div>
              <div class="status-info">
                <div class="status-title">Overall Status</div>
                <div class="status-value">{$healthStatus.status}</div>
              </div>
            </div>

            <div class="status-card" class:healthy={$healthStatus.imessage_access}>
              <div class="status-icon">
                {$healthStatus.imessage_access ? "✓" : "✕"}
              </div>
              <div class="status-info">
                <div class="status-title">iMessage Access</div>
                <div class="status-value">
                  {$healthStatus.imessage_access ? "Granted" : "Denied"}
                </div>
              </div>
            </div>

            <div class="status-card info">
              <div class="status-icon">🧠</div>
              <div class="status-info">
                <div class="status-title">Memory Mode</div>
                <div class="status-value">{$healthStatus.memory_mode}</div>
              </div>
            </div>

            <div class="status-card info">
              <div class="status-icon">💾</div>
              <div class="status-info">
                <div class="status-title">Available Memory</div>
                <div class="status-value">{$healthStatus.memory_available_gb.toFixed(1)} GB</div>
              </div>
            </div>
          </div>
        </section>

        <section class="memory-section">
          <h2>Memory Details</h2>
          <div class="memory-grid">
            <div class="memory-item">
              <span class="label">System Available:</span>
              <span class="value">{$healthStatus.memory_available_gb.toFixed(2)} GB</span>
            </div>
            <div class="memory-item">
              <span class="label">System Used:</span>
              <span class="value">{$healthStatus.memory_used_gb.toFixed(2)} GB</span>
            </div>
            <div class="memory-item">
              <span class="label">JARVIS RSS:</span>
              <span class="value">{$healthStatus.jarvis_rss_mb.toFixed(1)} MB</span>
            </div>
            <div class="memory-item">
              <span class="label">JARVIS VMS:</span>
              <span class="value">{$healthStatus.jarvis_vms_mb.toFixed(1)} MB</span>
            </div>
          </div>
        </section>

        {#if $healthStatus.details && Object.keys($healthStatus.details).length > 0}
          <section class="details-section">
            <h2>Issues</h2>
            <div class="issues-list">
              {#each Object.entries($healthStatus.details) as [key, value]}
                <div class="issue-item">
                  <span class="issue-key">{key}:</span>
                  <span class="issue-value">{value}</span>
                </div>
              {/each}
            </div>
          </section>
        {/if}
      {:else}
        <section class="error-section">
          <p>Unable to fetch health status. Make sure the API is running.</p>
        </section>
      {/if}
    </div>
  {/if}
</div>

<style>
  .health-status {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
    background: var(--bg-primary);
  }

  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
  }

  h1 {
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
  }

  .refresh-btn {
    padding: 8px 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    cursor: pointer;
    color: var(--text-primary);
    font-size: 13px;
    transition: all 0.15s ease;
  }

  .refresh-btn:hover {
    background: var(--bg-hover);
  }

  .loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    height: 300px;
    color: var(--text-secondary);
  }

  .sections {
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  section {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 20px;
  }

  h2 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 16px 0;
  }

  .model-card {
    background: var(--bg-primary);
    border-radius: 8px;
    padding: 16px;
  }

  .model-header {
    margin-bottom: 12px;
  }

  .status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
    background: var(--bg-secondary);
    color: var(--text-secondary);
  }

  .status-badge.success {
    background: rgba(52, 199, 89, 0.2);
    color: #34c759;
  }

  .status-badge.loading {
    background: rgba(11, 147, 246, 0.2);
    color: var(--accent-color);
  }

  .status-badge.error {
    background: rgba(255, 95, 87, 0.2);
    color: var(--error-color);
  }

  .loading-progress {
    margin: 16px 0;
  }

  .progress-bar {
    width: 100%;
    height: 6px;
    background: var(--bg-secondary);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 8px;
  }

  .progress {
    height: 100%;
    background: var(--accent-color);
    transition: width 0.3s ease;
  }

  .progress-text {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .error-message {
    padding: 12px;
    background: rgba(255, 95, 87, 0.1);
    border-radius: 8px;
    color: var(--error-color);
    font-size: 13px;
    margin: 12px 0;
  }

  .model-details {
    margin: 12px 0;
  }

  .detail-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    padding: 4px 0;
  }

  .detail-row .label {
    color: var(--text-secondary);
  }

  .detail-row .value {
    color: var(--text-primary);
    font-weight: 500;
  }

  .model-actions {
    margin-top: 16px;
  }

  .action-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    background: var(--bg-secondary);
    color: var(--text-primary);
    font-size: 14px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .action-btn:hover:not(:disabled) {
    background: var(--bg-hover);
  }

  .action-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .action-btn.primary {
    background: var(--accent-color);
    border-color: var(--accent-color);
    color: white;
  }

  .action-btn.primary:hover:not(:disabled) {
    opacity: 0.9;
  }

  .action-btn.danger {
    border-color: var(--error-color);
    color: var(--error-color);
  }

  .action-btn.danger:hover:not(:disabled) {
    background: rgba(255, 95, 87, 0.1);
  }

  .model-hint {
    font-size: 12px;
    color: var(--text-secondary);
    margin: 12px 0 0 0;
  }

  .status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
  }

  .status-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    background: var(--bg-primary);
    border-radius: 8px;
    border-left: 3px solid var(--border-color);
  }

  .status-card.healthy {
    border-left-color: #34c759;
  }

  .status-card.info {
    border-left-color: var(--accent-color);
  }

  .status-card .status-icon {
    font-size: 20px;
  }

  .status-title {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .status-value {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
    text-transform: capitalize;
  }

  .memory-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }

  .memory-item {
    display: flex;
    justify-content: space-between;
    padding: 12px;
    background: var(--bg-primary);
    border-radius: 8px;
    font-size: 13px;
  }

  .memory-item .label {
    color: var(--text-secondary);
  }

  .memory-item .value {
    color: var(--text-primary);
    font-weight: 500;
    font-variant-numeric: tabular-nums;
  }

  .issues-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .issue-item {
    padding: 12px;
    background: rgba(255, 95, 87, 0.1);
    border-radius: 8px;
    font-size: 13px;
  }

  .issue-key {
    color: var(--error-color);
    font-weight: 500;
  }

  .issue-value {
    color: var(--text-primary);
    margin-left: 8px;
  }

  .error-section {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 20px;
    color: var(--text-secondary);
  }
</style>
