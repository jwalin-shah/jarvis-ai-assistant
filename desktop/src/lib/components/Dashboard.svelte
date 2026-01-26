<script lang="ts">
  import { createEventDispatcher, onMount } from "svelte";
  import { healthStatus, fetchHealthStatus, modelStatus } from "../stores/health";
  import { conversations, fetchConversations } from "../stores/conversations";
  import LoadingSpinner from "./LoadingSpinner.svelte";

  const dispatch = createEventDispatcher<{ navigate: "messages" | "health" }>();

  let loading = true;

  onMount(async () => {
    await Promise.all([fetchHealthStatus(), fetchConversations()]);
    loading = false;
  });

  $: totalMessages = $conversations.reduce((sum, c) => sum + c.message_count, 0);
  $: groupChats = $conversations.filter((c) => c.is_group).length;
  $: directChats = $conversations.filter((c) => !c.is_group).length;
</script>

<div class="dashboard">
  <header>
    <h1>Dashboard</h1>
    <p class="subtitle">JARVIS AI Assistant Overview</p>
  </header>

  {#if loading}
    <div class="loading">
      <LoadingSpinner size="large" />
      <span>Loading dashboard...</span>
    </div>
  {:else}
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-icon">💬</div>
        <div class="stat-content">
          <div class="stat-value">{$conversations.length}</div>
          <div class="stat-label">Conversations</div>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-icon">📝</div>
        <div class="stat-content">
          <div class="stat-value">{totalMessages.toLocaleString()}</div>
          <div class="stat-label">Total Messages</div>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-icon">👤</div>
        <div class="stat-content">
          <div class="stat-value">{directChats}</div>
          <div class="stat-label">Direct Chats</div>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-icon">👥</div>
        <div class="stat-content">
          <div class="stat-value">{groupChats}</div>
          <div class="stat-label">Group Chats</div>
        </div>
      </div>
    </div>

    <div class="sections">
      <section class="model-section">
        <h2>AI Model Status</h2>
        <div class="model-status" class:loaded={$modelStatus.state === "loaded"}>
          <div class="model-indicator">
            {#if $modelStatus.state === "loaded"}
              <span class="status-icon success">✓</span>
            {:else if $modelStatus.state === "loading"}
              <LoadingSpinner size="small" />
            {:else if $modelStatus.state === "error"}
              <span class="status-icon error">✕</span>
            {:else}
              <span class="status-icon">○</span>
            {/if}
          </div>
          <div class="model-info">
            <div class="model-state">
              {#if $modelStatus.state === "loaded"}
                Model Ready
              {:else if $modelStatus.state === "loading"}
                Loading... {$modelStatus.progress ? Math.round($modelStatus.progress * 100) : 0}%
              {:else if $modelStatus.state === "error"}
                Error: {$modelStatus.error || "Unknown"}
              {:else}
                Model Not Loaded
              {/if}
            </div>
            {#if $modelStatus.memory_usage_mb}
              <div class="model-memory">
                Using {$modelStatus.memory_usage_mb.toFixed(0)} MB
              </div>
            {/if}
            {#if $modelStatus.load_time_seconds}
              <div class="model-load-time">
                Loaded in {$modelStatus.load_time_seconds.toFixed(1)}s
              </div>
            {/if}
          </div>
        </div>
      </section>

      <section class="health-section">
        <h2>System Health</h2>
        {#if $healthStatus}
          <div class="health-grid">
            <div class="health-item" class:ok={$healthStatus.imessage_access}>
              <span class="health-icon">{$healthStatus.imessage_access ? "✓" : "✕"}</span>
              <span>iMessage Access</span>
            </div>
            <div class="health-item" class:ok={$healthStatus.status === "healthy"}>
              <span class="health-icon">{$healthStatus.status === "healthy" ? "✓" : "!"}</span>
              <span>System Status: {$healthStatus.status}</span>
            </div>
            <div class="health-item info">
              <span class="health-icon">🧠</span>
              <span>Memory Mode: {$healthStatus.memory_mode}</span>
            </div>
            <div class="health-item info">
              <span class="health-icon">💾</span>
              <span>Available: {$healthStatus.memory_available_gb.toFixed(1)} GB</span>
            </div>
          </div>
          <button class="view-health-btn" on:click={() => dispatch("navigate", "health")}>
            View Full Health Report →
          </button>
        {:else}
          <p class="no-health">Unable to fetch health status</p>
        {/if}
      </section>

      <section class="quick-actions">
        <h2>Quick Actions</h2>
        <div class="actions-grid">
          <button class="action-btn" on:click={() => dispatch("navigate", "messages")}>
            <span class="action-icon">💬</span>
            <span>View Messages</span>
          </button>
          <button class="action-btn" on:click={() => dispatch("navigate", "health")}>
            <span class="action-icon">🔧</span>
            <span>System Health</span>
          </button>
        </div>
      </section>
    </div>
  {/if}
</div>

<style>
  .dashboard {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
    background: var(--bg-primary);
  }

  header {
    margin-bottom: 24px;
  }

  h1 {
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
  }

  .subtitle {
    font-size: 14px;
    color: var(--text-secondary);
    margin: 4px 0 0 0;
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

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }

  .stat-card {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .stat-icon {
    font-size: 32px;
  }

  .stat-value {
    font-size: 24px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .stat-label {
    font-size: 13px;
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

  .model-status {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 12px 16px;
    background: var(--bg-primary);
    border-radius: 8px;
  }

  .model-indicator {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .status-icon {
    font-size: 20px;
    color: var(--text-secondary);
  }

  .status-icon.success {
    color: #34c759;
  }

  .status-icon.error {
    color: var(--error-color);
  }

  .model-state {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .model-memory,
  .model-load-time {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .health-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }

  .health-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: var(--bg-primary);
    border-radius: 8px;
    font-size: 13px;
    color: var(--text-secondary);
  }

  .health-item.ok {
    color: #34c759;
  }

  .health-icon {
    font-size: 14px;
  }

  .view-health-btn {
    background: transparent;
    border: 1px solid var(--border-color);
    color: var(--accent-color);
    padding: 8px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.15s ease;
  }

  .view-health-btn:hover {
    background: var(--bg-hover);
  }

  .no-health {
    color: var(--text-secondary);
    font-size: 14px;
    margin: 0;
  }

  .actions-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
  }

  .action-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 16px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
    color: var(--text-primary);
    font-size: 13px;
  }

  .action-btn:hover {
    background: var(--bg-hover);
    border-color: var(--accent-color);
  }

  .action-icon {
    font-size: 24px;
  }
</style>
