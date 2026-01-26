<script lang="ts">
  import { apiConnected, healthStatus } from "../stores/health";
  import { modelStatus } from "../stores/health";

  export let currentView: "messages" | "dashboard" | "health" = "messages";

  function setView(view: "messages" | "dashboard" | "health") {
    currentView = view;
  }
</script>

<nav class="sidebar">
  <div class="logo">
    <span class="logo-icon">J</span>
    <span class="logo-text">JARVIS</span>
  </div>

  <div class="nav-items">
    <button
      class="nav-item"
      class:active={currentView === "messages"}
      on:click={() => setView("messages")}
    >
      <span class="icon">💬</span>
      <span class="label">Messages</span>
    </button>

    <button
      class="nav-item"
      class:active={currentView === "dashboard"}
      on:click={() => setView("dashboard")}
    >
      <span class="icon">📊</span>
      <span class="label">Dashboard</span>
    </button>

    <button
      class="nav-item"
      class:active={currentView === "health"}
      on:click={() => setView("health")}
    >
      <span class="icon">🔧</span>
      <span class="label">Health</span>
    </button>
  </div>

  <div class="status-section">
    <div class="status-item" class:connected={$apiConnected} class:disconnected={!$apiConnected}>
      <span class="status-dot"></span>
      <span class="status-text">{$apiConnected ? "API Connected" : "API Disconnected"}</span>
    </div>

    <div class="status-item" class:loaded={$modelStatus.state === "loaded"}>
      <span class="status-dot"></span>
      <span class="status-text">
        {#if $modelStatus.state === "loaded"}
          Model Ready
        {:else if $modelStatus.state === "loading"}
          Loading {$modelStatus.progress ? Math.round($modelStatus.progress * 100) : 0}%
        {:else if $modelStatus.state === "error"}
          Model Error
        {:else}
          Model Unloaded
        {/if}
      </span>
    </div>

    {#if $healthStatus}
      <div class="status-item memory">
        <span class="status-text">{$healthStatus.memory_mode} Mode</span>
      </div>
    {/if}
  </div>
</nav>

<style>
  .sidebar {
    width: 200px;
    min-width: 200px;
    background: var(--bg-secondary);
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--border-color);
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
  }

  .logo-icon {
    width: 32px;
    height: 32px;
    background: var(--accent-color);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 18px;
  }

  .logo-text {
    font-weight: 600;
    font-size: 16px;
    color: var(--text-primary);
  }

  .nav-items {
    flex: 1;
    padding: 8px;
  }

  .nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    padding: 10px 12px;
    background: transparent;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    color: var(--text-secondary);
    font-size: 14px;
    text-align: left;
    transition: all 0.15s ease;
  }

  .nav-item:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .nav-item.active {
    background: var(--bg-active);
    color: var(--text-primary);
  }

  .icon {
    font-size: 16px;
  }

  .status-section {
    padding: 12px;
    border-top: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .status-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-secondary);
  }

  .status-item.connected .status-dot {
    background: #34c759;
  }

  .status-item.disconnected .status-dot {
    background: var(--error-color);
  }

  .status-item.loaded .status-dot {
    background: #34c759;
  }

  .memory {
    padding-left: 12px;
  }
</style>
