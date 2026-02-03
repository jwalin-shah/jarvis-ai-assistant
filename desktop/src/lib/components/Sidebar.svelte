<script lang="ts">
  import { healthStore } from "../stores/health";

  type ViewType = "messages" | "dashboard" | "health" | "settings" | "templates";

  let { currentView = $bindable<ViewType>("messages"), collapsed = $bindable(false) } = $props<{
    currentView?: ViewType;
    collapsed?: boolean;
  }>();

  function navigate(view: ViewType) {
    currentView = view;
  }

  function toggleCollapse() {
    collapsed = !collapsed;
  }
</script>

<aside class="sidebar glass" class:collapsed>
  <div class="logo">
    <span class="logo-icon">J</span>
    {#if !collapsed}
      <span class="logo-text">JARVIS</span>
    {/if}
    <button class="collapse-btn" onclick={toggleCollapse} title={collapsed ? "Expand sidebar" : "Collapse sidebar"}>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        {#if collapsed}
          <polyline points="9 18 15 12 9 6"></polyline>
        {:else}
          <polyline points="15 18 9 12 15 6"></polyline>
        {/if}
      </svg>
    </button>
  </div>

  <nav class="nav">
    <button
      class="nav-item"
      class:active={currentView === "dashboard"}
      onclick={() => navigate("dashboard")}
      title="Dashboard"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="3" width="7" height="7" />
        <rect x="14" y="3" width="7" height="7" />
        <rect x="14" y="14" width="7" height="7" />
        <rect x="3" y="14" width="7" height="7" />
      </svg>
      {#if !collapsed}<span>Dashboard</span>{/if}
    </button>

    <button
      class="nav-item"
      class:active={currentView === "messages"}
      onclick={() => navigate("messages")}
      title="Messages"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path
          d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"
        />
      </svg>
      {#if !collapsed}<span>Messages</span>{/if}
    </button>

    <button
      class="nav-item"
      class:active={currentView === "templates"}
      onclick={() => navigate("templates")}
      title="Template Builder"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="16" y1="13" x2="8" y2="13" />
        <line x1="16" y1="17" x2="8" y2="17" />
        <polyline points="10 9 9 9 8 9" />
      </svg>
      {#if !collapsed}<span>Templates</span>{/if}
    </button>

    <button
      class="nav-item"
      class:active={currentView === "health"}
      onclick={() => navigate("health")}
      title="Health Status"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path
          d="M22 12h-4l-3 9L9 3l-3 9H2"
        />
      </svg>
      {#if !collapsed}<span>Health</span>{/if}
    </button>

    <button
      class="nav-item"
      class:active={currentView === "settings"}
      onclick={() => navigate("settings")}
      title="Settings"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="3" />
        <path
          d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"
        />
      </svg>
      {#if !collapsed}<span>Settings</span>{/if}
    </button>
  </nav>

  <div class="status">
    {#if $healthStore.connected}
      <span class="status-dot connected"></span>
      {#if !collapsed}<span class="status-text">Connected</span>{/if}
    {:else}
      <span class="status-dot disconnected"></span>
      {#if !collapsed}<span class="status-text">Disconnected</span>{/if}
    {/if}
  </div>
</aside>

<style>
  .sidebar {
    width: 200px;
    min-width: 200px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    padding: 16px 0;
    transition: width 0.2s ease, min-width 0.2s ease;
  }

  .sidebar.collapsed {
    width: 60px;
    min-width: 60px;
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 16px;
    margin-bottom: 24px;
  }

  .sidebar.collapsed .logo {
    padding: 0 8px;
    justify-content: center;
  }

  .collapse-btn {
    margin-left: auto;
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s ease;
  }

  .collapse-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .collapse-btn svg {
    width: 16px;
    height: 16px;
  }

  .sidebar.collapsed .collapse-btn {
    margin-left: 0;
    margin-top: 8px;
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
    font-size: 18px;
  }

  .nav {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 0 8px;
  }

  .nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    background: transparent;
    border: none;
    border-radius: 8px;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 14px;
    transition: all 0.15s ease;
    text-align: left;
  }

  .nav-item:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .nav-item.active {
    background: var(--bg-active);
    color: var(--text-primary);
  }

  .nav-item svg {
    width: 20px;
    height: 20px;
    flex-shrink: 0;
  }

  .sidebar.collapsed .nav-item {
    justify-content: center;
    padding: 10px;
  }

  .status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    border-top: 1px solid var(--border-color);
    margin-top: auto;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .status-dot.connected {
    background: #34c759;
  }

  .status-dot.disconnected {
    background: var(--error-color);
  }

  .status-text {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .sidebar.collapsed .status {
    justify-content: center;
    padding: 12px 8px;
  }
</style>
