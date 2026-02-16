<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { WS_HTTP_BASE } from '../api/websocket';

  interface HealthStatus {
    status: 'healthy' | 'degraded' | 'unhealthy';
    imessage_access: boolean;
    memory_available_gb: number;
    memory_mode: string;
    model_loaded: boolean;
    permissions_ok: boolean;
    jarvis_rss_mb: number;
    model?: {
      id: string;
      display_name: string;
      loaded: boolean;
    };
    issues?: Record<string, string>;
  }

  interface DiagnosticResult {
    status: string;
    checks: {
      routers?: { status: string; registered?: number };
      schemas?: { status: string };
      imessage_sender?: { status: string };
      applescript?: { status: string; chat_count?: number };
      database?: { status: string };
    };
    issues: string[];
  }

  let health = $state<HealthStatus | null>(null);
  let diagnostic = $state<DiagnosticResult | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let refreshInterval: ReturnType<typeof setInterval>;

  async function fetchHealth() {
    try {
      const response = await fetch(`${WS_HTTP_BASE}/health`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      health = await response.json();
      error = null;
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to fetch health';
      health = null;
    }
  }

  async function fetchDiagnostic() {
    try {
      const response = await fetch(`${WS_HTTP_BASE}/health/diagnostic`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      diagnostic = await response.json();
    } catch (e) {
      // Diagnostic endpoint is optional, don't show error
    }
  }

  async function refresh() {
    loading = true;
    await Promise.all([fetchHealth(), fetchDiagnostic()]);
    loading = false;
  }

  onMount(() => {
    refresh();
    // Refresh every 30 seconds
    refreshInterval = setInterval(fetchHealth, 30000);
  });

  onDestroy(() => {
    clearInterval(refreshInterval);
  });

  function getStatusIcon(status: string) {
    switch (status) {
      case 'healthy':
      case 'ok':
        return '✅';
      case 'degraded':
        return '⚠️';
      case 'unhealthy':
      case 'error':
        return '❌';
      default:
        return '⏳';
    }
  }

  function getStatusColor(status: string) {
    switch (status) {
      case 'healthy':
      case 'ok':
        return 'var(--color-success)';
      case 'degraded':
        return 'var(--color-warning)';
      case 'unhealthy':
      case 'error':
        return 'var(--color-error)';
      default:
        return 'var(--text-secondary)';
    }
  }
</script>

<div class="health-monitor">
  <div class="header">
    <h3>System Health</h3>
    <button class="refresh-btn" onclick={refresh} disabled={loading}>
      {loading ? '⏳' : '🔄'}
    </button>
  </div>

  {#if error}
    <div class="alert error">
      <strong>Connection Error</strong>
      <p>{error}</p>
      <p class="hint">Make sure the backend is running (make launch)</p>
    </div>
  {:else if health}
    <div class="status-card" style="border-color: {getStatusColor(health.status)}">
      <div class="status-header">
        <span class="status-icon">{getStatusIcon(health.status)}</span>
        <span class="status-text" style="color: {getStatusColor(health.status)}">
          {health.status.toUpperCase()}
        </span>
      </div>

      <div class="metrics">
        <div class="metric">
          <span class="metric-label">iMessage Access</span>
          <span class="metric-value">{health.imessage_access ? '✅' : '❌'}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Memory</span>
          <span class="metric-value"
            >{health.memory_available_gb.toFixed(1)} GB ({health.memory_mode})</span
          >
        </div>
        <div class="metric">
          <span class="metric-label">Model</span>
          <span class="metric-value">{health.model_loaded ? '✅ Loaded' : '⏳ Not loaded'}</span>
        </div>
      </div>

      {#if health.issues && Object.keys(health.issues).length > 0}
        <div class="issues">
          <h4>Issues Detected</h4>
          {#each Object.entries(health.issues) as [key, value]}
            <div class="issue-item">
              <span class="issue-key">{key}:</span>
              <span class="issue-value">{value}</span>
            </div>
          {/each}
        </div>
      {/if}
    </div>

    {#if diagnostic}
      <div class="diagnostic-section">
        <h4>Diagnostic Checks</h4>
        {#each Object.entries(diagnostic.checks) as [name, result]}
          <div class="check-item">
            <span class="check-name">{name}:</span>
            <span class="check-status" style="color: {getStatusColor(result.status)}">
              {getStatusIcon(result.status)}
              {result.status}
            </span>
          </div>
        {/each}

        {#if diagnostic.issues.length > 0}
          <div class="diagnostic-issues">
            <h5>Detailed Issues</h5>
            {#each diagnostic.issues as issue}
              <div class="diagnostic-issue">• {issue}</div>
            {/each}
          </div>
        {/if}
      </div>
    {/if}
  {:else}
    <div class="loading">Loading health status...</div>
  {/if}
</div>

<style>
  .health-monitor {
    background: var(--surface-elevated);
    border-radius: var(--radius-lg);
    padding: var(--space-4);
    max-width: 400px;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--space-4);
  }

  h3 {
    margin: 0;
    font-size: var(--text-lg);
  }

  .refresh-btn {
    background: none;
    border: none;
    font-size: var(--text-lg);
    cursor: pointer;
    padding: var(--space-1);
  }

  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .status-card {
    border: 2px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: var(--space-4);
    margin-bottom: var(--space-4);
  }

  .status-header {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: var(--space-4);
    padding-bottom: var(--space-3);
    border-bottom: 1px solid var(--border-color);
  }

  .status-icon {
    font-size: var(--text-xl);
  }

  .status-text {
    font-weight: var(--font-weight-bold);
    font-size: var(--text-lg);
  }

  .metrics {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .metric-label {
    color: var(--text-secondary);
    font-size: var(--text-sm);
  }

  .metric-value {
    font-weight: var(--font-weight-medium);
  }

  .issues {
    margin-top: var(--space-4);
    padding-top: var(--space-3);
    border-top: 1px solid var(--border-color);
  }

  .issues h4 {
    color: var(--color-error);
    margin-bottom: var(--space-2);
  }

  .issue-item {
    font-size: var(--text-sm);
    margin-bottom: var(--space-1);
  }

  .issue-key {
    font-weight: var(--font-weight-medium);
  }

  .issue-value {
    color: var(--text-secondary);
  }

  .alert {
    padding: var(--space-4);
    border-radius: var(--radius-md);
    background: var(--color-error-bg, rgba(255, 59, 48, 0.1));
    border: 1px solid var(--color-error);
  }

  .alert strong {
    color: var(--color-error);
    display: block;
    margin-bottom: var(--space-2);
  }

  .hint {
    font-size: var(--text-sm);
    color: var(--text-secondary);
    margin-top: var(--space-2);
  }

  .diagnostic-section {
    margin-top: var(--space-4);
    padding-top: var(--space-4);
    border-top: 1px solid var(--border-color);
  }

  .diagnostic-section h4 {
    margin-bottom: var(--space-3);
  }

  .check-item {
    display: flex;
    justify-content: space-between;
    padding: var(--space-2) 0;
    font-size: var(--text-sm);
  }

  .check-name {
    text-transform: capitalize;
    color: var(--text-secondary);
  }

  .diagnostic-issues {
    margin-top: var(--space-3);
    padding: var(--space-3);
    background: var(--surface-base);
    border-radius: var(--radius-md);
  }

  .diagnostic-issues h5 {
    color: var(--color-error);
    margin-bottom: var(--space-2);
  }

  .diagnostic-issue {
    font-size: var(--text-xs);
    color: var(--text-secondary);
    margin-bottom: var(--space-1);
  }

  .loading {
    text-align: center;
    padding: var(--space-8);
    color: var(--text-secondary);
  }
</style>
