<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { healthStore, fetchHealth } from "../stores/health";
  import { conversationsStore } from "../stores/conversations.svelte";
  import {
    metricsStore,
    startMetricsPolling,
    stopMetricsPolling,
    type MetricsRequest,
  } from "../stores/metrics";

  type View = "messages" | "dashboard" | "health" | "settings" | "templates" | "network";
  let { onNavigate = (_view: View) => {} }: { onNavigate?: (view: View) => void } = $props();

  onMount(() => {
    fetchHealth();
    startMetricsPolling(10000);
  });

  onDestroy(() => {
    stopMetricsPolling();
  });

  let totalMessages = $derived(
    conversationsStore.conversations.reduce(
      (sum, c) => sum + c.message_count,
      0
    )
  );

  function formatTimestamp(ts: number): string {
    return new Date(ts * 1000).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }

  function formatLatency(ms: number): string {
    if (ms < 1) return "<1ms";
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }

  // Compute max latency for bar chart scaling
  function getMaxLatency(requests: MetricsRequest[]): number {
    if (requests.length === 0) return 100;
    return Math.max(...requests.map((r) => r.total_latency_ms), 100);
  }

  // Get latency phases in consistent order
  function getPhases(latency: Record<string, number>): [string, number][] {
    return Object.entries(latency).sort(([a], [b]) => a.localeCompare(b));
  }

  // Phase colors for latency breakdown bars
  const phaseColors: Record<string, string> = {
    embedding: "#007aff",
    search: "#5856d6",
    generate: "#34c759",
    total: "#ff9500",
    route: "#ff3b30",
    classify: "#af52de",
  };

  function getPhaseColor(phase: string): string {
    return phaseColors[phase] || "#8e8e93";
  }
</script>

<div class="dashboard">
  <h1>Dashboard</h1>

  <!-- Status cards -->
  <div class="cards">
    <button class="card" onclick={() => onNavigate("messages")}>
      <div class="card-icon messages">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
      </div>
      <div class="card-content">
        <h3>Conversations</h3>
        <p class="stat">{conversationsStore.conversations.length}</p>
        <p class="sub">{totalMessages.toLocaleString()} total messages</p>
      </div>
    </button>

    <button class="card" onclick={() => onNavigate("health")}>
      <div class="card-icon health" class:healthy={$healthStore.data?.status === "healthy"}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
        </svg>
      </div>
      <div class="card-content">
        <h3>System Health</h3>
        <p class="stat" class:healthy={$healthStore.data?.status === "healthy"}>
          {$healthStore.data?.status || "Unknown"}
        </p>
        <p class="sub">
          {#if $healthStore.data && $healthStore.data.memory_available_gb !== null}
            {$healthStore.data.memory_available_gb.toFixed(1)} GB available
          {:else}
            Checking...
          {/if}
        </p>
      </div>
    </button>

    <div class="card">
      <div class="card-icon model" class:loaded={$healthStore.data?.model_loaded}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
        </svg>
      </div>
      <div class="card-content">
        <h3>AI Model</h3>
        <p class="stat">{$healthStore.data?.model_loaded ? "Loaded" : "Not Loaded"}</p>
        <p class="sub">
          {$healthStore.data?.model?.display_name || "Unknown model"}
        </p>
      </div>
    </div>

    <div class="card">
      <div class="card-icon imessage" class:connected={$healthStore.data?.imessage_access}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="5" y="2" width="14" height="20" rx="2" ry="2" />
          <line x1="12" y1="18" x2="12.01" y2="18" />
        </svg>
      </div>
      <div class="card-content">
        <h3>iMessage</h3>
        <p class="stat">{$healthStore.data?.imessage_access ? "Connected" : "Not Connected"}</p>
        <p class="sub">
          {#if $healthStore.data?.imessage_access}
            Full Disk Access granted
          {:else}
            Grant access in System Settings
          {/if}
        </p>
      </div>
    </div>
  </div>

  <!-- Routing Metrics Section -->
  <div class="metrics-section">
    <h2>Routing Metrics</h2>

    {#if $metricsStore.loading && !$metricsStore.summary}
      <div class="metrics-loading">
        <span class="spinner"></span>
        Loading metrics...
      </div>
    {:else if $metricsStore.error && !$metricsStore.summary}
      <div class="metrics-empty">{$metricsStore.error}</div>
    {:else if $metricsStore.summary}
      <!-- Summary cards -->
      <div class="metric-cards">
        <div class="metric-card">
          <span class="metric-label">Total Requests</span>
          <span class="metric-value">{$metricsStore.summary.total_requests}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Avg Latency</span>
          <span class="metric-value">{formatLatency($metricsStore.summary.avg_latency_ms)}</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Cache Hit Rate</span>
          <span class="metric-value">{$metricsStore.summary.cache_hit_rate}%</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">P95 Latency</span>
          <span class="metric-value">{formatLatency($metricsStore.summary.p95_latency_ms)}</span>
        </div>
      </div>

      <!-- Decision breakdown -->
      {#if $metricsStore.summary.decisions && Object.keys($metricsStore.summary.decisions).length > 0}
        <div class="decision-breakdown">
          <h3>Decision Distribution</h3>
          <div class="decision-chips">
            {#each Object.entries($metricsStore.summary.decisions) as [decision, count]}
              <span class="decision-chip" class:generate={decision === "generate"}
                class:template={decision === "template"} class:cache={decision === "cache"}>
                {decision}: {count}
              </span>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Request log table -->
      {#if $metricsStore.recentRequests.length > 0}
        <div class="request-log">
          <h3>Recent Requests</h3>
          <div class="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Decision</th>
                  <th>Similarity</th>
                  <th>Cache</th>
                  <th>Latency</th>
                  <th>Breakdown</th>
                </tr>
              </thead>
              <tbody>
                {#each $metricsStore.recentRequests.slice(0, 50) as req (req.timestamp + req.query_hash)}
                  {@const maxLatency = getMaxLatency($metricsStore.recentRequests)}
                  <tr>
                    <td class="mono">{formatTimestamp(req.timestamp)}</td>
                    <td>
                      <span class="badge" class:generate={req.routing_decision === "generate"}
                        class:template={req.routing_decision === "template"}
                        class:cache={req.routing_decision === "cache"}>
                        {req.routing_decision}
                      </span>
                    </td>
                    <td class="mono">{req.similarity_score.toFixed(3)}</td>
                    <td>
                      {#if req.cache_hit}
                        <span class="cache-hit">HIT</span>
                      {:else}
                        <span class="cache-miss">MISS</span>
                      {/if}
                    </td>
                    <td class="mono">{formatLatency(req.total_latency_ms)}</td>
                    <td>
                      <div class="latency-bar"
                        title={getPhases(req.latency).map(([p, v]) => `${p}: ${formatLatency(v)}`).join(", ")}>
                        {#each getPhases(req.latency) as [phase, ms]}
                          <div
                            class="bar-segment"
                            style="width: {Math.max(2, (ms / maxLatency) * 120)}px; background: {getPhaseColor(phase)};"
                            title="{phase}: {formatLatency(ms)}"
                          ></div>
                        {/each}
                      </div>
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>

          <!-- Legend -->
          <div class="legend">
            {#each Object.entries(phaseColors) as [phase, color]}
              <span class="legend-item">
                <span class="legend-dot" style="background: {color};"></span>
                {phase}
              </span>
            {/each}
          </div>
        </div>
      {:else}
        <div class="metrics-empty">No routing requests recorded yet</div>
      {/if}
    {:else}
      <div class="metrics-empty">No metrics data available</div>
    {/if}
  </div>
</div>

<style>
  .dashboard {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
  }

  h1 {
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 24px;
  }

  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }

  .card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    display: flex;
    gap: 16px;
    cursor: pointer;
    transition: all 0.15s ease;
    text-align: left;
  }

  .card:hover {
    background: var(--bg-hover);
    border-color: var(--accent-color);
  }

  .card-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: var(--bg-active);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .card-icon svg {
    width: 24px;
    height: 24px;
  }

  .card-icon.messages {
    background: rgba(11, 147, 246, 0.2);
    color: var(--accent-color);
  }

  .card-icon.health {
    background: rgba(255, 95, 87, 0.2);
    color: var(--error-color);
  }

  .card-icon.health.healthy {
    background: rgba(52, 199, 89, 0.2);
    color: #34c759;
  }

  .card-icon.model {
    background: rgba(88, 86, 214, 0.2);
    color: var(--group-color);
  }

  .card-icon.model.loaded {
    background: rgba(52, 199, 89, 0.2);
    color: #34c759;
  }

  .card-icon.imessage {
    background: rgba(255, 95, 87, 0.2);
    color: var(--error-color);
  }

  .card-icon.imessage.connected {
    background: rgba(52, 199, 89, 0.2);
    color: #34c759;
  }

  .card-content h3 {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }

  .card-content .stat {
    font-size: 24px;
    font-weight: 600;
    text-transform: capitalize;
  }

  .card-content .stat.healthy {
    color: #34c759;
  }

  .card-content .sub {
    font-size: 13px;
    color: var(--text-secondary);
    margin-top: 4px;
  }

  /* Metrics Section */
  .metrics-section {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
  }

  .metrics-section h2 {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 16px;
  }

  .metrics-section h3 {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 12px;
  }

  .metrics-loading {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-secondary);
    font-size: 14px;
    padding: 24px 0;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid var(--border-color);
    border-top-color: var(--accent-color);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .metrics-empty {
    color: var(--text-secondary);
    text-align: center;
    padding: 24px;
    font-size: 14px;
  }

  /* Metric summary cards */
  .metric-cards {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
  }

  .metric-card {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
  }

  .metric-label {
    display: block;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
  }

  .metric-value {
    display: block;
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
  }

  /* Decision breakdown */
  .decision-breakdown {
    margin-bottom: 20px;
  }

  .decision-chips {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .decision-chip {
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 500;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
  }

  .decision-chip.generate {
    background: rgba(52, 199, 89, 0.15);
    color: #34c759;
    border-color: rgba(52, 199, 89, 0.3);
  }

  .decision-chip.template {
    background: rgba(88, 86, 214, 0.15);
    color: #5856d6;
    border-color: rgba(88, 86, 214, 0.3);
  }

  .decision-chip.cache {
    background: rgba(255, 149, 0, 0.15);
    color: #ff9500;
    border-color: rgba(255, 149, 0, 0.3);
  }

  /* Request log table */
  .request-log {
    margin-top: 16px;
  }

  .table-wrapper {
    overflow-x: auto;
    max-height: 400px;
    overflow-y: auto;
    border: 1px solid var(--border-color);
    border-radius: 8px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }

  thead {
    position: sticky;
    top: 0;
    z-index: 1;
  }

  th {
    background: var(--bg-primary);
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    color: var(--text-secondary);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border-color);
  }

  td {
    padding: 6px 12px;
    border-bottom: 1px solid var(--border-color);
    color: var(--text-primary);
  }

  tr:last-child td {
    border-bottom: none;
  }

  tr:hover td {
    background: var(--bg-hover);
  }

  .mono {
    font-family: "SF Mono", "Menlo", monospace;
    font-size: 12px;
  }

  /* Decision badges */
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
  }

  .badge.generate {
    background: rgba(52, 199, 89, 0.15);
    color: #34c759;
  }

  .badge.template {
    background: rgba(88, 86, 214, 0.15);
    color: #5856d6;
  }

  .badge.cache {
    background: rgba(255, 149, 0, 0.15);
    color: #ff9500;
  }

  .cache-hit {
    color: #34c759;
    font-weight: 600;
    font-size: 11px;
  }

  .cache-miss {
    color: var(--text-secondary);
    font-size: 11px;
  }

  /* Latency breakdown bar */
  .latency-bar {
    display: flex;
    height: 12px;
    border-radius: 3px;
    overflow: hidden;
    gap: 1px;
    min-width: 40px;
  }

  .bar-segment {
    height: 100%;
    min-width: 2px;
    border-radius: 2px;
    transition: width 0.2s ease;
  }

  /* Legend */
  .legend {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 12px;
    padding-top: 8px;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 2px;
  }
</style>
