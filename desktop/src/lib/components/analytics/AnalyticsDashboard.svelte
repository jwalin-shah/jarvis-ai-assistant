<script lang="ts">
  import { onMount } from "svelte";
  import { api } from "../../api/client";
  import type {
    AnalyticsOverview,
    AnalyticsTimeline,
    ActivityHeatmap,
    ContactsLeaderboard,
    TrendingPatterns,
    TimeRange,
  } from "../../api/types";
  import TimeSeriesChart from "./TimeSeriesChart.svelte";
  import HeatmapCalendar from "./HeatmapCalendar.svelte";
  import ContactLeaderboard from "./ContactLeaderboard.svelte";
  import SentimentGauge from "./SentimentGauge.svelte";
  import ResponseTimeChart from "./ResponseTimeChart.svelte";

  export let onClose: () => void;

  let loading = true;
  let error: string | null = null;
  let selectedTimeRange: TimeRange = "month";

  // Data states
  let overview: AnalyticsOverview | null = null;
  let timeline: AnalyticsTimeline | null = null;
  let heatmap: ActivityHeatmap | null = null;
  let leaderboard: ContactsLeaderboard | null = null;
  let trends: TrendingPatterns | null = null;

  const timeRangeOptions: { value: TimeRange; label: string }[] = [
    { value: "week", label: "Last Week" },
    { value: "month", label: "Last Month" },
    { value: "three_months", label: "Last 3 Months" },
    { value: "all_time", label: "All Time" },
  ];

  async function fetchData() {
    loading = true;
    error = null;
    try {
      // Fetch all data in parallel
      const [overviewData, timelineData, heatmapData, leaderboardData, trendsData] =
        await Promise.all([
          api.getAnalyticsOverview(selectedTimeRange),
          api.getAnalyticsTimeline("day", selectedTimeRange),
          api.getActivityHeatmap(selectedTimeRange),
          api.getContactsLeaderboard(selectedTimeRange, 10),
          api.getTrendingPatterns(selectedTimeRange),
        ]);
      overview = overviewData;
      timeline = timelineData;
      heatmap = heatmapData;
      leaderboard = leaderboardData;
      trends = trendsData;
    } catch (e) {
      if (e instanceof Error) {
        error = `Failed to load analytics: ${e.message}`;
      } else {
        error = "Failed to load analytics. Please try again.";
      }
    } finally {
      loading = false;
    }
  }

  function handleTimeRangeChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    selectedTimeRange = target.value as TimeRange;
    fetchData();
  }

  function handleBackdropClick(event: MouseEvent) {
    if (event.target === event.currentTarget) {
      onClose();
    }
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === "Escape") {
      onClose();
    }
  }

  function formatNumber(num: number): string {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  }

  function formatResponseTime(minutes: number | null): string {
    if (minutes === null) return "N/A";
    if (minutes < 60) return `${Math.round(minutes)} min`;
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  }

  function getTrendIcon(direction: string): string {
    if (direction === "increasing") return "trending_up";
    if (direction === "decreasing") return "trending_down";
    return "trending_flat";
  }

  function getTrendColor(direction: string): string {
    if (direction === "increasing") return "var(--success-color)";
    if (direction === "decreasing") return "var(--error-color)";
    return "var(--text-secondary)";
  }

  async function handleExport(format: "json" | "csv") {
    try {
      const blob = await api.exportAnalytics(format, selectedTimeRange);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `analytics_${selectedTimeRange}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error("Export failed:", e);
    }
  }

  onMount(() => {
    fetchData();
    window.addEventListener("keydown", handleKeydown);
    return () => {
      window.removeEventListener("keydown", handleKeydown);
    };
  });
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div class="modal-overlay" on:click={handleBackdropClick}>
  <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
    <div class="modal-header">
      <h2 id="modal-title">Analytics Dashboard</h2>
      <div class="header-controls">
        <select
          class="time-range-select"
          value={selectedTimeRange}
          on:change={handleTimeRangeChange}
          disabled={loading}
        >
          {#each timeRangeOptions as option}
            <option value={option.value}>{option.label}</option>
          {/each}
        </select>
        <div class="export-buttons">
          <button
            class="export-btn"
            on:click={() => handleExport("json")}
            disabled={loading}
            title="Export as JSON"
          >
            JSON
          </button>
          <button
            class="export-btn"
            on:click={() => handleExport("csv")}
            disabled={loading}
            title="Export as CSV"
          >
            CSV
          </button>
        </div>
        <button class="close-btn" on:click={onClose} aria-label="Close">
          &times;
        </button>
      </div>
    </div>

    <div class="modal-content">
      {#if loading}
        <div class="loading-state">
          <div class="loading-spinner"></div>
          <p>Loading analytics...</p>
        </div>
      {:else if error}
        <div class="error-state">
          <p class="error-message">{error}</p>
          <button class="retry-btn" on:click={fetchData}>Try Again</button>
        </div>
      {:else if overview}
        <div class="dashboard-content">
          <!-- Overview Stats Cards -->
          <section class="overview-section">
            <div class="stat-card primary">
              <div class="stat-value">{formatNumber(overview.total_messages)}</div>
              <div class="stat-label">Total Messages</div>
              {#if trends?.overall_trend}
                <div
                  class="stat-trend"
                  style="color: {getTrendColor(trends.overall_trend.direction)}"
                >
                  <span class="trend-arrow">
                    {trends.overall_trend.direction === "increasing" ? "+" : ""}
                    {trends.overall_trend.percentage_change.toFixed(1)}%
                  </span>
                </div>
              {/if}
            </div>
            <div class="stat-card sent">
              <div class="stat-value">{formatNumber(overview.sent_messages)}</div>
              <div class="stat-label">Sent</div>
            </div>
            <div class="stat-card received">
              <div class="stat-value">{formatNumber(overview.received_messages)}</div>
              <div class="stat-label">Received</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">{overview.active_conversations}</div>
              <div class="stat-label">Active Chats</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">{overview.avg_messages_per_day.toFixed(1)}</div>
              <div class="stat-label">Messages/Day</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">{formatResponseTime(overview.avg_response_time_minutes)}</div>
              <div class="stat-label">Avg Response</div>
            </div>
          </section>

          <!-- Sentiment & Peak Times -->
          <section class="metrics-row">
            <div class="metric-card sentiment-card">
              <h3>Overall Sentiment</h3>
              <SentimentGauge
                score={overview.sentiment.score}
                label={overview.sentiment.label}
              />
            </div>
            <div class="metric-card peak-times-card">
              <h3>Peak Activity</h3>
              <div class="peak-times">
                {#if overview.peak_hour !== null}
                  <div class="peak-item">
                    <span class="peak-label">Busiest Hour</span>
                    <span class="peak-value">
                      {overview.peak_hour === 0 ? "12 AM" :
                       overview.peak_hour === 12 ? "12 PM" :
                       overview.peak_hour < 12 ? `${overview.peak_hour} AM` :
                       `${overview.peak_hour - 12} PM`}
                    </span>
                  </div>
                {/if}
                {#if overview.peak_day}
                  <div class="peak-item">
                    <span class="peak-label">Busiest Day</span>
                    <span class="peak-value">{overview.peak_day}</span>
                  </div>
                {/if}
              </div>
            </div>
            <div class="metric-card comparison-card">
              <h3>vs Previous Period</h3>
              <div class="comparison-items">
                <div class="comparison-item">
                  <span class="comparison-label">Messages</span>
                  <span
                    class="comparison-value"
                    class:positive={overview.period_comparison.total_change_percent > 0}
                    class:negative={overview.period_comparison.total_change_percent < 0}
                  >
                    {overview.period_comparison.total_change_percent > 0 ? "+" : ""}
                    {overview.period_comparison.total_change_percent.toFixed(1)}%
                  </span>
                </div>
                <div class="comparison-item">
                  <span class="comparison-label">Sent</span>
                  <span
                    class="comparison-value"
                    class:positive={overview.period_comparison.sent_change_percent > 0}
                    class:negative={overview.period_comparison.sent_change_percent < 0}
                  >
                    {overview.period_comparison.sent_change_percent > 0 ? "+" : ""}
                    {overview.period_comparison.sent_change_percent.toFixed(1)}%
                  </span>
                </div>
                <div class="comparison-item">
                  <span class="comparison-label">Contacts</span>
                  <span
                    class="comparison-value"
                    class:positive={overview.period_comparison.contacts_change_percent > 0}
                    class:negative={overview.period_comparison.contacts_change_percent < 0}
                  >
                    {overview.period_comparison.contacts_change_percent > 0 ? "+" : ""}
                    {overview.period_comparison.contacts_change_percent.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </section>

          <!-- Time Series Chart -->
          {#if timeline}
            <section class="chart-section">
              <h3>Message Activity Over Time</h3>
              <TimeSeriesChart data={timeline.data} granularity={timeline.granularity} />
            </section>
          {/if}

          <!-- Activity Heatmap -->
          {#if heatmap}
            <section class="chart-section">
              <h3>Activity Calendar</h3>
              <HeatmapCalendar data={heatmap.data} stats={heatmap.stats} />
            </section>
          {/if}

          <!-- Two Column Layout -->
          <div class="two-column">
            <!-- Contact Leaderboard -->
            {#if leaderboard}
              <section class="chart-section">
                <h3>Top Contacts</h3>
                <ContactLeaderboard contacts={leaderboard.contacts} />
              </section>
            {/if}

            <!-- Trending Patterns -->
            {#if trends}
              <section class="chart-section">
                <h3>Trends & Patterns</h3>
                <div class="trends-list">
                  {#if trends.trending_contacts.length > 0}
                    <div class="trend-group">
                      <h4>Trending Contacts</h4>
                      {#each trends.trending_contacts.slice(0, 5) as contact}
                        <div class="trend-item">
                          <span class="trend-name">{contact.contact_name || "Unknown"}</span>
                          <span
                            class="trend-change"
                            style="color: {getTrendColor(contact.trend)}"
                          >
                            {contact.change_percent > 0 ? "+" : ""}
                            {contact.change_percent.toFixed(0)}%
                          </span>
                        </div>
                      {/each}
                    </div>
                  {/if}
                  {#if trends.anomalies.length > 0}
                    <div class="trend-group">
                      <h4>Unusual Activity</h4>
                      {#each trends.anomalies.slice(0, 3) as anomaly}
                        <div class="anomaly-item">
                          <span class="anomaly-date">{anomaly.date}</span>
                          <span class="anomaly-type" class:spike={anomaly.type === "spike"}>
                            {anomaly.type === "spike" ? "High" : "Low"}: {anomaly.value} msgs
                          </span>
                        </div>
                      {/each}
                    </div>
                  {/if}
                  {#if trends.seasonality.detected}
                    <div class="seasonality-note">
                      Pattern detected: {trends.seasonality.pattern}
                    </div>
                  {/if}
                </div>
              </section>
            {/if}
          </div>
        </div>
      {/if}
    </div>

    <div class="modal-footer">
      <button class="btn secondary" on:click={fetchData} disabled={loading}>
        Refresh
      </button>
      <button class="btn primary" on:click={onClose}>
        Close
      </button>
    </div>
  </div>
</div>

<style>
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    width: 95%;
    max-width: 1000px;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-color);
  }

  .modal-header h2 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .header-controls {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .time-range-select {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-primary);
    padding: 6px 10px;
    font-size: 13px;
    cursor: pointer;
  }

  .export-buttons {
    display: flex;
    gap: 4px;
  }

  .export-btn {
    background: var(--bg-hover);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-secondary);
    padding: 4px 8px;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .export-btn:hover:not(:disabled) {
    background: var(--bg-active);
    color: var(--text-primary);
  }

  .close-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 24px;
    cursor: pointer;
    padding: 0;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
  }

  .close-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .modal-content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
  }

  .loading-state,
  .error-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 300px;
    gap: 16px;
    color: var(--text-secondary);
  }

  .loading-spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-color);
    border-top-color: var(--accent-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .error-message {
    color: var(--error-color);
  }

  .retry-btn {
    background: var(--bg-hover);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
  }

  .dashboard-content {
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  /* Overview Stats */
  .overview-section {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
  }

  .stat-card {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
  }

  .stat-card.primary {
    border-color: var(--accent-color);
  }

  .stat-card.sent .stat-value {
    color: var(--accent-color);
  }

  .stat-card.received .stat-value {
    color: #34c759;
  }

  .stat-value {
    font-size: 24px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .stat-label {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 4px;
  }

  .stat-trend {
    font-size: 12px;
    margin-top: 4px;
  }

  /* Metrics Row */
  .metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
  }

  .metric-card {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 16px;
  }

  .metric-card h3 {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 12px 0;
  }

  .peak-times {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .peak-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .peak-label {
    font-size: 13px;
    color: var(--text-secondary);
  }

  .peak-value {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .comparison-items {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .comparison-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .comparison-label {
    font-size: 13px;
    color: var(--text-secondary);
  }

  .comparison-value {
    font-size: 14px;
    font-weight: 500;
  }

  .comparison-value.positive {
    color: #34c759;
  }

  .comparison-value.negative {
    color: #ff3b30;
  }

  /* Chart Sections */
  .chart-section {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 16px;
  }

  .chart-section h3 {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 12px 0;
  }

  /* Two Column */
  .two-column {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  /* Trends */
  .trends-list {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .trend-group h4 {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    margin: 0 0 8px 0;
    text-transform: uppercase;
  }

  .trend-item {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid var(--border-color);
  }

  .trend-item:last-child {
    border-bottom: none;
  }

  .trend-name {
    font-size: 13px;
    color: var(--text-primary);
  }

  .trend-change {
    font-size: 13px;
    font-weight: 500;
  }

  .anomaly-item {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid var(--border-color);
  }

  .anomaly-date {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .anomaly-type {
    font-size: 12px;
    color: var(--error-color);
  }

  .anomaly-type.spike {
    color: var(--success-color);
  }

  .seasonality-note {
    font-size: 12px;
    color: var(--text-secondary);
    padding: 8px;
    background: var(--bg-hover);
    border-radius: 4px;
  }

  /* Modal Footer */
  .modal-footer {
    display: flex;
    gap: 10px;
    padding: 16px 20px;
    border-top: 1px solid var(--border-color);
    justify-content: flex-end;
  }

  .btn {
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn.primary {
    background: var(--accent-color);
    color: white;
    border: none;
  }

  .btn.primary:hover:not(:disabled) {
    background: #0a82e0;
  }

  .btn.secondary {
    background: var(--bg-hover);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
  }

  .btn.secondary:hover:not(:disabled) {
    background: var(--bg-active);
  }

  /* Responsive */
  @media (max-width: 900px) {
    .overview-section {
      grid-template-columns: repeat(3, 1fr);
    }

    .metrics-row {
      grid-template-columns: 1fr;
    }

    .two-column {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 600px) {
    .overview-section {
      grid-template-columns: repeat(2, 1fr);
    }

    .stat-value {
      font-size: 20px;
    }
  }
</style>
