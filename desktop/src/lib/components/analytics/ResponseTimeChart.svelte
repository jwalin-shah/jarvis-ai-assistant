<script lang="ts">
  export let distribution: Record<number, number> = {};
  export let avgTime: number | null = null;

  // Convert hourly distribution to chart data
  $: chartData = Object.entries(distribution)
    .map(([hour, count]) => ({
      hour: parseInt(hour),
      count,
    }))
    .sort((a, b) => a.hour - b.hour);

  $: maxCount = Math.max(...chartData.map((d) => d.count), 1);

  function formatHour(hour: number): string {
    if (hour === 0) return "12am";
    if (hour === 12) return "12pm";
    return hour < 12 ? `${hour}am` : `${hour - 12}pm`;
  }

  function formatTime(minutes: number | null): string {
    if (minutes === null) return "N/A";
    if (minutes < 1) return "< 1 min";
    if (minutes < 60) return `${Math.round(minutes)} min`;
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  }

  function getBarHeight(count: number): number {
    return (count / maxCount) * 100;
  }

  function getBarColor(hour: number): string {
    // Color based on time of day
    if (hour >= 6 && hour < 12) return "#ff9f0a"; // morning - orange
    if (hour >= 12 && hour < 18) return "var(--accent-color)"; // afternoon - blue
    if (hour >= 18 && hour < 22) return "#34c759"; // evening - green
    return "#8e8e93"; // night - gray
  }
</script>

<div class="response-time-chart">
  {#if avgTime !== null}
    <div class="avg-time">
      <span class="avg-label">Average Response Time</span>
      <span class="avg-value">{formatTime(avgTime)}</span>
    </div>
  {/if}

  <div class="chart">
    {#if chartData.length > 0}
      {#each chartData as point}
        <div
          class="bar-wrapper"
          title="{formatHour(point.hour)}: {point.count} responses"
        >
          <div
            class="bar"
            style="height: {getBarHeight(point.count)}%; background-color: {getBarColor(point.hour)}"
          ></div>
        </div>
      {/each}
    {:else}
      <div class="no-data">No response time data available</div>
    {/if}
  </div>

  {#if chartData.length > 0}
    <div class="x-axis">
      <span>12am</span>
      <span>6am</span>
      <span>12pm</span>
      <span>6pm</span>
      <span>11pm</span>
    </div>

    <div class="time-legend">
      <span class="legend-item">
        <span class="legend-dot" style="background: #8e8e93"></span>
        Night
      </span>
      <span class="legend-item">
        <span class="legend-dot" style="background: #ff9f0a"></span>
        Morning
      </span>
      <span class="legend-item">
        <span class="legend-dot" style="background: var(--accent-color)"></span>
        Afternoon
      </span>
      <span class="legend-item">
        <span class="legend-dot" style="background: #34c759"></span>
        Evening
      </span>
    </div>
  {/if}
</div>

<style>
  .response-time-chart {
    width: 100%;
  }

  .avg-time {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: var(--bg-hover);
    border-radius: 6px;
    margin-bottom: 12px;
  }

  .avg-label {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .avg-value {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .chart {
    display: flex;
    align-items: flex-end;
    height: 80px;
    gap: 2px;
  }

  .bar-wrapper {
    flex: 1;
    height: 100%;
    display: flex;
    align-items: flex-end;
    cursor: pointer;
  }

  .bar {
    width: 100%;
    min-height: 2px;
    border-radius: 2px 2px 0 0;
    transition: height 0.3s ease;
  }

  .bar-wrapper:hover .bar {
    opacity: 0.8;
  }

  .no-data {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary);
    font-size: 13px;
  }

  .x-axis {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 10px;
    color: var(--text-secondary);
  }

  .time-legend {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 8px;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 10px;
    color: var(--text-secondary);
  }

  .legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }
</style>
