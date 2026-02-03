<script lang="ts">
  import { onMount } from "svelte";
  import type { TimelineDataPoint } from "../../api/types";

  export let data: TimelineDataPoint[];
  export let granularity: string = "day";

  let chartContainer: HTMLDivElement;

  function getMaxValue(): number {
    if (!data.length) return 1;
    return Math.max(...data.map((d) => d.total), 1);
  }

  function formatDate(point: TimelineDataPoint): string {
    if (point.hour !== undefined) {
      return point.hour === 0 ? "12am" :
             point.hour === 12 ? "12pm" :
             point.hour < 12 ? `${point.hour}am` : `${point.hour - 12}pm`;
    }
    if (point.date) {
      if (granularity === "week") {
        return point.date; // YYYY-WNN
      }
      if (granularity === "month") {
        return point.date; // YYYY-MM
      }
      // Format date as MM/DD
      const parts = point.date.split("-");
      return `${parts[1]}/${parts[2]}`;
    }
    return "";
  }

  function getBarHeight(value: number): number {
    const max = getMaxValue();
    return (value / max) * 100;
  }
</script>

<div class="chart-wrapper" bind:this={chartContainer}>
  <div class="chart">
    {#each data as point, i}
      <div class="bar-group" title="{formatDate(point)}: {point.total} total ({point.sent} sent, {point.received} received)">
        <div class="bar-stack" style="height: {getBarHeight(point.total)}%">
          <div
            class="bar sent"
            style="height: {point.total > 0 ? (point.sent / point.total) * 100 : 0}%"
          ></div>
          <div
            class="bar received"
            style="height: {point.total > 0 ? (point.received / point.total) * 100 : 0}%"
          ></div>
        </div>
      </div>
    {/each}
  </div>
  <div class="x-axis">
    {#if data.length <= 7}
      {#each data as point}
        <span class="x-label">{formatDate(point)}</span>
      {/each}
    {:else}
      <span class="x-label">{data.length > 0 ? formatDate(data[0]) : ""}</span>
      <span class="x-label">{data.length > 1 ? formatDate(data[Math.floor(data.length / 2)]) : ""}</span>
      <span class="x-label">{data.length > 0 ? formatDate(data[data.length - 1]) : ""}</span>
    {/if}
  </div>
  <div class="legend">
    <span class="legend-item">
      <span class="legend-color sent"></span>
      Sent
    </span>
    <span class="legend-item">
      <span class="legend-color received"></span>
      Received
    </span>
  </div>
</div>

<style>
  .chart-wrapper {
    width: 100%;
  }

  .chart {
    display: flex;
    align-items: flex-end;
    height: 150px;
    gap: 2px;
    padding: 8px 0;
  }

  .bar-group {
    flex: 1;
    height: 100%;
    display: flex;
    align-items: flex-end;
    min-width: 4px;
    cursor: pointer;
  }

  .bar-stack {
    width: 100%;
    display: flex;
    flex-direction: column;
    border-radius: 2px 2px 0 0;
    overflow: hidden;
    min-height: 2px;
    transition: height 0.3s ease;
  }

  .bar {
    width: 100%;
  }

  .bar.sent {
    background: var(--accent-color);
  }

  .bar.received {
    background: #34c759;
  }

  .x-axis {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
  }

  .x-label {
    font-size: 10px;
    color: var(--text-secondary);
  }

  .legend {
    display: flex;
    gap: 16px;
    justify-content: center;
    padding-top: 8px;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .legend-color {
    width: 12px;
    height: 12px;
    border-radius: 2px;
  }

  .legend-color.sent {
    background: var(--accent-color);
  }

  .legend-color.received {
    background: #34c759;
  }
</style>
