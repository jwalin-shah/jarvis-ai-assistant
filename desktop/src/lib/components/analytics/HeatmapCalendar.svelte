<script lang="ts">
  import type { HeatmapDataPoint } from "../../api/types";

  export let data: HeatmapDataPoint[];
  export let stats: {
    total_days: number;
    active_days: number;
    max_count: number;
    avg_count: number;
  };

  const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  function getColor(level: number): string {
    switch (level) {
      case 0: return "var(--bg-hover)";
      case 1: return "#0e4429";
      case 2: return "#006d32";
      case 3: return "#26a641";
      case 4: return "#39d353";
      default: return "var(--bg-hover)";
    }
  }

  function organizeData(): { weeks: HeatmapDataPoint[][]; months: { label: string; colSpan: number }[] } {
    if (!data.length) return { weeks: [], months: [] };

    // Sort data by date
    const sorted = [...data].sort((a, b) => a.date.localeCompare(b.date));

    // Group into weeks (columns)
    const weeks: HeatmapDataPoint[][] = [];
    let currentWeek: HeatmapDataPoint[] = [];
    let currentMonth = "";
    const months: { label: string; colSpan: number }[] = [];

    sorted.forEach((point, index) => {
      const date = new Date(point.date);
      const dayOfWeek = date.getDay();
      const monthYear = MONTHS[date.getMonth()];

      // Start a new week on Sunday or first item
      if (dayOfWeek === 0 && currentWeek.length > 0) {
        weeks.push(currentWeek);
        currentWeek = [];
      }

      // Track months
      if (monthYear !== currentMonth) {
        if (months.length > 0) {
          // Count weeks for previous month
        }
        months.push({ label: monthYear, colSpan: 1 });
        currentMonth = monthYear;
      } else if (months.length > 0 && dayOfWeek === 0) {
        months[months.length - 1].colSpan++;
      }

      currentWeek.push(point);
    });

    if (currentWeek.length > 0) {
      weeks.push(currentWeek);
    }

    return { weeks, months };
  }

  function getDayOfWeek(dateStr: string): number {
    return new Date(dateStr).getDay();
  }

  $: organized = organizeData();
</script>

<div class="heatmap-container">
  <div class="heatmap">
    <!-- Day labels -->
    <div class="day-labels">
      {#each DAYS as day, i}
        {#if i % 2 === 1}
          <span class="day-label">{day}</span>
        {:else}
          <span class="day-label"></span>
        {/if}
      {/each}
    </div>

    <!-- Calendar grid -->
    <div class="calendar-grid">
      {#each organized.weeks as week, weekIndex}
        <div class="week-column">
          {#each Array(7) as _, dayIndex}
            {@const point = week.find(d => getDayOfWeek(d.date) === dayIndex)}
            {#if point}
              <div
                class="day-cell"
                style="background-color: {getColor(point?.level || 0)}"
                title="{point?.date}: {point?.count} messages"
              ></div>
            {:else}
              <div class="day-cell empty"></div>
            {/if}
          {/each}
        </div>
      {/each}
    </div>
  </div>

  <!-- Stats summary -->
  <div class="stats-row">
    <span class="stat-item">
      <span class="stat-value">{stats.active_days}</span>
      <span class="stat-label">Active Days</span>
    </span>
    <span class="stat-item">
      <span class="stat-value">{stats.max_count}</span>
      <span class="stat-label">Max/Day</span>
    </span>
    <span class="stat-item">
      <span class="stat-value">{stats.avg_count.toFixed(1)}</span>
      <span class="stat-label">Avg/Day</span>
    </span>
    <span class="legend">
      <span class="legend-label">Less</span>
      {#each [0, 1, 2, 3, 4] as level}
        <span class="legend-cell" style="background-color: {getColor(level)}"></span>
      {/each}
      <span class="legend-label">More</span>
    </span>
  </div>
</div>

<style>
  .heatmap-container {
    width: 100%;
    overflow-x: auto;
  }

  .heatmap {
    display: flex;
    gap: 4px;
    min-width: 600px;
  }

  .day-labels {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding-top: 0;
  }

  .day-label {
    font-size: 9px;
    color: var(--text-secondary);
    height: 12px;
    line-height: 12px;
  }

  .calendar-grid {
    display: flex;
    gap: 3px;
    flex: 1;
  }

  .week-column {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  .day-cell {
    width: 12px;
    height: 12px;
    border-radius: 2px;
    cursor: pointer;
    transition: transform 0.15s;
  }

  .day-cell:hover:not(.empty) {
    transform: scale(1.2);
    outline: 1px solid var(--text-secondary);
  }

  .day-cell.empty {
    background-color: transparent;
  }

  .stats-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border-color);
  }

  .stat-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
  }

  .stat-value {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .stat-label {
    font-size: 10px;
    color: var(--text-secondary);
  }

  .legend {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .legend-label {
    font-size: 10px;
    color: var(--text-secondary);
  }

  .legend-cell {
    width: 12px;
    height: 12px;
    border-radius: 2px;
  }
</style>
