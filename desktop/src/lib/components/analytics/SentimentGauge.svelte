<script lang="ts">
  export let score: number; // -1 to 1
  export let label: string;

  // Convert score (-1 to 1) to angle (0 to 180)
  $: angle = ((score + 1) / 2) * 180;

  // Convert score to percentage for display
  $: percentage = Math.round(((score + 1) / 2) * 100);

  function getColor(score: number): string {
    if (score > 0.3) return "#34c759"; // positive
    if (score < -0.3) return "#ff3b30"; // negative
    return "#ff9f0a"; // neutral
  }

  function getLabelColor(label: string): string {
    if (label === "positive") return "#34c759";
    if (label === "negative") return "#ff3b30";
    return "#ff9f0a";
  }

  $: needleRotation = angle - 90;
</script>

<div class="gauge-container">
  <div class="gauge">
    <svg viewBox="0 0 200 120" class="gauge-svg">
      <!-- Background arc -->
      <path
        d="M 20 100 A 80 80 0 0 1 180 100"
        fill="none"
        stroke="var(--bg-hover)"
        stroke-width="20"
        stroke-linecap="round"
      />
      <!-- Gradient arc sections -->
      <path
        d="M 20 100 A 80 80 0 0 1 60 40"
        fill="none"
        stroke="#ff3b30"
        stroke-width="20"
        stroke-linecap="round"
        opacity="0.8"
      />
      <path
        d="M 60 35 A 80 80 0 0 1 140 35"
        fill="none"
        stroke="#ff9f0a"
        stroke-width="20"
        opacity="0.8"
      />
      <path
        d="M 140 40 A 80 80 0 0 1 180 100"
        fill="none"
        stroke="#34c759"
        stroke-width="20"
        stroke-linecap="round"
        opacity="0.8"
      />
      <!-- Needle -->
      <g transform="rotate({needleRotation}, 100, 100)">
        <line
          x1="100"
          y1="100"
          x2="100"
          y2="35"
          stroke="var(--text-primary)"
          stroke-width="3"
          stroke-linecap="round"
        />
        <circle cx="100" cy="100" r="8" fill="var(--text-primary)" />
        <circle cx="100" cy="100" r="4" fill="var(--bg-primary)" />
      </g>
    </svg>
  </div>
  <div class="gauge-value">
    <span class="score" style="color: {getColor(score)}">{score > 0 ? "+" : ""}{score.toFixed(2)}</span>
    <span class="label" style="color: {getLabelColor(label)}">{label}</span>
  </div>
  <div class="gauge-scale">
    <span>Negative</span>
    <span>Neutral</span>
    <span>Positive</span>
  </div>
</div>

<style>
  .gauge-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }

  .gauge {
    width: 150px;
    height: 90px;
  }

  .gauge-svg {
    width: 100%;
    height: 100%;
  }

  .gauge-value {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    margin-top: -16px;
  }

  .score {
    font-size: 24px;
    font-weight: 600;
  }

  .label {
    font-size: 12px;
    font-weight: 500;
    text-transform: capitalize;
  }

  .gauge-scale {
    display: flex;
    justify-content: space-between;
    width: 150px;
    font-size: 9px;
    color: var(--text-secondary);
  }
</style>
