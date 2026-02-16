<script lang="ts">
  import { createEventDispatcher, onMount } from "svelte";

  export let fromDate: Date = new Date(Date.now() - 90 * 24 * 60 * 60 * 1000); // 90 days ago
  export let toDate: Date = new Date();
  export let currentDate: Date = toDate;
  export let isPlaying: boolean = false;

  const dispatch = createEventDispatcher<{
    dateChange: Date;
    play: void;
    pause: void;
    reset: void;
  }>();

  let sliderValue = 100;
  let playInterval: ReturnType<typeof setInterval> | null = null;

  $: totalDays = Math.ceil((toDate.getTime() - fromDate.getTime()) / (24 * 60 * 60 * 1000));

  $: {
    const dayOffset = Math.floor((sliderValue / 100) * totalDays);
    currentDate = new Date(fromDate.getTime() + dayOffset * 24 * 60 * 60 * 1000);
    dispatch("dateChange", currentDate);
  }

  function formatDate(date: Date): string {
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  }

  function handleSliderChange(event: Event) {
    const target = event.target as HTMLInputElement;
    sliderValue = Number(target.value);
  }

  function togglePlay() {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  }

  function play() {
    isPlaying = true;
    dispatch("play");

    // Reset to beginning if at end
    if (sliderValue >= 100) {
      sliderValue = 0;
    }

    playInterval = setInterval(() => {
      if (sliderValue >= 100) {
        pause();
      } else {
        sliderValue = Math.min(100, sliderValue + 1);
      }
    }, 300);
  }

  function pause() {
    isPlaying = false;
    dispatch("pause");

    if (playInterval) {
      clearInterval(playInterval);
      playInterval = null;
    }
  }

  function reset() {
    pause();
    sliderValue = 100;
    dispatch("reset");
  }

  function stepBackward() {
    sliderValue = Math.max(0, sliderValue - 5);
  }

  function stepForward() {
    sliderValue = Math.min(100, sliderValue + 5);
  }

  onMount(() => {
    return () => {
      if (playInterval) {
        clearInterval(playInterval);
      }
    };
  });
</script>

<div class="time-slider">
  <div class="slider-header">
    <span class="date-label start">{formatDate(fromDate)}</span>
    <span class="current-date">{formatDate(currentDate)}</span>
    <span class="date-label end">{formatDate(toDate)}</span>
  </div>

  <div class="slider-container">
    <input
      type="range"
      min="0"
      max="100"
      step="1"
      value={sliderValue}
      on:input={handleSliderChange}
      class="slider"
    />
    <div class="slider-progress" style="width: {sliderValue}%"></div>
  </div>

  <div class="controls">
    <button class="control-btn" on:click={stepBackward} title="Step Backward">
      <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
        <path d="M6 6h2v12H6zm3.5 6l8.5 6V6z"/>
      </svg>
    </button>

    <button class="control-btn play-btn" on:click={togglePlay} title={isPlaying ? "Pause" : "Play"}>
      {#if isPlaying}
        <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
          <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
        </svg>
      {:else}
        <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
          <path d="M8 5v14l11-7z"/>
        </svg>
      {/if}
    </button>

    <button class="control-btn" on:click={stepForward} title="Step Forward">
      <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
        <path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z"/>
      </svg>
    </button>

    <div class="divider"></div>

    <button class="control-btn" on:click={reset} title="Reset to Present">
      <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
        <path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/>
      </svg>
    </button>
  </div>
</div>

<style>
  .time-slider {
    background: rgba(40, 40, 40, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 14px 18px;
    backdrop-filter: blur(8px);
  }

  .slider-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .date-label {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.4);
  }

  .current-date {
    font-size: 13px;
    font-weight: 600;
    color: #fff;
  }

  .slider-container {
    position: relative;
    height: 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    margin-bottom: 14px;
  }

  .slider {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    appearance: none;
    background: transparent;
    cursor: pointer;
    z-index: 2;
  }

  .slider::-webkit-slider-thumb {
    appearance: none;
    width: 16px;
    height: 16px;
    background: #fff;
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
    margin-top: -5px;
  }

  .slider::-moz-range-thumb {
    width: 16px;
    height: 16px;
    background: #fff;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
  }

  .slider-progress {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    background: var(--accent-color, #007AFF);
    border-radius: 3px;
    pointer-events: none;
    z-index: 1;
  }

  .controls {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
  }

  .control-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: rgba(255, 255, 255, 0.1);
    border: none;
    border-radius: 50%;
    color: rgba(255, 255, 255, 0.8);
    cursor: pointer;
    transition: all 0.15s;
  }

  .control-btn:hover {
    background: rgba(255, 255, 255, 0.2);
    color: #fff;
  }

  .control-btn.play-btn {
    width: 40px;
    height: 40px;
    background: var(--accent-color, #007AFF);
    color: #fff;
  }

  .control-btn.play-btn:hover {
    background: #0a82e0;
  }

  .divider {
    width: 1px;
    height: 24px;
    background: rgba(255, 255, 255, 0.1);
    margin: 0 4px;
  }
</style>
