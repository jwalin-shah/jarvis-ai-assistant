<script lang="ts">
  import { apiClient, APIError } from "../../api/client";
  import type { TimingSuggestion } from "../../api/types";

  interface Props {
    contactId: number;
    compact?: boolean;
    onSelect?: (suggestion: TimingSuggestion) => void;
  }

  let { contactId, compact = false, onSelect }: Props = $props();

  // State
  let suggestion: TimingSuggestion | null = $state(null);
  let isLoading = $state(false);
  let error = $state("");
  let showTooltip = $state(false);

  // Load suggestion on mount
  $effect(() => {
    loadSuggestion();
  });

  async function loadSuggestion() {
    isLoading = true;
    error = "";

    try {
      const response = await apiClient.getTimingSuggestions(contactId, undefined, undefined, 1);
      suggestion = response.suggestions[0] || null;
    } catch (e) {
      if (e instanceof APIError) {
        error = e.detail || e.message;
      } else if (e instanceof Error) {
        error = e.message;
      }
    } finally {
      isLoading = false;
    }
  }

  function handleClick() {
    if (suggestion && onSelect) {
      onSelect(suggestion);
    }
  }

  function formatTime(isoString: string): string {
    const date = new Date(isoString);
    const now = new Date();
    const tomorrow = new Date(now);
    tomorrow.setDate(tomorrow.getDate() + 1);

    const isToday = date.toDateString() === now.toDateString();
    const isTomorrow = date.toDateString() === tomorrow.toDateString();

    if (isToday) {
      return `Today ${date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}`;
    } else if (isTomorrow) {
      return `Tomorrow ${date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}`;
    } else {
      return date.toLocaleDateString(undefined, {
        weekday: "short",
        hour: "numeric",
        minute: "2-digit",
      });
    }
  }

  function getConfidenceColor(confidence: number): string {
    if (confidence >= 0.8) return "#34c759";
    if (confidence >= 0.6) return "#ff9500";
    if (confidence >= 0.4) return "#ffcc00";
    return "#8e8e93";
  }

  function getConfidenceLabel(confidence: number): string {
    if (confidence >= 0.8) return "Great time";
    if (confidence >= 0.6) return "Good time";
    if (confidence >= 0.4) return "OK time";
    return "Available";
  }
</script>

{#if isLoading}
  <span class="timing-badge loading" class:compact>
    <span class="spinner"></span>
  </span>
{:else if error}
  <span class="timing-badge error" class:compact title={error}>
    <span class="icon">!</span>
  </span>
{:else if suggestion}
  <button
    class="timing-badge"
    class:compact
    class:clickable={!!onSelect}
    onclick={handleClick}
    onmouseenter={() => (showTooltip = true)}
    onmouseleave={() => (showTooltip = false)}
    style="--confidence-color: {getConfidenceColor(suggestion.confidence)}"
  >
    <span class="icon">&#128337;</span>
    {#if !compact}
      <span class="time">{formatTime(suggestion.suggested_time)}</span>
    {/if}
    <span class="confidence-dot" title={getConfidenceLabel(suggestion.confidence)}></span>

    {#if showTooltip && !compact}
      <div class="tooltip">
        <div class="tooltip-header">
          <span class="tooltip-label">{getConfidenceLabel(suggestion.confidence)}</span>
          <span class="tooltip-confidence">
            {Math.round(suggestion.confidence * 100)}%
          </span>
        </div>
        {#if suggestion.reason}
          <p class="tooltip-reason">{suggestion.reason}</p>
        {/if}
        {#if onSelect}
          <p class="tooltip-action">Click to use this time</p>
        {/if}
      </div>
    {/if}
  </button>
{/if}

<style>
  .timing-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    background: var(--bg-secondary, #2a2a2a);
    border: 1px solid var(--border-color, #3a3a3a);
    border-radius: 6px;
    padding: 0.375rem 0.625rem;
    font-size: 0.75rem;
    color: var(--text-primary, #fff);
    position: relative;
    cursor: default;
  }

  .timing-badge.compact {
    padding: 0.25rem 0.5rem;
  }

  .timing-badge.clickable {
    cursor: pointer;
  }

  .timing-badge.clickable:hover {
    border-color: var(--accent-color, #007aff);
    background: var(--bg-tertiary, #3a3a3a);
  }

  .timing-badge.loading {
    min-width: 60px;
    justify-content: center;
  }

  .timing-badge.error {
    border-color: #ff3b30;
    color: #ff3b30;
  }

  .icon {
    font-size: 0.875rem;
    line-height: 1;
  }

  .spinner {
    width: 12px;
    height: 12px;
    border: 2px solid var(--border-color, #3a3a3a);
    border-top-color: var(--accent-color, #007aff);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .time {
    font-weight: 500;
  }

  .confidence-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: var(--confidence-color);
  }

  .tooltip {
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: var(--bg-primary, #1e1e1e);
    border: 1px solid var(--border-color, #3a3a3a);
    border-radius: 8px;
    padding: 0.75rem;
    min-width: 180px;
    max-width: 250px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    margin-bottom: 0.5rem;
    z-index: 100;
  }

  .tooltip::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 6px solid transparent;
    border-top-color: var(--border-color, #3a3a3a);
  }

  .tooltip-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.375rem;
  }

  .tooltip-label {
    font-weight: 600;
    color: var(--confidence-color);
  }

  .tooltip-confidence {
    font-size: 0.625rem;
    background: var(--bg-secondary, #2a2a2a);
    padding: 0.125rem 0.375rem;
    border-radius: 4px;
    color: var(--text-secondary, #999);
  }

  .tooltip-reason {
    margin: 0;
    font-size: 0.75rem;
    color: var(--text-secondary, #999);
    line-height: 1.4;
  }

  .tooltip-action {
    margin: 0.5rem 0 0 0;
    font-size: 0.625rem;
    color: var(--accent-color, #007aff);
    font-style: italic;
  }
</style>
