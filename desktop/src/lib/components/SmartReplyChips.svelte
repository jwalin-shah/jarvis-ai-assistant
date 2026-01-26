<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { api } from "../api/client";
  import type { SmartReplySuggestion } from "../api/types";

  // Props
  let { lastMessage = "", isFocused = false }: { lastMessage: string; isFocused: boolean } = $props();

  // State
  let suggestions = $state<SmartReplySuggestion[]>([]);
  let loading = $state(false);
  let error = $state<string | null>(null);
  let visible = $state(false);
  let toastMessage = $state<string | null>(null);
  let toastTimeout: ReturnType<typeof setTimeout> | null = null;

  // Fetch suggestions when lastMessage changes
  $effect(() => {
    if (lastMessage && lastMessage.trim()) {
      fetchSuggestions(lastMessage);
    } else {
      suggestions = [];
      visible = false;
    }
  });

  async function fetchSuggestions(message: string) {
    loading = true;
    error = null;
    visible = false;

    try {
      const response = await api.getSmartReplySuggestions(message, 3);
      suggestions = response.suggestions;
      // Trigger animation after a small delay
      setTimeout(() => {
        visible = true;
      }, 50);
    } catch (err) {
      error = err instanceof Error ? err.message : "Failed to fetch suggestions";
      suggestions = [];
    } finally {
      loading = false;
    }
  }

  async function handleChipClick(suggestion: SmartReplySuggestion) {
    try {
      await navigator.clipboard.writeText(suggestion.text);
      showToast("Copied to clipboard");
    } catch (err) {
      showToast("Failed to copy");
    }
  }

  function showToast(message: string) {
    toastMessage = message;
    if (toastTimeout) {
      clearTimeout(toastTimeout);
    }
    toastTimeout = setTimeout(() => {
      toastMessage = null;
    }, 2000);
  }

  // Keyboard shortcuts handler
  function handleKeydown(event: KeyboardEvent) {
    if (!isFocused || suggestions.length === 0) return;

    // Check if user pressed 1, 2, or 3
    const keyNum = parseInt(event.key);
    if (keyNum >= 1 && keyNum <= 3 && suggestions[keyNum - 1]) {
      event.preventDefault();
      handleChipClick(suggestions[keyNum - 1]);
    }
  }

  onMount(() => {
    window.addEventListener("keydown", handleKeydown);
  });

  onDestroy(() => {
    window.removeEventListener("keydown", handleKeydown);
    if (toastTimeout) {
      clearTimeout(toastTimeout);
    }
  });
</script>

{#if suggestions.length > 0}
  <div class="smart-reply-container" class:visible>
    <div class="chips-wrapper">
      {#each suggestions as suggestion, index (suggestion.text)}
        <button
          class="chip"
          onclick={() => handleChipClick(suggestion)}
          title="Press {index + 1} to copy"
          style="animation-delay: {index * 50}ms"
        >
          <span class="chip-text">{suggestion.text}</span>
          <span class="chip-shortcut">{index + 1}</span>
        </button>
      {/each}
    </div>
  </div>
{/if}

{#if toastMessage}
  <div class="toast">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M20 6L9 17l-5-5" />
    </svg>
    {toastMessage}
  </div>
{/if}

<style>
  .smart-reply-container {
    padding: 8px 16px 16px;
    opacity: 0;
    transform: translateY(10px);
    transition: opacity 0.2s ease, transform 0.2s ease;
  }

  .smart-reply-container.visible {
    opacity: 1;
    transform: translateY(0);
  }

  .chips-wrapper {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: flex-start;
  }

  .chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    background: var(--bg-secondary);
    border: 1px solid var(--accent-color);
    border-radius: 20px;
    color: var(--accent-color);
    font-size: 14px;
    cursor: pointer;
    transition: all 0.15s ease;
    opacity: 0;
    transform: translateY(8px);
    animation: chipFadeIn 0.25s ease forwards;
  }

  @keyframes chipFadeIn {
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .chip:hover {
    background: var(--accent-color);
    color: white;
  }

  .chip:active {
    transform: scale(0.98);
  }

  .chip-text {
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .chip-shortcut {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    opacity: 0.7;
  }

  .chip:hover .chip-shortcut {
    background: rgba(255, 255, 255, 0.25);
    opacity: 1;
  }

  .toast {
    position: fixed;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 20px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 14px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    animation: toastFadeIn 0.2s ease;
    z-index: 1000;
  }

  .toast svg {
    width: 16px;
    height: 16px;
    color: #34c759;
  }

  @keyframes toastFadeIn {
    from {
      opacity: 0;
      transform: translateX(-50%) translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }
  }
</style>
