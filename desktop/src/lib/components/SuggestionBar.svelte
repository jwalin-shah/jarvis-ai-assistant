<script lang="ts">
  import { apiClient, APIError } from "../api/client";
  import type { DraftReplyResponse } from "../api/types";
  import { jarvis } from "../socket/client";
  import Icon from "./Icon.svelte";

  interface Props {
    chatId: string;
    onSelect: (text: string) => void;
    onAccept?: (text: string) => void;
    onClose: () => void;
    initialSuggestions?: DraftReplyResponse["suggestions"];
  }

  let { chatId, onSelect, onAccept, onClose, initialSuggestions }: Props = $props();

  type BarState = "loading" | "streaming" | "results" | "error";
  let barState: BarState = $state("loading");
  let streamingText = $state("");
  let suggestions: DraftReplyResponse["suggestions"] = $state([]);
  let errorMessage = $state("");
  let abortController: AbortController | null = null;
  let runId = 0;

  // Allow more time for model loading + generation on Apple Silicon
  const STREAM_TIMEOUT_MS = 45000;
  const FALLBACK_TIMEOUT_MS = 30000;

  function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`${label} timed out`)), ms);
      promise.then(
        (value) => {
          clearTimeout(timer);
          resolve(value);
        },
        (err) => {
          clearTimeout(timer);
          reject(err);
        }
      );
    });
  }

  async function generateReplies() {
    const currentRun = ++runId;
    abortController?.abort();
    abortController = new AbortController();

    barState = "streaming";
    streamingText = "";
    errorMessage = "";

    try {
      // Stream tokens via socket for typewriter effect
      const result = await withTimeout(
        jarvis.generateDraftStream(
          { chat_id: chatId },
          (token: string) => {
            if (currentRun !== runId) return;
            streamingText += token;
          }
        ),
        STREAM_TIMEOUT_MS,
        "AI draft stream"
      );
      if (currentRun !== runId) return;
      if (result.gated || result.suggestions.length === 0) {
        throw new Error("No draft suggestions returned from stream");
      }
      suggestions = result.suggestions;
      barState = "results";
    } catch {
      if (currentRun !== runId) return;
      // Fall back to non-streaming API
      try {
        barState = "loading";
        const safeResponse = await withTimeout(
          apiClient.getDraftReplies(
            chatId,
            undefined,
            3,
            abortController!.signal,
            { preferSocket: false }
          ),
          FALLBACK_TIMEOUT_MS,
          "AI draft fallback"
        );
        if (currentRun !== runId) return;
        if (safeResponse.gated || safeResponse.suggestions.length === 0) {
          handleClose();
          return;
        }
        suggestions = safeResponse.suggestions;
        barState = "results";
      } catch (e2) {
        if (currentRun !== runId) return;
        if (e2 instanceof Error && e2.name === "AbortError") return;
        barState = "error";
        if (e2 instanceof APIError) {
          errorMessage = e2.detail || e2.message;
        } else if (e2 instanceof Error) {
          errorMessage = e2.message;
        } else {
          errorMessage = "Failed to generate suggestions";
        }
      }
    }
  }

  function handleClose() {
    abortController?.abort();
    streamingText = "";
    barState = "loading";
    onClose();
  }

  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === "Escape") {
      event.preventDefault();
      handleClose();
    }
  }

  function truncate(text: string, maxLen: number): string {
    if (text.length <= maxLen) return text;
    return text.slice(0, maxLen) + "...";
  }

  // Auto-generate on mount and re-generate when chatId changes.
  // If initialSuggestions were provided (from prefetch), use them directly.
  $effect(() => {
    void chatId; // track chatId reactivity
    if (initialSuggestions && initialSuggestions.length > 0) {
      suggestions = initialSuggestions;
      barState = "results";
    } else {
      generateReplies();
    }
    return () => {
      abortController?.abort();
    };
  });
</script>

<svelte:window onkeydown={handleKeyDown} />

<div class="suggestion-bar">
  <span class="bar-icon"><Icon name="sparkles" size={16} /></span>

  {#if barState === "streaming"}
    <button 
      class="bar-content streaming" 
      onclick={() => {
        if (streamingText) {
          onSelect(streamingText);
          onAccept?.(streamingText);
        }
      }}
      disabled={!streamingText}
      title="Click to use current draft"
    >
      {#if streamingText}
        <span class="streaming-text">{streamingText}<span class="cursor"></span></span>
      {:else}
        <div class="typing-indicator">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
        <span class="loading-text">AI is thinking...</span>
      {/if}
    </button>
  {/if}

  {#if barState === "loading"}
    <div class="bar-content loading">
      <div class="typing-indicator">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
      <span class="loading-text">Generating suggestions...</span>
    </div>
  {/if}

  {#if barState === "results"}
    <div class="bar-content chips">
      {#each suggestions as suggestion}
        <button
          class="chip"
          style:opacity={suggestion.confidence < 0.7 ? 0.85 : 1}
          onclick={() => {
            onSelect(suggestion.text);
            onAccept?.(suggestion.text);
          }}
          title="{suggestion.text} (confidence: {Math.round(suggestion.confidence * 100)}%)"
        >
          {truncate(suggestion.text, 100)}
        </button>
      {/each}
    </div>
    <button
      class="bar-btn"
      onclick={generateReplies}
      title="Regenerate suggestions"
      aria-label="Regenerate suggestions"
    >
      <Icon name="refresh-cw" size={14} />
    </button>
  {/if}

  {#if barState === "error"}
    <div class="bar-content error">
      <Icon name="alert-circle" size={14} />
      <span class="error-text">{errorMessage}</span>
      <button class="bar-btn" onclick={generateReplies} title="Retry">
        <Icon name="refresh-cw" size={14} />
      </button>
    </div>
  {/if}

  <div class="bar-actions">
    {#if barState === "results" || barState === "error"}
      <button
        class="bar-btn"
        onclick={generateReplies}
        title="Regenerate suggestions"
        aria-label="Regenerate suggestions"
      >
        <Icon name="refresh-cw" size={14} />
      </button>
    {/if}

    <button
      class="bar-btn close"
      onclick={handleClose}
      title="Close (Esc)"
      aria-label="Close suggestion bar"
    >
      <Icon name="x-circle" size={14} />
    </button>
  </div>
</div>

<style>
  .suggestion-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border-color);
    border-bottom: 1px solid var(--border-color);
    animation: slideUp 0.15s ease;
    overflow: hidden;
    position: relative;
    z-index: 50;
    pointer-events: auto;
  }

  @keyframes slideUp {
    from {
      opacity: 0;
      transform: translateY(8px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .bar-icon {
    display: flex;
    align-items: center;
    color: var(--accent-color);
    flex-shrink: 0;
  }

  .bar-content {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
    min-width: 0;
    overflow-x: auto;
  }

  .bar-content.streaming {
    font-size: 13px;
    color: var(--text-primary);
    overflow: hidden;
    background: none;
    border: none;
    padding: 0;
    margin: 0;
    cursor: pointer;
    text-align: left;
    font-family: inherit;
    width: 100%;
  }

  .bar-content.streaming:disabled {
    cursor: default;
  }

  .streaming-text {
    white-space: pre-wrap;
    word-break: break-word;
  }

  .cursor {
    display: inline-block;
    width: 2px;
    height: 1em;
    background: var(--accent-color);
    margin-left: 1px;
    vertical-align: text-bottom;
    animation: blink 0.6s step-end infinite;
  }

  @keyframes blink {
    50% {
      opacity: 0;
    }
  }

  .bar-content.loading {
    color: var(--text-secondary);
    font-size: 13px;
  }

  .bar-content.chips {
    gap: 6px;
  }

  .bar-content.error {
    color: var(--error-color, #ff3b30);
    font-size: 13px;
  }


  .loading-text,
  .error-text {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .chip {
    flex-shrink: 0;
    max-width: 300px;
    padding: 6px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    color: var(--text-primary);
    font-size: 13px;
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: all 0.15s ease;
  }

  .chip:hover {
    background: var(--bg-hover);
    border-color: var(--accent-color);
  }

  .bar-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: none;
    border: none;
    border-radius: 6px;
    color: var(--text-secondary);
    cursor: pointer;
    flex-shrink: 0;
    transition: all 0.15s ease;
  }

  .bar-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .bar-actions {
    display: flex;
    align-items: center;
    gap: 4px;
    flex-shrink: 0;
  }

  /* Hide scrollbar on the chips container */
  .bar-content.chips {
    scrollbar-width: none;
    -ms-overflow-style: none;
  }
  .bar-content.chips::-webkit-scrollbar {
    display: none;
  }

  .typing-indicator {
    display: flex;
    align-items: center;
    gap: 3px;
    padding: 0 4px;
  }

  .typing-dot {
    width: 6px;
    height: 6px;
    background: var(--text-secondary);
    border-radius: 50%;
    animation: typingBounce 1.4s ease-in-out infinite;
  }

  .typing-dot:nth-child(2) {
    animation-delay: 0.2s;
  }

  .typing-dot:nth-child(3) {
    animation-delay: 0.4s;
  }

  @keyframes typingBounce {
    0%, 60%, 100% {
      transform: translateY(0);
      opacity: 0.4;
    }
    30% {
      transform: translateY(-4px);
      opacity: 1;
    }
  }
</style>
