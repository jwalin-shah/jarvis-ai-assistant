<script lang="ts">
  import { onMount, type Snippet } from "svelte";
  import { toast } from "../stores/toast";
  import Icon from "./Icon.svelte";

  interface Props {
    children?: Snippet;
  }

  let { children }: Props = $props();

  let hasError = $state(false);
  let errorMessage = $state("");
  let errorStack = $state("");

  function handleError(event: ErrorEvent): void {
    hasError = true;
    errorMessage = event.error?.message || event.message || "Unknown error";
    errorStack = event.error?.stack || "";

    // Show toast notification
    toast.error("An unexpected error occurred", {
      description: errorMessage,
      duration: 8000,
    });

    // Prevent default browser error handling
    event.preventDefault();
  }

  function handleUnhandledRejection(event: PromiseRejectionEvent): void {
    hasError = true;
    const error = event.reason;
    errorMessage = error?.message || String(error) || "Unhandled promise rejection";
    errorStack = error?.stack || "";

    // Show toast notification
    toast.error("An unexpected error occurred", {
      description: errorMessage,
      duration: 8000,
    });

    // Prevent default browser error handling
    event.preventDefault();
  }

  function retry(): void {
    hasError = false;
    errorMessage = "";
    errorStack = "";
    location.reload();
  }

  function dismiss(): void {
    hasError = false;
    errorMessage = "";
    errorStack = "";
  }

  onMount(() => {
    window.addEventListener("error", handleError);
    window.addEventListener("unhandledrejection", handleUnhandledRejection);

    return () => {
      window.removeEventListener("error", handleError);
      window.removeEventListener("unhandledrejection", handleUnhandledRejection);
    };
  });
</script>

{#if hasError}
  <div class="error-boundary" role="alert">
    <div class="error-content">
      <div class="error-icon">
        <Icon name="alert-circle" size={48} />
      </div>
      <h2>Something went wrong</h2>
      <p class="error-message">{errorMessage}</p>
      {#if errorStack}
        <details class="error-details">
          <summary>Technical details</summary>
          <pre>{errorStack}</pre>
        </details>
      {/if}
      <div class="error-actions">
        <button class="btn-primary" onclick={retry}>
          <Icon name="refresh-cw" size={16} />
          Reload App
        </button>
        <button class="btn-secondary" onclick={dismiss}>
          Dismiss
        </button>
      </div>
    </div>
  </div>
{/if}

{#if children}
  {@render children()}
{/if}

<style>
  .error-boundary {
    position: fixed;
    inset: 0;
    z-index: var(--z-error);
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.85);
    backdrop-filter: blur(var(--blur-overlay));
    -webkit-backdrop-filter: blur(var(--blur-overlay));
  }

  .error-content {
    max-width: 480px;
    padding: 32px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-xl);
    text-align: center;
  }

  .error-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 80px;
    height: 80px;
    margin: 0 auto var(--space-5);
    background: var(--color-error-bg-hover);
    border-radius: var(--radius-full);
    color: var(--color-error);
  }

  h2 {
    margin: 0 0 12px;
    font-size: var(--text-xl);
    font-weight: 600;
    color: var(--text-primary);
  }

  .error-message {
    margin: 0 0 20px;
    font-size: var(--text-sm);
    color: var(--text-secondary);
    word-break: break-word;
  }

  .error-details {
    margin-bottom: 20px;
    text-align: left;
  }

  .error-details summary {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    cursor: pointer;
    padding: 8px;
  }

  .error-details pre {
    margin: 8px 0 0;
    padding: 12px;
    background: var(--bg-surface);
    border-radius: var(--radius-md);
    font-size: 11px;
    color: var(--text-secondary);
    overflow: auto;
    max-height: 200px;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .error-actions {
    display: flex;
    gap: 12px;
    justify-content: center;
  }

  .btn-primary,
  .btn-secondary {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    font-size: var(--text-sm);
    font-weight: 500;
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .btn-primary {
    background: var(--color-primary);
    border: none;
    color: white;
  }

  .btn-primary:hover {
    filter: brightness(1.1);
  }

  .btn-secondary {
    background: transparent;
    border: 1px solid var(--border-default);
    color: var(--text-secondary);
  }

  .btn-secondary:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
</style>
