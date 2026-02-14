<script lang="ts">
  import { onMount, type Snippet } from 'svelte';

  interface Props {
    children: Snippet;
    onError?: (error: Error, componentStack: string) => void;
    fallback?: Snippet<[Error]>;
  }

  let { children, onError, fallback }: Props = $props();

  let error = $state<Error | null>(null);
  let componentStack = $state('');

  onMount(() => {
    const handleError = (event: ErrorEvent) => {
      error = event.error;
      componentStack = event.error?.stack || '';
      onError?.(event.error, componentStack);
      event.preventDefault();
    };

    window.addEventListener('error', handleError);
    return () => window.removeEventListener('error', handleError);
  });

  function reset() {
    error = null;
    componentStack = '';
  }
</script>

{#if error}
  {#if fallback}
    {@render fallback(error)}
  {:else}
    <div class="error-boundary" role="alert">
      <div class="error-icon">⚠️</div>
      <h2>Something went wrong</h2>
      <p class="error-message">{error.message}</p>
      <details class="error-details">
        <summary>Technical Details</summary>
        <pre>{componentStack}</pre>
      </details>
      <button class="retry-btn" onclick={reset}>Try Again</button>
    </div>
  {/if}
{:else}
  {@render children()}
{/if}

<style>
  .error-boundary {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: var(--space-8);
    text-align: center;
    background: var(--surface-elevated);
    border-radius: var(--radius-lg);
    margin: var(--space-4);
  }

  .error-icon {
    font-size: 3rem;
    margin-bottom: var(--space-4);
  }

  h2 {
    color: var(--text-primary);
    margin-bottom: var(--space-2);
  }

  .error-message {
    color: var(--text-secondary);
    margin-bottom: var(--space-4);
    max-width: 400px;
  }

  .error-details {
    margin: var(--space-4) 0;
    width: 100%;
    max-width: 600px;
  }

  .error-details summary {
    cursor: pointer;
    color: var(--text-secondary);
    font-size: var(--text-sm);
  }

  .error-details pre {
    text-align: left;
    background: var(--surface-base);
    padding: var(--space-4);
    border-radius: var(--radius-md);
    font-size: var(--text-xs);
    overflow-x: auto;
    max-height: 200px;
    overflow-y: auto;
    margin-top: var(--space-2);
  }

  .retry-btn {
    padding: var(--space-2) var(--space-4);
    background: var(--color-primary);
    color: white;
    border: none;
    border-radius: var(--radius-md);
    cursor: pointer;
    font-weight: var(--font-weight-medium);
  }

  .retry-btn:hover {
    opacity: 0.9;
  }
</style>
