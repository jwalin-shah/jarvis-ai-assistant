<script lang="ts">
  import { onMount, type Snippet } from 'svelte';
  import { announce } from '../stores/keyboard';

  interface Props {
    children: Snippet;
    onError?: (error: Error, componentStack: string) => void;
    fallback?: Snippet<[Error]>;
    context?: string; // Component context for better error messages
    allowReset?: boolean; // Whether to show reset button
    onReset?: () => void; // Callback when user clicks reset
  }

  let { children, onError, fallback, context = 'component', allowReset = true, onReset }: Props = $props();

  let error = $state<Error | null>(null);
  let componentStack = $state('');
  let errorId = $state('');

  // Error categorization for better UX
  const ERROR_CATEGORIES: Record<string, { message: string; suggestion: string }> = {
    'NetworkError': {
      message: 'Connection issue detected',
      suggestion: 'Check your internet connection and try again.'
    },
    'TypeError': {
      message: 'Something unexpected happened',
      suggestion: 'This might be a temporary issue. Please try again.'
    },
    'RangeError': {
      message: 'Data processing error',
      suggestion: 'The data might be too large. Try refreshing the view.'
    },
    'VirtualScrollError': {
      message: 'Display error',
      suggestion: 'Try scrolling to reset the view, or reload the page.'
    },
    'default': {
      message: 'Something went wrong',
      suggestion: 'Please try again or reload the page if the issue persists.'
    }
  };

  onMount(() => {
    const handleError = (event: ErrorEvent) => {
      error = event.error;
      componentStack = event.error?.stack || '';
      errorId = generateErrorId();
      
      // Log error for debugging
      console.error(`[ErrorBoundary${context ? `:${context}` : ''}]`, {
        error: event.error,
        errorId,
        timestamp: new Date().toISOString(),
      });
      
      // Announce error to screen readers
      announce(`Error: ${getErrorMessage(event.error)}`);
      
      onError?.(event.error, componentStack);
      event.preventDefault();
    };

    // Handle unhandled promise rejections
    const handleRejection = (event: PromiseRejectionEvent) => {
      console.error('[ErrorBoundary] Unhandled promise rejection:', event.reason);
    };

    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleRejection);
    
    return () => {
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleRejection);
    };
  });

  function generateErrorId(): string {
    return `err_${Date.now().toString(36)}_${Math.random().toString(36).substr(2, 5)}`;
  }

  function getErrorCategory(err: Error): string {
    const name = err?.name || 'Error';
    if (name in ERROR_CATEGORIES) return name;
    if (err?.message?.includes('virtual') || err?.message?.includes('scroll')) {
      return 'VirtualScrollError';
    }
    return 'default';
  }

  function getErrorMessage(err: Error): string {
    return ERROR_CATEGORIES[getErrorCategory(err)]?.message || ERROR_CATEGORIES.default.message;
  }

  function getErrorSuggestion(err: Error): string {
    return ERROR_CATEGORIES[getErrorCategory(err)]?.suggestion || ERROR_CATEGORIES.default.suggestion;
  }

  function reset() {
    error = null;
    componentStack = '';
    onReset?.();
    announce('Error cleared, retrying');
  }

  function handleReload() {
    window.location.reload();
  }
</script>

{#if error}
  {#if fallback}
    {@render fallback(error)}
  {:else}
    <div class="error-boundary" role="alert" aria-live="polite">
      <div class="error-icon">⚠️</div>
      <h2>{getErrorMessage(error)}</h2>
      <p class="error-suggestion">{getErrorSuggestion(error)}</p>
      <p class="error-message">{error.message}</p>
      {#if errorId}
        <p class="error-id">Error ID: {errorId}</p>
      {/if}
      <details class="error-details">
        <summary>Technical Details</summary>
        <pre>{componentStack}</pre>
      </details>
      <div class="error-actions">
        {#if allowReset}
          <button class="retry-btn" onclick={reset}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
              <polyline points="23 4 23 10 17 10"></polyline>
              <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
            </svg>
            Try Again
          </button>
        {/if}
        <button class="reload-btn" onclick={handleReload}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
            <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path>
            <path d="M21 3v5h-5"></path>
            <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path>
            <path d="M3 21v-5h5"></path>
          </svg>
          Reload Page
        </button>
      </div>
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

  .error-suggestion {
    color: var(--text-secondary);
    font-size: var(--text-base);
    margin-bottom: var(--space-2);
    max-width: 400px;
  }

  .error-id {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    font-family: var(--font-family-mono, monospace);
    margin-bottom: var(--space-4);
  }

  .error-actions {
    display: flex;
    gap: var(--space-3);
    margin-top: var(--space-4);
  }

  .retry-btn,
  .reload-btn {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-4);
    border: none;
    border-radius: var(--radius-md);
    cursor: pointer;
    font-weight: var(--font-weight-medium);
    transition: opacity var(--duration-fast) var(--ease-out);
  }

  .retry-btn {
    background: var(--color-primary);
    color: white;
  }

  .reload-btn {
    background: var(--surface-base);
    color: var(--text-primary);
    border: 1px solid var(--border-default);
  }

  .retry-btn:hover,
  .reload-btn:hover {
    opacity: 0.9;
  }

  .retry-btn svg,
  .reload-btn svg {
    flex-shrink: 0;
  }
</style>
