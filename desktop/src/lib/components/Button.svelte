<script lang="ts">
  import type { HTMLButtonAttributes } from 'svelte/elements';
  import type { Snippet } from 'svelte';

  interface Props extends HTMLButtonAttributes {
    variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
    size?: 'sm' | 'md' | 'lg';
    loading?: boolean;
    fullWidth?: boolean;
    children: Snippet;
  }

  let {
    variant = 'secondary',
    size = 'md',
    loading = false,
    fullWidth = false,
    children,
    class: className = '',
    disabled,
    ...rest
  }: Props = $props();

  const variantClasses = {
    primary: 'btn-primary',
    secondary: 'btn-secondary',
    ghost: 'btn-ghost',
    danger: 'btn-danger',
  };

  const sizeClasses = {
    sm: 'btn-sm',
    md: 'btn-md',
    lg: 'btn-lg',
  };
</script>

<button
  class="btn {variantClasses[variant]} {sizeClasses[size]} {className}"
  class:loading
  class:full-width={fullWidth}
  disabled={disabled || loading}
  {...rest}
>
  {#if loading}
    <span class="spinner" aria-hidden="true"></span>
  {/if}
  {@render children()}
</button>

<style>
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-2);
    font-weight: var(--font-weight-medium);
    border-radius: var(--radius-md);
    transition:
      transform var(--duration-fast) var(--ease-out),
      box-shadow var(--duration-fast) var(--ease-out),
      background var(--duration-fast) var(--ease-out),
      border-color var(--duration-fast) var(--ease-out),
      opacity var(--duration-fast) var(--ease-out);
    cursor: pointer;
    border: none;
    font-family: var(--font-family-sans);
    line-height: var(--line-height-tight);
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn:not(:disabled):hover {
    transform: translateY(-1px);
  }

  .btn:not(:disabled):active {
    transform: scale(0.98);
  }

  /* Primary */
  .btn-primary {
    background: var(--color-primary);
    color: white;
  }

  .btn-primary:hover:not(:disabled) {
    background: var(--color-primary-hover);
    box-shadow: var(--shadow-md);
  }

  /* Secondary */
  .btn-secondary {
    background: var(--surface-elevated);
    color: var(--text-primary);
    border: 1px solid var(--border-default);
  }

  .btn-secondary:hover:not(:disabled) {
    background: var(--surface-hover);
    border-color: var(--color-primary);
  }

  /* Ghost */
  .btn-ghost {
    background: transparent;
    color: var(--text-secondary);
  }

  .btn-ghost:hover:not(:disabled) {
    background: var(--surface-hover);
    color: var(--text-primary);
  }

  /* Danger */
  .btn-danger {
    background: var(--color-error);
    color: white;
  }

  .btn-danger:hover:not(:disabled) {
    background: #e6352b;
    box-shadow: 0 4px 12px rgba(255, 59, 48, 0.3);
  }

  /* Sizes */
  .btn-sm {
    padding: var(--space-2) var(--space-3);
    font-size: var(--text-sm);
    min-height: 32px;
  }

  .btn-md {
    padding: var(--space-3) var(--space-4);
    font-size: var(--text-base);
    min-height: 40px;
  }

  .btn-lg {
    padding: var(--space-4) var(--space-5);
    font-size: var(--text-lg);
    min-height: 48px;
  }

  /* Full width */
  .full-width {
    width: 100%;
  }

  /* Loading spinner */
  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid currentColor;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  /* Reduce motion */
  :global(:root.reduce-motion) .spinner {
    animation: none;
    border: 2px solid currentColor;
    border-top-color: transparent;
    opacity: 0.5;
  }
</style>
