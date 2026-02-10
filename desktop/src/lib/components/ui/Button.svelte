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
    gap: 8px;
    font-weight: 500;
    border-radius: 8px;
    transition:
      transform 150ms cubic-bezier(0.33, 1, 0.68, 1),
      box-shadow 150ms cubic-bezier(0.33, 1, 0.68, 1),
      background 150ms cubic-bezier(0.33, 1, 0.68, 1),
      border-color 150ms cubic-bezier(0.33, 1, 0.68, 1),
      opacity 150ms cubic-bezier(0.33, 1, 0.68, 1);
    cursor: pointer;
    border: none;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', Roboto,
      Helvetica, Arial, sans-serif;
    line-height: 1.25;
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
    background: var(--color-primary, #007aff);
    color: white;
  }

  .btn-primary:hover:not(:disabled) {
    background: var(--color-primary-hover, #0056b3);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  /* Secondary */
  .btn-secondary {
    background: var(--surface-elevated, #141414);
    color: var(--text-primary, #ffffff);
    border: 1px solid var(--border-default, rgba(255, 255, 255, 0.12));
  }

  .btn-secondary:hover:not(:disabled) {
    background: var(--surface-hover, #2c2c2e);
    border-color: var(--color-primary, #007aff);
  }

  /* Ghost */
  .btn-ghost {
    background: transparent;
    color: var(--text-secondary, rgba(255, 255, 255, 0.6));
  }

  .btn-ghost:hover:not(:disabled) {
    background: var(--surface-hover, #2c2c2e);
    color: var(--text-primary, #ffffff);
  }

  /* Danger */
  .btn-danger {
    background: var(--color-error, #ff3b30);
    color: white;
  }

  .btn-danger:hover:not(:disabled) {
    background: #e6352b;
    box-shadow: 0 4px 12px rgba(255, 59, 48, 0.3);
  }

  /* Sizes */
  .btn-sm {
    padding: 8px 12px;
    font-size: 13px;
    min-height: 32px;
  }

  .btn-md {
    padding: 12px 16px;
    font-size: 15px;
    min-height: 40px;
  }

  .btn-lg {
    padding: 16px 20px;
    font-size: 17px;
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
