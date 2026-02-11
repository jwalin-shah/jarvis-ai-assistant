<script lang="ts">
  import { toasts, dismissToast, type ToastType } from "../stores/toast";
  import Icon from "./Icon.svelte";

  function getIcon(type: ToastType): "check" | "x-circle" | "alert-circle" | "message-circle" {
    switch (type) {
      case "success":
        return "check";
      case "error":
        return "x-circle";
      case "warning":
        return "alert-circle";
      default:
        return "message-circle";
    }
  }
</script>

<div class="toast-container" role="region" aria-label="Notifications">
  {#each $toasts as toast, i (toast.id)}
    <div
      class="toast toast-{toast.type}"
      role="alert"
      style="--index: {i}"
    >
      <div class="toast-icon">
        <Icon name={getIcon(toast.type)} size={18} />
      </div>
      <div class="toast-content">
        <p class="toast-message">{toast.message}</p>
        {#if toast.description}
          <p class="toast-description">{toast.description}</p>
        {/if}
      </div>
      {#if toast.action}
        <button class="toast-action" onclick={toast.action.onClick}>
          {toast.action.label}
        </button>
      {/if}
      {#if toast.dismissible}
        <button
          class="toast-dismiss"
          onclick={() => dismissToast(toast.id)}
          aria-label="Dismiss"
        >
          <Icon name="x-circle" size={16} />
        </button>
      {/if}
    </div>
  {/each}
</div>

<style>
  .toast-container {
    position: fixed;
    top: 24px;
    right: 24px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    z-index: 9999;
    pointer-events: none;
    max-width: 400px;
  }

  .toast {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 14px 16px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-lg);
    pointer-events: auto;
    animation: toastEnter 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
  }

  @keyframes toastEnter {
    from {
      opacity: 0;
      transform: translateY(-100%) scale(0.9);
    }
    to {
      opacity: 1;
      transform: translateY(0) scale(1);
    }
  }

  .toast-icon {
    flex-shrink: 0;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-full);
  }

  .toast-success .toast-icon {
    background: rgba(52, 199, 89, 0.15);
    color: var(--color-success);
  }

  .toast-error .toast-icon {
    background: rgba(255, 59, 48, 0.15);
    color: var(--color-error);
  }

  .toast-warning .toast-icon {
    background: rgba(255, 149, 0, 0.15);
    color: var(--color-warning);
  }

  .toast-info .toast-icon {
    background: rgba(0, 122, 255, 0.15);
    color: var(--color-primary);
  }

  .toast-content {
    flex: 1;
    min-width: 0;
  }

  .toast-message {
    font-size: var(--text-sm);
    font-weight: 500;
    color: var(--text-primary);
    line-height: 1.4;
  }

  .toast-description {
    font-size: var(--text-xs);
    color: var(--text-secondary);
    margin-top: 4px;
    line-height: 1.4;
  }

  .toast-action {
    flex-shrink: 0;
    padding: 6px 12px;
    background: var(--color-primary);
    border: none;
    border-radius: var(--radius-sm);
    color: white;
    font-size: var(--text-xs);
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .toast-action:hover {
    filter: brightness(1.1);
  }

  .toast-dismiss {
    flex-shrink: 0;
    padding: 4px;
    background: transparent;
    border: none;
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
    cursor: pointer;
    transition: all 0.15s ease;
    opacity: 0;
  }

  .toast:hover .toast-dismiss {
    opacity: 1;
  }

  .toast-dismiss:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }

  /* Stagger animation for multiple toasts */
  .toast:nth-child(1) { animation-delay: 0ms; }
  .toast:nth-child(2) { animation-delay: 50ms; }
  .toast:nth-child(3) { animation-delay: 100ms; }

  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) {
    .toast {
      animation: none;
    }
  }
</style>
