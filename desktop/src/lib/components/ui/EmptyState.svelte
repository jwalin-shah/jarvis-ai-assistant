<script lang="ts">
  import type { Snippet } from 'svelte';
  import Button from './Button.svelte';

  interface Props {
    title: string;
    description?: string;
    icon: Snippet;
    action?: {
      label: string;
      onClick: () => void;
      variant?: 'primary' | 'secondary';
    };
  }

  let { title, description, icon, action }: Props = $props();
</script>

<div class="empty-state" role="status">
  <div class="icon-wrapper">
    {@render icon()}
  </div>
  <h3>{title}</h3>
  {#if description}
    <p>{description}</p>
  {/if}
  {#if action}
    <div class="action">
      <Button onclick={action.onClick} variant={action.variant ?? 'primary'}>
        {action.label}
      </Button>
    </div>
  {/if}
</div>

<style>
  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-3);
    color: var(--text-secondary);
    padding: var(--space-8);
    text-align: center;
    min-height: 300px;
  }

  .icon-wrapper {
    width: 64px;
    height: 64px;
    border-radius: var(--radius-xl);
    background: var(--surface-elevated);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-tertiary);
    margin-bottom: var(--space-2);
  }

  .icon-wrapper :global(svg) {
    width: 32px;
    height: 32px;
    stroke-width: 1.5;
  }

  h3 {
    font-size: var(--text-lg);
    font-weight: var(--font-weight-semibold);
    color: var(--text-primary);
    letter-spacing: var(--letter-spacing-tight);
    margin: 0;
  }

  p {
    font-size: var(--text-base);
    max-width: 280px;
    line-height: var(--line-height-normal);
    margin: 0;
  }

  .action {
    margin-top: var(--space-3);
  }
</style>
