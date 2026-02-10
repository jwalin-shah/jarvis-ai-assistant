<script lang="ts">
  interface Props {
    date: string;
  }

  let { date }: Props = $props();

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    if (date.toDateString() === today.toDateString()) {
      return 'Today';
    } else if (date.toDateString() === yesterday.toDateString()) {
      return 'Yesterday';
    } else {
      return date.toLocaleDateString([], {
        weekday: 'long',
        month: 'long',
        day: 'numeric',
      });
    }
  }
</script>

<div class="date-header">
  <span>{formatDate(date)}</span>
</div>

<style>
  .date-header {
    text-align: center;
    margin: var(--space-4) 0 var(--space-2);
  }

  .date-header span {
    background: var(--surface-elevated);
    padding: var(--space-1) var(--space-3);
    border-radius: var(--radius-full);
    font-size: var(--text-xs);
    color: var(--text-secondary);
    font-weight: var(--font-weight-medium);
  }
</style>
