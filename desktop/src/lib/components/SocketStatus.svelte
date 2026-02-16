<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { jarvis } from '../socket';
  import type { ConnectionInfo } from '../socket';

  let connectionInfo = $state<ConnectionInfo>(jarvis.getConnectionInfo());
  let unsubConnectionInfo: (() => void) | null = null;
  let showDetails = $state(false);

  onMount(() => {
    unsubConnectionInfo = jarvis.on<ConnectionInfo>('connection_info_changed', (info) => {
      connectionInfo = info;
    });
  });

  onDestroy(() => {
    unsubConnectionInfo?.();
  });

  function reconnect() {
    void jarvis.connect();
  }

  function getStatusIcon(state: string) {
    switch (state) {
      case 'connected':
        return '✅';
      case 'connecting':
        return '⏳';
      case 'disconnected':
        return '❌';
      default:
        return '⚪';
    }
  }

  function getStatusColor(state: string) {
    switch (state) {
      case 'connected':
        return 'var(--color-success)';
      case 'connecting':
        return 'var(--color-warning)';
      case 'disconnected':
        return 'var(--color-error)';
      default:
        return 'var(--text-secondary)';
    }
  }
</script>

<div class="socket-status" class:connected={connectionInfo.state === 'connected'}>
  <button class="status-btn" onclick={() => showDetails = !showDetails}>
    <span class="status-icon" style="color: {getStatusColor(connectionInfo.state)}">
      {getStatusIcon(connectionInfo.state)}
    </span>
    <span class="status-text">
      Socket {connectionInfo.state}
    </span>
  </button>

  {#if showDetails}
    <div class="details-popup">
      <div class="detail-row">
        <span class="detail-label">State:</span>
        <span class="detail-value" style="color: {getStatusColor(connectionInfo.state)}">
          {connectionInfo.state}
        </span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Transport:</span>
        <span class="detail-value">{connectionInfo.transport}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Fallback:</span>
        <span class="detail-value">{connectionInfo.isFallback ? 'yes' : 'no'}</span>
      </div>

      {#if connectionInfo.state !== 'connected'}
        <button class="reconnect-btn" onclick={reconnect}>
          Reconnect
        </button>
      {/if}

      <div class="help-text">
        {#if connectionInfo.state === 'disconnected'}
          <p>Socket disconnected. Click reconnect or restart the app.</p>
        {:else if connectionInfo.state === 'connecting'}
          <p>Connecting to socket server...</p>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .socket-status {
    position: relative;
  }

  .status-btn {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-1) var(--space-2);
    background: var(--surface-base);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    cursor: pointer;
    font-size: var(--text-sm);
  }

  .status-btn:hover {
    background: var(--surface-elevated);
  }

  .status-icon {
    font-size: var(--text-sm);
  }

  .status-text {
    text-transform: capitalize;
  }

  .latency {
    color: var(--text-secondary);
    font-size: var(--text-xs);
  }

  .details-popup {
    position: absolute;
    top: 100%;
    right: 0;
    margin-top: var(--space-2);
    padding: var(--space-4);
    background: var(--surface-elevated);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-lg);
    min-width: 250px;
    z-index: 100;
  }

  .detail-row {
    display: flex;
    justify-content: space-between;
    padding: var(--space-1) 0;
    font-size: var(--text-sm);
  }

  .detail-label {
    color: var(--text-secondary);
  }

  .detail-row.error .detail-value {
    color: var(--color-error);
  }

  .reconnect-btn {
    width: 100%;
    margin-top: var(--space-3);
    padding: var(--space-2);
    background: var(--color-primary);
    color: white;
    border: none;
    border-radius: var(--radius-md);
    cursor: pointer;
    font-size: var(--text-sm);
  }

  .reconnect-btn:hover {
    opacity: 0.9;
  }

  .help-text {
    margin-top: var(--space-3);
    padding-top: var(--space-3);
    border-top: 1px solid var(--border-color);
  }

  .help-text p {
    font-size: var(--text-xs);
    color: var(--text-secondary);
    margin: 0;
  }
</style>
