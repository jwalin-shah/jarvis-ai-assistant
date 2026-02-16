<script lang="ts">
  import type { GraphNode, KnowledgeNode } from '../../api/types';

  interface Props {
    node: GraphNode | KnowledgeNode;
    x: number;
    y: number;
  }

  let { node, x, y }: Props = $props();

  // Check if this is a knowledge graph entity node
  let isEntity = $derived('node_type' in node && node.node_type === 'entity');

  let tooltipEl = $state<HTMLDivElement | null>(null);

  let tooltipStyle = $derived(
    (() => {
      let left = x + 15;
      let top = y - 10;
      if (tooltipEl) {
        const rect = tooltipEl.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        if (left + rect.width > vw - 8) left = x - rect.width - 15;
        if (top + rect.height > vh - 8) top = vh - rect.height - 8;
        if (top < 8) top = 8;
        if (left < 8) left = 8;
      }
      return `left: ${left}px; top: ${top}px;`;
    })()
  );

  function formatDate(dateStr: string | null): string {
    if (!dateStr) return 'N/A';
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      });
    } catch {
      return 'N/A';
    }
  }

  function formatSentiment(score: number): { text: string; color: string } {
    if (score >= 0.3) return { text: 'Positive', color: '#34c759' };
    if (score <= -0.3) return { text: 'Negative', color: '#ff3b30' };
    return { text: 'Neutral', color: '#ff9f0a' };
  }

  function formatResponseTime(minutes: number | null): string {
    if (minutes === null) return 'N/A';
    if (minutes < 60) return `${Math.round(minutes)} min`;
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  }

  let sentiment = $derived(
    !isEntity ? formatSentiment((node as GraphNode).sentiment_score) : { text: '', color: '' }
  );
</script>

<div class="tooltip" style={tooltipStyle} bind:this={tooltipEl}>
  <div class="tooltip-header">
    <div class="node-indicator" style="background: {node.color}"></div>
    <h3>{node.label}</h3>
  </div>

  <div class="tooltip-content">
    {#if isEntity}
      <!-- Entity node (knowledge graph) -->
      <div class="stat-row">
        <span class="stat-label">Type</span>
        <span class="stat-value capitalize"
          >{('category' in node && node.category) || 'Entity'}</span
        >
      </div>
      {#if 'metadata' in node && node.metadata.edge_type}
        <div class="stat-row">
          <span class="stat-label">Relationship</span>
          <span class="stat-value capitalize"
            >{String(node.metadata.edge_type).replace(/_/g, ' ')}</span
          >
        </div>
      {/if}
    {:else}
      <!-- Contact node -->
      {@const contactNode = node as GraphNode}
      <div class="stat-row">
        <span class="stat-label">Relationship</span>
        <span class="stat-value capitalize">{contactNode.relationship_type}</span>
      </div>

      <div class="stat-row">
        <span class="stat-label">Messages</span>
        <span class="stat-value">{contactNode.message_count.toLocaleString()}</span>
      </div>

      <div class="stat-row">
        <span class="stat-label">Sentiment</span>
        <span class="stat-value" style="color: {sentiment.color}">
          {(contactNode.sentiment_score * 100).toFixed(0)}% ({sentiment.text})
        </span>
      </div>

      {#if contactNode.response_time_avg !== null}
        <div class="stat-row">
          <span class="stat-label">Avg Response</span>
          <span class="stat-value">{formatResponseTime(contactNode.response_time_avg)}</span>
        </div>
      {/if}

      <div class="stat-row">
        <span class="stat-label">Last Contact</span>
        <span class="stat-value">{formatDate(contactNode.last_contact)}</span>
      </div>

      {#if contactNode.cluster_id !== null}
        <div class="stat-row">
          <span class="stat-label">Cluster</span>
          <span class="stat-value">#{contactNode.cluster_id + 1}</span>
        </div>
      {/if}
    {/if}
  </div>

  {#if !isEntity}
    {@const contactNode = node as GraphNode}
    {#if contactNode.metadata.sent !== undefined || contactNode.metadata.received !== undefined}
      <div class="tooltip-footer">
        <div class="message-breakdown">
          <span class="sent">Sent: {contactNode.metadata.sent ?? 0}</span>
          <span class="received">Received: {contactNode.metadata.received ?? 0}</span>
        </div>
      </div>
    {/if}
  {/if}
</div>

<style>
  .tooltip {
    position: fixed;
    z-index: var(--z-tooltip);
    padding: 12px 16px;
    background: rgba(40, 40, 40, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(8px);
    min-width: 200px;
    max-width: 280px;
    pointer-events: none;
    animation: fadeIn 0.15s ease-out;
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(5px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .tooltip-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }

  .node-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: #fff;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .tooltip-content {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
  }

  .stat-label {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.6);
  }

  .stat-value {
    font-size: 12px;
    font-weight: 500;
    color: #fff;
    text-align: right;
  }

  .stat-value.capitalize {
    text-transform: capitalize;
  }

  .tooltip-footer {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
  }

  .message-breakdown {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
  }

  .sent {
    color: #4ecdc4;
  }

  .received {
    color: #96ceb4;
  }
</style>
