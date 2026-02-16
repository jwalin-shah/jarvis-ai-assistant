<script lang="ts">
  import type { GraphNode } from '../../api/types';

  interface Props {
    nodes: GraphNode[];
    colors: Record<string, string>;
    onfilter?: (types: string[]) => void;
  }

  let { nodes, colors, onfilter }: Props = $props();

  let expanded = $state(true);
  let selectedTypes: Set<string> = $state(new Set());

  // Calculate relationship type distribution
  let relationshipCounts = $derived(
    nodes.reduce(
      (acc, node) => {
        const type = node.relationship_type;
        acc[type] = (acc[type] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    )
  );

  let sortedTypes = $derived(
    Object.entries(relationshipCounts)
      .sort((a, b) => b[1] - a[1])
      .map(([type, count]) => ({ type, count, color: colors[type] || colors.unknown }))
  );

  function toggleType(type: string) {
    if (selectedTypes.has(type)) {
      selectedTypes.delete(type);
    } else {
      selectedTypes.add(type);
    }
    selectedTypes = selectedTypes;
    onfilter?.(Array.from(selectedTypes));
  }

  function selectAll() {
    selectedTypes = new Set(sortedTypes.map((t) => t.type));
    onfilter?.(Array.from(selectedTypes));
  }

  function clearSelection() {
    selectedTypes = new Set();
    onfilter?.([]);
  }
</script>

<div class="legend" class:collapsed={!expanded}>
  <button class="legend-toggle" onclick={() => (expanded = !expanded)}>
    <span class="toggle-icon">{expanded ? 'v' : '>'}</span>
    <span class="toggle-text">Legend</span>
  </button>

  {#if expanded}
    <div class="legend-content">
      <div class="legend-header">
        <span class="legend-title">Relationship Types</span>
        <div class="legend-actions">
          {#if selectedTypes.size > 0 && selectedTypes.size < sortedTypes.length}
            <button class="action-btn" onclick={selectAll}>All</button>
          {/if}
          {#if selectedTypes.size > 0}
            <button class="action-btn" onclick={clearSelection}>Clear</button>
          {/if}
        </div>
      </div>

      <div class="legend-items">
        {#each sortedTypes as { type, count, color }}
          <button
            class="legend-item"
            class:selected={selectedTypes.has(type)}
            class:dimmed={selectedTypes.size > 0 && !selectedTypes.has(type)}
            onclick={() => toggleType(type)}
          >
            <span class="color-dot" style="background: {color}"></span>
            <span class="type-name">{type}</span>
            <span class="type-count">{count}</span>
          </button>
        {/each}
      </div>

      <div class="legend-footer">
        <span class="total-nodes">{nodes.length} contacts</span>
      </div>
    </div>
  {/if}
</div>

<style>
  .legend {
    position: absolute;
    bottom: 20px;
    left: 20px;
    background: rgba(40, 40, 40, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    backdrop-filter: blur(8px);
    min-width: 160px;
    max-width: 220px;
    overflow: hidden;
    transition: all 0.2s ease;
  }

  .legend.collapsed {
    min-width: auto;
  }

  .legend-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 10px 14px;
    background: none;
    border: none;
    color: #fff;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    text-align: left;
    transition: background 0.15s;
  }

  .legend-toggle:hover {
    background: rgba(255, 255, 255, 0.05);
  }

  .toggle-icon {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.5);
    transition: transform 0.2s;
  }

  .legend-content {
    padding: 0 14px 14px;
    animation: slideDown 0.2s ease-out;
  }

  @keyframes slideDown {
    from {
      opacity: 0;
      transform: translateY(-10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .legend-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }

  .legend-title {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .legend-actions {
    display: flex;
    gap: 6px;
  }

  .action-btn {
    padding: 2px 6px;
    background: rgba(255, 255, 255, 0.1);
    border: none;
    border-radius: 4px;
    color: rgba(255, 255, 255, 0.7);
    font-size: 10px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .action-btn:hover {
    background: rgba(255, 255, 255, 0.2);
    color: #fff;
  }

  .legend-items {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 8px;
    background: none;
    border: 1px solid transparent;
    border-radius: 6px;
    color: #fff;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
    width: 100%;
    text-align: left;
  }

  .legend-item:hover {
    background: rgba(255, 255, 255, 0.05);
  }

  .legend-item.selected {
    background: rgba(255, 255, 255, 0.1);
    border-color: rgba(255, 255, 255, 0.2);
  }

  .legend-item.dimmed {
    opacity: 0.4;
  }

  .color-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .type-name {
    flex: 1;
    text-transform: capitalize;
  }

  .type-count {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    font-weight: 500;
  }

  .legend-footer {
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
  }

  .total-nodes {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.4);
  }
</style>
