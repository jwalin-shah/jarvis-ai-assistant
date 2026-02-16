<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { LayoutType } from '../../api/types';

  export let showLabels: boolean = true;
  export let currentLayout: LayoutType = 'force';
  export let showFacts: boolean = false;

  const dispatch = createEventDispatcher<{
    resetZoom: void;
    toggleLabels: void;
    toggleFacts: void;
    reheat: void;
    changeLayout: LayoutType;
    search: string;
    filterRelationships: string[];
    export: 'png' | 'svg' | 'json' | 'html';
  }>();

  let searchQuery = '';
  let selectedRelationships: string[] = [];
  let searchTimer: ReturnType<typeof setTimeout> | null = null;

  const layouts: { value: LayoutType; label: string }[] = [
    { value: 'force', label: 'Force' },
    { value: 'hierarchical', label: 'Hierarchy' },
    { value: 'radial', label: 'Radial' },
  ];

  const relationshipTypes = [
    'family',
    'friend',
    'work',
    'acquaintance',
    'professional',
    'romantic',
    'unknown',
  ];

  function handleSearch() {
    if (searchTimer) clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      dispatch('search', searchQuery);
    }, 200);
  }

  function handleLayoutChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    dispatch('changeLayout', target.value as LayoutType);
  }

  function toggleRelationship(type: string) {
    if (selectedRelationships.includes(type)) {
      selectedRelationships = selectedRelationships.filter((r) => r !== type);
    } else {
      selectedRelationships = [...selectedRelationships, type];
    }
    dispatch('filterRelationships', selectedRelationships);
  }

  function clearFilters() {
    selectedRelationships = [];
    searchQuery = '';
    if (searchTimer) clearTimeout(searchTimer);
    dispatch('filterRelationships', []);
    dispatch('search', '');
  }
</script>

<div class="controls">
  <div class="controls-left">
    <div class="search-box">
      <input
        type="text"
        placeholder="Search contacts..."
        bind:value={searchQuery}
        on:input={handleSearch}
      />
      {#if searchQuery}
        <button
          class="clear-search"
          on:click={() => {
            searchQuery = '';
            handleSearch();
          }}
        >
          &times;
        </button>
      {/if}
    </div>

    <select class="layout-select" value={currentLayout} on:change={handleLayoutChange}>
      {#each layouts as layout}
        <option value={layout.value}>{layout.label}</option>
      {/each}
    </select>
  </div>

  <div class="controls-center">
    <div class="relationship-filters">
      {#each relationshipTypes as type}
        <button
          class="filter-chip"
          class:active={selectedRelationships.includes(type)}
          on:click={() => toggleRelationship(type)}
        >
          {type}
        </button>
      {/each}
      {#if selectedRelationships.length > 0}
        <button class="clear-filters" on:click={clearFilters}>Clear</button>
      {/if}
    </div>
  </div>

  <div class="controls-right">
    <button class="control-btn" on:click={() => dispatch('resetZoom')} title="Reset Zoom">
      <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
        <path
          d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0016 9.5 6.5 6.5 0 109.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"
        />
        <path d="M12 10h-2v2H9v-2H7V9h2V7h1v2h2v1z" />
      </svg>
    </button>

    <button
      class="control-btn"
      class:active={showLabels}
      on:click={() => dispatch('toggleLabels')}
      title="Toggle Labels"
    >
      <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
        <path d="M9 17v-2h4v2H9zm8-10H7v2h10V7zm-4 4H7v2h6v-2z" />
      </svg>
    </button>

    <button
      class="control-btn"
      class:active={showFacts}
      on:click={() => dispatch('toggleFacts')}
      title="Show Facts (Knowledge Graph)"
    >
      <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
        <path
          d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"
        />
      </svg>
    </button>

    <button class="control-btn" on:click={() => dispatch('reheat')} title="Reheat Simulation">
      <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
        <path
          d="M17.65 6.35A7.958 7.958 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"
        />
      </svg>
    </button>

    <div class="divider"></div>

    <button class="control-btn export" on:click={() => dispatch('export', 'png')} title="Export">
      <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
        <path
          d="M19 12v7H5v-7H3v7c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2v-7h-2zm-6 .67l2.59-2.58L17 11.5l-5 5-5-5 1.41-1.41L11 12.67V3h2v9.67z"
        />
      </svg>
    </button>
  </div>
</div>

<style>
  .controls {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-color);
    gap: 16px;
    flex-wrap: wrap;
  }

  .controls-left,
  .controls-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .controls-center {
    flex: 1;
    display: flex;
    justify-content: center;
  }

  .search-box {
    position: relative;
    display: flex;
    align-items: center;
  }

  .search-box input {
    width: 180px;
    padding: 6px 28px 6px 10px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 13px;
  }

  .search-box input:focus {
    outline: none;
    border-color: var(--accent-color);
  }

  .clear-search {
    position: absolute;
    right: 6px;
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 16px;
    padding: 0;
    line-height: 1;
  }

  .layout-select {
    padding: 6px 10px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 13px;
    cursor: pointer;
  }

  .relationship-filters {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    justify-content: center;
  }

  .filter-chip {
    padding: 4px 10px;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    background: var(--bg-primary);
    color: var(--text-secondary);
    font-size: 11px;
    cursor: pointer;
    text-transform: capitalize;
    transition: all 0.15s;
  }

  .filter-chip:hover {
    border-color: var(--accent-color);
  }

  .filter-chip.active {
    background: var(--accent-color);
    border-color: var(--accent-color);
    color: white;
  }

  .clear-filters {
    padding: 4px 10px;
    border: none;
    border-radius: 12px;
    background: var(--bg-hover);
    color: var(--text-secondary);
    font-size: 11px;
    cursor: pointer;
  }

  .control-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: var(--bg-primary);
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s;
  }

  .control-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .control-btn.active {
    background: var(--accent-color);
    border-color: var(--accent-color);
    color: white;
  }

  .control-btn.export {
    background: var(--accent-color);
    border-color: var(--accent-color);
    color: white;
  }

  .control-btn.export:hover {
    background: #0a82e0;
  }

  .divider {
    width: 1px;
    height: 20px;
    background: var(--border-color);
    margin: 0 4px;
  }

  @media (max-width: 768px) {
    .controls {
      flex-direction: column;
      align-items: stretch;
    }

    .controls-left,
    .controls-right,
    .controls-center {
      justify-content: center;
    }

    .search-box input {
      width: 100%;
    }
  }
</style>
