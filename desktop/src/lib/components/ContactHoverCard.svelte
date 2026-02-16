<script lang="ts">
  import { onMount } from 'svelte';
  import { getApiBaseUrl } from '../config/runtime';
  import { formatParticipant } from '../db';
  import type { ContactProfile } from '../types';

  let { identifier, visible = $bindable(false), x = 0, y = 0 } = $props<{
    identifier: string;
    visible: boolean;
    x: number;
    y: number;
  }>();

  let profile = $state<ContactProfile | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);

  const API_BASE = getApiBaseUrl();

  async function fetchProfile() {
    if (!identifier) return;
    loading = true;
    error = null;
    try {
      const response = await fetch(`${API_BASE}/graph/contact/${encodeURIComponent(identifier)}`);
      if (!response.ok) throw new Error('Failed to load profile');
      profile = await response.json();
    } catch (err) {
      console.error('Error fetching contact profile:', err);
      error = 'Profile unavailable';
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    if (visible && identifier) {
      fetchProfile();
    }
  });

  function getInitials(name: string): string {
    const parts = name.trim().split(/\s+/);
    if (parts.length >= 2) {
      return `${parts[0]![0]}${parts[parts.length - 1]![0]}`.toUpperCase();
    }
    return parts[0]?.[0]?.toUpperCase() || '?';
  }
</script>

{#if visible}
  <div class="hover-card" style="left: {x}px; top: {y}px;" transition:fade={{ duration: 100 }}>
    {#if loading}
      <div class="loading">
        <div class="spinner"></div>
        <span>Loading profile...</span>
      </div>
    {:else if error}
      <div class="error">{error}</div>
    {:else if profile}
      <div class="header">
        <div class="avatar">
          {getInitials(profile.contact_name || formatParticipant(identifier))}
        </div>
        <div class="name-info">
          <h3>{profile.contact_name || formatParticipant(identifier)}</h3>
          <p class="relationship">{profile.relationship || 'Unknown relationship'}</p>
        </div>
      </div>

      <div class="stats">
        <div class="stat">
          <span class="label">Messages</span>
          <span class="value">{profile.message_count}</span>
        </div>
        <div class="stat">
          <span class="label">Formality</span>
          <span class="value">{profile.formality}</span>
        </div>
      </div>

      {#if profile.top_topics && profile.top_topics.length > 0}
        <div class="section">
          <h4>Topics</h4>
          <div class="tags">
            {#each profile.top_topics.slice(0, 3) as topic}
              <span class="tag">{topic}</span>
            {/each}
          </div>
        </div>
      {/if}

      {#if profile.extracted_facts && profile.extracted_facts.length > 0}
        <div class="section">
          <h4>Extracted Facts</h4>
          <ul class="facts">
            {#each profile.extracted_facts.slice(0, 2) as fact}
              <li>{fact.subject} {fact.predicate.replace('_', ' ')} {fact.value}</li>
            {/each}
          </ul>
        </div>
      {/if}
    {:else}
      <div class="empty">No profile data available</div>
    {/if}
  </div>
{/if}

<style>
  .hover-card {
    position: fixed;
    z-index: 1000;
    width: 280px;
    background: var(--surface-elevated);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-xl);
    padding: var(--space-4);
    pointer-events: none;
    animation: fadeIn 0.15s ease-out;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .loading, .error, .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-2);
    padding: var(--space-4);
    color: var(--text-secondary);
    font-size: var(--text-sm);
  }

  .spinner {
    width: 20px;
    height: 20px;
    border: 2px solid var(--border-default);
    border-top-color: var(--color-primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .header {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    margin-bottom: var(--space-4);
  }

  .avatar {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: var(--color-primary);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: var(--font-weight-semibold);
    font-size: var(--text-lg);
  }

  .name-info h3 {
    margin: 0;
    font-size: var(--text-base);
    font-weight: var(--font-weight-semibold);
    color: var(--text-primary);
  }

  .relationship {
    margin: 0;
    font-size: var(--text-xs);
    color: var(--text-secondary);
    text-transform: capitalize;
  }

  .stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-2);
    margin-bottom: var(--space-4);
    padding: var(--space-2);
    background: var(--surface-base);
    border-radius: var(--radius-md);
  }

  .stat {
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .stat .label {
    font-size: 10px;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .stat .value {
    font-size: var(--text-sm);
    font-weight: var(--font-weight-medium);
    color: var(--text-primary);
  }

  .section {
    margin-bottom: var(--space-3);
  }

  .section h4 {
    margin: 0 0 var(--space-1) 0;
    font-size: 11px;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }

  .tag {
    padding: 2px 8px;
    background: var(--bg-active);
    color: var(--accent-color);
    border-radius: var(--radius-full);
    font-size: 10px;
    font-weight: var(--font-weight-medium);
  }

  .facts {
    margin: 0;
    padding: 0;
    list-style: none;
    font-size: var(--text-xs);
    color: var(--text-secondary);
  }

  .facts li {
    padding: 2px 0;
    border-bottom: 1px solid var(--border-subtle);
  }

  .facts li:last-child {
    border-bottom: none;
  }
</style>
