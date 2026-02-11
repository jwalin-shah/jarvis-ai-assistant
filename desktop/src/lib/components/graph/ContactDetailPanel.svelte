<script lang="ts">
  import { api } from "../../api/client";
  import type { ContactProfileDetail, ContactFact, GraphNode } from "../../api/types";

  export let node: GraphNode | null = null;
  export let visible: boolean = false;
  export let onClose: (() => void) | null = null;

  let profile: ContactProfileDetail | null = null;
  let loading = false;
  let error: string | null = null;

  // Fact category icons
  const categoryIcons: Record<string, string> = {
    relationship: "👥",
    location: "📍",
    work: "💼",
    preference: "⭐",
    event: "📅",
  };

  // Group facts by category
  function groupFacts(facts: ContactFact[]): Record<string, ContactFact[]> {
    const groups: Record<string, ContactFact[]> = {};
    for (const fact of facts) {
      if (!groups[fact.category]) {
        groups[fact.category] = [];
      }
      groups[fact.category]!.push(fact);
    }
    return groups;
  }

  async function loadProfile(contactId: string) {
    loading = true;
    error = null;
    try {
      profile = await api.getContactProfile(contactId);
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load profile";
      profile = null;
    } finally {
      loading = false;
    }
  }

  $: if (node && visible) {
    loadProfile(node.id);
  }

  $: groupedFacts = profile?.facts ? groupFacts(profile.facts) : {};

  function handleClose() {
    visible = false;
    if (onClose) onClose();
  }
</script>

{#if visible && node}
  <div class="panel-overlay" on:click={handleClose} on:keydown={(e) => e.key === "Escape" && handleClose()} role="button" tabindex="-1">
    <div class="detail-panel" on:click|stopPropagation role="presentation">
      <div class="panel-header">
        <div class="contact-info">
          <div class="contact-avatar" style="background-color: {node.color}">
            {node.label.charAt(0).toUpperCase()}
          </div>
          <div>
            <h2>{node.label || "Unknown"}</h2>
            {#if profile}
              <span class="relationship-badge">{profile.relationship}</span>
              <span class="formality-badge">{profile.formality}</span>
            {/if}
          </div>
        </div>
        <button class="close-btn" on:click={handleClose}>&times;</button>
      </div>

      {#if loading}
        <div class="loading">Loading profile...</div>
      {:else if error}
        <div class="error">{error}</div>
      {:else if profile}
        <div class="panel-body">
          <!-- Style Guide -->
          <section>
            <h3>Communication Style</h3>
            <p class="style-guide">{profile.style_guide}</p>
          </section>

          <!-- Stats -->
          <section class="stats-grid">
            <div class="stat">
              <span class="stat-value">{profile.message_count}</span>
              <span class="stat-label">Messages</span>
            </div>
            <div class="stat">
              <span class="stat-value">{profile.avg_message_length.toFixed(0)}</span>
              <span class="stat-label">Avg Length</span>
            </div>
            <div class="stat">
              <span class="stat-value">{(profile.formality_score * 100).toFixed(0)}%</span>
              <span class="stat-label">Formality</span>
            </div>
            {#if profile.avg_response_time_minutes != null}
              <div class="stat">
                <span class="stat-value">{profile.avg_response_time_minutes.toFixed(0)}m</span>
                <span class="stat-label">Avg Response</span>
              </div>
            {/if}
          </section>

          <!-- Topics -->
          {#if profile.top_topics.length > 0}
            <section>
              <h3>Topics</h3>
              <div class="tags">
                {#each profile.top_topics as topic}
                  <span class="tag">{topic}</span>
                {/each}
              </div>
            </section>
          {/if}

          <!-- Facts -->
          {#if Object.keys(groupedFacts).length > 0}
            <section>
              <h3>Knowledge</h3>
              {#each Object.entries(groupedFacts) as [category, facts]}
                <div class="fact-group">
                  <h4>{categoryIcons[category] || "📌"} {category}</h4>
                  {#each facts as fact}
                    <div class="fact-item">
                      <span class="fact-subject">{fact.subject}</span>
                      <span class="fact-predicate">{fact.predicate.replace(/_/g, " ")}</span>
                      {#if fact.value}
                        <span class="fact-value">({fact.value})</span>
                      {/if}
                      <span class="fact-confidence" title="Confidence: {(fact.confidence * 100).toFixed(0)}%">
                        {fact.confidence >= 0.8 ? "●" : fact.confidence >= 0.5 ? "◐" : "○"}
                      </span>
                    </div>
                  {/each}
                </div>
              {/each}
            </section>
          {/if}
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .panel-overlay {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    width: 380px;
    z-index: 1000;
    background: rgba(0, 0, 0, 0.1);
  }

  .detail-panel {
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    width: 380px;
    background: var(--bg-primary, #1e1e1e);
    border-left: 1px solid var(--border-color, #333);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 16px;
    border-bottom: 1px solid var(--border-color, #333);
  }

  .contact-info {
    display: flex;
    gap: 12px;
    align-items: center;
  }

  .contact-avatar {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 20px;
    font-weight: 600;
    flex-shrink: 0;
  }

  h2 {
    margin: 0;
    font-size: 16px;
    color: var(--text-primary, #fff);
  }

  .relationship-badge, .formality-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    margin-top: 4px;
    margin-right: 4px;
  }

  .relationship-badge {
    background: var(--accent-bg, #2a4a7f);
    color: var(--accent-text, #8ab4f8);
  }

  .formality-badge {
    background: var(--badge-bg, #333);
    color: var(--text-secondary, #aaa);
  }

  .close-btn {
    background: none;
    border: none;
    color: var(--text-secondary, #aaa);
    font-size: 24px;
    cursor: pointer;
    padding: 0;
    line-height: 1;
  }

  .panel-body {
    padding: 16px;
    flex: 1;
  }

  section {
    margin-bottom: 20px;
  }

  h3 {
    margin: 0 0 8px;
    font-size: 13px;
    text-transform: uppercase;
    color: var(--text-secondary, #aaa);
    letter-spacing: 0.5px;
  }

  .style-guide {
    font-size: 13px;
    color: var(--text-primary, #ddd);
    line-height: 1.5;
    margin: 0;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }

  .stat {
    display: flex;
    flex-direction: column;
    padding: 8px 12px;
    background: var(--card-bg, #252525);
    border-radius: 8px;
  }

  .stat-value {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary, #fff);
  }

  .stat-label {
    font-size: 11px;
    color: var(--text-secondary, #aaa);
  }

  .tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .tag {
    padding: 4px 10px;
    background: var(--tag-bg, #2a2a2a);
    border-radius: 16px;
    font-size: 12px;
    color: var(--text-primary, #ddd);
  }

  .fact-group {
    margin-bottom: 12px;
  }

  .fact-group h4 {
    margin: 0 0 6px;
    font-size: 13px;
    color: var(--text-primary, #ddd);
    text-transform: capitalize;
  }

  .fact-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 0;
    font-size: 12px;
    color: var(--text-secondary, #bbb);
  }

  .fact-subject {
    font-weight: 600;
    color: var(--text-primary, #ddd);
  }

  .fact-predicate {
    color: var(--text-secondary, #888);
  }

  .fact-value {
    color: var(--text-secondary, #aaa);
    font-style: italic;
  }

  .fact-confidence {
    margin-left: auto;
    font-size: 10px;
  }

  .loading, .error {
    padding: 32px 16px;
    text-align: center;
    color: var(--text-secondary, #aaa);
  }

  .error {
    color: var(--error-color, #f44);
  }
</style>
