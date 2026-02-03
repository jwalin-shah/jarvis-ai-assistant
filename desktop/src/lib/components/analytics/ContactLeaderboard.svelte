<script lang="ts">
  import type { LeaderboardContact } from "../../api/types";

  export let contacts: LeaderboardContact[];

  function getTrendIcon(trend: string): string {
    if (trend === "increasing") return "arrow_upward";
    if (trend === "decreasing") return "arrow_downward";
    return "remove";
  }

  function getTrendColor(trend: string): string {
    if (trend === "increasing") return "#34c759";
    if (trend === "decreasing") return "#ff3b30";
    return "var(--text-secondary)";
  }

  function formatResponseTime(minutes: number | null): string {
    if (minutes === null) return "-";
    if (minutes < 60) return `${Math.round(minutes)}m`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h`;
  }

  function getRankBadgeClass(rank: number): string {
    if (rank === 1) return "gold";
    if (rank === 2) return "silver";
    if (rank === 3) return "bronze";
    return "";
  }
</script>

<div class="leaderboard">
  {#each contacts as contact}
    <div class="contact-row">
      <div class="rank-badge" class:gold={contact.rank === 1} class:silver={contact.rank === 2} class:bronze={contact.rank === 3}>
        {contact.rank}
      </div>
      <div class="contact-info">
        <span class="contact-name">{contact.contact_name || "Unknown"}</span>
        <span class="contact-stats">
          {contact.total_messages} msgs
          <span class="separator">|</span>
          {Math.round(contact.engagement_score)} engagement
        </span>
      </div>
      <div class="contact-metrics">
        <div class="metric" title="Sentiment">
          <span class="metric-value" class:positive={contact.sentiment_score > 0.2} class:negative={contact.sentiment_score < -0.2}>
            {contact.sentiment_score > 0 ? "+" : ""}{contact.sentiment_score.toFixed(2)}
          </span>
        </div>
        <div class="metric" title="Response Time">
          <span class="metric-value">{formatResponseTime(contact.avg_response_time_minutes)}</span>
        </div>
        <div class="trend" style="color: {getTrendColor(contact.trend)}" title="Activity trend">
          {#if contact.trend === "increasing"}
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M7 14l5-5 5 5H7z"/>
            </svg>
          {:else if contact.trend === "decreasing"}
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M7 10l5 5 5-5H7z"/>
            </svg>
          {:else}
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <rect x="4" y="11" width="16" height="2"/>
            </svg>
          {/if}
        </div>
      </div>
    </div>
  {/each}
  {#if contacts.length === 0}
    <div class="empty-state">No contacts to display</div>
  {/if}
</div>

<style>
  .leaderboard {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 300px;
    overflow-y: auto;
  }

  .contact-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px;
    background: var(--bg-secondary);
    border-radius: 6px;
    transition: background-color 0.15s;
  }

  .contact-row:hover {
    background: var(--bg-hover);
  }

  .rank-badge {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    font-size: 12px;
    font-weight: 600;
    background: var(--bg-hover);
    color: var(--text-secondary);
  }

  .rank-badge.gold {
    background: linear-gradient(135deg, #ffd700, #ffb700);
    color: #000;
  }

  .rank-badge.silver {
    background: linear-gradient(135deg, #c0c0c0, #a0a0a0);
    color: #000;
  }

  .rank-badge.bronze {
    background: linear-gradient(135deg, #cd7f32, #b5651d);
    color: #fff;
  }

  .contact-info {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }

  .contact-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .contact-stats {
    font-size: 11px;
    color: var(--text-secondary);
  }

  .separator {
    margin: 0 4px;
    opacity: 0.5;
  }

  .contact-metrics {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .metric {
    text-align: center;
  }

  .metric-value {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .metric-value.positive {
    color: #34c759;
  }

  .metric-value.negative {
    color: #ff3b30;
  }

  .trend {
    display: flex;
    align-items: center;
  }

  .empty-state {
    text-align: center;
    padding: 24px;
    color: var(--text-secondary);
    font-size: 13px;
  }
</style>
