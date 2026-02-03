<script lang="ts">
  import Icon from "./Icon.svelte";

  type EmptyStateType = "no-messages" | "no-conversation" | "no-results" | "error" | "offline";

  interface Props {
    type: EmptyStateType;
    title?: string;
    description?: string;
    action?: {
      label: string;
      onClick: () => void;
    };
  }

  let { type, title, description, action }: Props = $props();

  const defaults: Record<EmptyStateType, { title: string; description: string; illustration: string }> = {
    "no-messages": {
      title: "No messages yet",
      description: "Start a conversation to see messages here",
      illustration: "messages",
    },
    "no-conversation": {
      title: "Select a conversation",
      description: "Choose a conversation from the sidebar to view messages",
      illustration: "select",
    },
    "no-results": {
      title: "No results found",
      description: "Try adjusting your search or filters",
      illustration: "search",
    },
    "error": {
      title: "Something went wrong",
      description: "We couldn't load this content. Please try again.",
      illustration: "error",
    },
    "offline": {
      title: "You're offline",
      description: "Check your internet connection and try again",
      illustration: "offline",
    },
  };

  const config = $derived({
    title: title ?? defaults[type].title,
    description: description ?? defaults[type].description,
    illustration: defaults[type].illustration,
  });
</script>

<div class="empty-state">
  <div class="illustration">
    {#if config.illustration === "messages"}
      <svg viewBox="0 0 120 120" fill="none">
        <circle cx="60" cy="60" r="50" fill="var(--bg-surface)" />
        <rect x="30" y="40" width="40" height="24" rx="12" fill="var(--color-primary)" opacity="0.2" />
        <rect x="50" y="56" width="40" height="24" rx="12" fill="var(--bg-elevated)" />
        <circle cx="42" cy="52" r="3" fill="var(--text-tertiary)" class="dot dot-1" />
        <circle cx="52" cy="52" r="3" fill="var(--text-tertiary)" class="dot dot-2" />
        <circle cx="62" cy="52" r="3" fill="var(--text-tertiary)" class="dot dot-3" />
      </svg>
    {:else if config.illustration === "select"}
      <svg viewBox="0 0 120 120" fill="none">
        <circle cx="60" cy="60" r="50" fill="var(--bg-surface)" />
        <rect x="25" y="35" width="30" height="50" rx="6" fill="var(--bg-elevated)" />
        <rect x="30" y="42" width="20" height="4" rx="2" fill="var(--text-tertiary)" />
        <rect x="30" y="50" width="16" height="3" rx="1.5" fill="var(--text-tertiary)" opacity="0.5" />
        <rect x="30" y="58" width="20" height="4" rx="2" fill="var(--text-tertiary)" />
        <rect x="30" y="66" width="14" height="3" rx="1.5" fill="var(--text-tertiary)" opacity="0.5" />
        <path d="M65 60 L85 60" stroke="var(--color-primary)" stroke-width="2" stroke-dasharray="4 2" class="arrow-line" />
        <path d="M82 55 L90 60 L82 65" stroke="var(--color-primary)" stroke-width="2" fill="none" class="arrow-head" />
      </svg>
    {:else if config.illustration === "search"}
      <svg viewBox="0 0 120 120" fill="none">
        <circle cx="60" cy="60" r="50" fill="var(--bg-surface)" />
        <circle cx="52" cy="52" r="20" stroke="var(--text-tertiary)" stroke-width="4" fill="none" />
        <line x1="66" y1="66" x2="82" y2="82" stroke="var(--text-tertiary)" stroke-width="4" stroke-linecap="round" />
        <line x1="44" y1="52" x2="60" y2="52" stroke="var(--text-tertiary)" stroke-width="2" stroke-linecap="round" opacity="0.5" />
        <line x1="52" y1="44" x2="52" y2="60" stroke="var(--text-tertiary)" stroke-width="2" stroke-linecap="round" opacity="0.5" />
      </svg>
    {:else if config.illustration === "error"}
      <svg viewBox="0 0 120 120" fill="none">
        <circle cx="60" cy="60" r="50" fill="var(--bg-surface)" />
        <circle cx="60" cy="60" r="25" stroke="var(--color-error)" stroke-width="4" fill="none" opacity="0.3" />
        <line x1="50" y1="50" x2="70" y2="70" stroke="var(--color-error)" stroke-width="4" stroke-linecap="round" />
        <line x1="70" y1="50" x2="50" y2="70" stroke="var(--color-error)" stroke-width="4" stroke-linecap="round" />
      </svg>
    {:else if config.illustration === "offline"}
      <svg viewBox="0 0 120 120" fill="none">
        <circle cx="60" cy="60" r="50" fill="var(--bg-surface)" />
        <path d="M40 70 Q60 50 80 70" stroke="var(--text-tertiary)" stroke-width="4" fill="none" stroke-linecap="round" opacity="0.3" />
        <path d="M48 78 Q60 64 72 78" stroke="var(--text-tertiary)" stroke-width="4" fill="none" stroke-linecap="round" opacity="0.5" />
        <circle cx="60" cy="86" r="4" fill="var(--text-tertiary)" />
        <line x1="35" y1="45" x2="85" y2="85" stroke="var(--color-error)" stroke-width="3" stroke-linecap="round" />
      </svg>
    {/if}
  </div>

  <h3 class="title">{config.title}</h3>
  <p class="description">{config.description}</p>

  {#if action}
    <button class="action-btn" onclick={action.onClick}>
      {action.label}
    </button>
  {/if}
</div>

<style>
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px 24px;
    text-align: center;
    animation: fadeIn 0.3s ease;
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(8px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .illustration {
    width: 120px;
    height: 120px;
    margin-bottom: 24px;
  }

  .illustration svg {
    width: 100%;
    height: 100%;
  }

  /* Animated dots for messages illustration */
  .dot {
    animation: dotBounce 1.4s ease-in-out infinite;
  }

  .dot-1 { animation-delay: 0ms; }
  .dot-2 { animation-delay: 160ms; }
  .dot-3 { animation-delay: 320ms; }

  @keyframes dotBounce {
    0%, 60%, 100% {
      transform: translateY(0);
      opacity: 0.4;
    }
    30% {
      transform: translateY(-4px);
      opacity: 1;
    }
  }

  /* Animated arrow for select illustration */
  .arrow-line {
    animation: arrowPulse 2s ease-in-out infinite;
  }

  .arrow-head {
    animation: arrowPulse 2s ease-in-out infinite;
  }

  @keyframes arrowPulse {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
  }

  .title {
    font-size: var(--text-lg);
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 8px;
  }

  .description {
    font-size: var(--text-sm);
    color: var(--text-secondary);
    max-width: 280px;
    line-height: 1.5;
  }

  .action-btn {
    margin-top: 20px;
    padding: 10px 20px;
    background: var(--color-primary);
    border: none;
    border-radius: var(--radius-md);
    color: white;
    font-size: var(--text-sm);
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .action-btn:hover {
    filter: brightness(1.1);
    transform: translateY(-1px);
  }

  /* Reduced motion */
  @media (prefers-reduced-motion: reduce) {
    .empty-state {
      animation: none;
    }
    .dot,
    .arrow-line,
    .arrow-head {
      animation: none;
      opacity: 0.7;
    }
  }
</style>
