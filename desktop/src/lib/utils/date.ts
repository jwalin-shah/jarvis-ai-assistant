/**
 * Date formatting utilities
 */

const dateFormatCache = new Map<string, string>();
const dateStringCache = new Map<number, string>();

/**
 * Format a date string to a relative date (Today, Yesterday, or full date)
 */
export function formatDate(dateStr: string): string {
  let cached = dateFormatCache.get(dateStr);
  if (cached) return cached;

  const date = new Date(dateStr);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  let formatted: string;
  if (date.toDateString() === today.toDateString()) {
    formatted = 'Today';
  } else if (date.toDateString() === yesterday.toDateString()) {
    formatted = 'Yesterday';
  } else {
    formatted = date.toLocaleDateString([], {
      weekday: 'long',
      month: 'long',
      day: 'numeric',
    });
  }

  dateFormatCache.set(dateStr, formatted);
  return formatted;
}

/**
 * Format a date for conversation list (Today = time, Yesterday = "Yesterday", older = date)
 */
export function formatConversationDate(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));

  if (days === 0) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } else if (days === 1) {
    return 'Yesterday';
  } else if (days < 7) {
    return date.toLocaleDateString([], { weekday: 'short' });
  } else {
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  }
}

/**
 * Get date string for a message (for comparison)
 */
export function getMessageDateString(messageId: number, dateStr: string): string {
  let cached = dateStringCache.get(messageId);
  if (!cached) {
    cached = new Date(dateStr).toDateString();
    dateStringCache.set(messageId, cached);
  }
  return cached;
}

/**
 * Format a date string to a compact relative time (e.g., "2m", "1h", "3d")
 */
export function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHr = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHr / 24);
  const diffWeek = Math.floor(diffDay / 7);

  if (diffSec < 60) return 'now';
  if (diffMin < 60) return `${diffMin}m`;
  if (diffHr < 24) return `${diffHr}h`;
  if (diffDay < 7) return `${diffDay}d`;
  if (diffWeek < 52) return `${diffWeek}w`;
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

/**
 * Format a date string to a full readable timestamp for tooltips
 */
export function formatFullTimestamp(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleString([], {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Clear all date format caches
 */
export function clearDateCaches(): void {
  dateFormatCache.clear();
  dateStringCache.clear();
}
