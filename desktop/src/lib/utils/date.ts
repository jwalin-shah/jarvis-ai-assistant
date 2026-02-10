/**
 * Date formatting utilities
 */

const timeFormatCache = new Map<string, string>();
const dateFormatCache = new Map<string, string>();
const dateStringCache = new Map<number, string>();

/**
 * Format a date string to time (HH:MM)
 */
export function formatTime(dateStr: string): string {
  let cached = timeFormatCache.get(dateStr);
  if (cached) return cached;

  const formatted = new Date(dateStr).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
  timeFormatCache.set(dateStr, formatted);
  return formatted;
}

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
 * Clear all date format caches
 */
export function clearDateCaches(): void {
  timeFormatCache.clear();
  dateFormatCache.clear();
  dateStringCache.clear();
}
