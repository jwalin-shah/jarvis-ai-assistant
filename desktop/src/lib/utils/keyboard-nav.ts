/**
 * Shared vim-style keyboard navigation utility.
 * Handles j/k/g/G/Escape for list navigation in both
 * ConversationList and MessageView.
 */

export type NavAction =
  | { type: 'next'; index: number }
  | { type: 'prev'; index: number }
  | { type: 'first' }
  | { type: 'last'; index: number }
  | { type: 'escape' }
  | null;

/**
 * Compute a navigation action from a keyboard event.
 * Returns null if the key isn't a navigation key.
 *
 * @param key - event.key
 * @param shiftKey - event.shiftKey
 * @param currentIndex - currently focused index (-1 = none)
 * @param maxIndex - last valid index (items.length - 1)
 */
export function getNavAction(
  key: string,
  shiftKey: boolean,
  currentIndex: number,
  maxIndex: number
): NavAction {
  if (maxIndex < 0) return null;

  switch (key) {
    case 'j':
    case 'ArrowDown':
      if (currentIndex < maxIndex) {
        return { type: 'next', index: currentIndex + 1 };
      }
      return null;

    case 'k':
    case 'ArrowUp':
      if (currentIndex > 0) {
        return { type: 'prev', index: currentIndex - 1 };
      }
      if (currentIndex === -1) {
        return { type: 'prev', index: maxIndex };
      }
      return null;

    case 'g':
      if (!shiftKey) {
        return { type: 'first' };
      }
      return null;

    case 'G':
      if (shiftKey) {
        return { type: 'last', index: maxIndex };
      }
      return null;

    case 'Escape':
      return { type: 'escape' };

    default:
      return null;
  }
}

/** Guard: returns true if event target is a text input (should skip nav). */
export function isTypingInInput(event: KeyboardEvent): boolean {
  return event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement;
}
