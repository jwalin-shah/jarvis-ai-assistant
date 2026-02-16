/**
 * Keyboard navigation store for managing focus and navigation state
 */

import { writable, derived, get } from 'svelte/store';

export type FocusZone = 'sidebar' | 'conversations' | 'messages' | 'compose' | 'modal' | null;

interface KeyboardState {
  activeZone: FocusZone;
  conversationIndex: number;
  messageIndex: number;
  isVimMode: boolean;
}

const initialState: KeyboardState = {
  activeZone: null,
  conversationIndex: -1,
  messageIndex: -1,
  isVimMode: false,
};

const keyboardStore = writable<KeyboardState>(initialState);

// Derived stores
export const activeZone = derived(keyboardStore, ($state) => $state.activeZone);
export const conversationIndex = derived(keyboardStore, ($state) => $state.conversationIndex);
export const messageIndex = derived(keyboardStore, ($state) => $state.messageIndex);
export const isVimMode = derived(keyboardStore, ($state) => $state.isVimMode);

/**
 * Set the active focus zone
 */
export function setActiveZone(zone: FocusZone): void {
  keyboardStore.update((state) => ({ ...state, activeZone: zone }));
}

/**
 * Set conversation selection index
 */
export function setConversationIndex(index: number): void {
  keyboardStore.update((state) => ({ ...state, conversationIndex: index }));
}

/**
 * Move conversation selection
 */
export function moveConversationSelection(delta: number, maxIndex: number): number {
  const state = get(keyboardStore);
  const newIndex = Math.max(0, Math.min(state.conversationIndex + delta, maxIndex));
  keyboardStore.update((s) => ({ ...s, conversationIndex: newIndex }));
  return newIndex;
}

/**
 * Set message selection index
 */
export function setMessageIndex(index: number): void {
  keyboardStore.update((state) => ({ ...state, messageIndex: index }));
}

/**
 * Move message selection
 */
export function moveMessageSelection(delta: number, maxIndex: number): number {
  const state = get(keyboardStore);
  const newIndex = Math.max(0, Math.min(state.messageIndex + delta, maxIndex));
  keyboardStore.update((s) => ({ ...s, messageIndex: newIndex }));
  return newIndex;
}

/**
 * Toggle vim mode
 */
export function toggleVimMode(): void {
  keyboardStore.update((state) => ({ ...state, isVimMode: !state.isVimMode }));
}

/**
 * Reset keyboard state
 */
export function resetKeyboardState(): void {
  keyboardStore.set(initialState);
}

/**
 * Global keyboard shortcut handler
 * Returns true if the event was handled
 */
export function handleGlobalKeydown(
  event: KeyboardEvent,
  callbacks: {
    onEscape?: () => void;
    onEnter?: () => void;
    onArrowUp?: () => void;
    onArrowDown?: () => void;
    onArrowLeft?: () => void;
    onArrowRight?: () => void;
    onTab?: (shift: boolean) => void;
    onVimJ?: () => void;
    onVimK?: () => void;
    onVimG?: () => void;
    onVimShiftG?: () => void;
  }
): boolean {
  const state = get(keyboardStore);

  // Don't handle if typing in an input
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
    // Only handle Escape in inputs
    if (event.key === 'Escape' && callbacks.onEscape) {
      callbacks.onEscape();
      return true;
    }
    return false;
  }

  switch (event.key) {
    case 'Escape':
      if (callbacks.onEscape) {
        callbacks.onEscape();
        return true;
      }
      break;

    case 'Enter':
      if (callbacks.onEnter) {
        event.preventDefault();
        callbacks.onEnter();
        return true;
      }
      break;

    case 'ArrowUp':
      if (callbacks.onArrowUp) {
        event.preventDefault();
        callbacks.onArrowUp();
        return true;
      }
      break;

    case 'ArrowDown':
      if (callbacks.onArrowDown) {
        event.preventDefault();
        callbacks.onArrowDown();
        return true;
      }
      break;

    case 'ArrowLeft':
      if (callbacks.onArrowLeft) {
        event.preventDefault();
        callbacks.onArrowLeft();
        return true;
      }
      break;

    case 'ArrowRight':
      if (callbacks.onArrowRight) {
        event.preventDefault();
        callbacks.onArrowRight();
        return true;
      }
      break;

    case 'Tab':
      if (callbacks.onTab) {
        event.preventDefault();
        callbacks.onTab(event.shiftKey);
        return true;
      }
      break;

    // Vim-style navigation
    case 'j':
      if (state.isVimMode && callbacks.onVimJ) {
        event.preventDefault();
        callbacks.onVimJ();
        return true;
      }
      break;

    case 'k':
      if (state.isVimMode && callbacks.onVimK) {
        event.preventDefault();
        callbacks.onVimK();
        return true;
      }
      break;

    case 'g':
      if (state.isVimMode && !event.shiftKey && callbacks.onVimG) {
        event.preventDefault();
        callbacks.onVimG();
        return true;
      }
      break;

    case 'G':
      if (state.isVimMode && event.shiftKey && callbacks.onVimShiftG) {
        event.preventDefault();
        callbacks.onVimShiftG();
        return true;
      }
      break;
  }

  return false;
}

// ARIA live region announcements
let announceElement: HTMLElement | null = null;

/**
 * Initialize the ARIA announcer element
 */
export function initAnnouncer(): void {
  if (typeof document === 'undefined') return;

  announceElement = document.getElementById('aria-announcer');
  if (!announceElement) {
    announceElement = document.createElement('div');
    announceElement.id = 'aria-announcer';
    announceElement.setAttribute('role', 'status');
    announceElement.setAttribute('aria-live', 'polite');
    announceElement.setAttribute('aria-atomic', 'true');
    announceElement.className = 'sr-only';
    document.body.appendChild(announceElement);
  }
}

/**
 * Clean up the ARIA announcer element
 * Call this on app unmount to prevent DOM leaks
 */
export function destroyAnnouncer(): void {
  if (announceElement && announceElement.parentNode) {
    announceElement.parentNode.removeChild(announceElement);
  }
  announceElement = null;
}

/**
 * Announce a message to screen readers
 */
export function announce(message: string, priority: 'polite' | 'assertive' = 'polite'): void {
  if (!announceElement) {
    initAnnouncer();
  }

  if (announceElement) {
    announceElement.setAttribute('aria-live', priority);
    announceElement.textContent = '';
    // Small delay to ensure the change is announced
    requestAnimationFrame(() => {
      if (announceElement) {
        announceElement.textContent = message;
      }
    });
  }
}
