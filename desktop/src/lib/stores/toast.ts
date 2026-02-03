/**
 * Toast notification store for user feedback
 */

import { writable, derived } from "svelte/store";

export type ToastType = "success" | "error" | "warning" | "info";

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  description?: string;
  duration: number;
  dismissible: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface ToastState {
  toasts: Toast[];
}

const initialState: ToastState = {
  toasts: [],
};

const toastStore = writable<ToastState>(initialState);

// Derived store for active toasts
export const toasts = derived(toastStore, ($state) => $state.toasts);

// Counter for unique IDs
let toastCounter = 0;

/**
 * Show a toast notification
 */
export function showToast(
  message: string,
  options: Partial<Omit<Toast, "id" | "message">> = {}
): string {
  const id = `toast-${++toastCounter}-${Date.now()}`;

  const toast: Toast = {
    id,
    message,
    type: options.type ?? "info",
    duration: options.duration ?? 4000,
    dismissible: options.dismissible ?? true,
    description: options.description,
    action: options.action,
  };

  toastStore.update((state) => ({
    ...state,
    toasts: [...state.toasts, toast],
  }));

  // Auto-dismiss after duration (if not 0)
  if (toast.duration > 0) {
    setTimeout(() => {
      dismissToast(id);
    }, toast.duration);
  }

  return id;
}

/**
 * Dismiss a specific toast
 */
export function dismissToast(id: string): void {
  toastStore.update((state) => ({
    ...state,
    toasts: state.toasts.filter((t) => t.id !== id),
  }));
}

/**
 * Dismiss all toasts
 */
export function dismissAllToasts(): void {
  toastStore.update((state) => ({
    ...state,
    toasts: [],
  }));
}

// Convenience functions
export const toast = {
  success: (message: string, options?: Partial<Omit<Toast, "id" | "message" | "type">>) =>
    showToast(message, { ...options, type: "success" }),

  error: (message: string, options?: Partial<Omit<Toast, "id" | "message" | "type">>) =>
    showToast(message, { ...options, type: "error", duration: options?.duration ?? 6000 }),

  warning: (message: string, options?: Partial<Omit<Toast, "id" | "message" | "type">>) =>
    showToast(message, { ...options, type: "warning" }),

  info: (message: string, options?: Partial<Omit<Toast, "id" | "message" | "type">>) =>
    showToast(message, { ...options, type: "info" }),

  promise: async <T>(
    promise: Promise<T>,
    messages: {
      loading: string;
      success: string | ((data: T) => string);
      error: string | ((err: Error) => string);
    }
  ): Promise<T> => {
    const id = showToast(messages.loading, { type: "info", duration: 0, dismissible: false });

    try {
      const result = await promise;
      dismissToast(id);
      const successMsg = typeof messages.success === "function"
        ? messages.success(result)
        : messages.success;
      showToast(successMsg, { type: "success" });
      return result;
    } catch (err) {
      dismissToast(id);
      const errorMsg = typeof messages.error === "function"
        ? messages.error(err as Error)
        : messages.error;
      showToast(errorMsg, { type: "error" });
      throw err;
    }
  },
};
