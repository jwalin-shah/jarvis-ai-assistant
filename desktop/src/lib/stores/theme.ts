/**
 * Theme store for managing dark/light/system theme preferences
 */

import { writable, get } from "svelte/store";

export type ThemeMode = "dark" | "light" | "system";

const STORAGE_KEY = "jarvis-theme";

// Get initial theme from localStorage or default to 'dark'
function getInitialTheme(): ThemeMode {
  if (typeof window === "undefined") return "dark";
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "dark" || stored === "light" || stored === "system") {
    return stored;
  }
  return "dark";
}

// Create the theme store
export const themeMode = writable<ThemeMode>(getInitialTheme());

// Media query for system preference
let mediaQuery: MediaQueryList | null = null;

/**
 * Apply the theme to the document
 */
function applyTheme(mode: ThemeMode): void {
  if (typeof window === "undefined") return;

  let shouldBeDark: boolean;

  if (mode === "system") {
    // Use system preference
    shouldBeDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  } else {
    shouldBeDark = mode === "dark";
  }

  // Toggle the 'light' class on the root element
  if (shouldBeDark) {
    document.documentElement.classList.remove("light");
  } else {
    document.documentElement.classList.add("light");
  }
}

/**
 * Set the theme mode and persist to localStorage
 */
export function setTheme(mode: ThemeMode): void {
  themeMode.set(mode);
  if (typeof window !== "undefined") {
    localStorage.setItem(STORAGE_KEY, mode);
  }
  applyTheme(mode);
}

/**
 * Initialize the theme system
 * Call this once on app mount
 */
export function initializeTheme(): () => void {
  if (typeof window === "undefined") return () => {};

  // Apply initial theme
  applyTheme(get(themeMode));

  // Set up media query listener for system theme changes
  mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

  const handleMediaChange = (e: MediaQueryListEvent) => {
    const currentMode = get(themeMode);
    if (currentMode === "system") {
      applyTheme("system");
    }
  };

  mediaQuery.addEventListener("change", handleMediaChange);

  // Subscribe to theme changes
  const unsubscribe = themeMode.subscribe((mode) => {
    applyTheme(mode);
  });

  // Return cleanup function
  return () => {
    unsubscribe();
    mediaQuery?.removeEventListener("change", handleMediaChange);
  };
}

/**
 * Get the current effective theme (resolved 'system' to actual theme)
 */
export function getEffectiveTheme(): "dark" | "light" {
  const mode = get(themeMode);
  if (mode === "system") {
    if (typeof window !== "undefined") {
      return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    return "dark";
  }
  return mode;
}
