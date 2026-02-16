/**
 * Theme store for managing dark/light/system theme preferences
 * with accent color customization and reduced motion support
 */

import { writable, get, derived } from 'svelte/store';

export type ThemeMode = 'dark' | 'light' | 'system';

// Accent color presets
export const accentColors = {
  blue: { name: 'Blue', value: '#007AFF', rgb: '0, 122, 255' },
  purple: { name: 'Purple', value: '#5856D6', rgb: '88, 86, 214' },
  pink: { name: 'Pink', value: '#FF2D55', rgb: '255, 45, 85' },
  red: { name: 'Red', value: '#FF3B30', rgb: '255, 59, 48' },
  orange: { name: 'Orange', value: '#FF9500', rgb: '255, 149, 0' },
  yellow: { name: 'Yellow', value: '#FFCC00', rgb: '255, 204, 0' },
  green: { name: 'Green', value: '#34C759', rgb: '52, 199, 89' },
  teal: { name: 'Teal', value: '#5AC8FA', rgb: '90, 200, 250' },
  indigo: { name: 'Indigo', value: '#5E5CE6', rgb: '94, 92, 230' },
} as const;

export type AccentColorKey = keyof typeof accentColors;

interface ThemeState {
  mode: ThemeMode;
  accentColor: AccentColorKey;
  reducedMotion: boolean;
}

const STORAGE_KEY = 'jarvis-theme';
const ACCENT_STORAGE_KEY = 'jarvis-accent-color';
const MOTION_STORAGE_KEY = 'jarvis-reduced-motion';

// Get initial values from localStorage
function getInitialState(): ThemeState {
  if (typeof window === 'undefined') {
    return { mode: 'dark', accentColor: 'blue', reducedMotion: false };
  }

  const storedMode = localStorage.getItem(STORAGE_KEY);
  const storedAccent = localStorage.getItem(ACCENT_STORAGE_KEY);
  const storedMotion = localStorage.getItem(MOTION_STORAGE_KEY);

  // Check system preference for reduced motion
  const systemPrefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  return {
    mode:
      storedMode === 'dark' || storedMode === 'light' || storedMode === 'system'
        ? storedMode
        : 'dark',
    accentColor:
      storedAccent && storedAccent in accentColors ? (storedAccent as AccentColorKey) : 'blue',
    reducedMotion: storedMotion !== null ? storedMotion === 'true' : systemPrefersReducedMotion,
  };
}

// Create the theme store
const themeState = writable<ThemeState>(getInitialState());

// Derived stores for individual values
export const themeMode = derived(themeState, ($state) => $state.mode);
export const accentColor = derived(themeState, ($state) => $state.accentColor);
export const reducedMotion = derived(themeState, ($state) => $state.reducedMotion);

// Media query for system preference
let mediaQuery: MediaQueryList | null = null;
let motionMediaQuery: MediaQueryList | null = null;

/**
 * Apply all theme settings to the document
 */
function applyTheme(state: ThemeState): void {
  if (typeof window === 'undefined') return;

  const { mode, accentColor: accent, reducedMotion: motion } = state;

  // Determine light/dark mode
  let shouldBeDark: boolean;
  if (mode === 'system') {
    shouldBeDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  } else {
    shouldBeDark = mode === 'dark';
  }

  // Toggle light class
  if (shouldBeDark) {
    document.documentElement.classList.remove('light');
  } else {
    document.documentElement.classList.add('light');
  }

  // Apply accent color
  const colorConfig = accentColors[accent];
  document.documentElement.style.setProperty('--color-primary', colorConfig.value);
  document.documentElement.style.setProperty('--color-primary-rgb', colorConfig.rgb);
  document.documentElement.style.setProperty('--accent-color', colorConfig.value);
  document.documentElement.style.setProperty('--bubble-me', colorConfig.value);
  document.documentElement.style.setProperty('--bg-bubble-me', colorConfig.value);

  // Apply reduced motion
  if (motion) {
    document.documentElement.classList.add('reduce-motion');
  } else {
    document.documentElement.classList.remove('reduce-motion');
  }
}

/**
 * Set the theme mode
 */
export function setTheme(mode: ThemeMode): void {
  themeState.update((state) => ({ ...state, mode }));
  if (typeof window !== 'undefined') {
    localStorage.setItem(STORAGE_KEY, mode);
  }
}

/**
 * Set the accent color
 */
export function setAccentColor(color: AccentColorKey): void {
  themeState.update((state) => ({ ...state, accentColor: color }));
  if (typeof window !== 'undefined') {
    localStorage.setItem(ACCENT_STORAGE_KEY, color);
  }
}

/**
 * Set reduced motion preference
 */
export function setReducedMotion(enabled: boolean): void {
  themeState.update((state) => ({ ...state, reducedMotion: enabled }));
  if (typeof window !== 'undefined') {
    localStorage.setItem(MOTION_STORAGE_KEY, String(enabled));
  }
}

/**
 * Initialize the theme system
 * Call this once on app mount
 */
export function initializeTheme(): () => void {
  if (typeof window === 'undefined') return () => {};

  // Apply initial theme
  applyTheme(get(themeState));

  // Set up media query listeners
  mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
  motionMediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');

  const handleMediaChange = () => {
    const currentState = get(themeState);
    if (currentState.mode === 'system') {
      applyTheme(currentState);
    }
  };

  const handleMotionChange = (e: MediaQueryListEvent) => {
    // Only auto-update if user hasn't explicitly set a preference
    const storedMotion = localStorage.getItem(MOTION_STORAGE_KEY);
    if (storedMotion === null) {
      themeState.update((state) => ({ ...state, reducedMotion: e.matches }));
    }
  };

  mediaQuery.addEventListener('change', handleMediaChange);
  motionMediaQuery.addEventListener('change', handleMotionChange);

  // Subscribe to theme state changes
  const unsubscribe = themeState.subscribe((state) => {
    applyTheme(state);
  });

  // Return cleanup function
  return () => {
    unsubscribe();
    mediaQuery?.removeEventListener('change', handleMediaChange);
    motionMediaQuery?.removeEventListener('change', handleMotionChange);
  };
}

/**
 * Get the current effective theme (resolved 'system' to actual theme)
 */
export function getEffectiveTheme(): 'dark' | 'light' {
  const state = get(themeState);
  if (state.mode === 'system') {
    if (typeof window !== 'undefined') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    return 'dark';
  }
  return state.mode;
}
