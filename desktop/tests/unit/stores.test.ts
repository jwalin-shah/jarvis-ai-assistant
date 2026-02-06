/**
 * Unit tests for Svelte stores: toast, keyboard, theme, digest, health
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { get } from "svelte/store";

// ---------------------------------------------------------------------------
// Mocks - must be declared before imports that use them
// ---------------------------------------------------------------------------

vi.mock("../../src/lib/api/client", () => ({
  api: {
    ping: vi.fn(),
    getHealth: vi.fn(),
    generateDigest: vi.fn(),
    getDailyDigest: vi.fn(),
    getWeeklyDigest: vi.fn(),
    exportDigest: vi.fn(),
    getDigestPreferences: vi.fn(),
    updateDigestPreferences: vi.fn(),
  },
}));

vi.mock("../../src/lib/socket", () => ({
  jarvis: {
    connect: vi.fn(),
    ping: vi.fn(),
  },
}));

// ==========================================================================
// Toast Store
// ==========================================================================

describe("toast store", () => {
  let toasts: typeof import("../../src/lib/stores/toast").toasts;
  let showToast: typeof import("../../src/lib/stores/toast").showToast;
  let dismissToast: typeof import("../../src/lib/stores/toast").dismissToast;
  let dismissAllToasts: typeof import("../../src/lib/stores/toast").dismissAllToasts;
  let toast: typeof import("../../src/lib/stores/toast").toast;

  beforeEach(async () => {
    vi.useFakeTimers();
    // Re-import fresh module each test to reset internal state
    vi.resetModules();
    const mod = await import("../../src/lib/stores/toast");
    toasts = mod.toasts;
    showToast = mod.showToast;
    dismissToast = mod.dismissToast;
    dismissAllToasts = mod.dismissAllToasts;
    toast = mod.toast;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts with an empty toast list", () => {
    expect(get(toasts)).toEqual([]);
  });

  it("showToast adds a toast with defaults", () => {
    const id = showToast("Hello");
    const items = get(toasts);

    expect(items).toHaveLength(1);
    expect(items[0].message).toBe("Hello");
    expect(items[0].type).toBe("info");
    expect(items[0].duration).toBe(4000);
    expect(items[0].dismissible).toBe(true);
    expect(items[0].id).toBe(id);
  });

  it("showToast respects custom options", () => {
    const action = { label: "Undo", onClick: vi.fn() };
    showToast("Oops", {
      type: "error",
      duration: 10000,
      dismissible: false,
      description: "Something went wrong",
      action,
    });

    const items = get(toasts);
    expect(items[0].type).toBe("error");
    expect(items[0].duration).toBe(10000);
    expect(items[0].dismissible).toBe(false);
    expect(items[0].description).toBe("Something went wrong");
    expect(items[0].action).toBe(action);
  });

  it("auto-dismisses after duration", () => {
    showToast("Temp", { duration: 3000 });
    expect(get(toasts)).toHaveLength(1);

    vi.advanceTimersByTime(3000);
    expect(get(toasts)).toHaveLength(0);
  });

  it("does not auto-dismiss when duration is 0", () => {
    showToast("Sticky", { duration: 0 });
    vi.advanceTimersByTime(60000);
    expect(get(toasts)).toHaveLength(1);
  });

  it("dismissToast removes only the targeted toast", () => {
    const id1 = showToast("First");
    const id2 = showToast("Second");
    expect(get(toasts)).toHaveLength(2);

    dismissToast(id1);
    const remaining = get(toasts);
    expect(remaining).toHaveLength(1);
    expect(remaining[0].id).toBe(id2);
  });

  it("dismissAllToasts clears every toast", () => {
    showToast("A");
    showToast("B");
    showToast("C");
    expect(get(toasts)).toHaveLength(3);

    dismissAllToasts();
    expect(get(toasts)).toHaveLength(0);
  });

  it("generates unique IDs for each toast", () => {
    const id1 = showToast("X");
    const id2 = showToast("Y");
    expect(id1).not.toBe(id2);
  });

  // -- convenience helpers --

  it("toast.success sets type to success", () => {
    toast.success("Done");
    expect(get(toasts)[0].type).toBe("success");
  });

  it("toast.error sets type to error with 6s default duration", () => {
    toast.error("Fail");
    const t = get(toasts)[0];
    expect(t.type).toBe("error");
    expect(t.duration).toBe(6000);
  });

  it("toast.error respects custom duration", () => {
    toast.error("Fail", { duration: 1000 });
    expect(get(toasts)[0].duration).toBe(1000);
  });

  it("toast.warning sets type to warning", () => {
    toast.warning("Watch out");
    expect(get(toasts)[0].type).toBe("warning");
  });

  it("toast.info sets type to info", () => {
    toast.info("FYI");
    expect(get(toasts)[0].type).toBe("info");
  });

  it("toast.promise resolves: replaces loading toast with success", async () => {
    const resolved = Promise.resolve(42);
    const result = toast.promise(resolved, {
      loading: "Loading...",
      success: "Got it!",
      error: "Nope",
    });

    // Loading toast present immediately
    expect(get(toasts)).toHaveLength(1);
    expect(get(toasts)[0].message).toBe("Loading...");
    expect(get(toasts)[0].duration).toBe(0);
    expect(get(toasts)[0].dismissible).toBe(false);

    const value = await result;
    expect(value).toBe(42);

    // Loading dismissed, success added
    const items = get(toasts);
    expect(items).toHaveLength(1);
    expect(items[0].type).toBe("success");
    expect(items[0].message).toBe("Got it!");
  });

  it("toast.promise rejects: replaces loading toast with error", async () => {
    const rejected = Promise.reject(new Error("boom"));
    const call = toast.promise(rejected, {
      loading: "Working...",
      success: "OK",
      error: (err: Error) => `Failed: ${err.message}`,
    });

    await expect(call).rejects.toThrow("boom");

    const items = get(toasts);
    expect(items).toHaveLength(1);
    expect(items[0].type).toBe("error");
    expect(items[0].message).toBe("Failed: boom");
  });

  it("toast.promise success with function message", async () => {
    const resolved = Promise.resolve({ count: 5 });
    await toast.promise(resolved, {
      loading: "Counting...",
      success: (data: { count: number }) => `Found ${data.count} items`,
      error: "Error",
    });

    const items = get(toasts);
    expect(items[0].message).toBe("Found 5 items");
  });
});

// ==========================================================================
// Keyboard Store
// ==========================================================================

describe("keyboard store", () => {
  let activeZone: typeof import("../../src/lib/stores/keyboard").activeZone;
  let conversationIndex: typeof import("../../src/lib/stores/keyboard").conversationIndex;
  let messageIndex: typeof import("../../src/lib/stores/keyboard").messageIndex;
  let isVimMode: typeof import("../../src/lib/stores/keyboard").isVimMode;
  let setActiveZone: typeof import("../../src/lib/stores/keyboard").setActiveZone;
  let setConversationIndex: typeof import("../../src/lib/stores/keyboard").setConversationIndex;
  let moveConversationSelection: typeof import("../../src/lib/stores/keyboard").moveConversationSelection;
  let setMessageIndex: typeof import("../../src/lib/stores/keyboard").setMessageIndex;
  let moveMessageSelection: typeof import("../../src/lib/stores/keyboard").moveMessageSelection;
  let toggleVimMode: typeof import("../../src/lib/stores/keyboard").toggleVimMode;
  let resetKeyboardState: typeof import("../../src/lib/stores/keyboard").resetKeyboardState;
  let handleGlobalKeydown: typeof import("../../src/lib/stores/keyboard").handleGlobalKeydown;

  beforeEach(async () => {
    vi.resetModules();
    const mod = await import("../../src/lib/stores/keyboard");
    activeZone = mod.activeZone;
    conversationIndex = mod.conversationIndex;
    messageIndex = mod.messageIndex;
    isVimMode = mod.isVimMode;
    setActiveZone = mod.setActiveZone;
    setConversationIndex = mod.setConversationIndex;
    moveConversationSelection = mod.moveConversationSelection;
    setMessageIndex = mod.setMessageIndex;
    moveMessageSelection = mod.moveMessageSelection;
    toggleVimMode = mod.toggleVimMode;
    resetKeyboardState = mod.resetKeyboardState;
    handleGlobalKeydown = mod.handleGlobalKeydown;
  });

  // -- initial state --

  it("has correct initial state", () => {
    expect(get(activeZone)).toBe(null);
    expect(get(conversationIndex)).toBe(-1);
    expect(get(messageIndex)).toBe(-1);
    expect(get(isVimMode)).toBe(false);
  });

  // -- setters --

  it("setActiveZone updates the active zone", () => {
    setActiveZone("sidebar");
    expect(get(activeZone)).toBe("sidebar");

    setActiveZone("messages");
    expect(get(activeZone)).toBe("messages");
  });

  it("setConversationIndex updates the index", () => {
    setConversationIndex(3);
    expect(get(conversationIndex)).toBe(3);
  });

  it("setMessageIndex updates the index", () => {
    setMessageIndex(7);
    expect(get(messageIndex)).toBe(7);
  });

  // -- movement --

  it("moveConversationSelection clamps within bounds", () => {
    setConversationIndex(0);

    // Move down
    const idx1 = moveConversationSelection(1, 5);
    expect(idx1).toBe(1);
    expect(get(conversationIndex)).toBe(1);

    // Move past max clamps to max
    const idx2 = moveConversationSelection(100, 5);
    expect(idx2).toBe(5);

    // Move below 0 clamps to 0
    const idx3 = moveConversationSelection(-100, 5);
    expect(idx3).toBe(0);
  });

  it("moveMessageSelection clamps within bounds", () => {
    setMessageIndex(2);

    const idx1 = moveMessageSelection(1, 10);
    expect(idx1).toBe(3);
    expect(get(messageIndex)).toBe(3);

    const idx2 = moveMessageSelection(-100, 10);
    expect(idx2).toBe(0);

    const idx3 = moveMessageSelection(999, 10);
    expect(idx3).toBe(10);
  });

  // -- vim mode --

  it("toggleVimMode flips the boolean", () => {
    expect(get(isVimMode)).toBe(false);
    toggleVimMode();
    expect(get(isVimMode)).toBe(true);
    toggleVimMode();
    expect(get(isVimMode)).toBe(false);
  });

  // -- reset --

  it("resetKeyboardState restores defaults", () => {
    setActiveZone("compose");
    setConversationIndex(5);
    setMessageIndex(3);
    toggleVimMode();

    resetKeyboardState();

    expect(get(activeZone)).toBe(null);
    expect(get(conversationIndex)).toBe(-1);
    expect(get(messageIndex)).toBe(-1);
    expect(get(isVimMode)).toBe(false);
  });

  // -- handleGlobalKeydown --
  //
  // The handler checks `event.target instanceof HTMLInputElement`.
  // In node env, these DOM classes don't exist, so we stub them globally
  // and create lightweight instances via Object.create for the instanceof
  // check to work.

  // Minimal stubs for HTMLInputElement / HTMLTextAreaElement
  class FakeHTMLInputElement {}
  class FakeHTMLTextAreaElement {}

  beforeEach(() => {
    // Ensure the global DOM classes exist so `instanceof` works inside the store
    if (typeof globalThis.HTMLInputElement === "undefined") {
      (globalThis as Record<string, unknown>).HTMLInputElement = FakeHTMLInputElement;
    }
    if (typeof globalThis.HTMLTextAreaElement === "undefined") {
      (globalThis as Record<string, unknown>).HTMLTextAreaElement = FakeHTMLTextAreaElement;
    }
  });

  function makeKeyEvent(
    key: string,
    overrides: Record<string, unknown> = {}
  ): KeyboardEvent {
    return {
      key,
      shiftKey: false,
      preventDefault: vi.fn(),
      target: {}, // plain object - not an input
      ...overrides,
    } as unknown as KeyboardEvent;
  }

  function makeInputTarget(): object {
    return Object.create(FakeHTMLInputElement.prototype);
  }

  function makeTextareaTarget(): object {
    return Object.create(FakeHTMLTextAreaElement.prototype);
  }

  it("dispatches Escape callback", () => {
    const onEscape = vi.fn();
    const handled = handleGlobalKeydown(makeKeyEvent("Escape"), { onEscape });
    expect(handled).toBe(true);
    expect(onEscape).toHaveBeenCalledOnce();
  });

  it("dispatches Enter callback and prevents default", () => {
    const onEnter = vi.fn();
    const event = makeKeyEvent("Enter");
    handleGlobalKeydown(event, { onEnter });
    expect(onEnter).toHaveBeenCalledOnce();
    expect(event.preventDefault).toHaveBeenCalled();
  });

  it("dispatches arrow key callbacks", () => {
    const onUp = vi.fn();
    const onDown = vi.fn();
    const onLeft = vi.fn();
    const onRight = vi.fn();

    handleGlobalKeydown(makeKeyEvent("ArrowUp"), { onArrowUp: onUp });
    handleGlobalKeydown(makeKeyEvent("ArrowDown"), { onArrowDown: onDown });
    handleGlobalKeydown(makeKeyEvent("ArrowLeft"), { onArrowLeft: onLeft });
    handleGlobalKeydown(makeKeyEvent("ArrowRight"), { onArrowRight: onRight });

    expect(onUp).toHaveBeenCalledOnce();
    expect(onDown).toHaveBeenCalledOnce();
    expect(onLeft).toHaveBeenCalledOnce();
    expect(onRight).toHaveBeenCalledOnce();
  });

  it("dispatches Tab with shift info", () => {
    const onTab = vi.fn();
    handleGlobalKeydown(makeKeyEvent("Tab", { shiftKey: true }), { onTab });
    expect(onTab).toHaveBeenCalledWith(true);

    handleGlobalKeydown(makeKeyEvent("Tab", { shiftKey: false }), { onTab });
    expect(onTab).toHaveBeenCalledWith(false);
  });

  it("returns false when no callback matches", () => {
    const handled = handleGlobalKeydown(makeKeyEvent("ArrowUp"), {});
    expect(handled).toBe(false);
  });

  it("ignores non-Escape keys when target is an input", () => {
    const onEnter = vi.fn();
    const handled = handleGlobalKeydown(
      makeKeyEvent("Enter", { target: makeInputTarget() }),
      { onEnter }
    );
    expect(handled).toBe(false);
    expect(onEnter).not.toHaveBeenCalled();
  });

  it("still handles Escape inside an input", () => {
    const onEscape = vi.fn();
    const handled = handleGlobalKeydown(
      makeKeyEvent("Escape", { target: makeInputTarget() }),
      { onEscape }
    );
    expect(handled).toBe(true);
    expect(onEscape).toHaveBeenCalledOnce();
  });

  it("ignores non-Escape keys when target is a textarea", () => {
    const onArrowUp = vi.fn();
    const handled = handleGlobalKeydown(
      makeKeyEvent("ArrowUp", { target: makeTextareaTarget() }),
      { onArrowUp }
    );
    expect(handled).toBe(false);
    expect(onArrowUp).not.toHaveBeenCalled();
  });

  // -- vim keys --

  it("vim j/k only fire in vim mode", () => {
    const onJ = vi.fn();
    const onK = vi.fn();

    // Not in vim mode: should not fire
    handleGlobalKeydown(makeKeyEvent("j"), { onVimJ: onJ });
    handleGlobalKeydown(makeKeyEvent("k"), { onVimK: onK });
    expect(onJ).not.toHaveBeenCalled();
    expect(onK).not.toHaveBeenCalled();

    // Enable vim mode
    toggleVimMode();

    handleGlobalKeydown(makeKeyEvent("j"), { onVimJ: onJ });
    handleGlobalKeydown(makeKeyEvent("k"), { onVimK: onK });
    expect(onJ).toHaveBeenCalledOnce();
    expect(onK).toHaveBeenCalledOnce();
  });

  it("vim g (go to top) fires in vim mode without shift", () => {
    toggleVimMode();
    const onG = vi.fn();
    const handled = handleGlobalKeydown(
      makeKeyEvent("g", { shiftKey: false }),
      { onVimG: onG }
    );
    expect(handled).toBe(true);
    expect(onG).toHaveBeenCalledOnce();
  });

  it("vim G (go to bottom) fires in vim mode with shift", () => {
    toggleVimMode();
    const onShiftG = vi.fn();
    const handled = handleGlobalKeydown(
      makeKeyEvent("G", { shiftKey: true }),
      { onVimShiftG: onShiftG }
    );
    expect(handled).toBe(true);
    expect(onShiftG).toHaveBeenCalledOnce();
  });
});

// ==========================================================================
// Theme Store
// ==========================================================================

describe("theme store", () => {
  let themeMode: typeof import("../../src/lib/stores/theme").themeMode;
  let accentColor: typeof import("../../src/lib/stores/theme").accentColor;
  let reducedMotion: typeof import("../../src/lib/stores/theme").reducedMotion;
  let setTheme: typeof import("../../src/lib/stores/theme").setTheme;
  let setAccentColor: typeof import("../../src/lib/stores/theme").setAccentColor;
  let setReducedMotion: typeof import("../../src/lib/stores/theme").setReducedMotion;
  let getEffectiveTheme: typeof import("../../src/lib/stores/theme").getEffectiveTheme;
  let accentColors: typeof import("../../src/lib/stores/theme").accentColors;

  // In-memory storage backing the localStorage mock
  let storage: Record<string, string>;

  // The theme store checks `typeof window !== "undefined"` at module load time
  // and inside setter functions. We need `window` to exist so those branches
  // are exercised. In node env, `window` is not defined, so we set it up here.
  let mockLocalStorage: {
    getItem: ReturnType<typeof vi.fn>;
    setItem: ReturnType<typeof vi.fn>;
    removeItem: ReturnType<typeof vi.fn>;
  };

  function setupBrowserGlobals(matchMediaFn?: (query: string) => { matches: boolean }) {
    storage = {};
    mockLocalStorage = {
      getItem: vi.fn((key: string) => storage[key] ?? null),
      setItem: vi.fn((key: string, val: string) => { storage[key] = val; }),
      removeItem: vi.fn((key: string) => { delete storage[key]; }),
    };

    const defaultMatchMedia = (query: string) => ({
      matches: query.includes("prefers-color-scheme: dark"),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    });

    const mediaFn = matchMediaFn
      ? (query: string) => ({
          ...defaultMatchMedia(query),
          matches: matchMediaFn(query).matches,
        })
      : defaultMatchMedia;

    // Stub window as a minimal object so `typeof window !== "undefined"` is true
    vi.stubGlobal("window", {
      matchMedia: vi.fn(mediaFn),
      localStorage: mockLocalStorage,
    });
    vi.stubGlobal("localStorage", mockLocalStorage);
    vi.stubGlobal("matchMedia", vi.fn(mediaFn));
    // Stub document.documentElement for applyTheme
    vi.stubGlobal("document", {
      documentElement: {
        classList: { add: vi.fn(), remove: vi.fn() },
        style: { setProperty: vi.fn() },
      },
    });
  }

  beforeEach(async () => {
    vi.resetModules();
    setupBrowserGlobals();

    const mod = await import("../../src/lib/stores/theme");
    themeMode = mod.themeMode;
    accentColor = mod.accentColor;
    reducedMotion = mod.reducedMotion;
    setTheme = mod.setTheme;
    setAccentColor = mod.setAccentColor;
    setReducedMotion = mod.setReducedMotion;
    getEffectiveTheme = mod.getEffectiveTheme;
    accentColors = mod.accentColors;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("defaults to dark mode, blue accent, no reduced motion", () => {
    expect(get(themeMode)).toBe("dark");
    expect(get(accentColor)).toBe("blue");
    expect(get(reducedMotion)).toBe(false);
  });

  it("reads stored theme mode from localStorage", async () => {
    vi.resetModules();
    storage["jarvis-theme"] = "light";
    const mod = await import("../../src/lib/stores/theme");
    expect(get(mod.themeMode)).toBe("light");
  });

  it("reads stored accent color from localStorage", async () => {
    vi.resetModules();
    storage["jarvis-accent-color"] = "purple";
    const mod = await import("../../src/lib/stores/theme");
    expect(get(mod.accentColor)).toBe("purple");
  });

  it("ignores invalid stored theme mode", async () => {
    vi.resetModules();
    storage["jarvis-theme"] = "neon";
    const mod = await import("../../src/lib/stores/theme");
    expect(get(mod.themeMode)).toBe("dark"); // falls back to default
  });

  it("ignores invalid stored accent color", async () => {
    vi.resetModules();
    storage["jarvis-accent-color"] = "rainbow";
    const mod = await import("../../src/lib/stores/theme");
    expect(get(mod.accentColor)).toBe("blue"); // falls back to default
  });

  it("reads stored reduced motion preference", async () => {
    vi.resetModules();
    storage["jarvis-reduced-motion"] = "true";
    const mod = await import("../../src/lib/stores/theme");
    expect(get(mod.reducedMotion)).toBe(true);
  });

  it("uses system prefers-reduced-motion when no stored preference", async () => {
    vi.resetModules();
    vi.unstubAllGlobals();
    // Set up globals where matchMedia always returns true (including reduced-motion)
    setupBrowserGlobals(() => ({ matches: true }));
    const mod = await import("../../src/lib/stores/theme");
    expect(get(mod.reducedMotion)).toBe(true);
  });

  it("setTheme updates store and persists to localStorage", () => {
    setTheme("light");
    expect(get(themeMode)).toBe("light");
    expect(mockLocalStorage.setItem).toHaveBeenCalledWith("jarvis-theme", "light");

    setTheme("system");
    expect(get(themeMode)).toBe("system");
    expect(mockLocalStorage.setItem).toHaveBeenCalledWith("jarvis-theme", "system");
  });

  it("setAccentColor updates store and persists to localStorage", () => {
    setAccentColor("green");
    expect(get(accentColor)).toBe("green");
    expect(mockLocalStorage.setItem).toHaveBeenCalledWith("jarvis-accent-color", "green");
  });

  it("setReducedMotion updates store and persists to localStorage", () => {
    setReducedMotion(true);
    expect(get(reducedMotion)).toBe(true);
    expect(mockLocalStorage.setItem).toHaveBeenCalledWith("jarvis-reduced-motion", "true");

    setReducedMotion(false);
    expect(get(reducedMotion)).toBe(false);
    expect(mockLocalStorage.setItem).toHaveBeenCalledWith("jarvis-reduced-motion", "false");
  });

  it("getEffectiveTheme returns dark/light for explicit modes", () => {
    setTheme("dark");
    expect(getEffectiveTheme()).toBe("dark");

    setTheme("light");
    expect(getEffectiveTheme()).toBe("light");
  });

  it("getEffectiveTheme resolves system mode via matchMedia", () => {
    setTheme("system");
    // matchMedia stub returns true for prefers-color-scheme: dark
    expect(getEffectiveTheme()).toBe("dark");
  });

  it("accentColors has the expected preset keys", () => {
    const keys = Object.keys(accentColors);
    expect(keys).toContain("blue");
    expect(keys).toContain("purple");
    expect(keys).toContain("green");
    expect(keys).toContain("red");
    expect(keys).toContain("orange");
    expect(keys).toContain("teal");
    expect(keys).toContain("indigo");
  });

  it("each accent color preset has name, value, and rgb", () => {
    for (const [, preset] of Object.entries(accentColors)) {
      expect(preset).toHaveProperty("name");
      expect(preset).toHaveProperty("value");
      expect(preset).toHaveProperty("rgb");
      expect(preset.value).toMatch(/^#[0-9A-Fa-f]{6}$/);
    }
  });
});

// ==========================================================================
// Digest Store
// ==========================================================================

describe("digest store", () => {
  let digestStore: typeof import("../../src/lib/stores/digest").digestStore;
  let fetchDigest: typeof import("../../src/lib/stores/digest").fetchDigest;
  let fetchDailyDigest: typeof import("../../src/lib/stores/digest").fetchDailyDigest;
  let fetchWeeklyDigest: typeof import("../../src/lib/stores/digest").fetchWeeklyDigest;
  let exportDigest: typeof import("../../src/lib/stores/digest").exportDigest;
  let fetchDigestPreferences: typeof import("../../src/lib/stores/digest").fetchDigestPreferences;
  let updateDigestPreferences: typeof import("../../src/lib/stores/digest").updateDigestPreferences;
  let clearDigestError: typeof import("../../src/lib/stores/digest").clearDigestError;
  let resetDigestStore: typeof import("../../src/lib/stores/digest").resetDigestStore;
  let mockApi: Record<string, ReturnType<typeof vi.fn>>;

  beforeEach(async () => {
    vi.resetModules();
    // Re-import mocked api
    const apiMod = await import("../../src/lib/api/client");
    mockApi = apiMod.api as unknown as Record<string, ReturnType<typeof vi.fn>>;

    const mod = await import("../../src/lib/stores/digest");
    digestStore = mod.digestStore;
    fetchDigest = mod.fetchDigest;
    fetchDailyDigest = mod.fetchDailyDigest;
    fetchWeeklyDigest = mod.fetchWeeklyDigest;
    exportDigest = mod.exportDigest;
    fetchDigestPreferences = mod.fetchDigestPreferences;
    updateDigestPreferences = mod.updateDigestPreferences;
    clearDigestError = mod.clearDigestError;
    resetDigestStore = mod.resetDigestStore;
  });

  it("has correct initial state", () => {
    const state = get(digestStore);
    expect(state).toEqual({
      loading: false,
      exporting: false,
      error: null,
      data: null,
      preferences: null,
      lastExport: null,
    });
  });

  // -- fetchDigest --

  it("fetchDigest sets loading then stores data on success", async () => {
    const fakeData = { period: "daily", generated_at: "now" };
    mockApi.generateDigest.mockResolvedValueOnce(fakeData);

    const promise = fetchDigest("daily");

    // After call starts, loading should be true (check via subscription)
    // After resolution:
    await promise;
    const state = get(digestStore);
    expect(state.loading).toBe(false);
    expect(state.data).toBe(fakeData);
    expect(state.error).toBe(null);
    expect(mockApi.generateDigest).toHaveBeenCalledWith({ period: "daily" });
  });

  it("fetchDigest stores error message on failure", async () => {
    mockApi.generateDigest.mockRejectedValueOnce(new Error("Network timeout"));
    await fetchDigest();

    const state = get(digestStore);
    expect(state.loading).toBe(false);
    expect(state.error).toBe("Network timeout");
    expect(state.data).toBe(null);
  });

  it("fetchDigest stores generic message for non-Error throws", async () => {
    mockApi.generateDigest.mockRejectedValueOnce("string error");
    await fetchDigest();

    const state = get(digestStore);
    expect(state.error).toBe("Failed to fetch digest");
  });

  // -- fetchDailyDigest --

  it("fetchDailyDigest calls api.getDailyDigest", async () => {
    const fakeData = { period: "daily" };
    mockApi.getDailyDigest.mockResolvedValueOnce(fakeData);

    await fetchDailyDigest();
    const state = get(digestStore);
    expect(state.data).toBe(fakeData);
    expect(state.loading).toBe(false);
    expect(mockApi.getDailyDigest).toHaveBeenCalled();
  });

  it("fetchDailyDigest stores error on failure", async () => {
    mockApi.getDailyDigest.mockRejectedValueOnce(new Error("fail"));
    await fetchDailyDigest();
    expect(get(digestStore).error).toBe("fail");
  });

  // -- fetchWeeklyDigest --

  it("fetchWeeklyDigest calls api.getWeeklyDigest", async () => {
    const fakeData = { period: "weekly" };
    mockApi.getWeeklyDigest.mockResolvedValueOnce(fakeData);

    await fetchWeeklyDigest();
    expect(get(digestStore).data).toBe(fakeData);
    expect(mockApi.getWeeklyDigest).toHaveBeenCalled();
  });

  it("fetchWeeklyDigest stores error on failure", async () => {
    mockApi.getWeeklyDigest.mockRejectedValueOnce(new Error("weekly fail"));
    await fetchWeeklyDigest();
    expect(get(digestStore).error).toBe("weekly fail");
  });

  // -- exportDigest --

  it("exportDigest sets exporting flag and returns response", async () => {
    const fakeExport = { success: true, format: "markdown", filename: "d.md", data: "..." };
    mockApi.exportDigest.mockResolvedValueOnce(fakeExport);

    const result = await exportDigest("daily", "markdown");
    const state = get(digestStore);

    expect(result).toBe(fakeExport);
    expect(state.exporting).toBe(false);
    expect(state.lastExport).toBe(fakeExport);
    expect(mockApi.exportDigest).toHaveBeenCalledWith({ period: "daily", format: "markdown" });
  });

  it("exportDigest returns null on failure", async () => {
    mockApi.exportDigest.mockRejectedValueOnce(new Error("export fail"));

    const result = await exportDigest();
    expect(result).toBeNull();
    expect(get(digestStore).exporting).toBe(false);
    expect(get(digestStore).error).toBe("export fail");
  });

  // -- preferences --

  it("fetchDigestPreferences stores preferences", async () => {
    const prefs = { enabled: true, schedule: "daily" };
    mockApi.getDigestPreferences.mockResolvedValueOnce(prefs);

    await fetchDigestPreferences();
    expect(get(digestStore).preferences).toBe(prefs);
  });

  it("fetchDigestPreferences stores error on failure", async () => {
    mockApi.getDigestPreferences.mockRejectedValueOnce(new Error("prefs fail"));
    await fetchDigestPreferences();
    expect(get(digestStore).error).toBe("prefs fail");
  });

  it("updateDigestPreferences stores updated preferences", async () => {
    const updatedPrefs = { enabled: false, schedule: "weekly" };
    mockApi.updateDigestPreferences.mockResolvedValueOnce(updatedPrefs);

    await updateDigestPreferences({ enabled: false });
    expect(get(digestStore).preferences).toBe(updatedPrefs);
  });

  it("updateDigestPreferences stores error on failure", async () => {
    mockApi.updateDigestPreferences.mockRejectedValueOnce(new Error("update fail"));
    await updateDigestPreferences({});
    expect(get(digestStore).error).toBe("update fail");
  });

  // -- utility --

  it("clearDigestError clears only the error field", async () => {
    mockApi.generateDigest.mockRejectedValueOnce(new Error("oops"));
    await fetchDigest();
    expect(get(digestStore).error).toBe("oops");

    clearDigestError();
    expect(get(digestStore).error).toBe(null);
  });

  it("resetDigestStore restores initial state", async () => {
    const fakeData = { period: "daily" };
    mockApi.generateDigest.mockResolvedValueOnce(fakeData);
    await fetchDigest();
    expect(get(digestStore).data).toBe(fakeData);

    resetDigestStore();
    expect(get(digestStore)).toEqual({
      loading: false,
      exporting: false,
      error: null,
      data: null,
      preferences: null,
      lastExport: null,
    });
  });
});

// ==========================================================================
// Health Store
// ==========================================================================

describe("health store", () => {
  // NOTE: The health store is tightly coupled to external APIs (socket + HTTP).
  // We test the initial state and the state transitions driven by the mocked APIs.
  // The `isTauri` check is evaluated at module load time based on `window.__TAURI__`,
  // so we test the HTTP-only path (non-Tauri env) and the initial state.

  let healthStore: typeof import("../../src/lib/stores/health").healthStore;
  let checkApiConnection: typeof import("../../src/lib/stores/health").checkApiConnection;
  let fetchHealth: typeof import("../../src/lib/stores/health").fetchHealth;
  let mockApi: Record<string, ReturnType<typeof vi.fn>>;

  beforeEach(async () => {
    vi.resetModules();
    const apiMod = await import("../../src/lib/api/client");
    mockApi = apiMod.api as unknown as Record<string, ReturnType<typeof vi.fn>>;

    const mod = await import("../../src/lib/stores/health");
    healthStore = mod.healthStore;
    checkApiConnection = mod.checkApiConnection;
    fetchHealth = mod.fetchHealth;
  });

  it("has correct initial state", () => {
    expect(get(healthStore)).toEqual({
      connected: false,
      loading: false,
      error: null,
      data: null,
      source: null,
    });
  });

  // In test env, window.__TAURI__ is not set, so isTauri = false.
  // All requests go through HTTP path.

  it("checkApiConnection returns true and sets connected on HTTP success", async () => {
    mockApi.ping.mockResolvedValueOnce({ status: "ok" });

    const result = await checkApiConnection();
    expect(result).toBe(true);

    const state = get(healthStore);
    expect(state.connected).toBe(true);
    expect(state.loading).toBe(false);
    expect(state.source).toBe("http");
    expect(state.error).toBe(null);
  });

  it("checkApiConnection returns false and sets error on HTTP failure", async () => {
    mockApi.ping.mockRejectedValueOnce(new Error("ECONNREFUSED"));

    const result = await checkApiConnection();
    expect(result).toBe(false);

    const state = get(healthStore);
    expect(state.connected).toBe(false);
    expect(state.loading).toBe(false);
    expect(state.error).toBe("ECONNREFUSED");
    expect(state.source).toBe(null);
  });

  it("checkApiConnection stores generic message for non-Error throws", async () => {
    mockApi.ping.mockRejectedValueOnce("string error");

    await checkApiConnection();
    expect(get(healthStore).error).toBe("Connection failed");
  });

  it("fetchHealth stores health data on HTTP success", async () => {
    const fakeHealth = {
      status: "healthy",
      imessage_access: true,
      memory_available_gb: 4.0,
      memory_used_gb: 3.5,
      memory_mode: "FULL",
      model_loaded: true,
      permissions_ok: true,
      details: {},
      jarvis_rss_mb: 200,
      jarvis_vms_mb: 1000,
    };
    mockApi.getHealth.mockResolvedValueOnce(fakeHealth);

    await fetchHealth();

    const state = get(healthStore);
    expect(state.data).toBe(fakeHealth);
    expect(state.connected).toBe(true);
    expect(state.loading).toBe(false);
    expect(state.source).toBe("http");
  });

  it("fetchHealth stores error on HTTP failure", async () => {
    mockApi.getHealth.mockRejectedValueOnce(new Error("Server down"));

    await fetchHealth();

    const state = get(healthStore);
    expect(state.loading).toBe(false);
    expect(state.error).toBe("Server down");
    expect(state.source).toBe(null);
  });

  it("fetchHealth stores generic error for non-Error throws", async () => {
    mockApi.getHealth.mockRejectedValueOnce(42);

    await fetchHealth();
    expect(get(healthStore).error).toBe("Failed to fetch health");
  });
});
