/**
 * Shared test fixtures for E2E tests.
 *
 * Extends the base Playwright test with custom fixtures that automatically
 * set up API mocking and provide helper utilities.
 */

import { test as base, expect, Page } from "@playwright/test";
import {
  setupApiMocks,
  setupApiMocksWithErrors,
  setupDisconnectedMocks,
  setupSocketMocks,
  setupSlowApiMocks,
  setupLargeDataMocks,
} from "../mocks";

/**
 * Performance metrics collected during tests
 */
export interface PerformanceMetrics {
  renderTime: number;
  heapUsed: number;
  heapTotal: number;
  domNodes: number;
  layoutShifts: number;
}

/**
 * Custom test fixtures
 */
type CustomFixtures = {
  /** Page with API mocks already set up */
  mockedPage: Page;
  /** Page with error mocks set up */
  errorPage: Page;
  /** Page with disconnected mocks set up */
  disconnectedPage: Page;
  /** Page with socket mocks for real-time features */
  socketMockedPage: Page;
  /** Page with slow API responses for testing loading states */
  slowPage: Page;
  /** Page with large dataset mocks for performance testing */
  largePage: Page;
};

/**
 * Extended test with custom fixtures
 */
export const test = base.extend<CustomFixtures>({
  mockedPage: async ({ page }, use) => {
    await setupApiMocks(page);
    await use(page);
  },
  errorPage: async ({ page }, use) => {
    await setupApiMocksWithErrors(page);
    await use(page);
  },
  disconnectedPage: async ({ page }, use) => {
    await setupDisconnectedMocks(page);
    await use(page);
  },
  socketMockedPage: async ({ page }, use) => {
    await setupApiMocks(page);
    await setupSocketMocks(page);
    await use(page);
  },
  slowPage: async ({ page }, use) => {
    await setupSlowApiMocks(page, 500);
    await use(page);
  },
  largePage: async ({ page }, use) => {
    await setupLargeDataMocks(page, 1000);
    await use(page);
  },
});

export { expect };

/**
 * Helper to wait for the app to fully load.
 * Waits for the sidebar and initial content to be visible.
 */
export async function waitForAppLoad(page: Page): Promise<void> {
  // Wait for sidebar to be visible
  await page.waitForSelector(".sidebar", { state: "visible" });

  // Wait for the JARVIS logo
  await page.waitForSelector(".logo-text", { state: "visible" });
}

/**
 * Helper to navigate to a specific view.
 */
export async function navigateToView(
  page: Page,
  view: "dashboard" | "messages" | "health" | "settings" | "templates" | "network"
): Promise<void> {
  // Click the nav button with the matching title
  const navButton = page.locator(`.nav-item[title="${capitalizeFirst(view)}"]`);

  // For "health", the title is "Health Status"
  if (view === "health") {
    await page.locator('.nav-item[title="Health Status"]').click();
  } else if (view === "templates") {
    await page.locator('.nav-item[title="Templates"]').click();
  } else {
    await navButton.click();
  }

  // Wait for view to be active
  const expectedTitle = view === "health" ? "Health Status" : capitalizeFirst(view);
  await page.waitForSelector(`.nav-item.active[title="${expectedTitle}"]`);
}

function capitalizeFirst(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Helper to select a conversation from the list.
 */
export async function selectConversation(
  page: Page,
  displayName: string
): Promise<void> {
  // Find the conversation by name and click it
  const conversation = page.locator(".conversation", {
    has: page.locator(".name", { hasText: displayName }),
  });
  await conversation.click();

  // Wait for it to become active
  await page.waitForSelector(".conversation.active");
}

/**
 * Helper to wait for messages to load in the message view.
 */
export async function waitForMessagesLoad(page: Page): Promise<void> {
  // Wait for either messages to appear or the "Select a conversation" placeholder
  await page.waitForSelector(".messages, .message-view .empty", {
    state: "visible",
  });
}

/**
 * Helper to get the connection status indicator (from sidebar).
 */
export async function getConnectionStatus(
  page: Page
): Promise<"connected" | "disconnected"> {
  const dot = page.locator(".sidebar .status-dot");
  const hasConnectedClass = await dot.evaluate((el) =>
    el.classList.contains("connected")
  );
  return hasConnectedClass ? "connected" : "disconnected";
}

/**
 * Helper to open global search (Cmd+K).
 */
export async function openGlobalSearch(page: Page): Promise<void> {
  await page.keyboard.press("Meta+k");
  await page.waitForSelector(".search-modal, .global-search", { state: "visible" });
}

/**
 * Helper to close global search.
 */
export async function closeGlobalSearch(page: Page): Promise<void> {
  await page.keyboard.press("Escape");
  await page.waitForSelector(".search-modal, .global-search", { state: "hidden" });
}

/**
 * Helper to type in search and wait for results.
 */
export async function searchAndWait(
  page: Page,
  query: string,
  expectResults: boolean = true
): Promise<void> {
  const searchInput = page.locator(".search-input, .search input");
  await searchInput.fill(query);

  // Wait for debounce and results
  await page.waitForTimeout(500);

  if (expectResults) {
    await page.waitForSelector(".result-item, .search-result");
  }
}

/**
 * Helper to measure render time for a selector.
 */
export async function measureRenderTime(
  page: Page,
  action: () => Promise<void>,
  selector: string
): Promise<number> {
  const startTime = Date.now();
  await action();
  await page.waitForSelector(selector, { state: "visible" });
  return Date.now() - startTime;
}

/**
 * Helper to collect performance metrics from the page.
 */
export async function collectPerformanceMetrics(
  page: Page
): Promise<PerformanceMetrics> {
  const metrics = await page.evaluate(() => {
    const perf = performance as Performance & {
      memory?: { usedJSHeapSize: number; totalJSHeapSize: number };
    };

    // Get layout shift score
    let layoutShifts = 0;
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const layoutEntry = entry as PerformanceEntry & { value?: number };
        layoutShifts += layoutEntry.value || 0;
      }
    });
    try {
      observer.observe({ type: "layout-shift", buffered: true });
    } catch {
      // Layout shift not supported
    }

    return {
      renderTime: performance.now(),
      heapUsed: perf.memory?.usedJSHeapSize || 0,
      heapTotal: perf.memory?.totalJSHeapSize || 0,
      domNodes: document.querySelectorAll("*").length,
      layoutShifts,
    };
  });

  return metrics;
}

/**
 * Helper to check accessibility attributes on an element.
 */
export async function checkAccessibility(
  page: Page,
  selector: string
): Promise<{
  hasRole: boolean;
  hasAriaLabel: boolean;
  hasTabIndex: boolean;
  isFocusable: boolean;
}> {
  const element = page.locator(selector).first();

  const role = await element.getAttribute("role");
  const ariaLabel = await element.getAttribute("aria-label");
  const tabIndex = await element.getAttribute("tabindex");
  const isFocusable = await element.evaluate((el) => {
    const focusableSelectors = [
      "a[href]",
      "button",
      "input",
      "select",
      "textarea",
      "[tabindex]",
    ];
    return focusableSelectors.some((s) => el.matches(s));
  });

  return {
    hasRole: !!role,
    hasAriaLabel: !!ariaLabel,
    hasTabIndex: tabIndex !== null,
    isFocusable,
  };
}

/**
 * Helper to test keyboard navigation through a list of elements.
 */
export async function testKeyboardNavigation(
  page: Page,
  containerSelector: string,
  itemSelector: string,
  key: "ArrowDown" | "ArrowUp" | "Tab" = "ArrowDown"
): Promise<number> {
  const container = page.locator(containerSelector);
  const items = container.locator(itemSelector);
  const count = await items.count();

  if (count === 0) return 0;

  // Focus the first item
  await items.first().focus();

  let navigatedCount = 0;

  for (let i = 0; i < count - 1; i++) {
    await page.keyboard.press(key);
    navigatedCount++;

    // Verify focus moved
    const focusedElement = page.locator(":focus");
    const isFocused = await focusedElement.count();
    if (isFocused === 0) break;
  }

  return navigatedCount;
}

/**
 * Helper to simulate slow network conditions.
 */
export async function simulateSlowNetwork(page: Page): Promise<void> {
  const client = await page.context().newCDPSession(page);
  await client.send("Network.emulateNetworkConditions", {
    offline: false,
    downloadThroughput: (500 * 1024) / 8, // 500 kbps
    uploadThroughput: (500 * 1024) / 8,
    latency: 300, // 300ms latency
  });
}

/**
 * Helper to simulate offline mode.
 */
export async function simulateOffline(page: Page): Promise<void> {
  await page.context().setOffline(true);
}

/**
 * Helper to restore online mode.
 */
export async function simulateOnline(page: Page): Promise<void> {
  await page.context().setOffline(false);
}

/**
 * Helper to wait for network idle (no pending requests).
 */
export async function waitForNetworkIdle(page: Page, timeout = 5000): Promise<void> {
  await page.waitForLoadState("networkidle", { timeout });
}

/**
 * Helper to check for console errors during a test.
 */
export async function collectConsoleErrors(
  page: Page,
  action: () => Promise<void>
): Promise<string[]> {
  const errors: string[] = [];

  page.on("console", (msg) => {
    if (msg.type() === "error") {
      errors.push(msg.text());
    }
  });

  await action();

  return errors;
}

/**
 * Helper to test theme switching.
 */
export async function switchTheme(
  page: Page,
  theme: "light" | "dark"
): Promise<void> {
  // Navigate to settings
  await navigateToView(page, "settings");

  // Find and click the theme toggle
  const themeToggle = page.locator(
    `[data-theme="${theme}"], .theme-toggle, button:has-text("${theme}")`
  );

  if ((await themeToggle.count()) > 0) {
    await themeToggle.first().click();
    await page.waitForTimeout(300); // Wait for transition
  }
}

/**
 * Helper to scroll to bottom of a container.
 */
export async function scrollToBottom(page: Page, selector: string): Promise<void> {
  await page.locator(selector).evaluate((el) => {
    el.scrollTop = el.scrollHeight;
  });
}

/**
 * Helper to scroll to top of a container.
 */
export async function scrollToTop(page: Page, selector: string): Promise<void> {
  await page.locator(selector).evaluate((el) => {
    el.scrollTop = 0;
  });
}

/**
 * Helper to get scroll position of a container.
 */
export async function getScrollPosition(
  page: Page,
  selector: string
): Promise<{ top: number; height: number; scrollHeight: number }> {
  return page.locator(selector).evaluate((el) => ({
    top: el.scrollTop,
    height: el.clientHeight,
    scrollHeight: el.scrollHeight,
  }));
}

/**
 * Helper to check if an element is in viewport.
 */
export async function isInViewport(page: Page, selector: string): Promise<boolean> {
  return page.locator(selector).first().isVisible();
}

/**
 * Helper to count visible elements (used for virtual scroll testing).
 */
export async function countVisibleElements(
  page: Page,
  selector: string
): Promise<number> {
  const elements = page.locator(selector);
  const count = await elements.count();
  let visibleCount = 0;

  for (let i = 0; i < count; i++) {
    const isVisible = await elements.nth(i).isVisible();
    if (isVisible) visibleCount++;
  }

  return visibleCount;
}
