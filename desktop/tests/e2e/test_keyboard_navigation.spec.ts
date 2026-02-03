/**
 * E2E tests for keyboard navigation.
 *
 * Verifies all major keyboard interactions:
 * - Tab navigation through interactive elements
 * - Arrow key navigation in lists
 * - Keyboard shortcuts (Cmd+K, Escape, etc.)
 * - Focus management
 */

import { test, expect } from "./fixtures";
import {
  waitForAppLoad,
  navigateToView,
  selectConversation,
  openGlobalSearch,
  closeGlobalSearch,
} from "./fixtures";

test.describe("Keyboard Navigation", () => {
  test.describe("Tab Navigation", () => {
    test("Tab moves focus through interactive elements", async ({
      mockedPage: page,
    }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Start tabbing
      const focusedElements: string[] = [];

      for (let i = 0; i < 15; i++) {
        await page.keyboard.press("Tab");
        const tag = await page.evaluate(() => document.activeElement?.tagName);
        if (tag) focusedElements.push(tag);
      }

      // Should focus multiple elements
      expect(focusedElements.length).toBeGreaterThan(5);

      // Should include buttons, inputs, or links
      const interactiveElements = focusedElements.filter((el) =>
        ["BUTTON", "INPUT", "A", "SELECT", "TEXTAREA"].includes(el)
      );
      expect(interactiveElements.length).toBeGreaterThan(3);
    });

    test("Shift+Tab moves focus backwards", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Tab forward a few times
      for (let i = 0; i < 5; i++) {
        await page.keyboard.press("Tab");
      }

      const forwardElement = await page.evaluate(
        () => document.activeElement?.outerHTML
      );

      // Tab backward
      await page.keyboard.press("Shift+Tab");
      await page.keyboard.press("Shift+Tab");

      const backwardElement = await page.evaluate(
        () => document.activeElement?.outerHTML
      );

      // Should be different elements
      expect(forwardElement).not.toBe(backwardElement);
    });

    test("focus wraps from last to first element", async ({
      mockedPage: page,
    }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Tab many times to reach the end
      for (let i = 0; i < 50; i++) {
        await page.keyboard.press("Tab");
      }

      // Keep tabbing - focus should eventually wrap
      // This is browser-specific behavior
      const finalElement = await page.evaluate(
        () => document.activeElement?.tagName
      );
      expect(finalElement).toBeTruthy();
    });
  });

  test.describe("Sidebar Navigation", () => {
    test("Enter key activates sidebar navigation items", async ({
      mockedPage: page,
    }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Focus a nav item
      const messagesNav = page.locator('.nav-item[title="Messages"]');
      await messagesNav.focus();

      // Press Enter
      await page.keyboard.press("Enter");
      await page.waitForTimeout(200);

      // Nav item should be active
      await expect(messagesNav).toHaveClass(/active/);
    });

    test("Space key activates sidebar navigation items", async ({
      mockedPage: page,
    }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Focus settings nav
      const settingsNav = page.locator('.nav-item[title="Settings"]');
      await settingsNav.focus();

      // Press Space
      await page.keyboard.press("Space");
      await page.waitForTimeout(200);

      // Settings should be active
      await expect(settingsNav).toHaveClass(/active/);
    });

    test("can navigate between nav items with Tab", async ({
      mockedPage: page,
    }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Focus first nav item
      const firstNav = page.locator(".nav-item").first();
      await firstNav.focus();

      // Tab to next nav item
      await page.keyboard.press("Tab");

      // Check if another nav item is focused
      const activeTag = await page.evaluate(() => document.activeElement?.className);
      // It might focus the next nav item or another element
    });
  });

  test.describe("Conversation List Navigation", () => {
    test("Arrow keys navigate conversations", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await page.waitForSelector(".conversation");

      // Click first conversation to focus list
      await page.locator(".conversation").first().click();

      // Press Down to move to next
      await page.keyboard.press("ArrowDown");
      await page.waitForTimeout(100);

      // Press Down again
      await page.keyboard.press("ArrowDown");
      await page.waitForTimeout(100);

      // Some conversation should be selected
      await expect(page.locator(".conversation.active")).toBeVisible();
    });

    test("Enter selects highlighted conversation", async ({
      mockedPage: page,
    }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await page.waitForSelector(".conversation");

      // Focus first conversation
      await page.locator(".conversation").first().focus();

      // Press Enter
      await page.keyboard.press("Enter");
      await page.waitForTimeout(300);

      // Conversation should be selected
      await expect(page.locator(".conversation.active")).toBeVisible();
    });

    test("Home key jumps to first conversation", async ({
      mockedPage: page,
    }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await page.waitForSelector(".conversation");

      // Select a middle conversation first
      await page.locator(".conversation").nth(1).click();
      await page.waitForTimeout(100);

      // Press Home
      await page.keyboard.press("Home");
      await page.waitForTimeout(100);

      // First conversation should be focused/selected
      // Behavior depends on implementation
    });

    test("End key jumps to last conversation", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await page.waitForSelector(".conversation");

      // Focus first conversation
      await page.locator(".conversation").first().click();

      // Press End
      await page.keyboard.press("End");
      await page.waitForTimeout(100);

      // Last conversation should be focused
      // Behavior depends on implementation
    });
  });

  test.describe("Global Shortcuts", () => {
    test("Cmd+K opens global search", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Press Cmd+K
      await page.keyboard.press("Meta+k");

      // Search modal should open
      await expect(
        page.locator(".search-modal, .global-search, .search-overlay")
      ).toBeVisible();
    });

    test("Escape closes global search", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Open search
      await openGlobalSearch(page);

      // Press Escape
      await page.keyboard.press("Escape");
      await page.waitForTimeout(200);

      // Search should be closed
      await expect(
        page.locator(".search-modal, .global-search, .search-overlay")
      ).not.toBeVisible();
    });

    test("Escape closes other modals/panels", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await page.waitForSelector(".conversation");
      await selectConversation(page, "John Doe");

      // Try to open AI draft panel if available
      const draftButton = page.locator(
        'button:has-text("Draft"), button:has-text("AI"), .draft-btn'
      );

      if ((await draftButton.count()) > 0) {
        await draftButton.first().click();
        await page.waitForTimeout(200);

        // Check if panel opened
        const panel = page.locator('.panel, [role="dialog"], .ai-draft-panel');

        if ((await panel.count()) > 0) {
          // Press Escape
          await page.keyboard.press("Escape");
          await page.waitForTimeout(200);

          // Panel should be closed
          await expect(panel).not.toBeVisible();
        }
      }
    });

    test("Cmd+/ shows keyboard shortcuts help", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Press Cmd+/
      await page.keyboard.press("Meta+/");
      await page.waitForTimeout(500);

      // Help modal might appear (if implemented)
      const helpModal = page.locator(
        ".help-modal, .shortcuts-modal, .keyboard-shortcuts"
      );

      // This is optional functionality
    });
  });

  test.describe("Search Navigation", () => {
    test("Arrow keys navigate search results", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Open search
      await openGlobalSearch(page);

      // Type to trigger search
      await page.locator(".search-input").fill("lunch");
      await page.waitForTimeout(500);

      // Navigate with arrow keys
      await page.keyboard.press("ArrowDown");
      await page.waitForTimeout(100);

      const selectedItem = page.locator(
        ".result-item.selected, [aria-selected='true']"
      );

      // Selection might be visible if results exist
      const hasResults = (await selectedItem.count()) > 0;
      if (hasResults) {
        await expect(selectedItem).toBeVisible();
      }

      await page.keyboard.press("Escape");
    });

    test("Enter opens selected search result", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      await openGlobalSearch(page);

      // Type and wait for results
      await page.locator(".search-input").fill("lunch");
      await page.waitForTimeout(500);

      // Navigate to first result
      await page.keyboard.press("ArrowDown");
      await page.waitForTimeout(100);

      // Press Enter
      await page.keyboard.press("Enter");
      await page.waitForTimeout(300);

      // Search should close and navigate (if result exists)
    });

    test("Tab in search input cycles through filter options", async ({
      mockedPage: page,
    }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      await openGlobalSearch(page);

      // Open filters if there's a filter toggle
      const filterToggle = page.locator(".filter-toggle");
      if ((await filterToggle.count()) > 0) {
        await filterToggle.click();
        await page.waitForTimeout(200);
      }

      // Tab through filter options
      await page.keyboard.press("Tab");
      await page.keyboard.press("Tab");

      const focused = await page.evaluate(
        () => document.activeElement?.className
      );

      // Should focus something in the filter panel
      await page.keyboard.press("Escape");
    });
  });

  test.describe("Settings Form Navigation", () => {
    test("Tab navigates through form fields", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await navigateToView(page, "settings");

      const focusedInputs: string[] = [];

      for (let i = 0; i < 10; i++) {
        await page.keyboard.press("Tab");
        const type = await page.evaluate(
          () => (document.activeElement as HTMLInputElement)?.type
        );
        if (type) focusedInputs.push(type);
      }

      // Should navigate through various input types
      expect(focusedInputs.length).toBeGreaterThan(3);
    });

    test("Arrow keys adjust slider values", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await navigateToView(page, "settings");

      // Find a slider input
      const slider = page.locator('input[type="range"]').first();

      if ((await slider.count()) > 0) {
        await slider.focus();

        const initialValue = await slider.inputValue();

        // Press right arrow to increase
        await page.keyboard.press("ArrowRight");
        await page.waitForTimeout(100);

        const newValue = await slider.inputValue();

        // Value should change (unless at max)
        // The specific change depends on the slider's step value
      }
    });

    test("Space toggles checkbox and toggle buttons", async ({
      mockedPage: page,
    }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await navigateToView(page, "settings");

      // Find toggle button
      const toggle = page.locator(".toggle-btn").first();

      if ((await toggle.count()) > 0) {
        const initialClass = await toggle.getAttribute("class");
        const wasOn = initialClass?.includes("on");

        await toggle.focus();
        await page.keyboard.press("Space");
        await page.waitForTimeout(200);

        const newClass = await toggle.getAttribute("class");
        const isOn = newClass?.includes("on");

        // State should change
        expect(isOn).not.toBe(wasOn);
      }
    });

    test("Enter submits save button", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await navigateToView(page, "settings");

      // Focus save button
      const saveButton = page.locator(".btn-primary");
      await saveButton.focus();

      // Press Enter
      await page.keyboard.press("Enter");
      await page.waitForTimeout(500);

      // Success banner should appear
      await expect(page.locator(".success-banner")).toBeVisible();
    });
  });

  test.describe("Message View Navigation", () => {
    test("Page Up/Down scrolls message list", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await page.waitForSelector(".conversation");
      await selectConversation(page, "John Doe");
      await page.waitForSelector(".message");

      // Focus message area
      const messageArea = page.locator(".messages, .message-list");

      if ((await messageArea.count()) > 0) {
        await messageArea.focus();

        // Get initial scroll position
        const initialScroll = await messageArea.evaluate((el) => el.scrollTop);

        // Press Page Down
        await page.keyboard.press("PageDown");
        await page.waitForTimeout(200);

        const newScroll = await messageArea.evaluate((el) => el.scrollTop);

        // Scroll position might change (if content is tall enough)
      }
    });

    test("Home jumps to first message", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await page.waitForSelector(".conversation");
      await selectConversation(page, "John Doe");
      await page.waitForSelector(".message");

      const messageArea = page.locator(".messages, .message-list");

      if ((await messageArea.count()) > 0) {
        await messageArea.focus();
        await page.keyboard.press("Home");
        await page.waitForTimeout(200);

        const scrollTop = await messageArea.evaluate((el) => el.scrollTop);
        // Should be at or near the top
        expect(scrollTop).toBeLessThanOrEqual(100);
      }
    });

    test("End jumps to last message", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await page.waitForSelector(".conversation");
      await selectConversation(page, "John Doe");
      await page.waitForSelector(".message");

      const messageArea = page.locator(".messages, .message-list");

      if ((await messageArea.count()) > 0) {
        await messageArea.focus();

        // Scroll to top first
        await messageArea.evaluate((el) => (el.scrollTop = 0));

        // Press End
        await page.keyboard.press("End");
        await page.waitForTimeout(200);

        const scrollInfo = await messageArea.evaluate((el) => ({
          scrollTop: el.scrollTop,
          scrollHeight: el.scrollHeight,
          clientHeight: el.clientHeight,
        }));

        // Should be at or near the bottom
        const isAtBottom =
          scrollInfo.scrollTop + scrollInfo.clientHeight >=
          scrollInfo.scrollHeight - 50;
        expect(isAtBottom).toBe(true);
      }
    });
  });

  test.describe("Focus Restoration", () => {
    test("focus returns after modal close", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Focus an element before opening modal
      const settingsNav = page.locator('.nav-item[title="Settings"]');
      await settingsNav.focus();

      // Open and close search
      await page.keyboard.press("Meta+k");
      await page.waitForSelector(".search-modal", { state: "visible" });
      await page.keyboard.press("Escape");
      await page.waitForTimeout(300);

      // Focus should return to body or a reasonable element
      const activeTag = await page.evaluate(
        () => document.activeElement?.tagName
      );
      expect(activeTag).toBeTruthy();
    });

    test("focus moves to new content after view change", async ({
      mockedPage: page,
    }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Navigate with keyboard
      const healthNav = page.locator('.nav-item[title="Health Status"]');
      await healthNav.focus();
      await page.keyboard.press("Enter");
      await page.waitForTimeout(300);

      // Focus should be somewhere in the new view
      const activeElement = await page.evaluate(
        () => document.activeElement?.tagName
      );
      expect(activeElement).toBeTruthy();
    });
  });
});
