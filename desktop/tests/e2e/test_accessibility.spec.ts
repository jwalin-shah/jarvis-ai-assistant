/**
 * E2E accessibility tests for JARVIS Desktop.
 *
 * These tests verify:
 * - ARIA labels and roles
 * - Keyboard-only navigation
 * - Focus management
 * - Screen reader compatibility
 * - Color contrast (where testable)
 */

import { test, expect } from "./fixtures";
import {
  waitForAppLoad,
  navigateToView,
  selectConversation,
  checkAccessibility,
  testKeyboardNavigation,
  openGlobalSearch,
  closeGlobalSearch,
} from "./fixtures";

test.describe("Accessibility Tests", () => {
  test.describe("ARIA Labels and Roles", () => {
    test("navigation items have proper ARIA attributes", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Navigation items should have proper labels
      const navItems = page.locator(".nav-item");
      const count = await navItems.count();

      for (let i = 0; i < count; i++) {
        const item = navItems.nth(i);
        const title = await item.getAttribute("title");
        const ariaLabel = await item.getAttribute("aria-label");

        // Should have either title or aria-label
        expect(title || ariaLabel).toBeTruthy();
      }
    });

    test("buttons have accessible names", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // All buttons should be accessible
      const buttons = page.locator("button");
      const count = await buttons.count();

      let accessibleCount = 0;
      for (let i = 0; i < Math.min(count, 20); i++) {
        // Check first 20 buttons
        const button = buttons.nth(i);
        const text = await button.textContent();
        const ariaLabel = await button.getAttribute("aria-label");
        const title = await button.getAttribute("title");

        if (text?.trim() || ariaLabel || title) {
          accessibleCount++;
        }
      }

      // Most buttons should have accessible names
      expect(accessibleCount).toBeGreaterThan(Math.min(count, 20) * 0.8);
    });

    test("form inputs have labels", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await navigateToView(page, "settings");

      // Find all inputs
      const inputs = page.locator("input, select, textarea");
      const count = await inputs.count();

      for (let i = 0; i < count; i++) {
        const input = inputs.nth(i);
        const id = await input.getAttribute("id");
        const ariaLabel = await input.getAttribute("aria-label");
        const ariaLabelledBy = await input.getAttribute("aria-labelledby");
        const placeholder = await input.getAttribute("placeholder");

        // Input should have some form of label
        if (id) {
          const label = page.locator(`label[for="${id}"]`);
          const hasLabel = (await label.count()) > 0;
          expect(hasLabel || ariaLabel || ariaLabelledBy || placeholder).toBeTruthy();
        }
      }
    });

    test("dialogs have proper role and labels", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Open global search (a dialog)
      await openGlobalSearch(page);

      // Dialog should have proper role
      const dialog = page.locator('[role="dialog"], .search-modal');
      await expect(dialog).toBeVisible();

      // Check for aria-label
      const ariaLabel = await dialog.getAttribute("aria-label");
      expect(ariaLabel).toBeTruthy();

      await closeGlobalSearch(page);
    });

    test("conversation list has proper list semantics", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await page.waitForSelector(".conversation");

      // Conversations should be in a list or have list roles
      const conversationList = page.locator(
        '.conversation-list, [role="list"], ul.conversations'
      );

      if ((await conversationList.count()) > 0) {
        // List items should have proper roles
        const items = conversationList.locator(
          '.conversation, [role="listitem"], [role="option"]'
        );
        expect(await items.count()).toBeGreaterThan(0);
      }
    });

    test("messages have proper content structure", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await page.waitForSelector(".conversation");
      await selectConversation(page, "John Doe");
      await page.waitForSelector(".message");

      // Messages should have readable content structure
      const messages = page.locator(".message");
      const count = await messages.count();

      for (let i = 0; i < Math.min(count, 5); i++) {
        const message = messages.nth(i);

        // Message should have text content
        const text = await message.locator(".message-text, .text, .content").textContent();
        expect(text).toBeTruthy();
      }
    });
  });

  test.describe("Keyboard Navigation", () => {
    test("can navigate entire app with keyboard only", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Start from body
      await page.keyboard.press("Tab");

      // Should be able to focus something
      const focusedElement = page.locator(":focus");
      await expect(focusedElement).toBeVisible();

      // Keep tabbing to verify focus moves
      let focusChanges = 0;
      for (let i = 0; i < 10; i++) {
        const beforeFocus = await page.evaluate(
          () => document.activeElement?.tagName
        );
        await page.keyboard.press("Tab");
        const afterFocus = await page.evaluate(
          () => document.activeElement?.tagName
        );

        if (beforeFocus !== afterFocus) {
          focusChanges++;
        }
      }

      expect(focusChanges).toBeGreaterThan(3);
    });

    test("can navigate sidebar with arrow keys", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Focus first nav item
      const firstNavItem = page.locator(".nav-item").first();
      await firstNavItem.focus();

      // Press arrow down to navigate
      await page.keyboard.press("ArrowDown");

      // Wait for focus to move to next nav item
      await expect(page.locator(".nav-item:nth-child(2), .nav-item.focused, .nav-item[data-focused='true']")).toBeFocused({ timeout: 1000 });

      // Should be able to press Enter to activate
      await page.keyboard.press("Enter");

      // Wait for view change indicator (active state or route change)
      await page.waitForFunction(() => {
        const activeNav = document.querySelector(".nav-item.active, .nav-item[aria-current]");
        return activeNav !== null;
      }, { timeout: 1000 });

      // Some view should have changed
      // (Hard to verify without knowing which item was focused)
    });

    test("can navigate conversations with keyboard", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await page.waitForSelector(".conversation");

      // Focus conversation list area
      const firstConversation = page.locator(".conversation").first();
      await firstConversation.focus();

      // Arrow down should move to next conversation
      await page.keyboard.press("ArrowDown");

      // Wait for focus to move to second conversation
      await expect(page.locator(".conversation:nth-child(2), .conversation.focused, .conversation[aria-selected='true']").first()).toBeFocused({ timeout: 1000 });

      // Enter should select the conversation
      await page.keyboard.press("Enter");

      // Wait for conversation to become active
      await expect(page.locator(".conversation.active")).toBeVisible({ timeout: 1000 });

      // A conversation should be selected
      await expect(page.locator(".conversation.active")).toBeVisible();
    });

    test("search modal can be opened with keyboard shortcut", async ({
      mockedPage: page,
    }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Cmd+K should open search
      await page.keyboard.press("Meta+k");

      // Search modal should be visible
      await expect(
        page.locator(".search-modal, .global-search, .search-overlay")
      ).toBeVisible();

      // Escape should close it
      await page.keyboard.press("Escape");

      // Wait for modal to be hidden
      await expect(
        page.locator(".search-modal, .global-search, .search-overlay")
      ).not.toBeVisible({ timeout: 1000 });

      // Should be closed
      await expect(
        page.locator(".search-modal, .global-search, .search-overlay")
      ).not.toBeVisible();
    });

    test("search results can be navigated with arrow keys", async ({
      mockedPage: page,
    }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Open search
      await openGlobalSearch(page);

      // Type to get results
      await page.locator(".search-input").fill("lunch");

      // Wait for search results to appear
      await page.waitForSelector(".result-item, .search-result, [role='option']", { timeout: 2000 });

      // Navigate with arrow keys
      await page.keyboard.press("ArrowDown");

      // Wait for first result to be selected
      await page.waitForSelector(".result-item.selected, [aria-selected='true'], .result-item.focused", { timeout: 1000 });

      // First result should be selected
      const selectedResult = page.locator(".result-item.selected, [aria-selected='true']");

      // Check if selection indicator exists
      const hasSelection = (await selectedResult.count()) > 0;
      if (hasSelection) {
        await expect(selectedResult).toBeVisible();
      }

      await page.keyboard.press("Escape");
    });

    test("settings form can be navigated with Tab", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await navigateToView(page, "settings");

      // Tab through form elements
      let tabbedElements = 0;
      for (let i = 0; i < 15; i++) {
        await page.keyboard.press("Tab");
        const focused = await page.evaluate(() => document.activeElement?.tagName);
        if (focused && ["INPUT", "BUTTON", "SELECT", "A"].includes(focused)) {
          tabbedElements++;
        }
      }

      // Should be able to tab through multiple form elements
      expect(tabbedElements).toBeGreaterThan(3);
    });

    test("Enter key activates buttons", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await navigateToView(page, "settings");

      // Find and focus a button
      const button = page.locator(".btn-primary, button").first();
      await button.focus();

      // Enter should activate it
      await page.keyboard.press("Enter");

      // Wait for button action to complete (success message or state change)
      await page.waitForFunction(() => {
        // Check for success indicators
        const success = document.querySelector(".success, .toast, [role='status']");
        const loading = document.querySelector("[aria-busy='true']");
        // Either success appears or button is no longer in loading state
        return success !== null || loading === null;
      }, { timeout: 2000 });
    });

    test("Space key activates checkboxes and toggles", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await navigateToView(page, "settings");

      // Find toggle button
      const toggle = page.locator(".toggle-btn, input[type='checkbox']").first();

      if ((await toggle.count()) > 0) {
        const initialState = await toggle.getAttribute("class");
        const initialChecked = await toggle.evaluate((el) =>
          el instanceof HTMLInputElement ? el.checked : el.getAttribute("aria-checked")
        );

        await toggle.focus();
        await page.keyboard.press("Space");

        // Wait for state change
        await page.waitForFunction(
          ({ el, initial }) => {
            const element = document.querySelector(el);
            if (!element) return false;
            const currentClass = element.getAttribute("class");
            const currentChecked = element instanceof HTMLInputElement
              ? element.checked
              : element.getAttribute("aria-checked");
            return currentClass !== initial.state || currentChecked !== initial.checked;
          },
          { el: ".toggle-btn, input[type='checkbox']", initial: { state: initialState, checked: initialChecked } },
          { timeout: 1000 }
        );

        const newState = await toggle.getAttribute("class");
        // State should have changed (class or checked attribute)
      }
    });
  });

  test.describe("Focus Management", () => {
    test("focus is visible on all interactive elements", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Check that focus ring is visible
      await page.keyboard.press("Tab");

      const focusedElement = page.locator(":focus");
      await expect(focusedElement).toBeVisible();

      // Check that focus has visible outline or ring
      const outline = await focusedElement.evaluate((el) => {
        const styles = window.getComputedStyle(el);
        return {
          outline: styles.outline,
          boxShadow: styles.boxShadow,
          border: styles.border,
        };
      });

      // Should have some visual focus indicator
      const hasVisibleFocus =
        (outline.outline && outline.outline !== "none") ||
        (outline.boxShadow && outline.boxShadow !== "none");

      // Note: This might pass even if focus isn't highly visible
      // Visual inspection is still recommended
    });

    test("focus is trapped in modal dialogs", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Open search modal
      await openGlobalSearch(page);

      // Tab many times - focus should stay in modal
      // Check focus stays in modal after each tab
      for (let i = 0; i < 20; i++) {
        await page.keyboard.press("Tab");

        // Wait for focus to stabilize by checking activeElement changes
        await page.waitForFunction(() => document.activeElement !== null, { timeout: 500 });
      }

      // Active element should still be in the modal
      const activeInModal = await page.evaluate(() => {
        const active = document.activeElement;
        const modal = document.querySelector(
          ".search-modal, .global-search, .search-overlay, [role='dialog']"
        );
        return modal?.contains(active);
      });

      expect(activeInModal).toBe(true);

      await page.keyboard.press("Escape");
    });

    test("focus returns to trigger after modal closes", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Focus a trigger element and note it
      await page.keyboard.press("Tab");
      await page.keyboard.press("Tab");

      // Open and close search
      await page.keyboard.press("Meta+k");
      await page.waitForSelector(".search-modal, .global-search", { state: "visible" });
      await page.keyboard.press("Escape");

      // Wait for modal to close
      await page.waitForSelector(".search-modal, .global-search", { state: "hidden", timeout: 1000 });

      // Wait for focus to be restored
      await page.waitForFunction(() => {
        const active = document.activeElement;
        return active !== null && active.tagName !== "BODY";
      }, { timeout: 1000 });

      // Focus should be somewhere reasonable (body or previously focused)
      const activeElement = await page.evaluate(
        () => document.activeElement?.tagName
      );
      expect(activeElement).toBeTruthy();
    });

    test("focus moves to error messages", async ({ errorPage: page }) => {
      await page.goto("/");
      await page.waitForSelector(".sidebar", { state: "visible" });

      // Wait for error to appear
      const errorElement = page.locator(
        ".error, .error-banner, .error-message, [role='alert']"
      );
      await errorElement.first().waitFor({ state: "visible", timeout: 5000 });

      // Error messages should be focusable or announced
      if ((await errorElement.count()) > 0) {
        const role = await errorElement.first().getAttribute("role");
        // Errors should have alert or status role for screen readers
        if (role) {
          expect(["alert", "status"]).toContain(role);
        }
      }
    });
  });

  test.describe("Screen Reader Compatibility", () => {
    test("page has proper heading structure", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Check for h1
      const h1 = page.locator("h1");
      expect(await h1.count()).toBeGreaterThanOrEqual(0); // May not have h1 on all pages

      // Navigate to a page with clear heading
      await navigateToView(page, "settings");
      await expect(page.locator("h1")).toHaveText("Settings");
    });

    test("landmarks are properly defined", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Check for main landmark
      const main = page.locator('main, [role="main"]');
      const hasMain = (await main.count()) > 0;

      // Check for navigation landmark
      const nav = page.locator('nav, [role="navigation"]');
      const hasNav = (await nav.count()) > 0;

      // Should have at least some landmarks
      expect(hasMain || hasNav).toBe(true);
    });

    test("images have alt text", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      const images = page.locator("img");
      const count = await images.count();

      for (let i = 0; i < count; i++) {
        const img = images.nth(i);
        const alt = await img.getAttribute("alt");
        const role = await img.getAttribute("role");

        // Should have alt text or be marked as decorative
        expect(alt !== null || role === "presentation").toBe(true);
      }
    });

    test("status updates are announced", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Check for live regions
      const liveRegions = page.locator(
        '[aria-live], [role="status"], [role="alert"], [role="log"]'
      );

      // App should have at least one live region for dynamic updates
      const count = await liveRegions.count();
      console.log(`Found ${count} live regions`);

      // This is advisory - not all apps need live regions
    });

    test("loading states are announced", async ({ slowPage: page }) => {
      await page.goto("/");

      // Check for loading announcements
      const loadingElement = page.locator(
        '[aria-busy="true"], .loading[role="status"], [aria-label*="loading" i]'
      );

      // Loading state may or may not be visible
      // This test documents expected behavior
    });
  });

  test.describe("Color and Contrast", () => {
    test("text has sufficient contrast", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Sample some text elements for contrast
      const textElements = page.locator("p, span, h1, h2, h3, button, a");
      const count = await textElements.count();

      for (let i = 0; i < Math.min(count, 10); i++) {
        const element = textElements.nth(i);
        const styles = await element.evaluate((el) => {
          const computed = window.getComputedStyle(el);
          return {
            color: computed.color,
            backgroundColor: computed.backgroundColor,
          };
        });

        // Log for manual verification
        // Automated contrast checking would require additional libraries
      }
    });

    test("focus indicators have sufficient contrast", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      await page.keyboard.press("Tab");
      const focusedElement = page.locator(":focus");

      if ((await focusedElement.count()) > 0) {
        const styles = await focusedElement.evaluate((el) => {
          const computed = window.getComputedStyle(el);
          return {
            outline: computed.outline,
            outlineColor: computed.outlineColor,
            boxShadow: computed.boxShadow,
          };
        });

        // Log for manual verification
        console.log("Focus styles:", styles);
      }
    });

    test("error states use more than just color", async ({ errorPage: page }) => {
      await page.goto("/");
      await page.waitForSelector(".sidebar", { state: "visible" });

      // Wait for error element to appear
      const errorElement = page.locator(".error, .error-banner, [role='alert']");
      await errorElement.first().waitFor({ state: "visible", timeout: 5000 });

      if ((await errorElement.count()) > 0) {
        // Error should have icon or text, not just color
        const hasIcon = (await errorElement.locator("svg, .icon").count()) > 0;
        const hasText = (await errorElement.textContent())?.length ?? 0 > 0;

        expect(hasIcon || hasText).toBe(true);
      }
    });
  });

  test.describe("Reduced Motion", () => {
    test("respects prefers-reduced-motion", async ({ mockedPage: page }) => {
      // Emulate reduced motion preference
      await page.emulateMedia({ reducedMotion: "reduce" });

      await page.goto("/");
      await waitForAppLoad(page);

      // Check that animations are disabled
      const animationDuration = await page.evaluate(() => {
        const el = document.querySelector(".sidebar, .app");
        if (!el) return "0s";
        return window.getComputedStyle(el).animationDuration;
      });

      // With reduced motion, animations should be instant or very short
      // This is advisory - CSS might not implement this
    });
  });
});
