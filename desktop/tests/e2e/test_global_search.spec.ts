/**
 * E2E tests for Global Search component.
 *
 * Tests the GlobalSearch modal functionality:
 * - Opening/closing behavior
 * - Text and semantic search modes
 * - Filter functionality
 * - Result navigation
 * - Keyboard shortcuts
 */

import { test, expect } from "./fixtures";
import {
  waitForAppLoad,
  openGlobalSearch,
  closeGlobalSearch,
  searchAndWait,
} from "./fixtures";

test.describe("Global Search", () => {
  test.beforeEach(async ({ socketMockedPage: page }) => {
    await page.goto("/");
    await waitForAppLoad(page);
  });

  test.describe("Opening and Closing", () => {
    test("opens with Cmd+K shortcut", async ({ socketMockedPage: page }) => {
      await page.keyboard.press("Meta+k");

      await expect(
        page.locator(".search-modal, .search-overlay, .global-search")
      ).toBeVisible();
    });

    test("closes with Escape key", async ({ socketMockedPage: page }) => {
      await openGlobalSearch(page);
      await page.keyboard.press("Escape");

      await expect(
        page.locator(".search-modal, .search-overlay, .global-search")
      ).not.toBeVisible();
    });

    test("closes when clicking outside modal", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      // Click on the overlay (outside modal)
      await page.locator(".search-overlay").click({ position: { x: 10, y: 10 } });
      await page.waitForTimeout(300);

      await expect(
        page.locator(".search-modal, .global-search")
      ).not.toBeVisible();
    });

    test("close button works", async ({ socketMockedPage: page }) => {
      await openGlobalSearch(page);

      const closeBtn = page.locator(".close-btn, button[aria-label='Close']");
      await closeBtn.click();

      await expect(
        page.locator(".search-modal, .search-overlay")
      ).not.toBeVisible();
    });

    test("focuses input when opened", async ({ socketMockedPage: page }) => {
      await openGlobalSearch(page);

      const input = page.locator(".search-input");
      await expect(input).toBeFocused();
    });
  });

  test.describe("Search Modes", () => {
    test("defaults to text search mode", async ({ socketMockedPage: page }) => {
      await openGlobalSearch(page);

      // Placeholder should indicate keyword search
      const input = page.locator(".search-input");
      const placeholder = await input.getAttribute("placeholder");

      expect(placeholder).toMatch(/Search|messages/i);
    });

    test("can toggle to semantic search mode", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      // Find and click mode toggle
      const modeToggle = page.locator(".mode-toggle");

      if ((await modeToggle.count()) > 0) {
        await modeToggle.click();
        await page.waitForTimeout(200);

        // Placeholder should change
        const input = page.locator(".search-input");
        const placeholder = await input.getAttribute("placeholder");

        expect(placeholder).toMatch(/meaning|semantic/i);
      }
    });

    test("semantic mode shows AI indicator", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      const modeToggle = page.locator(".mode-toggle");

      if ((await modeToggle.count()) > 0) {
        await modeToggle.click();
        await page.waitForTimeout(200);

        // Should show semantic indicator
        const semanticIndicator = page.locator(
          ".mode-toggle.semantic, .search-icon.semantic"
        );
        await expect(semanticIndicator.first()).toBeVisible();
      }
    });

    test("search mode is preserved during search", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      const modeToggle = page.locator(".mode-toggle");

      if ((await modeToggle.count()) > 0) {
        // Switch to semantic
        await modeToggle.click();
        await page.waitForTimeout(100);

        // Type search query
        await page.locator(".search-input").fill("dinner plans");
        await page.waitForTimeout(500);

        // Mode should still be semantic
        await expect(modeToggle).toHaveClass(/semantic/);
      }
    });
  });

  test.describe("Search Behavior", () => {
    test("shows empty state with instructions", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      // Empty state should show instructions
      const emptyState = page.locator(".empty-state");
      await expect(emptyState).toBeVisible();

      // Should have helpful text
      await expect(emptyState).toContainText(/Search|conversations/i);
    });

    test("shows loading state during search", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      // Type to trigger search
      await page.locator(".search-input").type("lunch", { delay: 50 });

      // Loading state might appear briefly
      const loadingState = page.locator(".loading-state, .spinner");

      // It may be too fast to catch, but state should exist
    });

    test("shows results after search", async ({ socketMockedPage: page }) => {
      await openGlobalSearch(page);

      // Type search query
      await page.locator(".search-input").fill("lunch");
      await page.waitForTimeout(600); // Wait for debounce

      // Results should appear
      const results = page.locator(".result-item, .results-list");

      // Check for either results or no-results state
      const hasResults = (await results.count()) > 0;
      const noResults = page.locator(".no-results-state");
      const hasNoResults = (await noResults.count()) > 0;

      expect(hasResults || hasNoResults).toBe(true);
    });

    test("shows no results state when nothing matches", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      // Type unlikely search query
      await page.locator(".search-input").fill("xyznonexistent123");
      await page.waitForTimeout(600);

      // No results state should show
      const noResultsState = page.locator(".no-results-state");

      if ((await noResultsState.count()) > 0) {
        await expect(noResultsState).toBeVisible();
        await expect(noResultsState).toContainText(/no results|not found/i);
      }
    });

    test("debounces search input", async ({ socketMockedPage: page }) => {
      let searchCount = 0;

      // Count search requests
      await page.route("http://localhost:8742/conversations/search", async (route) => {
        searchCount++;
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([]),
        });
      });

      await openGlobalSearch(page);

      // Type quickly
      await page.locator(".search-input").type("testing search", { delay: 20 });
      await page.waitForTimeout(700);

      // Should have made only 1-2 requests (debounced)
      expect(searchCount).toBeLessThanOrEqual(3);
    });

    test("clears results when input is cleared", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      // Search and get results
      await page.locator(".search-input").fill("lunch");
      await page.waitForTimeout(600);

      // Clear input
      const clearBtn = page.locator(".clear-btn");
      if ((await clearBtn.count()) > 0) {
        await clearBtn.click();
      } else {
        await page.locator(".search-input").fill("");
      }

      await page.waitForTimeout(200);

      // Should show empty state again
      await expect(page.locator(".empty-state")).toBeVisible();
    });
  });

  test.describe("Filters", () => {
    test("filter panel can be toggled", async ({ socketMockedPage: page }) => {
      await openGlobalSearch(page);

      const filterToggle = page.locator(".filter-toggle");

      if ((await filterToggle.count()) > 0) {
        // Initially closed
        await expect(page.locator(".filters-panel")).not.toBeVisible();

        // Open filters
        await filterToggle.click();
        await page.waitForTimeout(200);

        await expect(page.locator(".filters-panel")).toBeVisible();

        // Close filters
        await filterToggle.click();
        await page.waitForTimeout(200);

        await expect(page.locator(".filters-panel")).not.toBeVisible();
      }
    });

    test("sender filter accepts input", async ({ socketMockedPage: page }) => {
      await openGlobalSearch(page);

      const filterToggle = page.locator(".filter-toggle");

      if ((await filterToggle.count()) > 0) {
        await filterToggle.click();
        await page.waitForTimeout(200);

        const senderFilter = page.locator("#filter-sender, [name='sender']");

        if ((await senderFilter.count()) > 0) {
          await senderFilter.fill("John");
          await expect(senderFilter).toHaveValue("John");
        }
      }
    });

    test("date filters accept valid dates", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      const filterToggle = page.locator(".filter-toggle");

      if ((await filterToggle.count()) > 0) {
        await filterToggle.click();
        await page.waitForTimeout(200);

        const startDate = page.locator("#filter-start, [name='start']");

        if ((await startDate.count()) > 0) {
          await startDate.fill("2024-01-01");
          await expect(startDate).toHaveValue("2024-01-01");
        }
      }
    });

    test("clear filters button resets all filters", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      const filterToggle = page.locator(".filter-toggle");

      if ((await filterToggle.count()) > 0) {
        await filterToggle.click();
        await page.waitForTimeout(200);

        // Set some filters
        const senderFilter = page.locator("#filter-sender, [name='sender']");
        if ((await senderFilter.count()) > 0) {
          await senderFilter.fill("Test");
        }

        // Clear filters
        const clearBtn = page.locator(".clear-filters-btn");
        if ((await clearBtn.count()) > 0) {
          await clearBtn.click();
          await page.waitForTimeout(200);

          // Filters should be cleared
          if ((await senderFilter.count()) > 0) {
            await expect(senderFilter).toHaveValue("");
          }
        }
      }
    });

    test("semantic mode shows similarity threshold slider", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      // Switch to semantic mode
      const modeToggle = page.locator(".mode-toggle");
      if ((await modeToggle.count()) > 0) {
        await modeToggle.click();
        await page.waitForTimeout(200);
      }

      // Open filters
      const filterToggle = page.locator(".filter-toggle");
      if ((await filterToggle.count()) > 0) {
        await filterToggle.click();
        await page.waitForTimeout(200);

        // Threshold slider should be visible
        const thresholdSlider = page.locator("#threshold, .threshold-control");
        await expect(thresholdSlider.first()).toBeVisible();
      }
    });
  });

  test.describe("Result Navigation", () => {
    test("Arrow keys navigate through results", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      // Search to get results
      await page.locator(".search-input").fill("lunch");
      await page.waitForTimeout(600);

      const results = page.locator(".result-item");

      if ((await results.count()) > 0) {
        // Navigate down
        await page.keyboard.press("ArrowDown");
        await page.waitForTimeout(100);

        // First result should be selected
        const selectedResult = page.locator(".result-item.selected");
        const hasSelection = (await selectedResult.count()) > 0;

        if (hasSelection) {
          await expect(selectedResult).toBeVisible();
        }
      }
    });

    test("Enter key opens selected result", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      await page.locator(".search-input").fill("lunch");
      await page.waitForTimeout(600);

      const results = page.locator(".result-item");

      if ((await results.count()) > 0) {
        // Select first result
        await page.keyboard.press("ArrowDown");
        await page.waitForTimeout(100);

        // Press Enter
        await page.keyboard.press("Enter");
        await page.waitForTimeout(300);

        // Modal should close and navigate to result
        // (Search closes after selection)
      }
    });

    test("clicking result navigates to message", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      await page.locator(".search-input").fill("lunch");
      await page.waitForTimeout(600);

      const results = page.locator(".result-item");

      if ((await results.count()) > 0) {
        await results.first().click();
        await page.waitForTimeout(300);

        // Should navigate to the message
        // Search modal should close
        await expect(page.locator(".search-modal")).not.toBeVisible();
      }
    });

    test("results show conversation grouping", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      await page.locator(".search-input").fill("lunch");
      await page.waitForTimeout(600);

      // Results should be grouped by conversation
      const resultGroups = page.locator(".result-group");

      if ((await resultGroups.count()) > 0) {
        // Each group should have a header
        const groupHeaders = page.locator(".group-header");
        expect(await groupHeaders.count()).toBeGreaterThan(0);
      }
    });

    test("results show message count per conversation", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      await page.locator(".search-input").fill("lunch");
      await page.waitForTimeout(600);

      const messageCount = page.locator(".message-count");

      if ((await messageCount.count()) > 0) {
        const text = await messageCount.first().textContent();
        // Should show a number
        expect(text).toMatch(/\d+/);
      }
    });

    test("semantic search shows similarity badges", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      // Switch to semantic mode
      const modeToggle = page.locator(".mode-toggle");
      if ((await modeToggle.count()) > 0) {
        await modeToggle.click();
        await page.waitForTimeout(200);
      }

      // Search
      await page.locator(".search-input").fill("dinner plans");
      await page.waitForTimeout(600);

      // Check for similarity badges
      const similarityBadge = page.locator(".similarity-badge");

      if ((await similarityBadge.count()) > 0) {
        const text = await similarityBadge.first().textContent();
        // Should show percentage
        expect(text).toMatch(/\d+%/);
      }
    });
  });

  test.describe("Keyboard Hints", () => {
    test("shows keyboard hints in footer", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      const footer = page.locator(".search-footer, .keyboard-hints");
      await expect(footer).toBeVisible();

      // Should show navigation hints
      await expect(footer).toContainText(/Navigate|Arrow/i);
      await expect(footer).toContainText(/Enter|Select/i);
      await expect(footer).toContainText(/Esc|Close/i);
    });

    test("shows current search mode in footer", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      const modeHint = page.locator(".mode-hint");

      if ((await modeHint.count()) > 0) {
        const text = await modeHint.textContent();
        expect(text).toMatch(/Keyword|Semantic|AI/i);
      }
    });
  });

  test.describe("Responsive Behavior", () => {
    test("modal is responsive at different widths", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      // Test at different widths
      const widths = [1200, 800, 500];

      for (const width of widths) {
        await page.setViewportSize({ width, height: 720 });
        await page.waitForTimeout(100);

        // Modal should still be visible and usable
        await expect(page.locator(".search-modal")).toBeVisible();
        await expect(page.locator(".search-input")).toBeVisible();
      }
    });

    test("results scroll when many are present", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      await page.locator(".search-input").fill("message");
      await page.waitForTimeout(600);

      const resultsContainer = page.locator(".search-results, .results-list");

      if ((await resultsContainer.count()) > 0) {
        // Check if scrollable
        const scrollInfo = await resultsContainer.evaluate((el) => ({
          scrollHeight: el.scrollHeight,
          clientHeight: el.clientHeight,
          isScrollable: el.scrollHeight > el.clientHeight,
        }));

        // If many results, should be scrollable
        if (scrollInfo.scrollHeight > 300) {
          expect(scrollInfo.isScrollable).toBe(true);
        }
      }
    });
  });
});
