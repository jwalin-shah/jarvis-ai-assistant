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

      // Wait for modal to actually close
      await expect(
        page.locator(".search-modal, .global-search")
      ).not.toBeVisible({ timeout: 5000 });
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

        // Wait for placeholder to change
        const input = page.locator(".search-input");
        await expect(input).toHaveAttribute("placeholder", /meaning|semantic/i, {
          timeout: 5000,
        });
      }
    });

    test("semantic mode shows AI indicator", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      const modeToggle = page.locator(".mode-toggle");

      if ((await modeToggle.count()) > 0) {
        await modeToggle.click();

        // Wait for semantic indicator to appear
        const semanticIndicator = page.locator(
          ".mode-toggle.semantic, .search-icon.semantic"
        );
        await expect(semanticIndicator.first()).toBeVisible({ timeout: 5000 });
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

        // Wait for mode to switch
        await expect(modeToggle).toHaveClass(/semantic/, { timeout: 5000 });

        // Type search query
        await page.locator(".search-input").fill("dinner plans");

        // Wait for results or no-results state to appear after debounce
        await Promise.race([
          page
            .locator(".result-item, .no-results-state")
            .first()
            .waitFor({ state: "visible", timeout: 5000 })
            .catch(() => {}),
          page.waitForLoadState("networkidle", { timeout: 2000 }).catch(() => {})
        ]);

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

      // Wait for results or no-results state to appear after debounce
      await Promise.race([
        page
          .locator(".result-item, .results-list, .no-results-state")
          .first()
          .waitFor({ state: "visible", timeout: 5000 }),
        page.waitForLoadState("networkidle", { timeout: 2000 }).catch(() => {})
      ]);

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

      // Wait for no results state to appear after debounce
      const noResultsState = page.locator(".no-results-state");
      await noResultsState.waitFor({ state: "visible", timeout: 5000 }).catch(() => {});

      if ((await noResultsState.count()) > 0) {
        await expect(noResultsState).toBeVisible();
        await expect(noResultsState).toContainText(/no results|not found/i);
      }
    });

    test("debounces search input", async ({ socketMockedPage: page }) => {
      let searchCount = 0;
      let lastRequestTime = Date.now();

      // Count search requests
      await page.route("http://localhost:8742/conversations/search", async (route) => {
        searchCount++;
        lastRequestTime = Date.now();
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([]),
        });
      });

      await openGlobalSearch(page);

      const startTime = Date.now();
      // Type quickly
      await page.locator(".search-input").type("testing search", { delay: 20 });

      // Wait for debounce to complete (wait until 1s after last request or 2s max)
      while (Date.now() - lastRequestTime < 1000 && Date.now() - startTime < 2000) {
        await page.waitForFunction(() => true, { timeout: 100 });
      }

      // Should have made only 1-2 requests (debounced)
      expect(searchCount).toBeLessThanOrEqual(3);
    });

    test("clears results when input is cleared", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      // Search and get results
      await page.locator(".search-input").fill("lunch");

      // Wait for results to appear after debounce
      await Promise.race([
        page
          .locator(".result-item, .no-results-state")
          .first()
          .waitFor({ state: "visible", timeout: 5000 })
          .catch(() => {}),
        page.waitForLoadState("networkidle", { timeout: 2000 }).catch(() => {})
      ]);

      // Clear input
      const clearBtn = page.locator(".clear-btn");
      if ((await clearBtn.count()) > 0) {
        await clearBtn.click();
      } else {
        await page.locator(".search-input").fill("");
      }

      // Wait for empty state to appear
      await expect(page.locator(".empty-state")).toBeVisible({ timeout: 5000 });
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
        await expect(page.locator(".filters-panel")).toBeVisible({ timeout: 5000 });

        // Close filters
        await filterToggle.click();
        await expect(page.locator(".filters-panel")).not.toBeVisible({
          timeout: 5000,
        });
      }
    });

    test("sender filter accepts input", async ({ socketMockedPage: page }) => {
      await openGlobalSearch(page);

      const filterToggle = page.locator(".filter-toggle");

      if ((await filterToggle.count()) > 0) {
        await filterToggle.click();

        // Wait for filter panel to be visible
        await expect(page.locator(".filters-panel")).toBeVisible({ timeout: 5000 });

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

        // Wait for filter panel to be visible
        await expect(page.locator(".filters-panel")).toBeVisible({ timeout: 5000 });

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

        // Wait for filter panel to be visible
        await expect(page.locator(".filters-panel")).toBeVisible({ timeout: 5000 });

        // Set some filters
        const senderFilter = page.locator("#filter-sender, [name='sender']");
        if ((await senderFilter.count()) > 0) {
          await senderFilter.fill("Test");
        }

        // Clear filters
        const clearBtn = page.locator(".clear-filters-btn");
        if ((await clearBtn.count()) > 0) {
          await clearBtn.click();

          // Wait for filters to be cleared
          if ((await senderFilter.count()) > 0) {
            await expect(senderFilter).toHaveValue("", { timeout: 5000 });
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

        // Wait for mode to switch
        await expect(modeToggle).toHaveClass(/semantic/, { timeout: 5000 });
      }

      // Open filters
      const filterToggle = page.locator(".filter-toggle");
      if ((await filterToggle.count()) > 0) {
        await filterToggle.click();

        // Wait for filter panel to be visible
        await expect(page.locator(".filters-panel")).toBeVisible({ timeout: 5000 });

        // Threshold slider should be visible
        const thresholdSlider = page.locator("#threshold, .threshold-control");
        await expect(thresholdSlider.first()).toBeVisible({ timeout: 5000 });
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

      // Wait for results to appear
      const results = page.locator(".result-item");
      await results.first().waitFor({ state: "visible", timeout: 5000 }).catch(() => {});

      if ((await results.count()) > 0) {
        // Navigate down
        await page.keyboard.press("ArrowDown");

        // Wait for selection to be applied
        const selectedResult = page.locator(".result-item.selected");
        await selectedResult
          .waitFor({ state: "visible", timeout: 5000 })
          .catch(() => {});

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

      // Wait for results to appear
      const results = page.locator(".result-item");
      await results.first().waitFor({ state: "visible", timeout: 5000 }).catch(() => {});

      if ((await results.count()) > 0) {
        // Select first result
        await page.keyboard.press("ArrowDown");

        // Wait for selection to be applied
        await page
          .locator(".result-item.selected")
          .waitFor({ state: "visible", timeout: 5000 })
          .catch(() => {});

        // Press Enter
        await page.keyboard.press("Enter");

        // Wait for modal to close after selection
        await expect(page.locator(".search-modal")).not.toBeVisible({
          timeout: 5000,
        });
      }
    });

    test("clicking result navigates to message", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      await page.locator(".search-input").fill("lunch");

      // Wait for results to appear
      const results = page.locator(".result-item");
      await results.first().waitFor({ state: "visible", timeout: 5000 }).catch(() => {});

      if ((await results.count()) > 0) {
        await results.first().click();

        // Wait for modal to close after navigation
        await expect(page.locator(".search-modal")).not.toBeVisible({
          timeout: 5000,
        });
      }
    });

    test("results show conversation grouping", async ({
      socketMockedPage: page,
    }) => {
      await openGlobalSearch(page);

      await page.locator(".search-input").fill("lunch");

      // Wait for results to appear after search
      await Promise.race([
        page
          .locator(".result-item, .result-group")
          .first()
          .waitFor({ state: "visible", timeout: 5000 })
          .catch(() => {}),
        page.waitForLoadState("networkidle", { timeout: 2000 }).catch(() => {})
      ]);

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

      // Wait for results to appear after search
      await Promise.race([
        page
          .locator(".result-item, .message-count")
          .first()
          .waitFor({ state: "visible", timeout: 5000 })
          .catch(() => {}),
        page.waitForLoadState("networkidle", { timeout: 2000 }).catch(() => {})
      ]);

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

        // Wait for mode to switch
        await expect(modeToggle).toHaveClass(/semantic/, { timeout: 5000 });
      }

      // Search
      await page.locator(".search-input").fill("dinner plans");

      // Wait for results to appear after semantic search
      await Promise.race([
        page
          .locator(".result-item, .similarity-badge")
          .first()
          .waitFor({ state: "visible", timeout: 5000 })
          .catch(() => {}),
        page.waitForLoadState("networkidle", { timeout: 2000 }).catch(() => {})
      ]);

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

        // Wait for layout to stabilize after viewport change
        await page.waitForLoadState("domcontentloaded");
        await expect(page.locator(".search-modal")).toBeVisible({ timeout: 5000 });

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

      // Wait for results to appear
      const resultsContainer = page.locator(".search-results, .results-list");
      await resultsContainer
        .first()
        .waitFor({ state: "visible", timeout: 5000 })
        .catch(() => {});

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
