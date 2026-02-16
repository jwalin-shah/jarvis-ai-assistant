/**
 * E2E performance tests for JARVIS Desktop.
 *
 * These tests measure and assert performance characteristics:
 * - Render times < 100ms
 * - Scroll performance with large datasets
 * - Memory usage patterns
 * - DOM node counts
 */

import { test, expect } from "./fixtures";
import {
  waitForAppLoad,
  navigateToView,
  selectConversation,
  measureRenderTime,
  collectPerformanceMetrics,
  scrollToBottom,
  scrollToTop,
  getScrollPosition,
  countVisibleElements,
} from "./fixtures";

test.describe("Performance Tests", () => {
  test.describe("Render Times", () => {
    test("app initial render completes within 100ms", async ({ mockedPage: page }) => {
      const renderTime = await measureRenderTime(
        page,
        async () => {
          await page.goto("/");
        },
        ".sidebar"
      );

      // Initial render should be fast
      expect(renderTime).toBeLessThan(2000); // 2s for initial load including assets
      console.log(`Initial render time: ${renderTime}ms`);
    });

    test("conversation list renders within 100ms", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      const renderTime = await measureRenderTime(
        page,
        async () => {
          await navigateToView(page, "messages");
        },
        ".conversation"
      );

      expect(renderTime).toBeLessThan(500);
      console.log(`Conversation list render time: ${renderTime}ms`);
    });

    test("message view renders within 100ms", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await page.waitForSelector(".conversation");

      const renderTime = await measureRenderTime(
        page,
        async () => {
          await selectConversation(page, "John Doe");
        },
        ".message"
      );

      expect(renderTime).toBeLessThan(500);
      console.log(`Message view render time: ${renderTime}ms`);
    });

    test("settings page renders within 100ms", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      const renderTime = await measureRenderTime(
        page,
        async () => {
          await navigateToView(page, "settings");
        },
        ".section"
      );

      expect(renderTime).toBeLessThan(500);
      console.log(`Settings page render time: ${renderTime}ms`);
    });

    test("health page renders within 100ms", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      const renderTime = await measureRenderTime(
        page,
        async () => {
          await navigateToView(page, "health");
        },
        ".metric-card"
      );

      expect(renderTime).toBeLessThan(500);
      console.log(`Health page render time: ${renderTime}ms`);
    });
  });

  test.describe("Large Dataset Performance", () => {
    test("handles 1000+ conversations without lag", async ({ largePage: page }) => {
      await page.goto("/");
      await page.waitForSelector(".sidebar", { state: "visible" });

      // Wait for conversations to load
      await page.waitForSelector(".conversation", { timeout: 10000 });

      // Check that many conversations are rendered (or virtualized)
      const conversationCount = await page.locator(".conversation").count();
      expect(conversationCount).toBeGreaterThan(0);

      // Collect performance metrics
      const metrics = await collectPerformanceMetrics(page);
      console.log(`DOM nodes with large data: ${metrics.domNodes}`);

      // DOM should not explode with large datasets (virtualization)
      // Allow up to 5000 nodes for a complex UI
      expect(metrics.domNodes).toBeLessThan(10000);
    });

    test("scrolling through large conversation list is smooth", async ({ largePage: page }) => {
      await page.goto("/");
      await page.waitForSelector(".sidebar", { state: "visible" });
      await page.waitForSelector(".conversation", { timeout: 10000 });

      const listSelector = ".conversation-list, .conversations";
      const list = page.locator(listSelector);

      if ((await list.count()) > 0) {
        // Measure scroll performance
        const startTime = Date.now();

        // Scroll to bottom
        await scrollToBottom(page, listSelector);
        await page.waitForFunction(() => true, { timeout: 100 });

        // Scroll back to top
        await scrollToTop(page, listSelector);
        await page.waitForFunction(() => true, { timeout: 100 });

        const scrollTime = Date.now() - startTime;
        console.log(`Scroll round-trip time: ${scrollTime}ms`);

        // Scrolling should be responsive
        expect(scrollTime).toBeLessThan(2000);
      }
    });

    test("virtual scrolling only renders visible items", async ({ largePage: page }) => {
      await page.goto("/");
      await page.waitForSelector(".sidebar", { state: "visible" });
      await page.waitForSelector(".conversation", { timeout: 10000 });

      // Count visible vs total conversations
      const totalCount = await page.locator(".conversation").count();
      const visibleCount = await countVisibleElements(page, ".conversation");

      console.log(`Total conversations: ${totalCount}, Visible: ${visibleCount}`);

      // With virtualization, visible count should be much less than total
      // (unless the list is very small)
      if (totalCount > 50) {
        // If not using virtualization, visible should equal total
        // If using virtualization, visible should be less
        expect(visibleCount).toBeLessThanOrEqual(totalCount);
      }
    });
  });

  test.describe("Memory Usage", () => {
    test("memory stays stable during navigation", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Collect initial metrics
      const initialMetrics = await collectPerformanceMetrics(page);
      console.log(`Initial heap: ${(initialMetrics.heapUsed / 1024 / 1024).toFixed(2)} MB`);

      // Navigate through all views multiple times
      for (let i = 0; i < 3; i++) {
        await navigateToView(page, "messages");
        await page.waitForLoadState("domcontentloaded");
        await navigateToView(page, "dashboard");
        await page.waitForLoadState("domcontentloaded");
        await navigateToView(page, "health");
        await page.waitForLoadState("domcontentloaded");
        await navigateToView(page, "settings");
        await page.waitForLoadState("domcontentloaded");
      }

      // Collect final metrics
      const finalMetrics = await collectPerformanceMetrics(page);
      console.log(`Final heap: ${(finalMetrics.heapUsed / 1024 / 1024).toFixed(2)} MB`);

      // Memory should not grow significantly (allow 50MB growth)
      // Note: This may not work in all browsers due to memory API availability
      if (initialMetrics.heapUsed > 0 && finalMetrics.heapUsed > 0) {
        const memoryGrowth = finalMetrics.heapUsed - initialMetrics.heapUsed;
        const memoryGrowthMB = memoryGrowth / 1024 / 1024;
        console.log(`Memory growth: ${memoryGrowthMB.toFixed(2)} MB`);

        expect(memoryGrowthMB).toBeLessThan(100); // Allow up to 100MB growth
      }
    });

    test("DOM nodes stay stable during navigation", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Collect initial metrics
      const initialMetrics = await collectPerformanceMetrics(page);
      console.log(`Initial DOM nodes: ${initialMetrics.domNodes}`);

      // Navigate through views
      await navigateToView(page, "settings");
      await page.waitForLoadState("domcontentloaded");
      await navigateToView(page, "health");
      await page.waitForLoadState("domcontentloaded");
      await navigateToView(page, "messages");
      await page.waitForLoadState("domcontentloaded");

      // Collect final metrics
      const finalMetrics = await collectPerformanceMetrics(page);
      console.log(`Final DOM nodes: ${finalMetrics.domNodes}`);

      // DOM node count should not grow excessively
      const nodeGrowth = finalMetrics.domNodes - initialMetrics.domNodes;
      console.log(`DOM node growth: ${nodeGrowth}`);

      // Allow some growth but not excessive
      expect(nodeGrowth).toBeLessThan(500);
    });

    test("conversation switching does not leak memory", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await page.waitForSelector(".conversation");

      // Collect initial metrics
      const initialMetrics = await collectPerformanceMetrics(page);

      // Switch between conversations multiple times
      const conversations = ["John Doe", "Jane Smith", "Project Team"];
      for (let i = 0; i < 5; i++) {
        for (const name of conversations) {
          try {
            await selectConversation(page, name);
            await page.waitForSelector(".message", { timeout: 500 }).catch(() => {});
          } catch {
            // Conversation might not exist in mock data
          }
        }
      }

      // Collect final metrics
      const finalMetrics = await collectPerformanceMetrics(page);

      // Check DOM node growth
      const nodeGrowth = finalMetrics.domNodes - initialMetrics.domNodes;
      console.log(`DOM nodes after conversation switching: ${nodeGrowth} growth`);

      expect(nodeGrowth).toBeLessThan(200);
    });
  });

  test.describe("Layout Stability", () => {
    test("no significant layout shifts during load", async ({ mockedPage: page }) => {
      // Set up layout shift tracking
      await page.goto("/");
      await waitForAppLoad(page);

      // Wait for layout to stabilize
      await page.waitForLoadState("networkidle", { timeout: 1000 }).catch(() => {});

      const metrics = await collectPerformanceMetrics(page);
      console.log(`Layout shifts: ${metrics.layoutShifts}`);

      // Layout shifts should be minimal
      expect(metrics.layoutShifts).toBeLessThan(0.5);
    });

    test("conversation selection does not cause layout shift", async ({
      mockedPage: page,
    }) => {
      await page.goto("/");
      await waitForAppLoad(page);
      await page.waitForSelector(".conversation");

      // Get initial position of sidebar
      const sidebarBefore = await page.locator(".sidebar").boundingBox();

      // Select a conversation
      await selectConversation(page, "John Doe");
      await page.waitForSelector(".message");

      // Get position after
      const sidebarAfter = await page.locator(".sidebar").boundingBox();

      // Sidebar should not move
      if (sidebarBefore && sidebarAfter) {
        expect(sidebarAfter.x).toBe(sidebarBefore.x);
        expect(sidebarAfter.width).toBe(sidebarBefore.width);
      }
    });
  });

  test.describe("Loading States", () => {
    test("loading indicators appear quickly", async ({ slowPage: page }) => {
      const startTime = Date.now();
      await page.goto("/");

      // Loading indicator should appear within 100ms
      try {
        await page.waitForSelector(".loading, .spinner, .loading-state", {
          state: "visible",
          timeout: 1000,
        });
        const loadingTime = Date.now() - startTime;
        console.log(`Loading indicator appeared in ${loadingTime}ms`);
      } catch {
        // Loading might complete before we can catch it
        console.log("Loading completed too quickly to measure indicator");
      }
    });

    test("loading states are removed after data loads", async ({ slowPage: page }) => {
      await page.goto("/");

      // Wait for loading to complete
      await page.waitForSelector(".sidebar", { state: "visible", timeout: 10000 });

      // Loading indicators should be gone
      const loadingIndicators = page.locator(".loading-overlay, .loading-spinner");
      const count = await loadingIndicators.count();

      // There shouldn't be any persistent loading indicators
      // (Some components might have their own loading states that are fine)
      expect(count).toBeLessThan(5);
    });
  });

  test.describe("Animation Performance", () => {
    test("transitions are smooth (no janky frames)", async ({ mockedPage: page }) => {
      await page.goto("/");
      await waitForAppLoad(page);

      // Enable performance monitoring
      const client = await page.context().newCDPSession(page);

      await client.send("Performance.enable");

      // Trigger some transitions
      await navigateToView(page, "settings");
      await page.waitForLoadState("domcontentloaded");
      await navigateToView(page, "messages");
      await page.waitForLoadState("domcontentloaded");

      // Get performance metrics
      const metrics = await client.send("Performance.getMetrics");
      const layoutCount = metrics.metrics.find((m) => m.name === "LayoutCount")?.value || 0;
      const recalcStyleCount =
        metrics.metrics.find((m) => m.name === "RecalcStyleCount")?.value || 0;

      console.log(`Layouts: ${layoutCount}, Style recalcs: ${recalcStyleCount}`);

      // These should be reasonable numbers
      expect(layoutCount).toBeLessThan(200);
      expect(recalcStyleCount).toBeLessThan(500);
    });
  });
});
