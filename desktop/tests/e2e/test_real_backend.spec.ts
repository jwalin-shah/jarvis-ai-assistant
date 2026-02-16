/**
 * E2E tests against the REAL backend (no mocks).
 *
 * Prerequisites:
 * 1. Start the API: make api-dev (port 8742)
 * 2. Start the frontend: pnpm dev (port 1420)
 * 3. Run: pnpm exec playwright test test_real_backend.spec.ts --headed
 *
 * These tests verify the full integration works end-to-end.
 */

import { test, expect, Page } from "@playwright/test";

// Run tests sequentially since they share the same backend
test.describe.configure({ mode: "serial" });

// Use the base test (no mocking fixtures)
test.describe("Real Backend Integration", () => {
  test.beforeEach(async ({ page }) => {
    // Go to the app
    await page.goto("/");

    // Wait for sidebar to load
    await page.waitForSelector(".sidebar", { state: "visible", timeout: 10000 });
  });

  test("connects to real backend and shows connected status", async ({ page }) => {
    // Wait for sidebar connection status specifically (not other status-text elements)
    const sidebarStatus = page.locator(".sidebar .status-text");
    await expect(sidebarStatus).toHaveText("Connected", {
      timeout: 15000,
    });
  });

  test("loads real conversations from iMessage database", async ({ page }) => {
    // Click Messages in sidebar
    await page.click('.nav-item[title="Messages"]');

    // Wait for conversations to load (real data)
    await page.waitForSelector(".conversation-list .conversation", { timeout: 15000 });

    // Should have at least one conversation
    const conversations = page.locator(".conversation-list .conversation");
    const count = await conversations.count();
    expect(count).toBeGreaterThan(0);

    console.log(`Found ${count} real conversations`);
  });

  test("loads messages when conversation is selected", async ({ page }) => {
    // Click Messages
    await page.click('.nav-item[title="Messages"]');

    // Wait for and click first conversation
    await page.waitForSelector(".conversation-list .conversation", { timeout: 20000 });
    await page.locator(".conversation-list .conversation").first().click();

    // Wait for message view to show (either messages or empty state)
    await page.waitForSelector(".message-view", { timeout: 20000 });

    // Check the message view is visible
    await expect(page.locator(".message-view")).toBeVisible();
  });

  test("health page shows real system status", async ({ page }) => {
    // Click Health
    await page.click('.nav-item[title="Health Status"]');

    // Wait for health data to load
    await page.waitForSelector(".health-status", { timeout: 15000 });

    // Should show memory info (use first() to avoid multiple matches)
    await expect(page.locator(".health-status").getByText("Memory").first()).toBeVisible();
  });

  test("settings page shows real model info", async ({ page }) => {
    // Click Settings
    await page.click('.nav-item[title="Settings"]');

    // Wait for settings to load
    await page.waitForSelector(".settings", { timeout: 15000 });

    // Should show model heading (use heading role for specificity)
    await expect(page.getByRole("heading", { name: "Model" })).toBeVisible();
  });

  test("debug endpoint returns trace data", async ({ page }) => {
    // Make a direct API call to debug endpoint
    const response = await page.request.get("http://localhost:8742/debug/status");
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data).toHaveProperty("memory_rss_mb");
    expect(data).toHaveProperty("model_loaded");

    console.log("System status:", data);
  });

  test("can generate AI draft (if model loaded)", async ({ page }) => {
    // This test requires the model to be loaded
    // Skip if model not loaded

    // First check if model is loaded
    const statusResponse = await page.request.get(
      "http://localhost:8742/debug/status"
    );
    const status = await statusResponse.json();

    if (!status.model_loaded) {
      console.log("Skipping AI draft test - model not loaded");
      test.skip();
      return;
    }

    // Navigate to messages
    await page.click('.nav-item[title="Messages"]');
    await page.waitForSelector(".conversation-list .conversation", { timeout: 15000 });

    // Click first conversation
    await page.locator(".conversation-list .conversation").first().click();
    await page.waitForSelector(".message-view .header", { timeout: 15000 });

    // Click AI Draft button if visible
    const aiDraftButton = page.locator('button:has-text("AI Draft")');
    if (await aiDraftButton.isVisible()) {
      await aiDraftButton.click();

      // Wait for draft to generate (could take a while)
      await page.waitForSelector(".draft-content, .draft-error", {
        timeout: 60000,
      });

      console.log("AI Draft generation completed");
    } else {
      console.log("AI Draft button not visible for this conversation");
    }
  });
});

/**
 * Performance measurement tests
 */
test.describe("Performance Metrics", () => {
  test("measures conversation list load time", async ({ page }) => {
    const startTime = Date.now();

    await page.goto("/");
    await page.waitForSelector(".sidebar", { state: "visible" });
    await page.click('.nav-item[title="Messages"]');
    await page.waitForSelector(".conversation-list .conversation", { timeout: 15000 });

    const loadTime = Date.now() - startTime;
    console.log(`Conversation list loaded in ${loadTime}ms`);

    // Should load within 5 seconds
    expect(loadTime).toBeLessThan(5000);
  });

  test("measures message load time", async ({ page }) => {
    await page.goto("/");
    await page.waitForSelector(".sidebar", { state: "visible" });
    await page.click('.nav-item[title="Messages"]');
    await page.waitForSelector(".conversation-list .conversation", { timeout: 15000 });

    const startTime = Date.now();
    await page.locator(".conversation-list .conversation").first().click();
    await page.waitForSelector(".message-view .messages, .message-view .empty-state", { timeout: 15000 });

    const loadTime = Date.now() - startTime;
    console.log(`Messages loaded in ${loadTime}ms`);

    // Should load within 3 seconds
    expect(loadTime).toBeLessThan(3000);
  });
});
