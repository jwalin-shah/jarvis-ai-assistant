/**
 * Interactive E2E test - clicks through everything and tests with real user
 *
 * Run: npx playwright test test_interactive.spec.ts --headed --project=chromium
 */

import { test, expect } from "@playwright/test";

test.describe("Interactive Full App Test", () => {
  test("full app walkthrough with Sangati Shah", async ({ page }) => {
    // Slow down so we can see what's happening
    test.slow();

    // Go to app
    await page.goto("/");
    console.log("1. Loaded app");

    // Wait for sidebar
    await page.waitForSelector(".sidebar", { state: "visible", timeout: 10000 });
    console.log("2. Sidebar visible");

    // Wait for connected status (in sidebar)
    await page.waitForSelector(".status-dot.connected", { timeout: 15000 });
    console.log("3. Connected to backend");

    // Test sidebar collapse
    const collapseBtn = page.locator(".collapse-btn");
    if (await collapseBtn.isVisible()) {
      await collapseBtn.click();
      await page.waitForTimeout(500);
      console.log("4. Collapsed sidebar");
      await collapseBtn.click();
      await page.waitForTimeout(500);
      console.log("5. Expanded sidebar");
    }

    // Click Dashboard
    await page.click('.nav-item:has-text("Dashboard")');
    await page.waitForTimeout(1000);
    console.log("6. Viewed Dashboard");

    // Click Messages
    await page.click('.nav-item:has-text("Messages")');
    await page.waitForSelector(".conversation", { timeout: 15000 });
    console.log("7. Viewed Messages - conversations loaded");

    // Find and click Sangati Shah
    const sangatiConvo = page.locator('.conversation:has-text("Sangati")');
    const sangatiExists = await sangatiConvo.count() > 0;

    if (sangatiExists) {
      await sangatiConvo.first().click();
      await page.waitForTimeout(1000);
      console.log("8. Clicked on Sangati Shah conversation");

      // Wait for messages to load
      await page.waitForSelector(".bubble, .empty-state", { timeout: 10000 });
      console.log("9. Messages loaded");

      // Check for suggested replies (smart-reply-container with chips)
      const suggestedReplies = page.locator(".smart-reply-container .chip");
      await page.waitForTimeout(3000); // Wait for suggestions to load

      const chipCount = await suggestedReplies.count();
      if (chipCount > 0) {
        console.log(`10. Found ${chipCount} suggested replies!`);

        // Click first suggestion to copy it
        await suggestedReplies.first().click();
        await page.waitForTimeout(1000);
        console.log("11. Clicked a suggested reply (copied to clipboard)");
      } else {
        console.log("10. No suggested replies shown yet");
      }

      // Check if compose area exists (textarea with placeholder iMessage)
      const composeArea = page.locator('textarea.compose-input');
      if (await composeArea.count() > 0) {
        // Type a test message
        await composeArea.fill("Test from Playwright - hi Sangati!");
        console.log("12. Typed test message in compose area");

        // Wait a moment for send button to enable
        await page.waitForTimeout(500);

        // Look for send button (enabled)
        const sendBtn = page.locator('.send-button:not([disabled])');
        if (await sendBtn.count() > 0) {
          // Click send
          await sendBtn.click();
          await page.waitForTimeout(2000);
          console.log("13. Sent message to Sangati Shah!");
        } else {
          console.log("13. Send button still disabled - skipping send");
        }
      } else {
        console.log("12. Compose area not found");
      }

      // Test AI Draft button
      const aiButton = page.locator('button:has-text("AI Draft")');
      if (await aiButton.count() > 0) {
        await aiButton.click();
        await page.waitForTimeout(3000);
        console.log("14. Clicked AI Draft button");
      }
    } else {
      console.log("8. Sangati Shah not found, clicking first conversation");
      await page.locator(".conversation").first().click();
    }

    // Click Health
    await page.click('.nav-item:has-text("Health")');
    await page.waitForTimeout(1000);
    console.log("15. Viewed Health page");

    // Click Settings
    await page.click('.nav-item:has-text("Settings")');
    await page.waitForTimeout(1000);
    console.log("16. Viewed Settings page");

    // Click Templates
    await page.click('.nav-item:has-text("Templates")');
    await page.waitForTimeout(1000);
    console.log("17. Viewed Templates page");

    // Back to Messages
    await page.click('.nav-item:has-text("Messages")');
    await page.waitForTimeout(500);
    console.log("18. Back to Messages");

    // Test search (Cmd+K)
    await page.keyboard.press("Meta+k");
    await page.waitForTimeout(1000);
    const searchModal = page.locator(".search-modal, .global-search");
    if (await searchModal.count() > 0) {
      console.log("19. Search modal opened");
      await page.keyboard.press("Escape");
    }

    console.log("=== Interactive test complete! ===");
  });
});
