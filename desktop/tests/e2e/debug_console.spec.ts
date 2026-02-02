/**
 * Debug test to capture console output
 */
import { test, expect } from "@playwright/test";
import { setupApiMocks } from "../mocks";

test("capture console output", async ({ page }) => {
  // Collect all console messages
  const messages: string[] = [];
  const errors: string[] = [];

  page.on("console", (msg) => {
    const text = `[${msg.type()}] ${msg.text()}`;
    messages.push(text);
    if (msg.type() === "error") {
      errors.push(msg.text());
    }
  });

  page.on("pageerror", (error) => {
    errors.push(`PageError: ${error.message}`);
  });

  // Setup API mocks
  await setupApiMocks(page);

  // Navigate
  await page.goto("/");

  // Wait a bit for any async errors
  await page.waitForTimeout(5000);

  // Log all console messages
  console.log("=== Console Messages ===");
  messages.forEach((m) => console.log(m));

  console.log("\n=== Errors ===");
  errors.forEach((e) => console.log(e));

  // Take screenshot
  await page.screenshot({ path: "debug-screenshot.png" });

  // Get page content
  const html = await page.content();
  console.log("\n=== Page HTML (first 2000 chars) ===");
  console.log(html.substring(0, 2000));

  // This test always passes - it's just for debugging
  expect(true).toBe(true);
});
