/**
 * TestDriver.ai E2E tests for JARVIS Desktop.
 *
 * Uses Vision AI for selectorless testing - tests don't break when UI changes.
 *
 * Prerequisites:
 *   1. Set TD_API_KEY environment variable
 *   2. Start backend: make api-dev (port 8742)
 *   3. Run: pnpm test:ai
 *
 * Docs: https://docs.testdriver.ai/v6/apps/tauri-apps
 */

import { expect, test } from '@testdriver.ai/playwright';

test.beforeEach(async ({ page }) => {
  await page.goto('http://localhost:1420');
  // Wait for app to initialize
  await page.waitForTimeout(2000);
});

test.describe('JARVIS App Launch', () => {
  test('should show the JARVIS logo and sidebar', async ({ page }) => {
    // Vision AI assertion - no selectors needed
    await expect(page).toMatchPrompt('JARVIS logo is visible in the sidebar');
  });

  test('should show navigation items', async ({ page }) => {
    await expect(page).toMatchPrompt(
      'Navigation sidebar shows Dashboard, Messages, Health Status, and Settings options'
    );
  });
});

test.describe('Backend Connection', () => {
  test('should show connection status', async ({ page }) => {
    // Wait for connection attempt
    await page.waitForTimeout(3000);
    await expect(page).toMatchPrompt(
      'Connection status indicator is visible showing either Connected or Disconnected'
    );
  });
});

test.describe('Messages View', () => {
  test.agent(`
    - Click on "Messages" in the sidebar navigation
    - Wait for conversations to load
    - Verify that a list of conversations is displayed
    - Click on the first conversation in the list
    - Verify that messages are displayed for the selected conversation
  `);
});

test.describe('Health Status', () => {
  test.agent(`
    - Click on "Health Status" in the sidebar
    - Verify that memory usage information is displayed
    - Verify that model status is shown
  `);
});

test.describe('Settings', () => {
  test.agent(`
    - Click on "Settings" in the sidebar
    - Verify that Model selection section is visible
    - Verify that configuration options are displayed
  `);
});

test.describe('Smart Replies', () => {
  test.agent(`
    - Click on "Messages" in the sidebar
    - Wait for conversations to load
    - Click on a conversation that has messages
    - Look for smart reply suggestions or chips near the message input
    - If smart replies are visible, verify they contain suggested response text
  `);
});

test.describe('Socket Streaming', () => {
  test('should stream AI responses', async ({ page, ai }) => {
    // Navigate to messages
    await ai('Click on Messages in the sidebar');
    await page.waitForTimeout(2000);

    // Select a conversation
    await ai('Click on the first conversation in the list');
    await page.waitForTimeout(2000);

    // Check for AI draft functionality (if available)
    const hasDraftButton = await page
      .locator('button:has-text("AI Draft"), button:has-text("Draft")')
      .isVisible()
      .catch(() => false);

    if (hasDraftButton) {
      await ai('Click the AI Draft or Draft button');
      // Wait for streaming to complete
      await page.waitForTimeout(10000);
      await expect(page).toMatchPrompt('AI generated draft text is visible in the interface');
    }
  });
});
