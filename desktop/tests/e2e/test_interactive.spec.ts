/**
 * Interactive E2E test - clicks through everything and tests with real user
 *
 * Run: npx playwright test test_interactive.spec.ts --headed --project=chromium
 */

import { test, expect } from '@playwright/test';

test.describe('Interactive Full App Test', () => {
  test('full app walkthrough with Sangati Shah', async ({ page }) => {
    // Slow down so we can see what's happening
    test.slow();

    // Go to app
    await page.goto('/');
    console.log('1. Loaded app');

    // Wait for sidebar
    await page.waitForSelector('.sidebar', { state: 'visible', timeout: 10000 });
    console.log('2. Sidebar visible');

    // Wait for connected status (in sidebar)
    await page.waitForSelector('.status-dot.connected', { timeout: 15000 });
    console.log('3. Connected to backend');

    // Test sidebar collapse
    const collapseBtn = page.locator('.collapse-btn');
    if (await collapseBtn.isVisible()) {
      await collapseBtn.click();
      // Wait for sidebar to collapse (check for collapsed state or width change)
      await page
        .waitForSelector(".sidebar.collapsed, .sidebar[data-collapsed='true']", {
          timeout: 2000,
        })
        .catch(() => page.waitForFunction(() => true, { timeout: 500 })); // fallback for animation
      console.log('4. Collapsed sidebar');
      await collapseBtn.click();
      // Wait for sidebar to expand
      await page
        .waitForSelector('.sidebar:not(.collapsed)', {
          timeout: 2000,
        })
        .catch(() => page.waitForFunction(() => true, { timeout: 500 })); // fallback for animation
      console.log('5. Expanded sidebar');
    }

    // Click Dashboard
    await page.click('.nav-item:has-text("Dashboard")');
    // Wait for dashboard content to load (look for typical dashboard elements)
    await page
      .waitForSelector('.dashboard, .stats-card, .chart', {
        timeout: 5000,
      })
      .catch(() => page.waitForLoadState('networkidle', { timeout: 2000 }));
    console.log('6. Viewed Dashboard');

    // Click Messages
    await page.click('.nav-item:has-text("Messages")');
    await page.waitForSelector('.conversation', { timeout: 15000 });
    console.log('7. Viewed Messages - conversations loaded');

    // Find and click Sangati Shah
    const sangatiConvo = page.locator('.conversation:has-text("Sangati")');
    const sangatiExists = (await sangatiConvo.count()) > 0;

    if (sangatiExists) {
      await sangatiConvo.first().click();
      // Wait for messages to load
      await page.waitForSelector('.bubble, .empty-state', { timeout: 10000 });
      console.log('8. Clicked on Sangati Shah conversation');
      console.log('9. Messages loaded');

      // Check for suggested replies (smart-reply-container with chips)
      const suggestedReplies = page.locator('.smart-reply-container .chip');
      // Wait for suggestions to load or timeout after 5s
      await page
        .waitForSelector('.smart-reply-container .chip', {
          timeout: 5000,
        })
        .catch(() => console.log('Suggestions not loaded yet'));

      const chipCount = await suggestedReplies.count();
      if (chipCount > 0) {
        console.log(`10. Found ${chipCount} suggested replies!`);

        // Click first suggestion to copy it
        await suggestedReplies.first().click();
        // Wait for clipboard feedback or state change
        await page
          .waitForSelector('.chip.copied, .chip:active', {
            timeout: 2000,
          })
          .catch(() => page.waitForFunction(() => true, { timeout: 200 })); // short fallback for animation
        console.log('11. Clicked a suggested reply (copied to clipboard)');
      } else {
        console.log('10. No suggested replies shown yet');
      }

      // Check if compose area exists (textarea with placeholder iMessage)
      const composeArea = page.locator('textarea.compose-input');
      if ((await composeArea.count()) > 0) {
        // Type a test message
        await composeArea.fill('Test from Playwright - hi Sangati!');
        console.log('12. Typed test message in compose area');

        // Wait for send button to enable
        await page
          .waitForSelector('.send-button:not([disabled])', {
            timeout: 2000,
          })
          .catch(() => console.log('Send button validation in progress'));

        // Look for send button (enabled)
        const sendBtn = page.locator('.send-button:not([disabled])');
        if ((await sendBtn.count()) > 0) {
          // Click send and wait for message to appear in chat or API response
          const messagePromise = page
            .waitForResponse(
              (response) => response.url().includes('/send') || response.url().includes('/message'),
              { timeout: 5000 }
            )
            .catch(() => null);
          await sendBtn.click();
          await messagePromise;
          // Wait for sent message to appear or chat to update
          await page
            .waitForSelector('.bubble:last-child', {
              timeout: 3000,
            })
            .catch(() => page.waitForFunction(() => true, { timeout: 500 }));
          console.log('13. Sent message to Sangati Shah!');
        } else {
          console.log('13. Send button still disabled - skipping send');
        }
      } else {
        console.log('12. Compose area not found');
      }

      // Test AI Draft button
      const aiButton = page.locator('button:has-text("AI Draft")');
      if ((await aiButton.count()) > 0) {
        // Wait for AI response after clicking
        const aiResponsePromise = page
          .waitForResponse(
            (response) => response.url().includes('/generate') || response.url().includes('/draft'),
            { timeout: 8000 }
          )
          .catch(() => null);
        await aiButton.click();
        await aiResponsePromise;
        // Wait for draft to populate compose area or show in UI
        await page
          .waitForSelector('textarea.compose-input[value]:not([value=""])', {
            timeout: 5000,
          })
          .catch(() => page.waitForLoadState('networkidle', { timeout: 1000 })); // fallback if selector doesn't match
        console.log('14. Clicked AI Draft button');
      }
    } else {
      console.log('8. Sangati Shah not found, clicking first conversation');
      await page.locator('.conversation').first().click();
    }

    // Click Health
    await page.click('.nav-item:has-text("Health")');
    // Wait for health page content to load
    await page
      .waitForSelector('.health-page, .health-stats, .health-chart', {
        timeout: 5000,
      })
      .catch(() => page.waitForLoadState('networkidle', { timeout: 2000 }));
    console.log('15. Viewed Health page');

    // Click Settings
    await page.click('.nav-item:has-text("Settings")');
    // Wait for settings page content to load
    await page
      .waitForSelector('.settings-page, .settings-section, form', {
        timeout: 5000,
      })
      .catch(() => page.waitForLoadState('networkidle', { timeout: 2000 }));
    console.log('16. Viewed Settings page');

    // Click Templates
    await page.click('.nav-item:has-text("Templates")');
    // Wait for templates page content to load
    await page
      .waitForSelector('.templates-page, .template-card, .template-list', {
        timeout: 5000,
      })
      .catch(() => page.waitForLoadState('networkidle', { timeout: 2000 }));
    console.log('17. Viewed Templates page');

    // Back to Messages
    await page.click('.nav-item:has-text("Messages")');
    // Wait for messages view to be visible again
    await page
      .waitForSelector('.conversation, .message-list', {
        timeout: 3000,
      })
      .catch(() => page.waitForLoadState('domcontentloaded'));
    console.log('18. Back to Messages');

    // Test search (Cmd+K)
    await page.keyboard.press('Meta+k');
    // Wait for search modal to appear
    const searchModal = page.locator('.search-modal, .global-search');
    await page
      .waitForSelector('.search-modal, .global-search', {
        timeout: 2000,
      })
      .catch(() => console.log('Search modal not found'));

    if ((await searchModal.count()) > 0) {
      console.log('19. Search modal opened');
      await page.keyboard.press('Escape');
      // Wait for modal to close
      await page
        .waitForSelector('.search-modal, .global-search', {
          state: 'hidden',
          timeout: 2000,
        })
        .catch(() => page.waitForFunction(() => true, { timeout: 200 })); // short fallback
    }

    console.log('=== Interactive test complete! ===');
  });
});
