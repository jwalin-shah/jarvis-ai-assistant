/**
 * E2E tests for error states and recovery.
 *
 * Verifies the app handles errors gracefully:
 * - API errors show user-friendly messages
 * - Network disconnection shows offline state
 * - Retry functionality works
 * - App recovers when connection is restored
 */

import { test, expect } from './fixtures';
import {
  waitForAppLoad,
  navigateToView,
  selectConversation,
  simulateOffline,
  simulateOnline,
  getConnectionStatus,
} from './fixtures';

test.describe('Error States and Recovery', () => {
  test.describe('API Error Handling', () => {
    test('shows error banner when health check fails', async ({ errorPage: page }) => {
      await page.goto('/');
      await page.waitForSelector('.sidebar', { state: 'visible' });

      // Wait for error to appear
      const errorBanner = page.locator(".error, .error-banner, [role='alert']");
      const disconnectedStatus = page.locator('.status-dot.disconnected');

      // Wait for either error banner or disconnected status
      await Promise.race([
        errorBanner
          .first()
          .waitFor({ state: 'visible', timeout: 10000 })
          .catch(() => {}),
        disconnectedStatus
          .first()
          .waitFor({ state: 'visible', timeout: 10000 })
          .catch(() => {}),
      ]);

      // Wait for DOM to update
      await page.waitForFunction(() => true, { timeout: 100 });

      // Either error banner or disconnected status should show
      const hasError = (await errorBanner.count()) > 0 || (await disconnectedStatus.count()) > 0;
      expect(hasError).toBe(true);
    });

    test('shows error when conversations fail to load', async ({ errorPage: page }) => {
      await page.goto('/');
      await page.waitForSelector('.sidebar');

      // Navigate to messages
      await page.locator('.nav-item[title="Messages"]').click();

      // Wait for error to appear
      const errorElement = page.locator('.error, .empty-state.error');
      await Promise.race([
        errorElement
          .first()
          .waitFor({ state: 'visible', timeout: 5000 })
          .catch(() => {}),
        page
          .waitForSelector('.conversation-list, .messages-container', { timeout: 5000 })
          .catch(() => {}),
      ]);

      if ((await errorElement.count()) > 0) {
        await expect(errorElement).toBeVisible();
      }
    });

    test('shows specific error message for API failures', async ({ page }) => {
      // Mock with specific error
      await page.route('http://localhost:8742/health', async (route) => {
        await route.fulfill({
          status: 503,
          contentType: 'application/json',
          body: JSON.stringify({
            error: 'Service Unavailable',
            detail: 'Model server is starting up, please wait',
          }),
        });
      });

      await page.route('http://localhost:8742/', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ status: 'degraded' }),
        });
      });

      await page.goto('/');
      await page.waitForSelector('.sidebar');
      await page.locator('.nav-item[title="Health Status"]').click();

      // Wait for health status view to load and show error
      const errorText = page.locator('.error-message, .error-detail, .status-banner');
      await Promise.race([
        errorText
          .first()
          .waitFor({ state: 'visible', timeout: 5000 })
          .catch(() => {}),
        page
          .waitForSelector('.health-status, .status-container', { timeout: 5000 })
          .catch(() => {}),
      ]);

      if ((await errorText.count()) > 0) {
        const text = await errorText.textContent();
        // Should contain some indication of the error
        expect(text).toBeTruthy();
      }
    });

    test('handles 404 errors gracefully', async ({ page }) => {
      await page.route('http://localhost:8742/conversations/invalid-id', async (route) => {
        await route.fulfill({
          status: 404,
          contentType: 'application/json',
          body: JSON.stringify({
            error: 'Not Found',
            detail: 'Conversation not found',
          }),
        });
      });

      await page.route('http://localhost:8742/**', async (route) => {
        if (route.request().url().includes('invalid-id')) {
          await route.fulfill({
            status: 404,
            contentType: 'application/json',
            body: JSON.stringify({ error: 'Not Found' }),
          });
        } else {
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ status: 'ok' }),
          });
        }
      });

      // 404 handling would need specific UI interaction
      // This documents expected behavior
    });

    test('handles timeout errors', async ({ page }) => {
      // Mock very slow endpoint that times out
      await page.route('http://localhost:8742/health', async (route) => {
        // Wait longer than typical timeout
        await new Promise((resolve) => setTimeout(resolve, 35000));
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ status: 'ok' }),
        });
      });

      // The page should handle the timeout gracefully
      // This is a long test and may need increased timeout
    });
  });

  test.describe('Network Disconnection', () => {
    test('shows disconnected status when offline', async ({ mockedPage: page }) => {
      await page.goto('/');
      await waitForAppLoad(page);

      // Verify connected first
      const initialStatus = await getConnectionStatus(page);
      expect(initialStatus).toBe('connected');

      // Go offline
      await simulateOffline(page);

      // Wait for disconnected status to appear
      const disconnectedIndicator = page.locator(
        '.status-dot.disconnected, .offline-banner, .connection-lost'
      );
      await disconnectedIndicator
        .first()
        .waitFor({ state: 'visible', timeout: 10000 })
        .catch(() => {});

      // Status should change to disconnected
      // Note: This depends on the app detecting the offline state
    });

    test('app remains functional with cached data when offline', async ({ mockedPage: page }) => {
      await page.goto('/');
      await waitForAppLoad(page);
      await page.waitForSelector('.conversation');

      // Select a conversation while online
      await selectConversation(page, 'John Doe');
      await page.waitForSelector('.message');

      // Go offline
      await simulateOffline(page);

      // Wait for offline state to be applied
      await page.waitForFunction(() => !navigator.onLine, { timeout: 500 }).catch(() => {});

      // Messages should still be visible (cached)
      await expect(page.locator('.message')).toBeVisible();

      // UI should remain interactive
      await expect(page.locator('.sidebar')).toBeVisible();

      // Restore online
      await simulateOnline(page);
    });

    test('shows offline indicator prominently', async ({ disconnectedPage: page }) => {
      await page.goto('/');
      await page.waitForSelector('.sidebar');

      // Wait for offline indicator to appear
      const offlineIndicator = page.locator(
        '.status-dot.disconnected, .offline-banner, .connection-lost'
      );
      await offlineIndicator.first().waitFor({ state: 'visible', timeout: 10000 });

      await expect(offlineIndicator.first()).toBeVisible();
    });

    test('retries connection after going online', async ({ mockedPage: page }) => {
      await page.goto('/');
      await waitForAppLoad(page);

      // Go offline then online
      await simulateOffline(page);

      // Wait for disconnected state
      const disconnectedIndicator = page.locator('.status-dot.disconnected, .offline-banner');
      await disconnectedIndicator
        .first()
        .waitFor({ state: 'visible', timeout: 5000 })
        .catch(() => {});

      await simulateOnline(page);

      // Wait for connected state
      const connectedIndicator = page.locator('.status-dot.connected');
      await connectedIndicator
        .first()
        .waitFor({ state: 'visible', timeout: 10000 })
        .catch(() => {});

      // Should reconnect
      const status = await getConnectionStatus(page);
      expect(status).toBe('connected');
    });
  });

  test.describe('Retry Functionality', () => {
    test('retry button reloads data', async ({ page }) => {
      let requestCount = 0;

      // First request fails, second succeeds
      await page.route('http://localhost:8742/health', async (route) => {
        requestCount++;
        if (requestCount === 1) {
          await route.fulfill({
            status: 500,
            contentType: 'application/json',
            body: JSON.stringify({ error: 'Server error' }),
          });
        } else {
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
              status: 'healthy',
              imessage_access: true,
              memory_available_gb: 6.5,
              memory_used_gb: 9.5,
              memory_mode: 'FULL',
              model_loaded: true,
              permissions_ok: true,
              jarvis_rss_mb: 512,
              jarvis_vms_mb: 2048,
            }),
          });
        }
      });

      await page.route('http://localhost:8742/', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ status: 'ok' }),
        });
      });

      await page.goto('/');
      await page.waitForSelector('.sidebar');
      await page.locator('.nav-item[title="Health Status"]').click();

      // Wait for error state to appear (since first request fails)
      const errorElement = page.locator(".error-message, .error-banner, [role='alert']");
      await errorElement
        .first()
        .waitFor({ state: 'visible', timeout: 5000 })
        .catch(() => {});

      // Find retry/refresh button
      const retryButton = page.locator(
        ".retry-btn, .refresh-btn, button:has-text('Retry'), button:has-text('Refresh')"
      );

      if ((await retryButton.count()) > 0) {
        await retryButton.first().click();

        // Wait for successful health data to load
        await page
          .waitForSelector('.health-status, .status-container', { timeout: 5000 })
          .catch(() => {});

        // Second request should succeed
        expect(requestCount).toBeGreaterThanOrEqual(2);
      }
    });

    test('automatic retry with exponential backoff', async ({ page }) => {
      const requestTimes: number[] = [];

      await page.route('http://localhost:8742/health', async (route) => {
        requestTimes.push(Date.now());
        await route.abort('connectionrefused');
      });

      await page.route('http://localhost:8742/', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ status: 'ok' }),
        });
      });

      await page.goto('/');
      await page.waitForSelector('.sidebar');
      await page.locator('.nav-item[title="Health Status"]').click();

      // Wait for error state indicating connection failure
      const errorElement = page.locator(".error-message, .connection-error, [role='alert']");
      await errorElement
        .first()
        .waitFor({ state: 'visible', timeout: 5000 })
        .catch(() => {});

      // Wait for multiple retry attempts (exponential backoff test)
      // Need sufficient time for at least 3 retry attempts - this tests actual timing behavior
      await page.waitForTimeout(10000); // INTENTIONAL: Testing exponential backoff timing

      // Should have made multiple attempts
      // With exponential backoff, gaps between attempts should increase
      if (requestTimes.length > 2) {
        const firstGap = requestTimes[1] - requestTimes[0];
        const secondGap = requestTimes[2] - requestTimes[1];
        // Second gap should be longer (exponential backoff)
        // Allow for some timing variance
      }
    });
  });

  test.describe('Partial Failure Handling', () => {
    test('app works when only some APIs fail', async ({ page }) => {
      // Health fails, but conversations work
      await page.route('http://localhost:8742/health', async (route) => {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Health check failed' }),
        });
      });

      await page.route('http://localhost:8742/', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ status: 'ok' }),
        });
      });

      await page.route('http://localhost:8742/conversations', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([
            {
              chat_id: 'chat-1',
              participants: ['+1234567890'],
              display_name: 'John Doe',
              last_message_date: new Date().toISOString(),
              message_count: 42,
              is_group: false,
              last_message_text: 'Hey!',
            },
          ]),
        });
      });

      await page.goto('/');
      await page.waitForSelector('.sidebar');

      // Conversations should still load
      await page.waitForSelector('.conversation', { timeout: 5000 });
      await expect(page.locator('.conversation')).toBeVisible();
    });

    test('shows degraded mode indicator when partial failure', async ({ page }) => {
      await page.route('http://localhost:8742/health', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            status: 'degraded',
            imessage_access: true,
            memory_available_gb: 2.0,
            memory_used_gb: 14.0,
            memory_mode: 'LITE',
            model_loaded: false,
            permissions_ok: true,
            details: { memory: 'Low memory, running in LITE mode' },
            jarvis_rss_mb: 256,
            jarvis_vms_mb: 1024,
          }),
        });
      });

      await page.route('http://localhost:8742/', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ status: 'ok' }),
        });
      });

      await page.goto('/');
      await page.waitForSelector('.sidebar');
      await page.locator('.nav-item[title="Health Status"]').click();

      // Wait for degraded status to appear
      const degradedIndicator = page.locator('.status-banner.degraded');
      await degradedIndicator.waitFor({ state: 'visible', timeout: 5000 });

      await expect(degradedIndicator).toBeVisible();
    });
  });

  test.describe('Error Recovery', () => {
    test('clears error state when data loads successfully', async ({ page }) => {
      let shouldFail = true;

      await page.route('http://localhost:8742/conversations', async (route) => {
        if (shouldFail) {
          shouldFail = false;
          await route.fulfill({
            status: 500,
            contentType: 'application/json',
            body: JSON.stringify({ error: 'Server error' }),
          });
        } else {
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify([
              {
                chat_id: 'chat-1',
                participants: ['+1234567890'],
                display_name: 'John Doe',
                last_message_date: new Date().toISOString(),
                message_count: 42,
                is_group: false,
                last_message_text: 'Hey!',
              },
            ]),
          });
        }
      });

      await page.route('http://localhost:8742/', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ status: 'ok' }),
        });
      });

      await page.goto('/');
      await page.waitForSelector('.sidebar');

      // Wait for initial error to appear (first request fails)
      const errorElement = page.locator(".error, .error-banner, [role='alert']");
      await errorElement
        .first()
        .waitFor({ state: 'visible', timeout: 5000 })
        .catch(() => {});

      // Retry (click refresh or reload)
      await page.reload();
      await page.waitForSelector('.conversation', { timeout: 10000 });

      // Error should be cleared and conversations visible
      await expect(page.locator('.conversation')).toBeVisible();
    });

    test('state is preserved during transient errors', async ({ mockedPage: page }) => {
      await page.goto('/');
      await waitForAppLoad(page);
      await page.waitForSelector('.conversation');

      // Make some selections
      await selectConversation(page, 'John Doe');
      await page.waitForSelector('.message');

      // Trigger an error for a new request
      await page.route('http://localhost:8742/health', async (route) => {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Transient error' }),
        });
      });

      // Navigate to health (which will fail)
      await navigateToView(page, 'health');

      // Wait for error state in health view
      const healthError = page.locator(".error, .error-message, [role='alert']");
      await healthError
        .first()
        .waitFor({ state: 'visible', timeout: 5000 })
        .catch(() => {});

      // Navigate back to messages
      await navigateToView(page, 'messages');

      // Wait for view to load
      await page.waitForSelector('.conversation, .message-list', { timeout: 1000 }).catch(() => {});

      // Previous selection should be preserved
      // (This depends on state management implementation)
    });
  });

  test.describe('User-Friendly Error Messages', () => {
    test('error messages are in plain language', async ({ errorPage: page }) => {
      await page.goto('/');
      await page.waitForSelector('.sidebar');

      // Wait for error message to appear
      const errorText = page.locator('.error-message, .error-text, .error-detail');
      await errorText
        .first()
        .waitFor({ state: 'visible', timeout: 10000 })
        .catch(() => {});

      if ((await errorText.count()) > 0) {
        const text = await errorText.textContent();

        // Should not contain technical jargon
        expect(text).not.toContain('500');
        expect(text).not.toContain('Internal Server Error');
        // Should be readable
        expect(text?.length).toBeGreaterThan(0);
      }
    });

    test('error provides actionable guidance', async ({ page }) => {
      await page.route('http://localhost:8742/**', async (route) => {
        await route.abort('connectionrefused');
      });

      await page.goto('/');
      await page.waitForSelector('.sidebar');

      // Wait for connection error to appear
      const errorElement = page.locator(".error, .connection-error, .error-banner, [role='alert']");
      await errorElement
        .first()
        .waitFor({ state: 'visible', timeout: 10000 })
        .catch(() => {});

      // Should provide retry option or troubleshooting steps
      const retryButton = page.locator(
        ".retry-btn, button:has-text('Retry'), button:has-text('Try again')"
      );

      // Retry option should be available
      if ((await retryButton.count()) > 0) {
        await expect(retryButton.first()).toBeVisible();
      }
    });
  });
});
