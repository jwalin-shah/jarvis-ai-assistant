/**
 * Playwright config for TestDriver.ai Vision AI tests.
 *
 * Uses natural language prompts instead of CSS selectors.
 * Tests are more resilient to UI changes.
 *
 * Requires TD_API_KEY environment variable.
 */

import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  // Test directory - TestDriver tests only
  testDir: './tests/testdriver',

  // Run sequentially for vision AI (needs consistent screenshots)
  fullyParallel: false,
  workers: 1,

  // Longer timeouts for AI processing
  timeout: 120000,
  expect: {
    timeout: 30000,
  },

  // Fail on CI if test.only left in
  forbidOnly: !!process.env.CI,

  // Retry on failure
  retries: process.env.CI ? 2 : 1,

  // Reporter
  reporter: [['html', { outputFolder: 'test-results/testdriver-report' }], ['list']],

  // Shared settings
  use: {
    baseURL: 'http://localhost:1420',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    // Larger viewport for better AI vision
    viewport: { width: 1440, height: 900 },
  },

  // Single browser for consistency
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  // Start Vite dev server
  webServer: {
    command: 'pnpm dev',
    url: 'http://localhost:1420',
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
    env: {
      VITE_PLAYWRIGHT: 'true',
    },
  },

  // Output directory
  outputDir: 'test-results/testdriver-artifacts',
});
