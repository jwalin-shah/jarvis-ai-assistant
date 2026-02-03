import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright configuration for JARVIS Desktop E2E tests.
 *
 * Tests run against the Vite dev server (port 1420) with mocked API responses.
 * The FastAPI backend is NOT required - all API calls are intercepted and mocked.
 *
 * Test categories:
 * - Core user flows (app launch, navigation, conversations, messages)
 * - Component integration (SmartReplyChips, GlobalSearch, etc.)
 * - Performance tests (render times, scroll performance, memory)
 * - Accessibility tests (ARIA, keyboard navigation, screen reader)
 */
export default defineConfig({
  // Test directory
  testDir: "./tests/e2e",

  // Run tests in parallel for speed
  fullyParallel: true,

  // Fail the build on CI if you accidentally left test.only in the source code
  forbidOnly: !!process.env.CI,

  // Retry failed tests on CI
  retries: process.env.CI ? 2 : 0,

  // Limit parallel workers on CI
  workers: process.env.CI ? 1 : undefined,

  // Reporter configuration
  reporter: [
    ["html", { outputFolder: "test-results/html-report" }],
    ["list"],
    // JSON reporter for CI integration
    ...(process.env.CI ? [["json", { outputFile: "test-results/results.json" }] as const] : []),
  ],

  // Global setup for common test utilities
  globalSetup: undefined,
  globalTeardown: undefined,

  // Shared settings for all projects
  use: {
    // Base URL for navigation
    baseURL: "http://localhost:1420",

    // Collect trace on first retry
    trace: "on-first-retry",

    // Screenshot on failure
    screenshot: "only-on-failure",

    // Video on failure
    video: "retain-on-failure",

    // Enable accessibility testing attributes
    bypassCSP: true,

    // Viewport for consistent testing
    viewport: { width: 1280, height: 720 },

    // Locale for consistent date/time formatting
    locale: "en-US",

    // Timezone for consistent testing
    timezoneId: "America/Los_Angeles",
  },

  // Test timeout
  timeout: 30000,

  // Expect timeout for assertions
  expect: {
    timeout: 5000,
    // Custom matchers for accessibility
    toHaveScreenshot: {
      maxDiffPixels: 100,
    },
  },

  // Configure projects for different browsers and test categories
  projects: [
    // Default browser tests
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "webkit",
      use: { ...devices["Desktop Safari"] },
    },
    // Accessibility-focused project with extra settings
    {
      name: "a11y",
      use: {
        ...devices["Desktop Chrome"],
        // Reduced motion for accessibility testing
        reducedMotion: "reduce",
      },
      testMatch: /accessibility|a11y/,
    },
    // Performance testing project
    {
      name: "performance",
      use: {
        ...devices["Desktop Chrome"],
        // Enable performance metrics collection
        launchOptions: {
          args: ["--enable-precise-memory-info"],
        },
      },
      testMatch: /performance/,
    },
    // Mobile viewport testing
    {
      name: "mobile",
      use: {
        ...devices["iPhone 14"],
      },
      testMatch: /responsive|mobile/,
    },
  ],

  // Start the Vite dev server before running tests
  webServer: {
    command: "pnpm run dev",
    url: "http://localhost:1420",
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
    stdout: "pipe",
    stderr: "pipe",
  },

  // Output directory for test artifacts
  outputDir: "test-results/artifacts",
});
