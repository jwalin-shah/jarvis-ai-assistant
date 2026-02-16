# JARVIS Desktop E2E Tests

Comprehensive End-to-End testing suite for the JARVIS Desktop application using Playwright.

## Overview

This test suite covers:
- **Core User Flows**: App launch, navigation, conversation management, messaging
- **Component Integration**: SmartReplyChipsV2, GlobalSearch, ConversationList
- **Performance Testing**: Render times, scroll performance, memory usage
- **Accessibility Testing**: ARIA labels, keyboard navigation, screen reader compatibility
- **Error Handling**: API errors, network disconnection, recovery flows

## Quick Start

```bash
# Install dependencies
pnpm install

# Install Playwright browsers
pnpm exec playwright install

# Run all E2E tests
pnpm test:e2e

# Run tests in UI mode (interactive)
pnpm test:e2e:ui

# Run tests with visible browser
pnpm test:e2e:headed
```

## Test Commands

| Command | Description |
|---------|-------------|
| `pnpm test:e2e` | Run all E2E tests |
| `pnpm test:e2e:ui` | Run tests in Playwright UI mode |
| `pnpm test:e2e:headed` | Run tests with visible browser |
| `pnpm test:e2e:debug` | Run tests in debug mode |
| `pnpm test:e2e:chromium` | Run tests only on Chromium |
| `pnpm test:e2e:webkit` | Run tests only on WebKit (Safari) |
| `pnpm test:e2e:a11y` | Run accessibility tests |
| `pnpm test:e2e:performance` | Run performance tests |
| `pnpm test:e2e:mobile` | Run tests with mobile viewport |
| `pnpm test:e2e:fast` | Run tests with parallel workers |
| `pnpm test:e2e:ci` | Run tests for CI environment |
| `pnpm test:e2e:report` | View HTML test report |

## Test Structure

```
tests/
├── e2e/
│   ├── fixtures.ts                    # Test fixtures and helpers
│   ├── test_app_launch.spec.ts        # App startup tests
│   ├── test_conversation_list.spec.ts # Conversation list tests
│   ├── test_message_view.spec.ts      # Message display tests
│   ├── test_search.spec.ts            # Search functionality tests
│   ├── test_global_search.spec.ts     # Global search modal tests
│   ├── test_ai_draft.spec.ts          # AI draft panel tests
│   ├── test_settings.spec.ts          # Settings page tests
│   ├── test_health_status.spec.ts     # Health monitoring tests
│   ├── test_smart_reply_chips.spec.ts # SmartReplyChipsV2 tests
│   ├── test_keyboard_navigation.spec.ts # Keyboard navigation tests
│   ├── test_accessibility.spec.ts     # A11y tests
│   ├── test_performance.spec.ts       # Performance tests
│   ├── test_error_recovery.spec.ts    # Error handling tests
│   └── README.md                      # This file
├── mocks/
│   ├── index.ts                       # Mock exports
│   ├── api-data.ts                    # Mock data
│   └── api-handlers.ts                # API route handlers
└── fixtures/
    └── (test data files)
```

## Test Categories

### Core User Flow Tests

Tests that verify the main user journeys:
- App opens without errors
- Navigation between views works
- Conversations load and display correctly
- Messages display with proper formatting
- Search filters and returns results
- Settings can be viewed and modified

### Component Integration Tests

Tests for specific components:

**SmartReplyChipsV2** (`test_smart_reply_chips.spec.ts`)
- Chip display and loading states
- Keyboard shortcuts (1, 2, 3)
- Click to copy functionality
- Refresh/regenerate
- Error handling

**GlobalSearch** (`test_global_search.spec.ts`)
- Opening/closing behavior
- Text vs semantic search modes
- Filter functionality
- Result navigation
- Keyboard shortcuts

### Performance Tests

Performance requirements (`test_performance.spec.ts`):
- Initial render < 2s
- View transitions < 500ms
- Scroll remains smooth with 1000+ items
- Memory stays stable during navigation
- DOM node count stays reasonable

### Accessibility Tests

A11y compliance (`test_accessibility.spec.ts`):
- ARIA labels on interactive elements
- Keyboard-only navigation
- Focus management
- Screen reader compatibility
- Color contrast verification

### Error Recovery Tests

Error handling (`test_error_recovery.spec.ts`):
- API error display
- Network disconnection handling
- Retry functionality
- Graceful degradation
- User-friendly error messages

## Fixtures

The test suite provides custom fixtures:

```typescript
// mockedPage - Page with all API endpoints mocked
test("example", async ({ mockedPage: page }) => {
  await page.goto("/");
  // All API calls are intercepted
});

// errorPage - Page with error responses
test("error handling", async ({ errorPage: page }) => {
  // API calls return errors
});

// disconnectedPage - Simulates offline state
test("offline mode", async ({ disconnectedPage: page }) => {
  // Network is unavailable
});

// socketMockedPage - Includes socket/WebSocket mocks
test("real-time features", async ({ socketMockedPage: page }) => {
  // Socket endpoints are mocked
});

// slowPage - Simulates slow network
test("loading states", async ({ slowPage: page }) => {
  // API responses are delayed
});

// largePage - Large dataset mocks
test("performance with large data", async ({ largePage: page }) => {
  // 1000+ conversations/messages
});
```

## Helper Functions

Available helper functions in `fixtures.ts`:

```typescript
// Navigation
await waitForAppLoad(page);
await navigateToView(page, "messages");
await selectConversation(page, "John Doe");

// Global Search
await openGlobalSearch(page);
await closeGlobalSearch(page);
await searchAndWait(page, "lunch");

// Performance
const renderTime = await measureRenderTime(page, action, selector);
const metrics = await collectPerformanceMetrics(page);

// Accessibility
const a11y = await checkAccessibility(page, selector);
const count = await testKeyboardNavigation(page, container, item, key);

// Network
await simulateOffline(page);
await simulateOnline(page);
await waitForNetworkIdle(page);

// Scroll
await scrollToBottom(page, selector);
await scrollToTop(page, selector);
const position = await getScrollPosition(page, selector);
```

## Mock Data

Mock data is defined in `tests/mocks/api-data.ts`:

- `mockConversations` - Sample conversation list
- `mockMessagesChat1`, `mockMessagesChat3` - Message data
- `mockHealthResponse` - Health check data
- `mockSettingsResponse` - Settings data
- `mockSmartReplies` - Smart reply suggestions
- `mockSearchResults` - Search results
- `generateLargeConversationList(count)` - Generate large datasets
- `generateLargeMessageList(count)` - Generate large message lists

## Running in CI

For CI environments:

```bash
# Run with CI-specific settings
CI=true pnpm test:e2e

# Or use the dedicated command
pnpm test:e2e:ci
```

CI configuration in `playwright.config.ts`:
- Single worker (to avoid flakiness)
- 2 retries on failure
- JSON reporter for CI integration
- Screenshots/videos on failure only

## Writing New Tests

### Basic Test Structure

```typescript
import { test, expect } from "./fixtures";
import { waitForAppLoad, navigateToView } from "./fixtures";

test.describe("Feature Name", () => {
  test.beforeEach(async ({ mockedPage: page }) => {
    await page.goto("/");
    await waitForAppLoad(page);
  });

  test("should do something", async ({ mockedPage: page }) => {
    // Arrange
    await navigateToView(page, "messages");

    // Act
    await page.locator(".button").click();

    // Assert
    await expect(page.locator(".result")).toBeVisible();
  });
});
```

### Adding New Mock Endpoints

In `tests/mocks/api-handlers.ts`:

```typescript
export async function setupApiMocks(page: Page): Promise<void> {
  // Add your new endpoint
  await page.route(`${API_BASE}/your-endpoint`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(yourMockData),
    });
  });
}
```

## Troubleshooting

### Tests failing locally but passing in CI

- Check network conditions (slow network affects timeouts)
- Verify all browsers are installed: `pnpm exec playwright install`
- Try running with `--headed` to see what's happening

### Flaky tests

- Add explicit waits: `await page.waitForSelector()`
- Avoid `waitForTimeout` where possible
- Use `toBeVisible()` over `toHaveCount()` for elements

### Performance tests failing

- Performance tests may fail in debug mode or slow machines
- Run with `--project=performance` only
- Increase thresholds if running in a slow environment

### Mocks not working

- Ensure routes are set up before navigation
- Check API base URL matches (`http://localhost:8742`)
- Route patterns use wildcards correctly (`*` for any segment)

## Related Documentation

- [Playwright Documentation](https://playwright.dev/docs/intro)
- [Testing Best Practices](https://playwright.dev/docs/best-practices)
- [Accessibility Testing](https://playwright.dev/docs/accessibility-testing)
