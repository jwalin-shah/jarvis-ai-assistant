/**
 * API route handlers for mocking the FastAPI backend in E2E tests.
 *
 * These handlers intercept HTTP requests to localhost:8742 and return
 * mock data, allowing tests to run without the actual backend.
 */

import { Page, Route } from "@playwright/test";
import {
  mockConversations,
  mockDraftReplyResponse,
  mockHealthResponse,
  mockMessagesChat1,
  mockMessagesChat3,
  mockModels,
  mockPingResponse,
  mockSettingsResponse,
  mockSmartReplies,
  mockSearchResults,
  mockSemanticSearchResults,
  mockSummaryResponse,
  mockTemplates,
  mockDigestResponse,
  mockPriorityInbox,
  generateLargeConversationList,
  generateLargeMessageList,
} from "./api-data";

const API_BASE = "http://localhost:8742";

/**
 * Setup all API mocks for a page.
 * Call this in beforeEach to ensure all API calls are intercepted.
 */
export async function setupApiMocks(page: Page): Promise<void> {
  // Health endpoint
  await page.route(`${API_BASE}/health`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockHealthResponse),
    });
  });

  // Ping endpoint
  await page.route(`${API_BASE}/`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockPingResponse),
    });
  });

  // Conversations list
  await page.route(`${API_BASE}/conversations`, async (route: Route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockConversations),
      });
    } else {
      await route.continue();
    }
  });

  // Individual conversation
  await page.route(
    `${API_BASE}/conversations/chat-*`,
    async (route: Route) => {
      const url = route.request().url();

      // Check if it's a messages request
      if (url.includes("/messages")) {
        const chatId = url.split("/conversations/")[1].split("/")[0];
        const messages =
          chatId === "chat-1"
            ? mockMessagesChat1
            : chatId === "chat-3"
              ? mockMessagesChat3
              : [];

        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(messages),
        });
      } else {
        // It's a conversation detail request
        const chatId = url.split("/conversations/")[1].split("?")[0];
        const conversation = mockConversations.find((c) => c.chat_id === chatId);

        if (conversation) {
          await route.fulfill({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify(conversation),
          });
        } else {
          await route.fulfill({
            status: 404,
            contentType: "application/json",
            body: JSON.stringify({ error: "Not found", detail: "Conversation not found" }),
          });
        }
      }
    }
  );

  // Settings endpoints
  await page.route(`${API_BASE}/settings`, async (route: Route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockSettingsResponse),
      });
    } else if (route.request().method() === "PUT") {
      // Return the updated settings (just echo back for tests)
      const body = route.request().postDataJSON();
      const updated = {
        ...mockSettingsResponse,
        ...body,
        generation: { ...mockSettingsResponse.generation, ...body?.generation },
        behavior: { ...mockSettingsResponse.behavior, ...body?.behavior },
      };
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(updated),
      });
    } else {
      await route.continue();
    }
  });

  // Models list
  await page.route(`${API_BASE}/settings/models`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockModels),
    });
  });

  // Model download
  await page.route(
    `${API_BASE}/settings/models/*/download`,
    async (route: Route) => {
      const url = route.request().url();
      const modelId = url.split("/models/")[1].split("/download")[0];

      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          model_id: modelId,
          status: "completed",
          progress: 100,
          error: null,
        }),
      });
    }
  );

  // Model activate
  await page.route(
    `${API_BASE}/settings/models/*/activate`,
    async (route: Route) => {
      const url = route.request().url();
      const modelId = url.split("/models/")[1].split("/activate")[0];

      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          success: true,
          model_id: modelId,
          error: null,
        }),
      });
    }
  );

  // Draft replies endpoint
  await page.route(
    `${API_BASE}/conversations/*/drafts`,
    async (route: Route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockDraftReplyResponse),
      });
    }
  );
}

/**
 * Setup API mocks that simulate errors.
 * Use this to test error handling in the UI.
 */
export async function setupApiMocksWithErrors(page: Page): Promise<void> {
  // Health endpoint returns error
  await page.route(`${API_BASE}/health`, async (route: Route) => {
    await route.fulfill({
      status: 500,
      contentType: "application/json",
      body: JSON.stringify({
        error: "Internal Server Error",
        detail: "Failed to check system health",
      }),
    });
  });

  // Ping endpoint returns error (simulates backend down)
  await page.route(`${API_BASE}/`, async (route: Route) => {
    await route.abort("connectionrefused");
  });

  // Conversations list returns error
  await page.route(`${API_BASE}/conversations`, async (route: Route) => {
    await route.fulfill({
      status: 500,
      contentType: "application/json",
      body: JSON.stringify({
        error: "Database Error",
        detail: "Failed to fetch conversations",
      }),
    });
  });
}

/**
 * Setup mocks for disconnected state.
 */
export async function setupDisconnectedMocks(page: Page): Promise<void> {
  // All requests fail with connection refused
  await page.route(`${API_BASE}/**`, async (route: Route) => {
    await route.abort("connectionrefused");
  });
}

/**
 * Setup WebSocket/socket mocks for real-time features.
 * This mocks the v2 API endpoints and smart reply features.
 */
export async function setupSocketMocks(page: Page): Promise<void> {
  // V2 API - Generate replies (smart replies)
  await page.route(`${API_BASE}/v2/conversations/*/replies`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockSmartReplies),
    });
  });

  // Suggestions endpoint
  await page.route(`${API_BASE}/suggestions`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockSmartReplies),
    });
  });

  // Summary endpoint
  await page.route(`${API_BASE}/drafts/summarize`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockSummaryResponse),
    });
  });

  // Search messages endpoint
  await page.route(`${API_BASE}/conversations/search`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockSearchResults),
    });
  });

  // Semantic search endpoint
  await page.route(`${API_BASE}/search/semantic`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockSemanticSearchResults),
    });
  });

  // Templates endpoints
  await page.route(`${API_BASE}/templates`, async (route: Route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockTemplates),
      });
    } else if (route.request().method() === "POST") {
      const body = route.request().postDataJSON();
      await route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify({ id: "template-new", ...body }),
      });
    } else {
      await route.continue();
    }
  });

  // Priority inbox endpoint
  await page.route(`${API_BASE}/priority`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockPriorityInbox),
    });
  });

  // Digest endpoints
  await page.route(`${API_BASE}/digest/*`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockDigestResponse),
    });
  });
}

/**
 * Setup API mocks with intentionally slow responses for testing loading states.
 */
export async function setupSlowApiMocks(page: Page, delayMs: number = 500): Promise<void> {
  // Health endpoint with delay
  await page.route(`${API_BASE}/health`, async (route: Route) => {
    await new Promise((resolve) => setTimeout(resolve, delayMs));
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockHealthResponse),
    });
  });

  // Ping endpoint with delay
  await page.route(`${API_BASE}/`, async (route: Route) => {
    await new Promise((resolve) => setTimeout(resolve, delayMs));
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockPingResponse),
    });
  });

  // Conversations list with delay
  await page.route(`${API_BASE}/conversations`, async (route: Route) => {
    if (route.request().method() === "GET") {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockConversations),
      });
    } else {
      await route.continue();
    }
  });

  // Messages with delay
  await page.route(`${API_BASE}/conversations/*/messages`, async (route: Route) => {
    await new Promise((resolve) => setTimeout(resolve, delayMs));
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockMessagesChat1),
    });
  });

  // Settings with delay
  await page.route(`${API_BASE}/settings`, async (route: Route) => {
    await new Promise((resolve) => setTimeout(resolve, delayMs));
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockSettingsResponse),
    });
  });

  // Models with delay
  await page.route(`${API_BASE}/settings/models`, async (route: Route) => {
    await new Promise((resolve) => setTimeout(resolve, delayMs));
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockModels),
    });
  });
}

/**
 * Setup API mocks with large datasets for performance testing.
 */
export async function setupLargeDataMocks(page: Page, count: number = 1000): Promise<void> {
  const largeConversations = generateLargeConversationList(count);

  // Health endpoint
  await page.route(`${API_BASE}/health`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockHealthResponse),
    });
  });

  // Ping endpoint
  await page.route(`${API_BASE}/`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockPingResponse),
    });
  });

  // Large conversations list
  await page.route(`${API_BASE}/conversations`, async (route: Route) => {
    if (route.request().method() === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(largeConversations),
      });
    } else {
      await route.continue();
    }
  });

  // Large message lists
  await page.route(`${API_BASE}/conversations/*/messages`, async (route: Route) => {
    const largeMessages = generateLargeMessageList(count);
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(largeMessages),
    });
  });

  // Settings
  await page.route(`${API_BASE}/settings`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockSettingsResponse),
    });
  });

  // Models
  await page.route(`${API_BASE}/settings/models`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockModels),
    });
  });
}

/**
 * Setup mocks that simulate intermittent failures (for retry testing).
 */
export async function setupIntermittentFailureMocks(
  page: Page,
  failureRate: number = 0.5
): Promise<void> {
  let requestCount = 0;

  await page.route(`${API_BASE}/**`, async (route: Route) => {
    requestCount++;
    const shouldFail = Math.random() < failureRate;

    if (shouldFail) {
      await route.fulfill({
        status: 500,
        contentType: "application/json",
        body: JSON.stringify({
          error: "Internal Server Error",
          detail: "Simulated intermittent failure",
        }),
      });
    } else {
      // Pass through to other handlers or fulfill with default
      await route.continue();
    }
  });
}
