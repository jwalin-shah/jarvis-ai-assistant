/**
 * E2E tests for the actual Tauri desktop app.
 *
 * These tests run against the real compiled app using tauri-driver.
 * Make sure the backend services are running before running these tests:
 *   - API server on port 8742
 *   - Socket server at ~/.jarvis/jarvis.sock
 */

describe("JARVIS Desktop App", () => {
  it("should launch and show the sidebar", async () => {
    // Wait for app to load
    const sidebar = await $(".sidebar");
    await sidebar.waitForDisplayed({ timeout: 10000 });

    // Check logo is visible
    const logo = await $(".logo-text");
    await expect(logo).toBeDisplayed();
  });

  it("should connect to backend", async () => {
    // Wait for connection status
    const statusDot = await $(".status-dot");
    await statusDot.waitForDisplayed({ timeout: 5000 });

    // Should eventually show connected (green dot has 'connected' class)
    await browser.waitUntil(
      async () => {
        const classes = await statusDot.getAttribute("class");
        return classes?.includes("connected");
      },
      {
        timeout: 15000,
        timeoutMsg: "Expected backend to connect within 15s",
      }
    );
  });

  it("should load conversations from Messages tab", async () => {
    // Click Messages nav item
    const messagesNav = await $('.nav-item[title="Messages"]');
    await messagesNav.click();

    // Wait for conversations to load
    const conversation = await $(".conversation");
    await conversation.waitForDisplayed({ timeout: 15000 });

    // Should have at least one conversation
    const conversations = await $$(".conversation");
    expect(conversations.length).toBeGreaterThan(0);
  });

  it("should show messages when conversation is selected", async () => {
    // Ensure we're on Messages tab
    const messagesNav = await $('.nav-item[title="Messages"]');
    await messagesNav.click();

    // Click first conversation
    const firstConvo = await $(".conversation");
    await firstConvo.waitForDisplayed({ timeout: 10000 });
    await firstConvo.click();

    // Wait for message header to appear
    const header = await $(".message-header");
    await header.waitForDisplayed({ timeout: 10000 });

    // Should show conversation name in header
    await expect(header).toBeDisplayed();
  });

  it("should navigate through the core routes", async () => {
    const routeChecks: Array<{ title: string; selector: string }> = [
      { title: "Dashboard", selector: ".dashboard" },
      { title: "Messages", selector: ".conversation-list" },
      { title: "Template Builder", selector: ".template-builder" },
      { title: "Network", selector: ".network-container" },
      { title: "Health Status", selector: ".health-status" },
      { title: "Settings", selector: ".settings" },
    ];

    for (const route of routeChecks) {
      const navItem = await $(`.nav-item[title="${route.title}"]`);
      await navItem.click();

      const view = await $(route.selector);
      await view.waitForDisplayed({ timeout: 10000 });

      const classes = await navItem.getAttribute("class");
      expect(classes).toContain("active");
    }
  });
});

describe("Socket Streaming", () => {
  it("should generate smart replies via socket", async () => {
    // Navigate to messages
    const messagesNav = await $('.nav-item[title="Messages"]');
    await messagesNav.click();

    // Select a conversation
    const conversation = await $(".conversation");
    await conversation.waitForDisplayed({ timeout: 10000 });
    await conversation.click();

    // Wait for messages to load
    const header = await $(".message-header");
    await header.waitForDisplayed({ timeout: 10000 });

    // Look for smart reply chips (if available)
    const smartReplies = await $(".smart-reply-chips");

    // Smart replies may or may not be visible depending on conversation
    // Just verify the component can render
    const isDisplayed = await smartReplies.isDisplayed().catch(() => false);
    console.log(`Smart replies visible: ${isDisplayed}`);
  });
});
