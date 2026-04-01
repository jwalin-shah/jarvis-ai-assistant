import { test, expect } from '@playwright/test';

test('verify send button has aria-label', async ({ page, context }) => {
    // Add a mock for window.__TAURI__ to bypass DB checks in browser
    await context.addInitScript(`
        window.__TAURI__ = {
            core: {
                invoke: async (cmd, args) => {
                    console.log('Mock Tauri Invoke:', cmd, args);
                    if (cmd === 'plugin:sql|load') return 'sqlite:mock.db';
                    if (cmd === 'plugin:sql|execute') return { rowsAffected: 1, lastInsertId: 1 };
                    if (cmd === 'plugin:sql|select') return [];
                    return null;
                }
            },
            event: {
                listen: async () => () => {},
                emit: async () => {}
            }
        };
    `);

    console.log("Navigating to http://localhost:1420/");
    await page.goto("http://localhost:1420/");

    // Wait a little for rendering
    await page.waitForTimeout(2000);

    // Click the Chat button in sidebar to make sure we are there
    try {
        const chatBtn = page.getByRole("button", { name: "Chat" });
        await chatBtn.click();
        await page.waitForTimeout(1000);
    } catch (e) {
        console.log("Could not click Chat button in sidebar:", e);
    }

    // Check if the send button has the aria-label
    const sendButton = page.locator("button[aria-label='Send message']");

    // Take a screenshot
    await page.screenshot({ path: "/home/jules/verification/verification.png" });

    // Assert the button is present in the DOM (even if disabled)
    await expect(sendButton).toHaveCount(1);
    console.log("Success: Found send button with correct aria-label.");
});
