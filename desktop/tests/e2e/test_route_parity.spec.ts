import { test, expect, waitForAppLoad } from "./fixtures";

test.describe("Route Parity", () => {
  test("sidebar navigation renders the same core views", async ({ mockedPage: page }) => {
    await page.goto("/");
    await waitForAppLoad(page);

    const checks: Array<{
      title: string;
      visibleSelector: string;
    }> = [
      { title: "Dashboard", visibleSelector: ".dashboard" },
      { title: "Messages", visibleSelector: ".conversation-list" },
      { title: "Template Builder", visibleSelector: ".template-builder" },
      { title: "Network", visibleSelector: ".network-container" },
      { title: "Health Status", visibleSelector: ".health-status" },
      { title: "Settings", visibleSelector: ".settings" },
    ];

    for (const check of checks) {
      await page.locator(`.nav-item[title="${check.title}"]`).click();
      await expect(page.locator(`.nav-item[title="${check.title}"]`)).toHaveClass(/active/);
      await expect(page.locator(check.visibleSelector)).toBeVisible();
    }
  });
});
