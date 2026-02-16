import { defineConfig } from "vitest/config";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { resolve } from "path";

export default defineConfig({
  plugins: [svelte({ hot: false })],
  resolve: {
    alias: {
      "@": resolve(__dirname, "./src"),
      // Stub Tauri APIs in unit tests
      "@tauri-apps/api/event": resolve(__dirname, "tests/mocks/tauri-event.ts"),
      "@tauri-apps/api/core": resolve(__dirname, "tests/mocks/tauri-core.ts"),
      "@tauri-apps/plugin-sql": resolve(__dirname, "tests/mocks/tauri-sql.ts"),
    },
  },
  test: {
    globals: true,
    environment: "node",
    include: ["tests/unit/**/*.test.ts"],
    coverage: {
      provider: "v8",
      include: ["src/lib/**/*.ts"],
      exclude: ["src/lib/**/*.svelte"],
    },
  },
});
