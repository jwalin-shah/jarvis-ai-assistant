import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { resolve } from "path";

// Check if running in Tauri context (TAURI_ENV_PLATFORM is set during tauri dev/build)
const isTauri = !!process.env.TAURI_ENV_PLATFORM;

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [svelte()],
  // Prevent vite from obscuring rust errors
  clearScreen: false,
  // Tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    watch: {
      // Tell vite to ignore watching `src-tauri`
      ignored: ["**/src-tauri/**"],
    },
  },
  resolve: {
    alias: isTauri
      ? {
          "@": resolve(__dirname, "./src"),
        }
      : {
          "@": resolve(__dirname, "./src"),
          // Stub Tauri APIs when not running in Tauri context (for E2E tests)
          "@tauri-apps/api/event": resolve(__dirname, "tests/mocks/tauri-event.ts"),
          "@tauri-apps/api/core": resolve(__dirname, "tests/mocks/tauri-core.ts"),
          "@tauri-apps/plugin-sql": resolve(__dirname, "tests/mocks/tauri-sql.ts"),
        },
  },
  build: {
    target: "esnext",
    minify: "esbuild",
    esbuild: {
      drop: process.env.NODE_ENV === "production" ? ["console", "debugger"] : [],
    },
  },
});
