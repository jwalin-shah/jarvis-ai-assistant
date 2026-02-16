/**
 * WebdriverIO config for testing the actual Tauri desktop app.
 *
 * Uses tauri-driver to wrap the native app and expose WebDriver protocol.
 *
 * Usage:
 *   1. Build the app: npm run tauri build -- --debug
 *   2. Run tests: npx wdio wdio.tauri.conf.ts
 */

import type { Options } from "@wdio/types";
import { spawn, ChildProcess } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Path to the built Tauri app (bundled location)
const APP_PATH = path.join(
  __dirname,
  "src-tauri/target/debug/bundle/macos/JARVIS.app/Contents/MacOS/JARVIS"
);

let tauriDriver: ChildProcess | null = null;

export const config: Options.Testrunner = {
  // Runner
  runner: "local",

  // Test files
  specs: ["./tests/tauri/**/*.spec.ts"],

  // Capabilities - connect to tauri-driver
  capabilities: [
    {
      "tauri:options": {
        application: APP_PATH,
      },
    } as any,
  ],

  // WebDriver config - tauri-driver runs on port 4444
  hostname: "localhost",
  port: 4444,

  // Framework
  framework: "mocha",
  mochaOpts: {
    ui: "bdd",
    timeout: 60000,
  },

  // Reporters
  reporters: ["spec"],

  // Log level
  logLevel: "info",

  // Wait for backend services
  waitforTimeout: 10000,

  // Connection retry
  connectionRetryTimeout: 120000,
  connectionRetryCount: 3,

  // Start tauri-driver before tests
  onPrepare: async function () {
    console.log("Starting tauri-driver...");
    tauriDriver = spawn("tauri-driver", [], {
      stdio: ["ignore", "pipe", "pipe"],
    });

    tauriDriver.stdout?.on("data", (data) => {
      console.log(`[tauri-driver] ${data}`);
    });

    tauriDriver.stderr?.on("data", (data) => {
      console.error(`[tauri-driver] ${data}`);
    });

    // Wait for driver to be ready
    await new Promise((resolve) => setTimeout(resolve, 2000));
    console.log("tauri-driver started");
  },

  // Stop tauri-driver after tests
  onComplete: async function () {
    if (tauriDriver) {
      console.log("Stopping tauri-driver...");
      tauriDriver.kill();
      tauriDriver = null;
    }
  },
};
