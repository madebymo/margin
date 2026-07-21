import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "@playwright/test";

const frontendDir = path.dirname(fileURLToPath(import.meta.url));
const backendDir = path.resolve(frontendDir, "../backend");
const baseURL = "http://127.0.0.1:8766";
const projectPython = path.resolve(
  frontendDir,
  process.platform === "win32"
    ? "../.venv/Scripts/python.exe"
    : "../.venv/bin/python",
);
const pythonExecutable =
  process.env.PYTHON ||
  (existsSync(projectPython)
    ? projectPython
    : process.platform === "win32"
      ? "python"
      : "python3");

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  timeout: 60_000,
  reporter: process.env.CI
    ? [["github"], ["list"], ["html", { open: "never" }]]
    : [["list"], ["html", { open: "never" }]],
  expect: {
    timeout: 10_000,
  },
  projects: [
    { name: "chromium", use: { browserName: "chromium" } },
    { name: "firefox", use: { browserName: "firefox" } },
    { name: "webkit", use: { browserName: "webkit" } },
  ],
  use: {
    baseURL,
    reducedMotion: "reduce",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    video: "retain-on-failure",
    viewport: { width: 1440, height: 900 },
  },
  webServer: {
    command: [
      JSON.stringify(pythonExecutable),
      "-m uvicorn tests.browser_v2_app:app --host 127.0.0.1 --port 8766",
    ].join(" "),
    cwd: backendDir,
    env: {
      ...process.env,
      TUTOR_E2E_TEST_APP: "1",
    },
    url: `${baseURL}/api/v2/goals`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
