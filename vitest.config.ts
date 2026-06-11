import { defineConfig } from "vitest/config";

// Separate from vite.config.ts (which roots the dev server at demo/). Tests run
// from the project root over test/**.
//
// Two projects:
//   unit — CPU tests in plain Node (test/*.test.ts)
//   gpu  — WebGPU integration tests in headless Chromium via Playwright
//          (test/gpu/*.test.ts). Vite serves the modules, so WGSL `?raw`
//          imports work as in the browser build.
export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: "unit",
          root: ".",
          include: ["test/*.test.ts"],
          environment: "node",
          testTimeout: 30000,
        },
      },
      {
        test: {
          name: "gpu",
          root: ".",
          include: ["test/gpu/*.test.ts"],
          testTimeout: 60000,
          browser: {
            enabled: true,
            headless: true,
            provider: "playwright",
            screenshotFailures: false,
            instances: [
              {
                browser: "chromium",
                launch: {
                  // Full Chromium (not the headless shell) — the shell build can
                  // lack GPU support. Flags unlock WebGPU on Linux.
                  channel: "chromium",
                  args: [
                    "--enable-unsafe-webgpu",
                    "--enable-features=Vulkan",
                    "--use-angle=vulkan",
                    "--ignore-gpu-blocklist",
                  ],
                },
              },
            ],
          },
        },
      },
    ],
  },
});
