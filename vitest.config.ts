import { defineConfig } from "vitest/config";

// Separate from vite.config.ts (which roots the dev server at demo/). Tests run
// from the project root over test/**.
export default defineConfig({
  test: {
    root: ".",
    include: ["test/**/*.test.ts"],
    environment: "node",
    // The native Dawn (webgpu) addon is unstable in worker threads; use a single
    // forked child process with no isolation.
    pool: "forks",
    poolOptions: { forks: { singleFork: true, isolate: false } },
    testTimeout: 30000,
  },
});
