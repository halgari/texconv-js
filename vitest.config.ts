import { defineConfig } from "vitest/config";

// Separate from vite.config.ts (which roots the dev server at demo/). Tests run
// from the project root over test/**.
export default defineConfig({
  test: {
    root: ".",
    include: ["test/**/*.test.ts"],
    environment: "node",
  },
});
