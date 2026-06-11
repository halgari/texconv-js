import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

// Library build (npm run build:lib). Bundles src/index.ts and its WGSL `?raw`
// imports into dist/texconv-js.js for consumption as a package. The browser
// demo is built separately by vite.config.ts; type declarations are emitted by
// `npm run build:types` (tsconfig.build.json). `emptyOutDir: false` so the lib
// bundle and the .d.ts files can coexist in dist/ regardless of build order.
export default defineConfig({
  build: {
    outDir: "dist",
    emptyOutDir: false,
    target: "esnext",
    sourcemap: true,
    lib: {
      entry: fileURLToPath(new URL("src/index.ts", import.meta.url)),
      formats: ["es"],
      fileName: () => "texconv-js.js",
    },
    rollupOptions: {
      // No runtime dependencies; everything (incl. inlined WGSL) is bundled.
      output: { inlineDynamicImports: true },
    },
  },
});
