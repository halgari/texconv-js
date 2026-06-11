import { defineConfig } from "vite";

// Demo app config. WGSL shaders are imported as strings via Vite's `?raw` suffix,
// so no custom plugin is needed. `npm run dev` serves the demo at /demo.
export default defineConfig({
  root: "demo",
  publicDir: "../public",
  build: {
    outDir: "../dist-demo",
    emptyOutDir: true,
    target: "esnext",
  },
});
