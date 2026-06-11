# Replace node-webgpu (Dawn addon) GPU test harness with vitest browser mode

**Date:** 2026-06-11
**Status:** Approved

## Problem

GPU integration tests run headlessly in Node via the `webgpu` npm package
(node-webgpu, Dawn-based native addon). The addon is unstable: it segfaults at
teardown, crashes after a cumulative number of operations, and misbehaves in
worker threads. The test harness has accumulated workarounds — per-suite
process isolation (`test/gpu/run-all.mjs`), sequential execution, one retry per
crashed suite, explicit `process.exit()` before Dawn teardown
(`test/gpu/harness.ts`), and a single-fork pool in `vitest.config.ts` — and
still flakes.

The library's production target is the browser (`navigator.gpu`); the Dawn
addon exists only for the test/dev path. Chromium's in-browser Dawn is stable —
the flakiness lives in the Node addon bindings, not Dawn itself.

## Decision

Run the GPU suites as ordinary vitest tests inside headless Chromium, using
vitest browser mode with the Playwright provider. Remove the `webgpu` addon and
all crash-management machinery. The library becomes browser-only at runtime
(approved: no Node fallback retained).

Alternatives considered:

- **Deno's built-in WebGPU (wgpu):** stable and addon-free, but adds a second
  runtime and the `?raw` WGSL imports would need a workaround. Rejected.
- **Upgrade/retune node-webgpu:** least churn but doesn't fix the root cause.
  Rejected.

## Design

### 1. Dependencies

- Remove: `webgpu`.
- Add: `@vitest/browser`, `playwright` (devDependencies).
- Upgrade: `vitest` 2.1 → latest 3.x (browser mode is stable there; on 2.1 it
  is experimental). Unit tests are expected to be unaffected.

### 2. Test layout

One vitest config with two projects:

- **`unit`** — Node environment, runs `test/*.test.ts` (`cli.test.ts`,
  `dds.test.ts`) exactly as today.
- **`gpu`** — browser mode, Playwright provider, headless Chromium launched
  with WebGPU flags (`--enable-unsafe-webgpu --enable-features=Vulkan
  --use-angle=vulkan` on Linux). Vite serves the modules, so the `?raw` WGSL
  imports work without changes.

### 3. Suite conversion

- `bc7.suite.ts`, `bcn.suite.ts`, `pipeline.suite.ts`, `large.suite.ts` →
  ordinary vitest `test()` files under `test/gpu/`, using `expect` instead of
  the custom harness assertions.
- Delete: `test/gpu/run-all.mjs`, `test/gpu/all.run.ts`, `test/gpu/harness.ts`,
  and the single-fork pool workaround in `vitest.config.ts`.
- `test/wgsl-compile.mjs` → a small gpu-project test that creates each shader
  module and asserts `getCompilationInfo()` reports no errors.

### 4. Runtime cleanup

`src/gpu/device.ts`:

- Drop the dynamic `import("webgpu")` Node fallback; require `navigator.gpu`.
- Keep the progressive adapter acquisition and the actionable no-adapter error
  message (Linux Chromium flags guidance).
- Keep the numeric bitflag constants (`ShaderStage`, `BufferUsage`,
  `TextureUsage`, `MapMode`) — spec-fixed values, harmless, and avoid relying
  on globals.

### 5. Scripts and docs

- `test` → unit project; `test:gpu` → gpu project; `test:all` → both via a
  single `vitest run`.
- README: replace the Dawn/vite-node GPU-test section and Dawn troubleshooting
  with vitest-browser-mode instructions and the Chromium WebGPU flags.

## Error handling

- If no WebGPU adapter is available in the test browser, the device acquisition
  error already explains the Linux Chromium flag situation; tests fail with
  that message rather than hanging.
- Headless Chromium WebGPU on Linux can need flag adjustments per driver. The
  escape hatch is running the gpu project headed (one-line config change),
  documented in the README.

## Testing / success criteria

- `npm run test:all` passes repeatedly (e.g. 10 consecutive runs) with no
  retries, no process isolation, and no segfaults.
- `npm run test` (unit) behavior unchanged after the vitest 3 upgrade.
- `npm run build` succeeds; no references to the `webgpu` package remain in
  source, config, or docs.

## Out of scope

- CI setup (no CI exists in this repo; software-rendering flags for CI are a
  possible follow-up).
- Any changes to the GPU compression/resize code itself.
