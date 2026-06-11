// Runs each GPU suite in its OWN vite-node process, SEQUENTIALLY. Per-process
// isolation keeps the flaky node-webgpu (Dawn) addon under its cumulative crash
// threshold; sequential avoids the GPU/device contention that parallel Dawn
// instances cause. Uses the local vite-node binary directly (no npx resolution)
// and retries a crashed suite once for the occasional Dawn flake.
//
// The actual GPU compress is ~tens of ms; the per-suite cost is dominated by
// vite-node startup + the one-time ~2s Dawn device init in each process.
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const viteNode = resolve(here, "../../node_modules/.bin/vite-node");
const entry = join(here, "all.run.ts");
const SUITES = ["bc7", "bcn", "pipeline", "large"];

let failures = 0;
for (const name of SUITES) {
  let res = spawnSync(viteNode, [entry, name], { stdio: "inherit", timeout: 60_000 });
  if (res.status !== 0) {
    console.log(`  (retrying ${name} after Dawn flake)`);
    res = spawnSync(viteNode, [entry, name], { stdio: "inherit", timeout: 60_000 });
  }
  if (res.status !== 0) failures++;
}

console.log(`\n${SUITES.length} suite(s), ${failures} failed`);
process.exit(failures === 0 ? 0 : 1);
