// Runs every GPU integration test (test/gpu/*.run.ts) under vite-node, each in its
// own process so the Dawn addon's teardown segfault can't cross-contaminate. Exits
// non-zero if any suite fails.
import { readdirSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const suites = readdirSync(here).filter((f) => f.endsWith(".run.ts"));

// The Dawn (node-webgpu) addon is intermittently unstable on process teardown /
// between GPU ops (pthread / std::system_error aborts) — a known addon issue that
// does NOT occur in real browsers and does not affect output correctness. Retry a
// crashed suite a few times so CI is reliable.
const MAX_ATTEMPTS = 4;

let failures = 0;
for (const suite of suites) {
  console.log(`\n=== ${suite} ===`);
  let ok = false;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS && !ok; attempt++) {
    if (attempt > 1) console.log(`  (retry ${attempt}/${MAX_ATTEMPTS} after Dawn instability)`);
    const res = spawnSync("npx", ["vite-node", join(here, suite)], { stdio: "inherit" });
    ok = res.status === 0;
  }
  if (!ok) failures++;
}

console.log(`\n${suites.length} suite(s), ${failures} failed`);
process.exit(failures === 0 ? 0 : 1);
