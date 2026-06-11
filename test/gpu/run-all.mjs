// Runs every GPU integration test (test/gpu/*.run.ts) under vite-node, each in its
// own process so the Dawn addon's teardown segfault can't cross-contaminate. Exits
// non-zero if any suite fails.
import { readdirSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const suites = readdirSync(here).filter((f) => f.endsWith(".run.ts"));

let failures = 0;
for (const suite of suites) {
  console.log(`\n=== ${suite} ===`);
  const res = spawnSync("npx", ["vite-node", join(here, suite)], { stdio: "inherit" });
  if (res.status !== 0) failures++;
}

console.log(`\n${suites.length} suite(s), ${failures} failed`);
process.exit(failures === 0 ? 0 : 1);
