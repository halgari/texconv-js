//-------------------------------------------------------------------------------------
// GPU test entry. Acquires the Dawn device once and runs the requested suite(s).
//
// The node-webgpu (Dawn) addon gets unstable after a cumulative number of ops in a
// single process (random std::system_error / pthread aborts), so running every
// suite back-to-back here crashes partway. The runner (run-all.mjs) therefore
// invokes this with ONE suite name per process and runs those processes in
// parallel — keeping each process's op count low while hiding the ~2s device init.
//
//   vite-node all.run.ts          -> all suites (fast but may hit the Dawn crash)
//   vite-node all.run.ts bcn      -> just the bcn suite (what the runner does)
//-------------------------------------------------------------------------------------

import { getDevice } from "../../src/gpu/device.js";
import { section, report } from "./harness.js";
import { run as bc7 } from "./bc7.suite.js";
import { run as bcn } from "./bcn.suite.js";
import { run as pipeline } from "./pipeline.suite.js";
import { run as large } from "./large.suite.js";

const ALL: Record<string, (d: GPUDevice) => Promise<void>> = { bc7, bcn, pipeline, large };

const wanted = process.argv[2];
const names = wanted ? [wanted] : Object.keys(ALL);

const device = await getDevice();
for (const name of names) {
  const fn = ALL[name];
  if (!fn) throw new Error(`unknown suite '${name}' (have: ${Object.keys(ALL).join(", ")})`);
  section(name);
  await fn(device);
}
report();
