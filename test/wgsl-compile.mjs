// Quick WGSL compile check using the headless Dawn device.
// Usage: node test/wgsl-compile.mjs src/gpu/bc7.wgsl
import { readFileSync } from "node:fs";
import { create } from "webgpu";

const path = process.argv[2] ?? "src/gpu/bc7.wgsl";
const code = readFileSync(path, "utf8");

const gpu = create([]);
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice();

device.pushErrorScope("validation");
const mod = device.createShaderModule({ code });
const info = await mod.getCompilationInfo();
const err = await device.popErrorScope();

let hadError = false;
for (const m of info.messages) {
  if (m.type === "error") hadError = true;
  const loc = m.lineNum ? ` (line ${m.lineNum}:${m.linePos})` : "";
  console.log(`[${m.type}]${loc} ${m.message}`);
}
if (err) {
  hadError = true;
  console.log(`[device error] ${err.message}`);
}
console.log(hadError ? "\nFAILED" : "\nOK: compiled clean");
process.exit(hadError ? 1 : 0);
