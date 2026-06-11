//-------------------------------------------------------------------------------------
// WGSL compile check (replaces the old node wgsl-compile.mjs script): every shader
// must create a module with no validation errors or compilation error messages.
//-------------------------------------------------------------------------------------

import { beforeAll, expect, test } from "vitest";
import { getDevice } from "../../src/gpu/device.js";
import bc6hWgsl from "../../src/gpu/bc6h.wgsl?raw";
import bc7Wgsl from "../../src/gpu/bc7.wgsl?raw";
import bcnWgsl from "../../src/gpu/bcn.wgsl?raw";
import resizeWgsl from "../../src/gpu/resize.wgsl?raw";

const shaders = [
  { name: "bc6h.wgsl", code: bc6hWgsl },
  { name: "bc7.wgsl", code: bc7Wgsl },
  { name: "bcn.wgsl", code: bcnWgsl },
  { name: "resize.wgsl", code: resizeWgsl },
];

let device: GPUDevice;
beforeAll(async () => {
  device = await getDevice();
});

for (const { name, code } of shaders) {
  test(`${name} compiles clean`, async () => {
    device.pushErrorScope("validation");
    const mod = device.createShaderModule({ code });
    const info = await mod.getCompilationInfo();
    const err = await device.popErrorScope();

    const errors = info.messages
      .filter((m) => m.type === "error")
      .map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
    expect(errors).toEqual([]);
    expect(err?.message).toBeUndefined();
  });
}
