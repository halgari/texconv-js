//-------------------------------------------------------------------------------------
// gpu/resize.ts
//
// GPU box-filter resize and mip-chain generation (resize.wgsl). Operates on
// R8G8B8A8 images; mip generation halves each level with sRGB-correct averaging.
//-------------------------------------------------------------------------------------

import { DXGI_FORMAT, isSRGB } from "../dxgi.js";
import { makeImage, countMips, type Image } from "../image.js";
import { ShaderStage, BufferUsage, TextureUsage, MapMode } from "./device.js";
import resizeWgsl from "./resize.wgsl?raw";

interface Cached {
  layout: GPUBindGroupLayout;
  pipeline: GPUComputePipeline;
}
const cache = new WeakMap<GPUDevice, Cached>();

function getCached(device: GPUDevice): Cached {
  let c = cache.get(device);
  if (c) return c;
  const module = device.createShaderModule({ code: resizeWgsl, label: "resize" });
  const layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: ShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: ShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
      { binding: 2, visibility: ShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint: "resize" },
  });
  c = { layout, pipeline };
  cache.set(device, c);
  return c;
}

/** Box-filter resize an R8G8B8A8 image to (dstW, dstH) on the GPU. */
export async function resizeImage(
  device: GPUDevice,
  src: Image,
  dstW: number,
  dstH: number,
  srgb = isSRGB(src.format),
): Promise<Image> {
  if (src.format !== DXGI_FORMAT.R8G8B8A8_UNORM && src.format !== DXGI_FORMAT.R8G8B8A8_UNORM_SRGB) {
    throw new Error("resizeImage: source must be R8G8B8A8_UNORM(_SRGB)");
  }
  const { layout, pipeline } = getCached(device);

  const tex = device.createTexture({
    size: { width: src.width, height: src.height },
    format: "rgba8unorm",
    usage: TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture: tex },
    src.pixels as BufferSource,
    { bytesPerRow: src.rowPitch, rowsPerImage: src.height },
    { width: src.width, height: src.height },
  );

  const outBytes = dstW * dstH * 4;
  const outBuf = device.createBuffer({ size: outBytes, usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC });

  const uniform = device.createBuffer({ size: 32, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
  const ud = new ArrayBuffer(32);
  const uv = new DataView(ud);
  uv.setUint32(0, src.width, true);
  uv.setUint32(4, src.height, true);
  uv.setUint32(8, dstW, true);
  uv.setUint32(12, dstH, true);
  uv.setUint32(16, srgb ? 1 : 0, true);
  device.queue.writeBuffer(uniform, 0, ud);

  const bindGroup = device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: tex.createView() },
      { binding: 2, resource: { buffer: outBuf } },
    ],
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(dstW / 8), Math.ceil(dstH / 8));
  pass.end();
  device.queue.submit([enc.finish()]);

  const staging = device.createBuffer({ size: outBytes, usage: BufferUsage.COPY_DST | BufferUsage.MAP_READ });
  const enc2 = device.createCommandEncoder();
  enc2.copyBufferToBuffer(outBuf, 0, staging, 0, outBytes);
  device.queue.submit([enc2.finish()]);

  await staging.mapAsync(MapMode.READ);
  const mapped = new Uint8Array(staging.getMappedRange()).slice();
  staging.unmap();

  const dst = makeImage(src.format, dstW, dstH);
  // Output buffer is tightly packed (dstW*4 per row) == image rowPitch.
  dst.pixels.set(mapped.subarray(0, dst.pixels.length));

  outBuf.destroy();
  uniform.destroy();
  staging.destroy();
  tex.destroy();
  return dst;
}

/**
 * Generate a mip chain (base + downsampled levels) by repeated halving. `levels`
 * of 0 means a full chain down to 1x1. Returns images from base to smallest.
 */
export async function generateMipChain(
  device: GPUDevice,
  base: Image,
  levels = 0,
  srgb = isSRGB(base.format),
): Promise<Image[]> {
  const full = countMips(base.width, base.height);
  const count = levels <= 0 ? full : Math.min(levels, full);
  const chain: Image[] = [base];
  let prev = base;
  for (let i = 1; i < count; i++) {
    const w = Math.max(1, prev.width >> 1);
    const h = Math.max(1, prev.height >> 1);
    prev = await resizeImage(device, prev, w, h, srgb);
    chain.push(prev);
  }
  return chain;
}
