//-------------------------------------------------------------------------------------
// gpu/compressBCn.ts
//
// Orchestrator for the BC1/BC2/BC3/BC4/BC5 GPU encoders (bcn.wgsl). One thread per
// block, a single dispatch, then readback.
//-------------------------------------------------------------------------------------

import { DXGI_FORMAT, blockBytes, computePitch } from "../dxgi.js";
import { makeImage, type Image } from "../image.js";
import { ShaderStage, BufferUsage, TextureUsage, MapMode } from "./device.js";
import bcnWgsl from "./bcn.wgsl?raw";

const ENTRY: Partial<Record<DXGI_FORMAT, string>> = {
  [DXGI_FORMAT.BC1_UNORM]: "encodeBC1",
  [DXGI_FORMAT.BC1_UNORM_SRGB]: "encodeBC1",
  [DXGI_FORMAT.BC2_UNORM]: "encodeBC2",
  [DXGI_FORMAT.BC2_UNORM_SRGB]: "encodeBC2",
  [DXGI_FORMAT.BC3_UNORM]: "encodeBC3",
  [DXGI_FORMAT.BC3_UNORM_SRGB]: "encodeBC3",
  [DXGI_FORMAT.BC4_UNORM]: "encodeBC4Red",
  [DXGI_FORMAT.BC5_UNORM]: "encodeBC5",
};

interface Cached {
  layout: GPUBindGroupLayout;
  module: GPUShaderModule;
  pipelines: Map<string, GPUComputePipeline>;
  pipelineLayout: GPUPipelineLayout;
}
const cache = new WeakMap<GPUDevice, Cached>();

function getCached(device: GPUDevice): Cached {
  let c = cache.get(device);
  if (c) return c;
  const module = device.createShaderModule({ code: bcnWgsl, label: "bcn-encode" });
  const layout = device.createBindGroupLayout({
    label: "bcn-bgl",
    entries: [
      { binding: 0, visibility: ShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: ShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
      { binding: 2, visibility: ShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  c = { layout, module, pipelines: new Map(), pipelineLayout };
  cache.set(device, c);
  return c;
}

/** Encode an R8G8B8A8 image to a BC1/2/3/4/5 format on the GPU. */
export async function compressBCn(device: GPUDevice, src: Image, format: DXGI_FORMAT): Promise<Image> {
  const entryPoint = ENTRY[format];
  if (!entryPoint) throw new Error(`compressBCn: unsupported format ${DXGI_FORMAT[format]}`);
  if (src.format !== DXGI_FORMAT.R8G8B8A8_UNORM && src.format !== DXGI_FORMAT.R8G8B8A8_UNORM_SRGB) {
    throw new Error("compressBCn: source must be R8G8B8A8_UNORM(_SRGB)");
  }

  const { width, height } = src;
  const xblocks = Math.max(1, (width + 3) >> 2);
  const yblocks = Math.max(1, (height + 3) >> 2);
  const numBlocks = xblocks * yblocks;
  const bpb = blockBytes(format); // 8 (BC1/BC4) or 16
  const outBytes = numBlocks * bpb;

  const { layout, module, pipelines, pipelineLayout } = getCached(device);
  let pipeline = pipelines.get(entryPoint);
  if (!pipeline) {
    pipeline = device.createComputePipeline({ label: entryPoint, layout: pipelineLayout, compute: { module, entryPoint } });
    pipelines.set(entryPoint, pipeline);
  }

  const tex = device.createTexture({
    size: { width, height },
    format: "rgba8unorm",
    usage: TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture: tex },
    src.pixels as BufferSource,
    { bytesPerRow: src.rowPitch, rowsPerImage: height },
    { width, height },
  );

  const outBuf = device.createBuffer({ size: outBytes, usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC });

  const uniform = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
  const ud = new ArrayBuffer(16);
  const uv = new DataView(ud);
  uv.setUint32(0, width, true);
  uv.setUint32(4, height, true);
  uv.setUint32(8, xblocks, true);
  uv.setUint32(12, numBlocks, true);
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
  pass.dispatchWorkgroups(Math.ceil(numBlocks / 64));
  pass.end();
  device.queue.submit([enc.finish()]);

  const staging = device.createBuffer({ size: outBytes, usage: BufferUsage.COPY_DST | BufferUsage.MAP_READ });
  const enc2 = device.createCommandEncoder();
  enc2.copyBufferToBuffer(outBuf, 0, staging, 0, outBytes);
  device.queue.submit([enc2.finish()]);

  await staging.mapAsync(MapMode.READ);
  const mapped = new Uint8Array(staging.getMappedRange()).slice();
  staging.unmap();

  const dst = makeImage(format, width, height);
  void computePitch; // pitch == buffer pitch (xblocks*bpb); flat copy below
  dst.pixels.set(mapped.subarray(0, Math.min(mapped.length, dst.pixels.length)));

  outBuf.destroy();
  uniform.destroy();
  staging.destroy();
  tex.destroy();
  return dst;
}
