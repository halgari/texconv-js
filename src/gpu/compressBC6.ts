//-------------------------------------------------------------------------------------
// gpu/compressBC6.ts
//
// WebGPU port of DirectXTex's BCDirectCompute.cpp (GPUCompressBC) for BC6H.
// Runs TryModeG10CS once (mode 11..14 probe), then TryModeLE10CS for mode ids
// 0..9 with ping-pong error buffers (err1 -> err2 -> err1 -> ... -> err1),
// then EncodeBlockCS reading err1.
//
// Differences from the D3D11 original, all forced by WebGPU semantics (same as
// the BC7 port in compressBC.ts):
//   * The REF_DEVICE (barrier) shader path is used, so storage buffers are padded
//     to a multiple of 4 blocks; over-dispatched edge threads then write into
//     padding rather than out of bounds.
//   * Per-dispatch uniforms are uploaded with writeBuffer + one submit each.
//   * Blocks are processed in <= 65535-workgroup batches via g_start_block_id.
//-------------------------------------------------------------------------------------

import { DXGI_FORMAT, computePitch } from "../dxgi.js";
import { makeImage, type Image } from "../image.js";
import { ShaderStage, BufferUsage, TextureUsage, MapMode } from "./device.js";
import bc6hwgsl from "./bc6h.wgsl?raw";

export interface CompressBC6Options {
  /** Target format: BC6H_UF16 or BC6H_SF16. */
  format: DXGI_FORMAT;
}

const PASS = {
  tryModeG10: "TryModeG10CS",
  tryModeLE10: "TryModeLE10CS",
  encode: "EncodeBlockCS",
} as const;

type ModulePipelines = Record<keyof typeof PASS, GPUComputePipeline>;

const moduleCache = new WeakMap<GPUDevice, { layout: GPUBindGroupLayout; pipelines: ModulePipelines }>();

function getPipelines(device: GPUDevice) {
  let cached = moduleCache.get(device);
  if (cached) return cached;

  const module = device.createShaderModule({ code: bc6hwgsl, label: "bc6h-encode" });

  const layout = device.createBindGroupLayout({
    label: "bc6h-bgl",
    entries: [
      { binding: 0, visibility: ShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: ShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
      { binding: 2, visibility: ShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: ShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });

  const mk = (entryPoint: string) =>
    device.createComputePipeline({
      label: entryPoint,
      layout: pipelineLayout,
      compute: { module, entryPoint },
    });

  const pipelines: ModulePipelines = {
    tryModeG10: mk(PASS.tryModeG10),
    tryModeLE10: mk(PASS.tryModeLE10),
    encode: mk(PASS.encode),
  };

  cached = { layout, pipelines };
  moduleCache.set(device, cached);
  return cached;
}

/**
 * Encode an R32G32B32A32_FLOAT image to BC6H on the GPU. Returns the BC6H Image (mip 0).
 *
 * Note: like upstream (BC6HEncode.hlsl does max(pixel, 0) for both formats), negative
 * inputs are clamped to zero even for BC6H_SF16 — texconv -gpu behaves the same way.
 */
export async function compressBC6H(
  device: GPUDevice,
  src: Image,
  opts: CompressBC6Options,
): Promise<Image> {
  if (src.format !== DXGI_FORMAT.R32G32B32A32_FLOAT) {
    throw new Error("compressBC6H: source must be R32G32B32A32_FLOAT");
  }
  if (opts.format !== DXGI_FORMAT.BC6H_UF16 && opts.format !== DXGI_FORMAT.BC6H_SF16) {
    throw new Error("compressBC6H: target must be BC6H_UF16 or BC6H_SF16");
  }

  const { width, height } = src;
  const xblocks = Math.max(1, (width + 3) >> 2);
  const yblocks = Math.max(1, (height + 3) >> 2);
  const numBlocks = xblocks * yblocks;
  const paddedBlocks = Math.ceil(numBlocks / 4) * 4; // 4 blocks per group in the G10 pass
  const bufBytes = paddedBlocks * 16;

  const { layout, pipelines } = getPipelines(device);

  // Capture validation errors as a thrown exception rather than a silent async
  // failure (e.g. an over-cap dispatch would otherwise produce garbage quietly).
  device.pushErrorScope("validation");

  // --- Input texture (rgba32float; src.pixels is the raw float data) ---
  const tex = device.createTexture({
    size: { width, height },
    format: "rgba32float",
    usage: TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,
  });
  // rowPitch is 16 bytes/pixel for R32G32B32A32_FLOAT (tight rows are fine here).
  device.queue.writeTexture(
    { texture: tex },
    src.pixels as BufferSource,
    { bytesPerRow: src.rowPitch, rowsPerImage: height },
    { width, height },
  );
  const texView = tex.createView();

  // --- Storage buffers ---
  const mkStorage = (label: string) =>
    device.createBuffer({
      label,
      size: bufBytes,
      usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC,
    });
  const output = mkStorage("bc6h-output");
  const err1 = mkStorage("bc6h-err1");
  const err2 = mkStorage("bc6h-err2");

  const uniform = device.createBuffer({
    size: 32,
    usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
  });
  const uniformData = new ArrayBuffer(32);
  const uniformView = new DataView(uniformData);

  // g_format: 95 (UNSIGNED_F16) or 96 (SIGNED_F16) — the DXGI codes themselves.
  const formatCode = opts.format;

  const setUniform = (modeId: number, startBlock: number) => {
    uniformView.setUint32(0, width, true); // g_tex_width
    uniformView.setUint32(4, xblocks, true); // g_num_block_x
    uniformView.setUint32(8, formatCode, true); // g_format
    uniformView.setUint32(12, modeId, true); // g_mode_id
    uniformView.setUint32(16, startBlock, true); // g_start_block_id
    uniformView.setUint32(20, numBlocks, true); // g_num_total_blocks
    uniformView.setFloat32(24, 1.0, true); // g_alpha_weight (unused by BC6H; layout parity)
    uniformView.setUint32(28, 0, true); // reserved
    device.queue.writeBuffer(uniform, 0, uniformData);
  };

  const bindGroup = (inBuf: GPUBuffer, outBuf: GPUBuffer) =>
    device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: texView },
        { binding: 2, resource: { buffer: inBuf } },
        { binding: 3, resource: { buffer: outBuf } },
      ],
    });

  // One dispatch == set uniform, encode a single compute pass, submit. Separate
  // submits keep the per-dispatch uniform writes correctly ordered.
  const dispatch = (
    pipeline: GPUComputePipeline,
    modeId: number,
    inBuf: GPUBuffer,
    outBuf: GPUBuffer,
    groups: number,
    startBlock: number,
  ) => {
    setUniform(modeId, startBlock);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup(inBuf, outBuf));
    pass.dispatchWorkgroups(groups);
    pass.end();
    device.queue.submit([enc.finish()]);
  };

  // WebGPU caps workgroups at 65535 per dimension, so process blocks in batches
  // (using g_start_block_id, exactly as BCDirectCompute.cpp did). The binding
  // constraint is the 2-blocks-per-group passes (LE10/encode): batch/2 <= 65535.
  // Use a multiple of 4 so the 4-block-per-group G10 pass tiles evenly.
  const BATCH = 131068;

  for (let start = 0; start < paddedBlocks; start += BATCH) {
    const nb = Math.min(BATCH, paddedBlocks - start); // multiple of 4
    const groups4 = nb / 4; // G10: 4 blocks per workgroup
    const groups2 = nb / 2; // LE10 / encode: 2 blocks per workgroup

    // Pass 1: probe single-region modes 11..14 -> err1.
    dispatch(pipelines.tryModeG10, 0, err2, err1, groups4, start);

    // Pass 2: two-region modes 1..10 (g_mode_id 0..9), ping-pong err1<->err2.
    // mode_id 0: err1 -> err2, mode_id 1: err2 -> err1, ... mode_id 9: err2 -> err1.
    for (let i = 0; i < 10; ++i) {
      const inBuf = i & 1 ? err2 : err1;
      const outBuf = i & 1 ? err1 : err2;
      dispatch(pipelines.tryModeLE10, i, inBuf, outBuf, groups2, start);
    }

    // Pass 3: emit final blocks for this batch (best result is in err1).
    dispatch(pipelines.encode, 0, err1, output, groups2, start);
  }

  // --- Read back ---
  const readBytes = numBlocks * 16;
  const staging = device.createBuffer({
    size: readBytes,
    usage: BufferUsage.COPY_DST | BufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(output, 0, staging, 0, readBytes);
  device.queue.submit([enc.finish()]);

  const scopeError = await device.popErrorScope();
  if (scopeError) throw new Error(`compressBC6H: GPU validation error — ${scopeError.message}`);

  await staging.mapAsync(MapMode.READ);
  const mapped = new Uint8Array(staging.getMappedRange()).slice();
  staging.unmap();

  // Pack into the destination BC6H Image (block-raster order == buffer order).
  const dst = makeImage(opts.format, width, height);
  const { rowPitch } = computePitch(opts.format, width, height);
  // Buffer pitch and image rowPitch are both xblocks*16, so this is a flat copy.
  dst.pixels.set(mapped.subarray(0, Math.min(mapped.length, dst.pixels.length)));
  void rowPitch;

  // Cleanup transient GPU resources.
  output.destroy();
  err1.destroy();
  err2.destroy();
  uniform.destroy();
  staging.destroy();
  tex.destroy();

  return dst;
}
