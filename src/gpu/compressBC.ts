//-------------------------------------------------------------------------------------
// gpu/compressBC.ts
//
// WebGPU port of DirectXTex's BCDirectCompute.cpp (GPUCompressBC) for BC7.
// Runs the four-pass encoder (TryMode456 -> TryMode137 -> TryMode02 -> EncodeBlock)
// with ping-pong error buffers.
//
// Differences from the D3D11 original, all forced by WebGPU semantics:
//   * The REF_DEVICE (barrier) shader path is used, so storage buffers are padded
//     to a multiple of 4 blocks; over-dispatched edge threads then write into
//     padding rather than out of bounds. All blocks dispatch at once (no 64-batch).
//   * Per-dispatch uniforms are uploaded with writeBuffer + one submit each.
//-------------------------------------------------------------------------------------

import { DXGI_FORMAT, computePitch } from "../dxgi.js";
import { makeImage, type Image } from "../image.js";
import { ShaderStage, BufferUsage, TextureUsage, MapMode } from "./device.js";
import bc7wgsl from "./bc7.wgsl?raw";

export interface CompressBCOptions {
  /** Target format: BC7_UNORM or BC7_UNORM_SRGB. */
  format: DXGI_FORMAT;
  /** BC7 alpha weight (texconv -aw). Default 1. */
  alphaWeight?: number;
  /** Skip the 1/3/7 and 0/2 multi-subset passes (TEX_COMPRESS_BC7_QUICK). */
  quick?: boolean;
  /** Also try the 3-subset modes 0/2 (TEX_COMPRESS_BC7_USE_3SUBSETS). */
  use3Subsets?: boolean;
}

const PASS = {
  tryMode456: "TryMode456CS",
  tryMode137: "TryMode137CS",
  tryMode02: "TryMode02CS",
  encode: "EncodeBlockCS",
} as const;

type ModulePipelines = Record<keyof typeof PASS, GPUComputePipeline>;

const moduleCache = new WeakMap<GPUDevice, { layout: GPUBindGroupLayout; pipelines: ModulePipelines }>();

function getPipelines(device: GPUDevice) {
  let cached = moduleCache.get(device);
  if (cached) return cached;

  const module = device.createShaderModule({ code: bc7wgsl, label: "bc7-encode" });

  const layout = device.createBindGroupLayout({
    label: "bc7-bgl",
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
    tryMode456: mk(PASS.tryMode456),
    tryMode137: mk(PASS.tryMode137),
    tryMode02: mk(PASS.tryMode02),
    encode: mk(PASS.encode),
  };

  cached = { layout, pipelines };
  moduleCache.set(device, cached);
  return cached;
}

/** Encode an R8G8B8A8 image to BC7 on the GPU. Returns the BC7 Image (mip 0). */
export async function compressBC7(
  device: GPUDevice,
  src: Image,
  opts: CompressBCOptions,
): Promise<Image> {
  if (src.format !== DXGI_FORMAT.R8G8B8A8_UNORM && src.format !== DXGI_FORMAT.R8G8B8A8_UNORM_SRGB) {
    throw new Error("compressBC7: source must be R8G8B8A8_UNORM(_SRGB)");
  }
  const alphaWeight = opts.alphaWeight ?? 1;
  const quick = opts.quick ?? false;
  const mode137 = !quick;
  const mode02 = !quick && (opts.use3Subsets ?? false);

  const { width, height } = src;
  const xblocks = Math.max(1, (width + 3) >> 2);
  const yblocks = Math.max(1, (height + 3) >> 2);
  const numBlocks = xblocks * yblocks;
  const paddedBlocks = Math.ceil(numBlocks / 4) * 4; // 4 blocks per group in 456/encode
  const bufBytes = paddedBlocks * 16;

  const { layout, pipelines } = getPipelines(device);

  // Capture validation errors as a thrown exception rather than a silent async
  // failure (e.g. an over-cap dispatch would otherwise produce garbage quietly).
  device.pushErrorScope("validation");

  // --- Input texture (rgba8unorm; no implicit sRGB conversion) ---
  const tex = device.createTexture({
    size: { width, height },
    format: "rgba8unorm",
    usage: TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,
  });
  // writeTexture needs rows aligned only for the source data we provide (tight is fine).
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
  const output = mkStorage("bc7-output");
  const err1 = mkStorage("bc7-err1");
  const err2 = mkStorage("bc7-err2");

  const uniform = device.createBuffer({
    size: 32,
    usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
  });
  const uniformData = new ArrayBuffer(32);
  const uniformView = new DataView(uniformData);

  const formatCode = opts.format; // g_format (unused by the BC7 shader math, kept for parity)

  const setUniform = (modeId: number, startBlock: number) => {
    uniformView.setUint32(0, width, true); // g_tex_width
    uniformView.setUint32(4, xblocks, true); // g_num_block_x
    uniformView.setUint32(8, formatCode, true); // g_format
    uniformView.setUint32(12, modeId, true); // g_mode_id
    uniformView.setUint32(16, startBlock, true); // g_start_block_id
    uniformView.setUint32(20, numBlocks, true); // g_num_total_blocks
    uniformView.setFloat32(24, alphaWeight, true); // g_alpha_weight
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
  // constraint is the 1-block-per-group passes (137/02): batch must be <= 65535
  // blocks. Use a multiple of 4 so the 4-block-per-group passes tile evenly.
  const BATCH = 65532;
  const finalErr = mode137 || mode02 ? err2 : err1;

  for (let start = 0; start < paddedBlocks; start += BATCH) {
    const nb = Math.min(BATCH, paddedBlocks - start); // multiple of 4
    const groups4 = nb / 4; // 456 / encode: 4 blocks per workgroup
    const groups1 = nb; // 137 / 02: 1 block per workgroup

    // Pass 1: modes 4/5/6 -> err1
    dispatch(pipelines.tryMode456, 0, err2, err1, groups4, start);

    // Pass 2: modes 1, 3, 7 (ping-pong err1<->err2)
    if (mode137) {
      dispatch(pipelines.tryMode137, 1, err1, err2, groups1, start);
      dispatch(pipelines.tryMode137, 3, err2, err1, groups1, start);
      dispatch(pipelines.tryMode137, 7, err1, err2, groups1, start);
    }

    // Pass 3: modes 0, 2 (3-subset)
    if (mode02) {
      dispatch(pipelines.tryMode02, 0, err2, err1, groups1, start);
      dispatch(pipelines.tryMode02, 2, err1, err2, groups1, start);
    }

    // Pass 4: emit final blocks for this batch.
    dispatch(pipelines.encode, 0, finalErr, output, groups4, start);
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
  if (scopeError) throw new Error(`compressBC7: GPU validation error — ${scopeError.message}`);

  await staging.mapAsync(MapMode.READ);
  const mapped = new Uint8Array(staging.getMappedRange()).slice();
  staging.unmap();

  // Pack into the destination BC7 Image (block-raster order == buffer order).
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
