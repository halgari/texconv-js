//-------------------------------------------------------------------------------------
// gpu/device.ts
//
// WebGPU device acquisition that works both in the browser (navigator.gpu) and in
// headless Node (the Dawn-based `webgpu` package, used for tests).
//-------------------------------------------------------------------------------------

let cached: Promise<GPUDevice> | null = null;

/** Obtain the platform GPU object (browser navigator.gpu, or Dawn in Node). */
async function getGPU(): Promise<GPU> {
  if (typeof navigator !== "undefined" && navigator.gpu) {
    return navigator.gpu;
  }
  // Headless Node fallback (tests). Imported dynamically so browser bundles never
  // try to resolve the native module.
  try {
    const mod: any = await import(/* @vite-ignore */ "webgpu");
    const create = mod.create ?? mod.default?.create;
    if (typeof create === "function") {
      return create([]) as GPU;
    }
  } catch {
    /* fall through to error below */
  }
  throw new Error("WebGPU is not available (no navigator.gpu and no headless `webgpu` module)");
}

/**
 * Acquire (and memoize) a GPUDevice. Pass `fresh: true` to bypass the cache, e.g.
 * if a prior device was lost.
 */
export async function getDevice(fresh = false): Promise<GPUDevice> {
  if (cached && !fresh) return cached;
  cached = (async () => {
    const gpu = await getGPU();
    // Try progressively weaker requests: a real high-perf adapter, then any
    // adapter, then the CPU fallback adapter. On Linux Chromium a null adapter
    // usually means WebGPU/Vulkan is still gated behind flags.
    const adapter =
      (await gpu.requestAdapter({ powerPreference: "high-performance" })) ??
      (await gpu.requestAdapter()) ??
      (await gpu.requestAdapter({ forceFallbackAdapter: true }));
    if (!adapter) {
      throw new Error(
        "WebGPU reported support (navigator.gpu) but no GPU adapter is available. " +
          "On Linux Chromium this is usually a flag/driver issue — enable " +
          "chrome://flags/#enable-unsafe-webgpu (and #enable-vulkan), restart, and " +
          "check chrome://gpu shows WebGPU as enabled. See the README troubleshooting section.",
      );
    }
    const device = await adapter.requestDevice();
    device.lost.then(() => {
      // Drop the cache so the next call re-acquires.
      if (cached) cached = null;
    });
    return device;
  })();
  return cached;
}

/** Round `value` up to a multiple of `align`. */
export function alignUp(value: number, align: number): number {
  return Math.ceil(value / align) * align;
}

// WebGPU bitflag constants. In browsers these exist as globals (GPUBufferUsage,
// etc.), but the headless Dawn module doesn't define them globally, so we use the
// spec's fixed numeric values for portability.
export const ShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 } as const;
export const BufferUsage = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
} as const;
export const TextureUsage = {
  COPY_SRC: 0x01,
  COPY_DST: 0x02,
  TEXTURE_BINDING: 0x04,
  STORAGE_BINDING: 0x08,
  RENDER_ATTACHMENT: 0x10,
} as const;
export const MapMode = { READ: 0x0001, WRITE: 0x0002 } as const;
