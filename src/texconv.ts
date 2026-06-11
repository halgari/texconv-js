//-------------------------------------------------------------------------------------
// texconv.ts
//
// High-level conversion entry point, modeling a subset of the texconv pipeline:
//   load -> (resize) -> (generate mips) -> (convert/compress per level) -> ScratchImage
//
// All resize/mip/compression stages run on the GPU (WebGPU). Uncompressed output is
// limited to the R8G8B8A8 family for now (the common browser path).
//-------------------------------------------------------------------------------------

import { DXGI_FORMAT, isCompressed, isSRGB } from "./dxgi.js";
import { ScratchImage, type Image } from "./image.js";
import { getDevice } from "./gpu/device.js";
import { compressBC7 } from "./gpu/compressBC.js";
import { compressBCn } from "./gpu/compressBCn.js";
import { resizeImage, generateMipChain } from "./gpu/resize.js";

export interface ConvertOptions {
  /** Target DXGI format (texconv -f). */
  format: DXGI_FORMAT;
  /** Mip levels: 1 = base only (default), 0 = full chain, N = first N levels. */
  mipLevels?: number;
  /** Resize target (texconv -w / -h). Omit to keep source size. */
  width?: number;
  height?: number;
  /**
   * Treat the source data as sRGB when filtering (resize/mips). Defaults to the
   * source format's sRGB-ness; set explicitly to override (texconv -srgbi).
   */
  srgbFilter?: boolean;
  /** BC7 alpha weighting (texconv -aw). */
  alphaWeight?: number;
  /** BC7 quick mode — modes 4/5/6 only (texconv -bc q). */
  bc7Quick?: boolean;
  /** BC7 also try 3-subset modes 0/2 (texconv -bc x). */
  bc7Use3Subsets?: boolean;
}

const R8_FAMILY = new Set([DXGI_FORMAT.R8G8B8A8_UNORM, DXGI_FORMAT.R8G8B8A8_UNORM_SRGB]);

/**
 * Convert a source Image to the requested format/shape. The source must be in the
 * R8G8B8A8 family (what the browser image loader produces).
 */
export async function convert(source: Image, options: ConvertOptions): Promise<ScratchImage> {
  if (!R8_FAMILY.has(source.format)) {
    throw new Error(`convert: source must be R8G8B8A8(_SRGB); got ${DXGI_FORMAT[source.format]}`);
  }

  const wantResize = !!(options.width || options.height);
  const mipLevels = options.mipLevels ?? 1;
  const wantMips = mipLevels !== 1;
  const srgbFilter = options.srgbFilter ?? isSRGB(source.format);

  const needGPU = wantResize || wantMips || isCompressed(options.format);
  const device = needGPU ? await getDevice() : null;

  // 1. Resize (if requested).
  let base = source;
  if (wantResize && device) {
    const dstW = options.width ?? source.width;
    const dstH = options.height ?? source.height;
    base = await resizeImage(device, source, dstW, dstH, srgbFilter);
  }

  // 2. Mip chain.
  let levels: Image[];
  if (wantMips && device) {
    levels = await generateMipChain(device, base, mipLevels, srgbFilter);
  } else {
    levels = [base];
  }

  // 3. Convert/compress each level to the target format.
  const out: Image[] = [];
  for (const lvl of levels) {
    out.push(await convertLevel(lvl, options, device));
  }

  return out.length === 1 ? ScratchImage.from2D(out[0]) : ScratchImage.fromMipChain(out);
}

/** Convert a single R8G8B8A8 level to the target format. */
async function convertLevel(
  level: Image,
  options: ConvertOptions,
  device: GPUDevice | null,
): Promise<Image> {
  const { format } = options;

  if (isCompressed(format)) {
    if (!device) throw new Error("convert: GPU device required for compression");
    return compressGPU(device, level, options);
  }

  if (format === level.format) return level;

  if (R8_FAMILY.has(format)) {
    // Reinterpret-only: same bytes, different color-space tag.
    return { ...level, format };
  }

  throw new Error(
    `convert: uncompressed conversion ${DXGI_FORMAT[level.format]} -> ${DXGI_FORMAT[format]} not yet implemented`,
  );
}

/** Route a compressed-target conversion to the appropriate GPU encoder. */
async function compressGPU(device: GPUDevice, level: Image, options: ConvertOptions): Promise<Image> {
  switch (options.format) {
    case DXGI_FORMAT.BC7_UNORM:
    case DXGI_FORMAT.BC7_UNORM_SRGB:
      return compressBC7(device, level, {
        format: options.format,
        alphaWeight: options.alphaWeight ?? 1,
        quick: options.bc7Quick ?? false,
        use3Subsets: options.bc7Use3Subsets ?? false,
      });
    case DXGI_FORMAT.BC1_UNORM:
    case DXGI_FORMAT.BC1_UNORM_SRGB:
    case DXGI_FORMAT.BC2_UNORM:
    case DXGI_FORMAT.BC2_UNORM_SRGB:
    case DXGI_FORMAT.BC3_UNORM:
    case DXGI_FORMAT.BC3_UNORM_SRGB:
    case DXGI_FORMAT.BC4_UNORM:
    case DXGI_FORMAT.BC5_UNORM:
      return compressBCn(device, level, options.format);
    default:
      throw new Error(
        `convert: GPU compression to ${DXGI_FORMAT[options.format]} not yet implemented (BC6H deferred)`,
      );
  }
}
