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
import { compressBC6H } from "./gpu/compressBC6.js";
import { resizeImage, generateMipChain } from "./gpu/resize.js";
import { rgba8ToLinearFloat } from "./hdr.js";
import { colorKeyImage, flipImage, premultiplyAlpha, swizzleImage } from "./transform.js";

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
  /** Mirror horizontally / vertically (texconv -hflip / -vflip). */
  hflip?: boolean;
  vflip?: boolean;
  /** Channel swizzle mask, e.g. "abgr" or "rgb1" (texconv -swizzle). */
  swizzle?: string;
  /** Colorkey 0xRRGGBB: matching pixels become transparent black (texconv -c). */
  colorKey?: number;
  /** Premultiply color by alpha after mip generation (texconv -pmalpha). */
  premultiplyAlpha?: boolean;
  /** Undo premultiplied alpha on the source (texconv -alpha). */
  straightAlpha?: boolean;
}

const R8_FAMILY = new Set([DXGI_FORMAT.R8G8B8A8_UNORM, DXGI_FORMAT.R8G8B8A8_UNORM_SRGB]);
const BC6H = new Set([DXGI_FORMAT.BC6H_UF16, DXGI_FORMAT.BC6H_SF16]);

/**
 * Convert a source Image to the requested format/shape. The source must be in the
 * R8G8B8A8 family (what the browser image loader produces) or linear
 * R32G32B32A32_FLOAT (what the .hdr loader produces; only BC6H targets supported).
 */
export async function convert(source: Image, options: ConvertOptions): Promise<ScratchImage> {
  const floatSource = source.format === DXGI_FORMAT.R32G32B32A32_FLOAT;
  if (!floatSource && !R8_FAMILY.has(source.format)) {
    throw new Error(
      `convert: source must be R8G8B8A8(_SRGB) or R32G32B32A32_FLOAT; got ${DXGI_FORMAT[source.format]}`,
    );
  }
  if (floatSource && !BC6H.has(options.format)) {
    throw new Error(
      `convert: HDR float source currently supports only BC6H targets; got ${DXGI_FORMAT[options.format]}`,
    );
  }

  // The swizzle/colorkey/premultiply transforms are RGBA8-only. Reject the
  // combinations that would otherwise fail deep inside a transform with an
  // opaque assertion: any of them on a float source, and premultiply on a BC6H
  // target (where it runs after the linear-float expansion below). flip is
  // format-generic and always allowed.
  const rgba8OnlyTransform =
    !!options.swizzle ||
    options.colorKey !== undefined ||
    !!options.premultiplyAlpha ||
    !!options.straightAlpha;
  if (floatSource && rgba8OnlyTransform) {
    throw new Error(
      "convert: -swizzle/-c/-pmalpha/-alpha are not supported for HDR float sources (RGBA8-only transforms)",
    );
  }
  if (BC6H.has(options.format) && options.premultiplyAlpha) {
    throw new Error(
      "convert: -pmalpha is not supported for BC6H targets (premultiply runs in linear float after expansion)",
    );
  }
  // CPU transforms, in upstream texconv's relative order (undo-pmalpha, flip,
  // swizzle, colorkey). Deviation: upstream resizes before swizzle/colorkey; we
  // apply them pre-resize so the RGBA8 and HDR-float paths share one code path —
  // only filtered-edge pixels can differ.
  let src = source;
  if (options.straightAlpha) src = premultiplyAlpha(src, true);
  if (options.hflip || options.vflip) src = flipImage(src, !!options.hflip, !!options.vflip);
  if (options.swizzle) src = swizzleImage(src, options.swizzle);
  if (options.colorKey !== undefined) src = colorKeyImage(src, options.colorKey);

  // BC6H is a linear HDR format: expand 8-bit sources to linear float up front so
  // resize/mips/encode all run in linear light (texconv converts to float the same way).
  if (!floatSource && BC6H.has(options.format)) {
    src = rgba8ToLinearFloat(src, options.srgbFilter ?? true);
  }

  const wantResize = !!(options.width || options.height);
  const mipLevels = options.mipLevels ?? 1;
  const wantMips = mipLevels !== 1;
  const srgbFilter = options.srgbFilter ?? isSRGB(src.format);

  const needGPU = wantResize || wantMips || isCompressed(options.format);
  const device = needGPU ? await getDevice() : null;

  // 1. Resize (if requested).
  let base = src;
  if (wantResize && device) {
    const dstW = options.width ?? src.width;
    const dstH = options.height ?? src.height;
    base = await resizeImage(device, src, dstW, dstH, srgbFilter);
  }

  // 2. Mip chain.
  let levels: Image[];
  if (wantMips && device) {
    levels = await generateMipChain(device, base, mipLevels, srgbFilter);
  } else {
    levels = [base];
  }

  // 3. Convert/compress each level to the target format. Premultiplication runs
  // after mip generation (matching upstream texconv's stage order).
  const out: Image[] = [];
  for (let lvl of levels) {
    if (options.premultiplyAlpha) lvl = premultiplyAlpha(lvl);
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

  if (format === DXGI_FORMAT.B8G8R8A8_UNORM || format === DXGI_FORMAT.B8G8R8A8_UNORM_SRGB) {
    // Same layout as RGBA8 with red/blue swapped (texconv -f BGRA).
    return { ...swizzleImage(level, "bgra"), format };
  }

  throw new Error(
    `convert: uncompressed conversion ${DXGI_FORMAT[level.format]} -> ${DXGI_FORMAT[format]} not yet implemented`,
  );
}

/** Route a compressed-target conversion to the appropriate GPU encoder. */
async function compressGPU(device: GPUDevice, level: Image, options: ConvertOptions): Promise<Image> {
  switch (options.format) {
    case DXGI_FORMAT.BC6H_UF16:
    case DXGI_FORMAT.BC6H_SF16:
      return compressBC6H(device, level, { format: options.format });
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
        `convert: GPU compression to ${DXGI_FORMAT[options.format]} not yet implemented`,
      );
  }
}
