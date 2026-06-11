//-------------------------------------------------------------------------------------
// texconv.ts
//
// High-level conversion entry point, modeling a subset of the texconv pipeline:
//   load -> (resize) -> (generate mips) -> (convert/compress) -> assemble ScratchImage
//
// The GPU block-compression stages are wired in from later milestones; M1 supports
// the uncompressed passthrough path so the end-to-end load->DDS flow runs today.
//-------------------------------------------------------------------------------------

import { DXGI_FORMAT, isCompressed } from "./dxgi.js";
import { ScratchImage, type Image } from "./image.js";

export interface ConvertOptions {
  /** Target DXGI format (texconv -f). */
  format: DXGI_FORMAT;
  /** Generate a full mip chain (texconv -m 0) or a fixed count. 1 = base only. */
  mipLevels?: number;
  /** Resize target (texconv -w / -h). Omit to keep source size. */
  width?: number;
  height?: number;
  /** Treat input as sRGB / write output as sRGB (texconv -srgbi/-srgbo/-srgb). */
  srgbIn?: boolean;
  srgbOut?: boolean;
  /** BC7 alpha weighting (texconv -aw). */
  alphaWeight?: number;
}

/**
 * Convert a source Image to the requested format/shape. Throws for stages not yet
 * implemented so callers get a clear signal rather than silent wrong output.
 */
export async function convert(source: Image, options: ConvertOptions): Promise<ScratchImage> {
  if (options.width || options.height) {
    throw new Error("convert: resize not yet implemented (M5)");
  }
  if ((options.mipLevels ?? 1) !== 1) {
    throw new Error("convert: mip generation not yet implemented (M5)");
  }

  if (isCompressed(options.format)) {
    throw new Error(
      `convert: GPU block compression to ${DXGI_FORMAT[options.format]} not yet wired (M2+)`,
    );
  }

  if (options.format === source.format) {
    return ScratchImage.from2D(source);
  }

  if (
    (source.format === DXGI_FORMAT.R8G8B8A8_UNORM &&
      options.format === DXGI_FORMAT.R8G8B8A8_UNORM_SRGB) ||
    (source.format === DXGI_FORMAT.R8G8B8A8_UNORM_SRGB &&
      options.format === DXGI_FORMAT.R8G8B8A8_UNORM)
  ) {
    // Reinterpret-only: same bytes, different color-space tag.
    const reinterpreted: Image = { ...source, format: options.format };
    return ScratchImage.from2D(reinterpreted);
  }

  throw new Error(
    `convert: uncompressed conversion ${DXGI_FORMAT[source.format]} -> ${DXGI_FORMAT[options.format]} not yet implemented`,
  );
}
