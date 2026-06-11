//-------------------------------------------------------------------------------------
// transform.ts
//
// CPU image transforms mirroring texconv's pre/post-processing stages:
// flip (-hflip/-vflip), channel swizzle (-swizzle), colorkey (-c), and
// premultiply/straighten alpha (-pmalpha/-alpha). All operate on R8G8B8A8 images
// except flip, which is generic over pixel size (works on HDR float images too).
//-------------------------------------------------------------------------------------

import { DXGI_FORMAT, bitsPerPixel } from "./dxgi.js";
import { makeImage, type Image } from "./image.js";

function assertRGBA8(img: Image, op: string): void {
  if (img.format !== DXGI_FORMAT.R8G8B8A8_UNORM && img.format !== DXGI_FORMAT.R8G8B8A8_UNORM_SRGB) {
    throw new Error(`${op}: source must be R8G8B8A8_UNORM(_SRGB); got ${DXGI_FORMAT[img.format]}`);
  }
}

/** Mirror an image horizontally and/or vertically (texconv -hflip / -vflip). */
export function flipImage(src: Image, horizontal: boolean, vertical: boolean): Image {
  if (!horizontal && !vertical) return src;
  const bpp = bitsPerPixel(src.format) / 8;
  if (!bpp) throw new Error(`flipImage: unsupported format ${DXGI_FORMAT[src.format]}`);
  const dst = makeImage(src.format, src.width, src.height);
  for (let y = 0; y < src.height; y++) {
    const sy = vertical ? src.height - 1 - y : y;
    const srcRow = src.pixels.subarray(sy * src.rowPitch, sy * src.rowPitch + src.width * bpp);
    const dstRow = dst.pixels.subarray(y * dst.rowPitch);
    if (!horizontal) {
      dstRow.set(srcRow);
    } else {
      for (let x = 0; x < src.width; x++) {
        const sx = (src.width - 1 - x) * bpp;
        for (let b = 0; b < bpp; b++) dstRow[x * bpp + b] = srcRow[sx + b];
      }
    }
  }
  return dst;
}

/**
 * Parse a texconv -swizzle mask into per-channel sources. Accepts 1-4 chars from
 * rgba / xyzw / 0 / 1 (case-insensitive); like upstream, a short mask replicates
 * its last element into the remaining positions. Source codes: 0-3 = channel
 * index, 4 = constant 0, 5 = constant 1.
 */
export function parseSwizzle(mask: string): number[] {
  if (!mask || mask.length > 4) throw new Error(`swizzle: mask must be 1-4 chars; got '${mask}'`);
  const codes: Record<string, number> = { r: 0, x: 0, g: 1, y: 1, b: 2, z: 2, a: 3, w: 3, "0": 4, "1": 5 };
  const out = [0, 1, 2, 3];
  let last = 0;
  for (let j = 0; j < 4; j++) {
    if (j < mask.length) {
      const code = codes[mask[j].toLowerCase()];
      if (code === undefined) throw new Error(`swizzle: invalid mask char '${mask[j]}'`);
      last = code;
    }
    out[j] = last;
  }
  return out;
}

/** Rearrange channels per a parsed swizzle (texconv -swizzle). */
export function swizzleImage(src: Image, mask: string): Image {
  assertRGBA8(src, "swizzleImage");
  const sel = parseSwizzle(mask);
  const dst = makeImage(src.format, src.width, src.height);
  for (let y = 0; y < src.height; y++) {
    const s = y * src.rowPitch;
    const d = y * dst.rowPitch;
    for (let x = 0; x < src.width; x++) {
      for (let c = 0; c < 4; c++) {
        const code = sel[c];
        dst.pixels[d + x * 4 + c] =
          code === 4 ? 0 : code === 5 ? 255 : src.pixels[s + x * 4 + code];
      }
    }
  }
  return dst;
}

/**
 * Colorkey (texconv -c): pixels whose RGB is within upstream's 0.2 tolerance of
 * `key` (0xRRGGBB) become transparent black; all other pixels get opaque alpha,
 * exactly as texconv's TransformImage lambda does.
 */
export function colorKeyImage(src: Image, key: number): Image {
  assertRGBA8(src, "colorKeyImage");
  const kr = (key >> 16) & 0xff;
  const kg = (key >> 8) & 0xff;
  const kb = key & 0xff;
  const tol = 0.2 * 255;
  const dst = makeImage(src.format, src.width, src.height);
  for (let y = 0; y < src.height; y++) {
    const s = y * src.rowPitch;
    const d = y * dst.rowPitch;
    for (let x = 0; x < src.width; x++) {
      const i = s + x * 4;
      const o = d + x * 4;
      if (
        Math.abs(src.pixels[i] - kr) <= tol &&
        Math.abs(src.pixels[i + 1] - kg) <= tol &&
        Math.abs(src.pixels[i + 2] - kb) <= tol
      ) {
        dst.pixels[o] = 0; dst.pixels[o + 1] = 0; dst.pixels[o + 2] = 0; dst.pixels[o + 3] = 0;
      } else {
        dst.pixels[o] = src.pixels[i];
        dst.pixels[o + 1] = src.pixels[i + 1];
        dst.pixels[o + 2] = src.pixels[i + 2];
        dst.pixels[o + 3] = 255;
      }
    }
  }
  return dst;
}

/** Premultiply color by alpha (texconv -pmalpha) or undo it (texconv -alpha). */
export function premultiplyAlpha(src: Image, undo = false): Image {
  assertRGBA8(src, "premultiplyAlpha");
  const dst = makeImage(src.format, src.width, src.height);
  for (let y = 0; y < src.height; y++) {
    const s = y * src.rowPitch;
    const d = y * dst.rowPitch;
    for (let x = 0; x < src.width; x++) {
      const i = s + x * 4;
      const o = d + x * 4;
      const a = src.pixels[i + 3];
      const scale = undo ? (a === 0 ? 0 : 255 / a) : a / 255;
      for (let c = 0; c < 3; c++) {
        dst.pixels[o + c] = Math.min(255, Math.round(src.pixels[i + c] * scale));
      }
      dst.pixels[o + 3] = a;
    }
  }
  return dst;
}
