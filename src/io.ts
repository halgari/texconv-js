//-------------------------------------------------------------------------------------
// io.ts
//
// Browser-native image decoding. PNG/JPG/BMP/WebP/GIF are decoded by the browser
// (createImageBitmap) and read back as RGBA8 via an OffscreenCanvas, so no WIC port
// is needed for the MVP input path.
//-------------------------------------------------------------------------------------

import { DXGI_FORMAT, computePitch } from "./dxgi.js";
import type { Image } from "./image.js";

export interface LoadOptions {
  /**
   * Color-manage the decode to sRGB ("default", the standard) so images carrying a
   * Display-P3 / Adobe-RGB profile (common in phone photos, screenshots, many PNGs)
   * decode to correct sRGB values. Set false to read the raw encoded bytes ("none")
   * — only correct when the source is already plain sRGB and you want byte-exact
   * texconv/WIC-style passthrough.
   */
  colorManage?: boolean;
}

/** Decode a PNG/JPG/etc. blob to an R8G8B8A8_UNORM Image (mip 0). */
export async function loadImageRGBA8(
  source: Blob | ImageBitmapSource,
  opts: LoadOptions = {},
): Promise<Image> {
  const bitmap = await createImageBitmap(source as ImageBitmapSource, {
    premultiplyAlpha: "none",
    colorSpaceConversion: opts.colorManage === false ? "none" : "default",
  });
  try {
    const { width, height } = bitmap;
    const rgba = drawToRGBA8(bitmap, width, height);
    const { rowPitch, slicePitch } = computePitch(DXGI_FORMAT.R8G8B8A8_UNORM, width, height);
    return { width, height, format: DXGI_FORMAT.R8G8B8A8_UNORM, rowPitch, slicePitch, pixels: rgba };
  } finally {
    bitmap.close();
  }
}

function drawToRGBA8(bitmap: ImageBitmap | HTMLCanvasElement, width: number, height: number): Uint8Array {
  const canvas =
    typeof OffscreenCanvas !== "undefined"
      ? new OffscreenCanvas(width, height)
      : Object.assign(document.createElement("canvas"), { width, height });
  const ctx = (canvas as OffscreenCanvas).getContext("2d", {
    willReadFrequently: true,
    colorSpace: "srgb",
  }) as OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null;
  if (!ctx) throw new Error("io: failed to acquire 2D context");
  ctx.drawImage(bitmap as CanvasImageSource, 0, 0);
  const imageData = ctx.getImageData(0, 0, width, height);
  return new Uint8Array(imageData.data.buffer.slice(0));
}

/** Wrap an existing RGBA8 byte array (e.g. from ImageData) as an Image. */
export function imageFromRGBA8(pixels: Uint8Array, width: number, height: number): Image {
  const { rowPitch, slicePitch } = computePitch(DXGI_FORMAT.R8G8B8A8_UNORM, width, height);
  if (pixels.byteLength < slicePitch) throw new Error("io: pixel buffer too small");
  return { width, height, format: DXGI_FORMAT.R8G8B8A8_UNORM, rowPitch, slicePitch, pixels };
}

/** Trigger a browser download of a byte buffer as a file. */
export function downloadBytes(bytes: Uint8Array, filename: string, mime = "application/octet-stream"): void {
  const blob = new Blob([bytes as BlobPart], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
