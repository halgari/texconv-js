//-------------------------------------------------------------------------------------
// hdr.ts
//
// Radiance RGBE (.hdr) reader — the standard HDR input for BC6H workflows (browsers
// cannot decode .hdr natively). Produces a linear-light R32G32B32A32_FLOAT Image.
// Mirrors DirectXTexHDR.cpp behavior for the common case: "-Y h +X w" orientation,
// 32-bit_rle_rgbe format, both new-style (scanline RLE) and flat/old-style data.
//-------------------------------------------------------------------------------------

import { DXGI_FORMAT } from "./dxgi.js";
import { makeImage, type Image } from "./image.js";

/** True if a filename looks like a Radiance HDR file. */
export function isHDRFile(name: string): boolean {
  return /\.hdr$/i.test(name);
}

/** Parse a Radiance .hdr file into a linear R32G32B32A32_FLOAT Image. */
export function loadHDR(data: Uint8Array): Image {
  let pos = 0;

  const readLine = (): string => {
    let end = pos;
    while (end < data.length && data[end] !== 0x0a) end++;
    const line = new TextDecoder().decode(data.subarray(pos, end));
    pos = end + 1;
    return line;
  };

  const magic = readLine();
  if (!magic.startsWith("#?RADIANCE") && !magic.startsWith("#?RGBE")) {
    throw new Error("HDR: missing #?RADIANCE signature");
  }

  let exposure = 1;
  for (;;) {
    const line = readLine();
    if (line === "") break; // blank line ends the header
    if (line.startsWith("#")) continue;
    const [key, value] = line.split("=", 2);
    if (key === "FORMAT" && value !== "32-bit_rle_rgbe") {
      throw new Error(`HDR: unsupported FORMAT '${value}'`);
    }
    if (key === "EXPOSURE") {
      const e = Number(value);
      if (e > 0) exposure *= e;
    }
  }

  const res = readLine().trim().split(/\s+/);
  // Standard orientation only: "-Y height +X width".
  if (res.length !== 4 || res[0] !== "-Y" || res[2] !== "+X") {
    throw new Error(`HDR: unsupported resolution line '${res.join(" ")}'`);
  }
  const height = Number(res[1]);
  const width = Number(res[3]);
  if (!(width > 0 && height > 0)) throw new Error("HDR: bad dimensions");

  const img = makeImage(DXGI_FORMAT.R32G32B32A32_FLOAT, width, height);
  const out = new Float32Array(img.pixels.buffer);
  const invExposure = 1 / exposure;
  const scanline = new Uint8Array(width * 4); // RGBE per pixel

  for (let y = 0; y < height; y++) {
    pos = readScanline(data, pos, scanline, width);
    let o = y * width * 4;
    for (let x = 0; x < width; x++) {
      const r = scanline[x * 4];
      const g = scanline[x * 4 + 1];
      const b = scanline[x * 4 + 2];
      const e = scanline[x * 4 + 3];
      // rgbe -> float: value = (mantissa) * 2^(e-136); zero exponent = black.
      const scale = e !== 0 ? Math.pow(2, e - 136) * invExposure : 0;
      out[o++] = r * scale;
      out[o++] = g * scale;
      out[o++] = b * scale;
      out[o++] = 1;
    }
  }
  return img;
}

/** Decode one scanline (RGBE interleaved into `dst`), returning the new read offset. */
function readScanline(data: Uint8Array, pos: number, dst: Uint8Array, width: number): number {
  if (pos + 4 > data.length) throw new Error("HDR: truncated file");

  // New-style RLE scanline: 0x02 0x02 then 16-bit width, then 4 component planes.
  if (data[pos] === 2 && data[pos + 1] === 2 && ((data[pos + 2] << 8) | data[pos + 3]) === width && width >= 8 && width < 32768) {
    pos += 4;
    for (let c = 0; c < 4; c++) {
      let x = 0;
      while (x < width) {
        if (pos >= data.length) throw new Error("HDR: truncated RLE scanline");
        let count = data[pos++];
        if (count > 128) {
          // Run of a single value.
          count -= 128;
          if (x + count > width) throw new Error("HDR: RLE run overflow");
          const v = data[pos++];
          for (let i = 0; i < count; i++) dst[(x + i) * 4 + c] = v;
        } else {
          // Literal span.
          if (count === 0 || x + count > width) throw new Error("HDR: bad RLE literal");
          for (let i = 0; i < count; i++) dst[(x + i) * 4 + c] = data[pos++];
        }
        x += count;
      }
    }
    return pos;
  }

  // Flat / old-style RLE: pixels in RGBE order; (1,1,1,n) repeats the previous pixel.
  let x = 0;
  let shift = 0;
  while (x < width) {
    if (pos + 4 > data.length) throw new Error("HDR: truncated scanline");
    const r = data[pos], g = data[pos + 1], b = data[pos + 2], e = data[pos + 3];
    pos += 4;
    if (r === 1 && g === 1 && b === 1) {
      // Old-style run: repeat previous pixel e<<shift times.
      if (x === 0) throw new Error("HDR: old-style run with no previous pixel");
      const count = e << shift;
      if (x + count > width) throw new Error("HDR: old-style run overflow");
      const px = (x - 1) * 4;
      for (let i = 0; i < count; i++) {
        dst[(x + i) * 4] = dst[px];
        dst[(x + i) * 4 + 1] = dst[px + 1];
        dst[(x + i) * 4 + 2] = dst[px + 2];
        dst[(x + i) * 4 + 3] = dst[px + 3];
      }
      x += count;
      shift += 8;
    } else {
      dst[x * 4] = r;
      dst[x * 4 + 1] = g;
      dst[x * 4 + 2] = b;
      dst[x * 4 + 3] = e;
      x++;
      shift = 0;
    }
  }
  return pos;
}

/**
 * Expand an R8G8B8A8 image to linear R32G32B32A32_FLOAT (sRGB-decoding the color
 * channels when `srgb`), e.g. to feed an LDR source to the BC6H encoder.
 */
export function rgba8ToLinearFloat(src: Image, srgb = true): Image {
  if (src.format !== DXGI_FORMAT.R8G8B8A8_UNORM && src.format !== DXGI_FORMAT.R8G8B8A8_UNORM_SRGB) {
    throw new Error("rgba8ToLinearFloat: source must be R8G8B8A8_UNORM(_SRGB)");
  }
  const img = makeImage(DXGI_FORMAT.R32G32B32A32_FLOAT, src.width, src.height);
  const out = new Float32Array(img.pixels.buffer);
  for (let y = 0; y < src.height; y++) {
    const row = y * src.rowPitch;
    let o = y * src.width * 4;
    for (let x = 0; x < src.width; x++) {
      const i = row + x * 4;
      for (let c = 0; c < 3; c++) {
        const v = src.pixels[i + c] / 255;
        out[o++] = srgb ? (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4)) : v;
      }
      out[o++] = src.pixels[i + 3] / 255;
    }
  }
  return img;
}

/**
 * Tonemap a linear float image to displayable R8G8B8A8 (Reinhard + sRGB encode) —
 * used by the demo to preview HDR/BC6H content. Not part of the conversion pipeline.
 */
export function tonemapToRGBA8(src: Image): Image {
  if (src.format !== DXGI_FORMAT.R32G32B32A32_FLOAT) {
    throw new Error("tonemapToRGBA8: source must be R32G32B32A32_FLOAT");
  }
  const img = makeImage(DXGI_FORMAT.R8G8B8A8_UNORM, src.width, src.height);
  const f = new Float32Array(src.pixels.buffer, src.pixels.byteOffset, src.width * src.height * 4);
  for (let i = 0; i < f.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const lin = Math.max(0, f[i + c]);
      const tone = lin / (1 + lin); // Reinhard
      const s = tone <= 0.0031308 ? tone * 12.92 : 1.055 * Math.pow(tone, 1 / 2.4) - 0.055;
      img.pixels[i + c] = Math.round(Math.min(1, Math.max(0, s)) * 255);
    }
    img.pixels[i + 3] = Math.round(Math.min(1, Math.max(0, f[i + 3])) * 255);
  }
  return img;
}
