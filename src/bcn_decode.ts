//-------------------------------------------------------------------------------------
// bcn_decode.ts
//
// Reference decoders for BC1/BC2/BC3/BC4/BC5, used for verification and preview.
// Matches the D3D/DirectXTex decode (and thus the bcn.wgsl encoder palettes).
//-------------------------------------------------------------------------------------

import { DXGI_FORMAT } from "./dxgi.js";
import { makeImage, type Image } from "./image.js";

function expand565(w: number): [number, number, number] {
  const r = (w >> 11) & 31;
  const g = (w >> 5) & 63;
  const b = w & 31;
  return [(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)];
}

/** Decode a BC1 color block (8 bytes at offset) into 16 RGBA pixels. */
function decodeColorBlock(bytes: Uint8Array, offset: number, out: Uint8Array, forceFourColor: boolean): void {
  const c0 = bytes[offset] | (bytes[offset + 1] << 8);
  const c1 = bytes[offset + 2] | (bytes[offset + 3] << 8);
  const e0 = expand565(c0);
  const e1 = expand565(c1);
  const pal: [number, number, number, number][] = [
    [e0[0], e0[1], e0[2], 255],
    [e1[0], e1[1], e1[2], 255],
    [0, 0, 0, 255],
    [0, 0, 0, 255],
  ];
  if (c0 > c1 || forceFourColor) {
    for (let i = 0; i < 3; i++) {
      pal[2][i] = Math.round((2 * e0[i] + e1[i]) / 3);
      pal[3][i] = Math.round((e0[i] + 2 * e1[i]) / 3);
    }
  } else {
    for (let i = 0; i < 3; i++) pal[2][i] = Math.round((e0[i] + e1[i]) / 2);
    pal[3] = [0, 0, 0, 0]; // transparent black
  }
  const idx = bytes[offset + 4] | (bytes[offset + 5] << 8) | (bytes[offset + 6] << 16) | (bytes[offset + 7] << 24);
  for (let p = 0; p < 16; p++) {
    const e = pal[(idx >>> (p * 2)) & 3];
    const o = p * 4;
    out[o] = e[0]; out[o + 1] = e[1]; out[o + 2] = e[2]; out[o + 3] = e[3];
  }
}

/** Decode a BC4 block (8 bytes) into 16 channel values [0,255]. */
function decodeAlphaBlock(bytes: Uint8Array, offset: number, out: number[]): void {
  const r0 = bytes[offset];
  const r1 = bytes[offset + 1];
  const pal = new Array<number>(8);
  pal[0] = r0;
  pal[1] = r1;
  if (r0 > r1) {
    for (let k = 1; k < 7; k++) pal[k + 1] = (r0 * (7 - k) + r1 * k) / 7;
  } else {
    for (let k = 1; k < 5; k++) pal[k + 1] = (r0 * (5 - k) + r1 * k) / 5;
    pal[6] = 0;
    pal[7] = 255;
  }
  // 48 bits of 3-bit indices begin at bit 16 of the 64-bit block.
  let lo = (bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16) | (bytes[offset + 3] << 24)) >>> 0;
  let hi = (bytes[offset + 4] | (bytes[offset + 5] << 8) | (bytes[offset + 6] << 16) | (bytes[offset + 7] << 24)) >>> 0;
  for (let p = 0; p < 16; p++) {
    const bitpos = 16 + p * 3;
    let v: number;
    if (bitpos < 32) {
      v = (lo >>> bitpos) & 7;
      if (bitpos + 3 > 32) v |= (hi << (32 - bitpos)) & 7;
    } else {
      v = (hi >>> (bitpos - 32)) & 7;
    }
    out[p] = Math.round(pal[v]);
  }
}

type Decoder = (bytes: Uint8Array, blockOffset: number, out: Uint8Array) => void;

const block16 = new Uint8Array(64);
const alpha16: number[] = new Array(16);

function makeBlockDecoder(format: DXGI_FORMAT): { decoder: Decoder; bytes: number } {
  switch (format) {
    case DXGI_FORMAT.BC1_UNORM:
    case DXGI_FORMAT.BC1_UNORM_SRGB:
      return { bytes: 8, decoder: (b, o, out) => decodeColorBlock(b, o, out, false) };
    case DXGI_FORMAT.BC2_UNORM:
    case DXGI_FORMAT.BC2_UNORM_SRGB:
      return {
        bytes: 16,
        decoder: (b, o, out) => {
          decodeColorBlock(b, o + 8, out, true);
          for (let p = 0; p < 16; p++) {
            const nib = (b[o + (p >> 1)] >> ((p & 1) * 4)) & 0xf;
            out[p * 4 + 3] = (nib << 4) | nib;
          }
        },
      };
    case DXGI_FORMAT.BC3_UNORM:
    case DXGI_FORMAT.BC3_UNORM_SRGB:
      return {
        bytes: 16,
        decoder: (b, o, out) => {
          decodeColorBlock(b, o + 8, out, true);
          decodeAlphaBlock(b, o, alpha16);
          for (let p = 0; p < 16; p++) out[p * 4 + 3] = alpha16[p];
        },
      };
    case DXGI_FORMAT.BC4_UNORM:
      return {
        bytes: 8,
        decoder: (b, o, out) => {
          decodeAlphaBlock(b, o, alpha16);
          for (let p = 0; p < 16; p++) {
            out[p * 4] = alpha16[p];
            out[p * 4 + 1] = 0;
            out[p * 4 + 2] = 0;
            out[p * 4 + 3] = 255;
          }
        },
      };
    case DXGI_FORMAT.BC5_UNORM:
      return {
        bytes: 16,
        decoder: (b, o, out) => {
          decodeAlphaBlock(b, o, alpha16);
          const r = alpha16.slice();
          decodeAlphaBlock(b, o + 8, alpha16);
          for (let p = 0; p < 16; p++) {
            out[p * 4] = r[p];
            out[p * 4 + 1] = alpha16[p];
            out[p * 4 + 2] = 0;
            out[p * 4 + 3] = 255;
          }
        },
      };
    default:
      throw new Error(`bcn_decode: unsupported format ${DXGI_FORMAT[format]}`);
  }
}

/** Decode a BC1/2/3/4/5 image to R8G8B8A8_UNORM. */
export function decodeBCn(src: Image): Image {
  const { width, height } = src;
  const { decoder, bytes } = makeBlockDecoder(src.format);
  const dst = makeImage(DXGI_FORMAT.R8G8B8A8_UNORM, width, height);
  const xblocks = Math.max(1, (width + 3) >> 2);
  for (let by = 0; by * 4 < height; by++) {
    for (let bx = 0; bx < xblocks; bx++) {
      decoder(src.pixels, (by * xblocks + bx) * bytes, block16);
      for (let py = 0; py < 4; py++) {
        const y = by * 4 + py;
        if (y >= height) break;
        for (let px = 0; px < 4; px++) {
          const x = bx * 4 + px;
          if (x >= width) break;
          const d = y * dst.rowPitch + x * 4;
          const s = (py * 4 + px) * 4;
          dst.pixels[d] = block16[s];
          dst.pixels[d + 1] = block16[s + 1];
          dst.pixels[d + 2] = block16[s + 2];
          dst.pixels[d + 3] = block16[s + 3];
        }
      }
    }
  }
  return dst;
}
